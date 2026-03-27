"""Crazyflie Waypoint Navigation - PPO Training, Play & Evaluation.

Single-waypoint navigation with 5-phase state machine:
    TAKEOFF -> STABILIZE -> NAVIGATE -> HOVER -> LAND

Usage (from IsaacLab directory):
    cd ~/projects/isaac/IsaacLab
    source ~/projects/isaac/env_isaaclab/bin/activate

    # Train:
    python ~/Desktop/Lasitha/drone_rl_project/waypoint/waypoint_ppo.py \
        --mode train --num_envs 512 --max_iterations 1000 --headless

    # Play trained policy:
    python ~/Desktop/Lasitha/drone_rl_project/waypoint/waypoint_ppo.py \
        --mode play --checkpoint /path/to/model_1000.pt --num_envs 1

    # Evaluate:
    python ~/Desktop/Lasitha/drone_rl_project/waypoint/waypoint_ppo.py \
        --mode eval --checkpoint /path/to/model_1000.pt --num_episodes 50 --headless
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Crazyflie Waypoint Nav - PPO")
parser.add_argument("--mode", type=str, required=True, choices=["train", "play", "eval"])
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for play/eval/resume.")
parser.add_argument("--num_episodes", type=int, default=50, help="Episodes for eval mode.")
parser.add_argument("--start_pos", type=float, nargs=3, default=[-5.0, 0.0, 0.2])
parser.add_argument("--goal_pos", type=float, nargs=3, default=[5.0, 0.0, 0.2])
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after sim launch) ───────────────────────────────────────────────
import gymnasium as gym
import time
import torch
import numpy as np
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.markers import CUBOID_MARKER_CFG

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# ── Robot Config ─────────────────────────────────────────────────────────────
_crazyflie_spawn = CRAZYFLIE_CFG.spawn.copy()
_crazyflie_spawn.scale = (10.0, 10.0, 10.0)
_crazyflie_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=10.0,
    enable_gyroscopic_forces=False,
)
VISIBLE_CRAZYFLIE_CFG = CRAZYFLIE_CFG.replace(
    prim_path="/World/envs/env_.*/Robot",
    spawn=_crazyflie_spawn,
)


# ── Environment Config ───────────────────────────────────────────────────────

@configclass
class ForestNavEnvCfg(DirectRLEnvCfg):
    episode_length_s = 18.0
    decimation = 2
    action_space = 4
    observation_space = 26
    state_space = 0
    debug_vis = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=40.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = VISIBLE_CRAZYFLIE_CFG
    thrust_to_weight = 1.9
    moment_scale = 0.04

    # Low-level PID controller (frozen — not learned)
    max_velocity_xy: float = 2.0
    max_velocity_z: float = 1.0
    max_yaw_rate: float = 1.5
    pid_vel_kp: float = 0.25        # velocity error → desired tilt (rad per m/s)
    pid_att_kp: float = 6.0         # attitude error → torque (proportional)
    pid_att_kd: float = 1.0         # angular velocity damping (derivative)
    pid_vz_kp: float = 0.5          # vertical velocity error → thrust correction
    pid_yaw_kp: float = 0.4         # yaw rate error → yaw torque
    pid_max_tilt: float = 0.5       # max desired tilt angle (radians, ~28 deg)

    lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=3,
            vertical_fov_range=(-90.0, 0.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=90.0,
        ),
        max_distance=15.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    lidar_max_distance: float = 15.0

    goal_threshold: float = 0.5
    landing_speed_threshold: float = 1.0
    max_flight_height: float = 8.0
    min_flight_height: float = 0.05
    cruise_altitude: float = 1.5

    takeoff_altitude_tolerance: float = 0.5
    stabilize_duration: float = 1.0
    hover_position_tolerance: float = 0.5
    hover_duration: float = 2.0

    static_start_pos: tuple = (-5.0, 0.0, 0.2)
    static_goal_pos: tuple = (5.0, 0.0, 0.2)

    # Reward Scales
    takeoff_ascent_scale: float = 10.0
    takeoff_drift_penalty: float = -2.0
    stabilize_position_scale: float = 3.0
    stabilize_low_speed_scale: float = 2.0
    stabilize_altitude_scale: float = -2.0
    stabilize_ang_vel_scale: float = 1.0
    nav_xy_progress_scale: float = 15.0
    nav_velocity_align_scale: float = 12.0
    nav_lateral_penalty_scale: float = -2.0
    nav_altitude_scale: float = -2.0
    nav_stability_scale: float = 3.0
    nav_max_speed: float = 2.0
    nav_speed_penalty_scale: float = -3.0
    hover_position_scale: float = 3.0
    hover_low_speed_scale: float = 2.0
    hover_altitude_scale: float = -2.0
    land_descent_scale: float = 5.0
    land_xy_stability_scale: float = 15.0
    land_drift_penalty: float = -10.0
    land_precision_scale: float = 200.0
    land_max_descent_speed: float = 0.3
    land_descent_speed_penalty: float = -15.0
    land_controlled_descent_scale: float = 5.0
    land_altitude_penalty_scale: float = -1.5
    goal_reached_bonus: float = 750.0
    time_penalty: float = -1.5
    ang_vel_reward_scale: float = -0.5
    yaw_rate_penalty_scale: float = -1.0
    upright_reward_scale: float = 2.0


# ── Environment ──────────────────────────────────────────────────────────────

class ForestNavEnv(DirectRLEnv):
    cfg: ForestNavEnvCfg

    TAKEOFF = 0
    STABILIZE = 1
    NAVIGATE = 2
    HOVER = 3
    LAND = 4

    def __init__(self, cfg: ForestNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_xy_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_z_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_alt = torch.zeros(self.num_envs, device=self.device)

        self._phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._stabilize_timer = torch.zeros(self.num_envs, device=self.device)
        self._hover_timer = torch.zeros(self.num_envs, device=self.device)

        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "takeoff_ascent", "takeoff_drift",
                "stabilize_position", "stabilize_speed", "stabilize_altitude", "stabilize_ang_vel",
                "nav_xy_progress", "nav_velocity_align", "nav_lateral", "nav_altitude", "nav_stability", "nav_speed_penalty",
                "hover_position", "hover_speed", "hover_altitude",
                "land_descent", "land_xy_stability", "land_drift", "land_precision", "land_descent_control", "land_controlled_descent", "land_altitude_penalty",
                "goal_bonus", "time_penalty", "ang_vel", "yaw_rate", "upright",
            ]
        }

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._lidar = RayCaster(self.cfg.lidar)
        self.scene.sensors["lidar"] = self._lidar

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.sim.set_camera_view(eye=[0.0, -18.0, 6.0], target=[0.0, 0.0, 2.0])

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # ── Auto-pilot for TAKEOFF and STABILIZE phases ──
        # Override RL actions so the drone reliably reaches cruise altitude
        altitude = self._robot.data.root_pos_w[:, 2]
        alt_err_to_cruise = self.cfg.cruise_altitude - altitude

        is_takeoff = self._phase == self.TAKEOFF
        is_stabilize = self._phase == self.STABILIZE

        # TAKEOFF: command upward velocity proportional to altitude error
        takeoff_vz = torch.clamp(alt_err_to_cruise * 1.5, 0.1, 1.0)
        self._actions[is_takeoff, 0] = 0.0  # no vx
        self._actions[is_takeoff, 1] = 0.0  # no vy
        self._actions[is_takeoff, 2] = takeoff_vz[is_takeoff]  # go up
        self._actions[is_takeoff, 3] = 0.0  # no yaw

        # STABILIZE: hold position (zero velocity)
        self._actions[is_stabilize, 0] = 0.0
        self._actions[is_stabilize, 1] = 0.0
        self._actions[is_stabilize, 2] = torch.clamp(alt_err_to_cruise[is_stabilize] * 0.5, -0.3, 0.3)
        self._actions[is_stabilize, 3] = 0.0

        # ── Decode velocity commands from RL actions ──
        vx_cmd = self._actions[:, 0] * self.cfg.max_velocity_xy
        vy_cmd = self._actions[:, 1] * self.cfg.max_velocity_xy
        vz_cmd = self._actions[:, 2] * self.cfg.max_velocity_z
        yaw_rate_cmd = self._actions[:, 3] * self.cfg.max_yaw_rate

        # ── Current state ──
        vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        gravity_b = self._robot.data.projected_gravity_b

        # ── OUTER LOOP: velocity error → desired attitude ──
        vx_err = vx_cmd - vel_b[:, 0]
        vy_err = vy_cmd - vel_b[:, 1]
        vz_err = vz_cmd - vel_b[:, 2]

        desired_roll = (self.cfg.pid_vel_kp * vy_err).clamp(
            -self.cfg.pid_max_tilt, self.cfg.pid_max_tilt
        )
        desired_pitch = (-self.cfg.pid_vel_kp * vx_err).clamp(
            -self.cfg.pid_max_tilt, self.cfg.pid_max_tilt
        )

        # ── Current attitude from projected gravity ──
        # Upright: gravity_b = [0, 0, -1]
        # Roll right by φ:  gravity_b_y = sin(φ)
        # Pitch fwd by θ:   gravity_b_x = -sin(θ)
        current_roll = gravity_b[:, 1]
        current_pitch = -gravity_b[:, 0]

        # ── INNER LOOP: attitude PD → torques (normalized to [-1, 1]) ──
        roll_torque = (
            self.cfg.pid_att_kp * (desired_roll - current_roll)
            - self.cfg.pid_att_kd * ang_vel_b[:, 0]
        ).clamp(-1.0, 1.0)

        pitch_torque = (
            self.cfg.pid_att_kp * (desired_pitch - current_pitch)
            - self.cfg.pid_att_kd * ang_vel_b[:, 1]
        ).clamp(-1.0, 1.0)

        yaw_torque = (
            self.cfg.pid_yaw_kp * (yaw_rate_cmd - ang_vel_b[:, 2])
        ).clamp(-1.0, 1.0)

        # ── THRUST: hover + vertical velocity correction ──
        hover_thrust = self._robot_weight
        thrust = hover_thrust * (1.0 + self.cfg.pid_vz_kp * vz_err)
        max_thrust = self.cfg.thrust_to_weight * self._robot_weight
        thrust = thrust.clamp(0.0, max_thrust)

        # ── Apply forces and torques ──
        self._thrust[:, 0, 2] = thrust
        self._moment[:, 0, 0] = self.cfg.moment_scale * self._robot_weight * roll_torque
        self._moment[:, 0, 1] = self.cfg.moment_scale * self._robot_weight * pitch_torque
        self._moment[:, 0, 2] = self.cfg.moment_scale * self._robot_weight * yaw_torque

    def _apply_action(self):
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )

    def _get_observations(self) -> dict:
        goal_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._goal_pos_w,
        )

        lidar_hits = self._lidar.data.ray_hits_w
        lidar_pos = self._lidar.data.pos_w.unsqueeze(1)
        lidar_distances = torch.linalg.norm(lidar_hits - lidar_pos, dim=-1)
        lidar_normalized = torch.clamp(
            lidar_distances / self.cfg.lidar_max_distance, 0.0, 1.0
        )

        phase_one_hot = torch.nn.functional.one_hot(
            self._phase.long(), num_classes=5
        ).float()

        obs = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            goal_pos_b,
            lidar_normalized,
            phase_one_hot,
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self._robot.data.root_pos_w
        dt = self.step_dt

        delta = self._goal_pos_w - root_pos
        xy_dist = torch.linalg.norm(delta[:, :2], dim=1)
        z_dist = torch.abs(delta[:, 2])
        dist_3d = torch.linalg.norm(delta, dim=1)

        lin_vel_b = self._robot.data.root_lin_vel_b
        lin_vel_w = self._robot.data.root_lin_vel_w
        speed = torch.linalg.norm(lin_vel_b, dim=1)
        horizontal_speed = torch.linalg.norm(lin_vel_w[:, :2], dim=1)
        altitude = root_pos[:, 2]
        alt_error = torch.abs(altitude - self.cfg.cruise_altitude)

        ang_vel_vec = self._robot.data.root_ang_vel_b
        ang_vel_magnitude = torch.linalg.norm(ang_vel_vec, dim=1)

        start_delta_xy = self._start_pos_w[:, :2] - root_pos[:, :2]
        start_xy_dist = torch.linalg.norm(start_delta_xy, dim=1)

        phase = self._phase.clone()

        is_takeoff = phase == self.TAKEOFF
        is_stabilize = phase == self.STABILIZE
        is_navigate = phase == self.NAVIGATE
        is_hover = phase == self.HOVER
        is_land = phase == self.LAND

        # Phase 0: TAKEOFF
        alt_progress = altitude - self._prev_alt
        self._prev_alt = altitude.clone()
        takeoff_ascent = torch.where(is_takeoff, alt_progress * self.cfg.takeoff_ascent_scale, torch.zeros_like(alt_progress))
        takeoff_drift = torch.where(is_takeoff, horizontal_speed * self.cfg.takeoff_drift_penalty * dt, torch.zeros_like(horizontal_speed))

        # Phase 1: STABILIZE
        stabilize_position = torch.where(is_stabilize, torch.exp(-start_xy_dist) * self.cfg.stabilize_position_scale * dt, torch.zeros_like(start_xy_dist))
        stabilize_speed = torch.where(is_stabilize, torch.exp(-speed) * self.cfg.stabilize_low_speed_scale * dt, torch.zeros_like(speed))
        stabilize_altitude = torch.where(is_stabilize, alt_error * self.cfg.stabilize_altitude_scale * dt, torch.zeros_like(alt_error))
        stabilize_ang_vel = torch.where(is_stabilize, torch.exp(-ang_vel_magnitude) * self.cfg.stabilize_ang_vel_scale * dt, torch.zeros_like(ang_vel_magnitude))

        # Phase 2: NAVIGATE
        xy_progress = self._prev_xy_dist - xy_dist
        self._prev_xy_dist = xy_dist.clone()
        nav_xy_progress = torch.where(is_navigate, xy_progress * self.cfg.nav_xy_progress_scale, torch.zeros_like(xy_progress))

        goal_dir_xy = delta[:, :2] / (xy_dist.unsqueeze(1) + 1e-6)
        vel_toward_goal = torch.sum(lin_vel_w[:, :2] * goal_dir_xy, dim=1)
        nav_velocity_align = torch.where(is_navigate, torch.clamp(vel_toward_goal, min=0.0) * self.cfg.nav_velocity_align_scale * dt, torch.zeros_like(vel_toward_goal))

        vel_lateral = lin_vel_w[:, :2] - vel_toward_goal.unsqueeze(1) * goal_dir_xy
        lateral_speed = torch.linalg.norm(vel_lateral, dim=1)
        nav_lateral_penalty = torch.where(is_navigate, lateral_speed * self.cfg.nav_lateral_penalty_scale * dt, torch.zeros_like(lateral_speed))

        nav_altitude = torch.where(is_navigate, alt_error * self.cfg.nav_altitude_scale * dt, torch.zeros_like(alt_error))

        gravity_z = self._robot.data.projected_gravity_b[:, 2]
        nav_stability = torch.where(is_navigate, (torch.exp(-ang_vel_magnitude) + (-gravity_z)) * self.cfg.nav_stability_scale * dt, torch.zeros_like(ang_vel_magnitude))

        excess_speed = torch.clamp(speed - self.cfg.nav_max_speed, min=0.0)
        nav_speed_penalty = torch.where(is_navigate, excess_speed * self.cfg.nav_speed_penalty_scale * dt, torch.zeros_like(excess_speed))

        # Phase 3: HOVER
        hover_position = torch.where(is_hover, torch.exp(-xy_dist) * self.cfg.hover_position_scale * dt, torch.zeros_like(xy_dist))
        hover_speed = torch.where(is_hover, torch.exp(-speed) * self.cfg.hover_low_speed_scale * dt, torch.zeros_like(speed))
        hover_altitude = torch.where(is_hover, alt_error * self.cfg.hover_altitude_scale * dt, torch.zeros_like(alt_error))

        # Phase 4: LAND
        descent_progress = self._prev_z_dist - z_dist
        self._prev_z_dist = z_dist.clone()
        land_descent = torch.where(is_land, descent_progress * self.cfg.land_descent_scale, torch.zeros_like(descent_progress))
        land_xy_stability = torch.where(is_land, torch.exp(-xy_dist) * self.cfg.land_xy_stability_scale * dt, torch.zeros_like(xy_dist))
        land_drift = torch.where(is_land, horizontal_speed * self.cfg.land_drift_penalty * dt, torch.zeros_like(horizontal_speed))
        land_precision = torch.where(is_land, torch.exp(-dist_3d * 5.0) * self.cfg.land_precision_scale * dt, torch.zeros_like(dist_3d))

        vertical_speed = torch.abs(lin_vel_w[:, 2])
        excess_descent = torch.clamp(vertical_speed - self.cfg.land_max_descent_speed, min=0.0)
        land_descent_control = torch.where(is_land, excess_descent * self.cfg.land_descent_speed_penalty * dt, torch.zeros_like(excess_descent))

        descent_vel = -lin_vel_w[:, 2]
        ideal_descent = torch.exp(-((descent_vel - 0.35) ** 2) / 0.05)
        land_controlled_descent = torch.where(is_land, ideal_descent * self.cfg.land_controlled_descent_scale * dt, torch.zeros_like(ideal_descent))

        land_altitude_penalty = torch.where(is_land, z_dist * self.cfg.land_altitude_penalty_scale * dt, torch.zeros_like(z_dist))

        goal_reached = is_land & (dist_3d < self.cfg.goal_threshold) & (speed < self.cfg.landing_speed_threshold)
        goal_bonus = goal_reached.float() * self.cfg.goal_reached_bonus

        # All-phase rewards
        time_penalty = torch.full_like(dist_3d, self.cfg.time_penalty * dt)
        ang_vel = torch.sum(torch.square(ang_vel_vec), dim=1)
        orient_scale = torch.ones_like(dist_3d)
        orient_scale = torch.where(is_navigate, 0.3 * orient_scale, orient_scale)
        orient_scale = torch.where(is_hover | is_land, 1.5 * orient_scale, orient_scale)
        ang_vel_penalty = ang_vel * self.cfg.ang_vel_reward_scale * dt * orient_scale
        yaw_rate = torch.abs(ang_vel_vec[:, 2])
        yaw_penalty = yaw_rate * self.cfg.yaw_rate_penalty_scale * dt
        upright_reward = -gravity_z * self.cfg.upright_reward_scale * dt * orient_scale

        # Phase transitions
        to_stabilize = is_takeoff & (alt_error < self.cfg.takeoff_altitude_tolerance)
        self._phase[to_stabilize] = self.STABILIZE
        self._stabilize_timer[to_stabilize] = 0.0

        self._stabilize_timer[self._phase == self.STABILIZE] += dt
        to_navigate = (self._phase == self.STABILIZE) & (self._stabilize_timer >= self.cfg.stabilize_duration)
        self._phase[to_navigate] = self.NAVIGATE

        altitude = self._robot.data.root_pos_w[:, 2]
        to_hover = (self._phase == self.NAVIGATE) & (xy_dist < self.cfg.hover_position_tolerance) & (speed < 1.5) & (altitude > 1.0)
        self._phase[to_hover] = self.HOVER
        self._hover_timer[to_hover] = 0.0

        self._hover_timer[self._phase == self.HOVER] += dt
        to_land = (self._phase == self.HOVER) & (self._hover_timer >= self.cfg.hover_duration) & (xy_dist < self.cfg.hover_position_tolerance)
        self._phase[to_land] = self.LAND

        reward = (
            takeoff_ascent + takeoff_drift
            + stabilize_position + stabilize_speed + stabilize_altitude + stabilize_ang_vel
            + nav_xy_progress + nav_velocity_align + nav_lateral_penalty + nav_altitude + nav_stability + nav_speed_penalty
            + hover_position + hover_speed + hover_altitude
            + land_descent + land_xy_stability + land_drift + land_precision + land_descent_control + land_controlled_descent + land_altitude_penalty + goal_bonus
            + time_penalty + ang_vel_penalty + yaw_penalty + upright_reward
        )

        # Logging
        self._episode_sums["takeoff_ascent"] += takeoff_ascent
        self._episode_sums["takeoff_drift"] += takeoff_drift
        self._episode_sums["stabilize_position"] += stabilize_position
        self._episode_sums["stabilize_speed"] += stabilize_speed
        self._episode_sums["stabilize_altitude"] += stabilize_altitude
        self._episode_sums["stabilize_ang_vel"] += stabilize_ang_vel
        self._episode_sums["nav_xy_progress"] += nav_xy_progress
        self._episode_sums["nav_velocity_align"] += nav_velocity_align
        self._episode_sums["nav_lateral"] += nav_lateral_penalty
        self._episode_sums["nav_altitude"] += nav_altitude
        self._episode_sums["nav_stability"] += nav_stability
        self._episode_sums["nav_speed_penalty"] += nav_speed_penalty
        self._episode_sums["hover_position"] += hover_position
        self._episode_sums["hover_speed"] += hover_speed
        self._episode_sums["hover_altitude"] += hover_altitude
        self._episode_sums["land_descent"] += land_descent
        self._episode_sums["land_xy_stability"] += land_xy_stability
        self._episode_sums["land_drift"] += land_drift
        self._episode_sums["land_precision"] += land_precision
        self._episode_sums["land_descent_control"] += land_descent_control
        self._episode_sums["land_controlled_descent"] += land_controlled_descent
        self._episode_sums["land_altitude_penalty"] += land_altitude_penalty
        self._episode_sums["goal_bonus"] += goal_bonus
        self._episode_sums["time_penalty"] += time_penalty
        self._episode_sums["ang_vel"] += ang_vel_penalty
        self._episode_sums["yaw_rate"] += yaw_penalty
        self._episode_sums["upright"] += upright_reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        too_low = (
            (self._robot.data.root_pos_w[:, 2] < self.cfg.min_flight_height)
            & (self._phase != self.TAKEOFF)
            & (self._phase != self.LAND)
        )
        too_high = self._robot.data.root_pos_w[:, 2] > self.cfg.max_flight_height
        flipped = self._robot.data.projected_gravity_b[:, 2] > 0.7

        goal_dist = torch.linalg.norm(self._goal_pos_w - self._robot.data.root_pos_w, dim=1)
        speed = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        goal_reached = (
            (self._phase == self.LAND)
            & (goal_dist < self.cfg.goal_threshold)
            & (speed < self.cfg.landing_speed_threshold)
        )

        touched_ground = (
            ((self._phase == self.HOVER) | (self._phase == self.LAND))
            & (self._robot.data.root_pos_w[:, 2] < 0.05)
        )

        died = too_low | too_high | flipped
        return died, time_out | goal_reached | touched_ground

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        final_distance = torch.linalg.norm(
            self._goal_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_distance.item()
        self.extras["log"]["Metrics/final_phase_mean"] = self._phase[env_ids].float().mean().item()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        num_reset = len(env_ids)

        start_local = torch.tensor(self.cfg.static_start_pos, dtype=torch.float32, device=self.device)
        goal_local = torch.tensor(self.cfg.static_goal_pos, dtype=torch.float32, device=self.device)

        start_pos = start_local.unsqueeze(0).expand(num_reset, -1).clone()
        goal_pos = goal_local.unsqueeze(0).expand(num_reset, -1).clone()

        start_pos[:, :2] += self._terrain.env_origins[env_ids, :2]
        goal_pos[:, :2] += self._terrain.env_origins[env_ids, :2]

        self._goal_pos_w[env_ids] = goal_pos
        self._start_pos_w[env_ids] = start_pos

        delta = goal_pos - start_pos
        self._prev_xy_dist[env_ids] = torch.linalg.norm(delta[:, :2], dim=1)
        self._prev_z_dist[env_ids] = torch.abs(delta[:, 2])
        self._prev_alt[env_ids] = start_pos[:, 2]

        self._phase[env_ids] = self.TAKEOFF
        self._stabilize_timer[env_ids] = 0.0
        self._hover_timer[env_ids] = 0.0

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = start_pos
        default_root_state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.5, 0.5, 0.5)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._goal_pos_w)


# ── PPO Config ───────────────────────────────────────────────────────────────

@configclass
class ForestNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 1024
    max_iterations = 1000
    save_interval = 100
    experiment_name = "crazyflie_forest_nav_ppo"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


# ── Gym Registration ─────────────────────────────────────────────────────────

gym.register(
    id="Isaac-CrazyflieWaypoint-PPO-v0",
    entry_point=f"{__name__}:ForestNavEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:ForestNavEnvCfg"},
)


# ── Train ────────────────────────────────────────────────────────────────────

def train():
    env_cfg = ForestNavEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.static_start_pos = tuple(args_cli.start_pos)
    env_cfg.static_goal_pos = tuple(args_cli.goal_pos)

    agent_cfg = ForestNavPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    log_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ppo", agent_cfg.experiment_name)
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    print(f"[INFO] Logging to: {log_dir}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Max iterations: {agent_cfg.max_iterations}")
    print(f"[INFO] Flight path: {env_cfg.static_start_pos} -> {env_cfg.static_goal_pos}")

    env = gym.make("Isaac-CrazyflieWaypoint-PPO-v0", cfg=env_cfg,
                    render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=os.path.join(log_dir, "videos"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length, disable_logger=True)

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    if args_cli.checkpoint:
        print(f"[INFO] Resuming from: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)

    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    elapsed = time.time() - start_time
    print(f"\n[INFO] Training complete! Duration: {elapsed:.1f}s")
    print(f"[INFO] Logs saved to: {log_dir}")

    env.close()


# ── Play ─────────────────────────────────────────────────────────────────────

def play():
    env_cfg = ForestNavEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 30.0

    env = gym.make("Isaac-CrazyflieWaypoint-PPO-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = ForestNavPPORunnerCfg()
    agent_cfg.max_iterations = 0

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    obs = env.get_observations()
    episode_count = 0
    total_reward = 0.0
    step_count = 0
    sim_dt = env_cfg.sim.dt * env_cfg.decimation

    print(f"\n{'='*60}")
    print(f"[INFO] PPO policy loaded! Drone is flying.")
    print(f"[INFO] Press Ctrl+C to stop.")
    print(f"{'='*60}\n")

    try:
        while simulation_app.is_running():
            step_start = time.time()
            with torch.no_grad():
                actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            total_reward += rewards.sum().item()
            step_count += 1

            if dones.any():
                episode_count += dones.sum().item()
                print(f"  Episode {int(episode_count)} done (step {step_count})")

            elapsed = time.time() - step_start
            sleep_time = sim_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    avg_reward = total_reward / max(episode_count, 1)
    print(f"\n[RESULT] Average reward: {avg_reward:.2f} over {int(episode_count)} episodes ({step_count} steps)")
    env.close()


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate():
    env_cfg = ForestNavEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 15.0
    env_cfg.debug_vis = False

    env = gym.make("Isaac-CrazyflieWaypoint-PPO-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = ForestNavPPORunnerCfg()
    agent_cfg.max_iterations = 0
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    obs = env.get_observations()

    total_episodes = 0
    episode_rewards = []
    final_distances = []
    episode_lengths = []
    crashes = 0
    goal_reached_count = 0
    env_rewards = np.zeros(args_cli.num_envs)
    env_steps = np.zeros(args_cli.num_envs, dtype=int)
    step_dt = env_cfg.sim.dt * env_cfg.decimation

    print(f"\n{'='*60}")
    print(f"EVALUATION: PPO | {args_cli.num_episodes} episodes")
    print(f"Checkpoint: {os.path.basename(args_cli.checkpoint)}")
    print(f"{'='*60}\n")

    while total_episodes < args_cli.num_episodes and simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)

        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy().flatten()
            dones_np = dones.cpu().numpy().flatten()
        else:
            rewards_np = np.array(rewards).flatten()
            dones_np = np.array(dones).flatten()

        env_rewards += rewards_np
        env_steps += 1

        for i in range(args_cli.num_envs):
            if dones_np[i] and total_episodes < args_cli.num_episodes:
                total_episodes += 1
                episode_rewards.append(env_rewards[i])
                episode_lengths.append(env_steps[i] * step_dt)

                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'extras'):
                    extras = env.unwrapped.extras.get("log", {})
                    dist = extras.get("Metrics/final_distance_to_goal", -1)
                    died_count = extras.get("Episode_Termination/died", 0)
                    final_distances.append(dist)
                    if died_count > 0:
                        crashes += 1
                    if dist >= 0 and dist < env_cfg.goal_threshold:
                        goal_reached_count += 1

                if total_episodes % 10 == 0 or total_episodes == args_cli.num_episodes:
                    print(f"  Episodes: {total_episodes}/{args_cli.num_episodes}")

                env_rewards[i] = 0.0
                env_steps[i] = 0

    n = len(episode_rewards)
    if n == 0:
        print("No episodes completed!")
        env.close()
        return

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = (goal_reached_count / n) * 100
    crash_rate = (crashes / n) * 100
    valid_distances = [d for d in final_distances if d >= 0]
    avg_distance = np.mean(valid_distances) if valid_distances else -1
    min_distance = np.min(valid_distances) if valid_distances else -1

    print(f"\n{'='*60}")
    print(f"RESULTS - PPO ({n} episodes)")
    print(f"{'='*60}")
    print(f"  Success Rate:       {success_rate:6.1f}%  ({goal_reached_count}/{n} landed at goal)")
    print(f"  Crash Rate:         {crash_rate:6.1f}%  ({crashes}/{n} crashed)")
    print(f"  Avg Final Distance: {avg_distance:6.3f} m")
    print(f"  Min Final Distance: {min_distance:6.3f} m")
    print(f"  Avg Reward:         {avg_reward:8.2f}  (+/- {std_reward:.2f})")
    print(f"  Avg Episode Length:  {avg_length:5.2f} s")
    print(f"{'='*60}\n")
    env.close()


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if args_cli.mode == "train":
        train()
    elif args_cli.mode == "play":
        play()
    elif args_cli.mode == "eval":
        evaluate()
    simulation_app.close()
