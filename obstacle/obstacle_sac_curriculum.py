"""Single-level Crazyflie obstacle avoidance + multi-waypoint navigation with SAC.

CURRICULUM APPROACH — trains one end-to-end policy (no hierarchical / no frozen LL).
The policy directly outputs velocity commands [vx, vy, vz, yaw_rate] → PID → physics.

Key differences from previous failed approaches:
  - 105-ray LiDAR (35 horiz × 3 vert) with 1D CNN feature extractor
  - Analytical ray-cylinder intersection so LiDAR actually sees obstacles
  - Curriculum: obstacles introduced gradually (0 → full count over training)
  - Proven reward structure from multi_waypoint_sac + simple obstacle penalties
  - No hierarchical architecture, no frozen LL, no OOD distribution mismatch

Usage (from IsaacLab directory):
    cd ~/projects/isaac/IsaacLab
    source ~/projects/isaac/env_isaaclab/bin/activate

    # Train (curriculum):
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/obstacle_sac_curriculum.py \
        --mode train --num_envs 256 --total_timesteps 16000000 --headless

    # Play:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/obstacle_sac_curriculum.py \
        --mode play --checkpoint /path/to/sac_final.zip

    # Eval:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/obstacle_sac_curriculum.py \
        --mode eval --checkpoint /path/to/sac_final.zip --num_episodes 50
"""
from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

# ── Argument Parser ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Crazyflie Obstacle Nav SAC — Curriculum")
parser.add_argument("--mode", type=str, default="train", choices=["train", "play", "eval"])
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--total_timesteps", type=int, default=16000000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--min_waypoints", type=int, default=3)
parser.add_argument("--max_waypoints", type=int, default=5)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after sim launch) ───────────────────────────────────────────────
import gymnasium as gym
import time
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

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


# ══════════════════════════════════════════════════════════════════════════════
# LiDAR CNN Feature Extractor
# ══════════════════════════════════════════════════════════════════════════════

class LidarCNNExtractor(BaseFeaturesExtractor):
    """1D CNN for 108-ray multi-channel LiDAR + MLP for state, fused together.

    Observation layout: [lidar_105, state_21]
    LiDAR is reshaped to (batch, 3, 35) for Conv1d processing.
    """

    def __init__(self, observation_space, features_dim=128, lidar_dim=105,
                 cnn_out_dim=32, lidar_channels=3):
        super().__init__(observation_space, features_dim)

        self.lidar_dim = lidar_dim
        self.state_dim = observation_space.shape[0] - lidar_dim
        self.lidar_channels = lidar_channels
        self.rays_per_channel = lidar_dim // lidar_channels

        # 1D CNN for LiDAR: (batch, 3, 35) -> (batch, cnn_out_dim)
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(lidar_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, cnn_out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # State MLP
        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
        )

        # Fusion
        self._features_dim = features_dim
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        lidar_flat = observations[:, :self.lidar_dim]
        state = observations[:, self.lidar_dim:]

        lidar = lidar_flat.view(-1, self.lidar_channels, self.rays_per_channel)
        lidar_features = self.lidar_cnn(lidar)
        state_features = self.state_mlp(state)

        return self.fusion(torch.cat([lidar_features, state_features], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# Robot Configuration
# ══════════════════════════════════════════════════════════════════════════════

_crazyflie_spawn = CRAZYFLIE_CFG.spawn.copy()
_crazyflie_spawn.scale = (10.0, 10.0, 10.0)
_crazyflie_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False, max_depenetration_velocity=10.0, enable_gyroscopic_forces=False,
)
VISIBLE_CRAZYFLIE_CFG = CRAZYFLIE_CFG.replace(
    prim_path="/World/envs/env_.*/Robot", spawn=_crazyflie_spawn,
)


# ══════════════════════════════════════════════════════════════════════════════
# Environment Configuration
# ══════════════════════════════════════════════════════════════════════════════

@configclass
class ObstacleCurriculumEnvCfg(DirectRLEnvCfg):
    """Single-level obstacle avoidance with curriculum learning."""

    episode_length_s = 40.0
    decimation = 2
    # obs: lidar(105) + lin_vel_b(3) + ang_vel_b(3) + gravity_b(3) + goal_pos_b(3)
    #      + phase_one_hot(5) + nearest_obstacle_dir(2) + nearest_obstacle_dist(1) + progress_frac(1)
    #    = 105 + 21 = 126   (LiDAR: 3ch × 35 horiz rays = 105)
    action_space = 4
    observation_space = 126
    state_space = 0
    debug_vis = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100, render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=70.0, replicate_physics=True)
    robot: ArticulationCfg = VISIBLE_CRAZYFLIE_CFG
    thrust_to_weight = 1.9
    moment_scale = 0.04

    # PID controller (same as multi_waypoint_sac training)
    max_velocity_xy: float = 2.0
    max_velocity_z: float = 1.0
    max_yaw_rate: float = 1.5
    pid_vel_kp: float = 0.25
    pid_att_kp: float = 6.0
    pid_att_kd: float = 1.0
    pid_vz_kp: float = 0.5
    pid_yaw_kp: float = 0.4
    pid_max_tilt: float = 0.5

    # LiDAR: 3 vertical channels × 36 horizontal = 108 rays
    # 10° horizontal resolution gives good obstacle coverage
    lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body", update_period=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)), ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=3, vertical_fov_range=(-90.0, 0.0),
            horizontal_fov_range=(0.0, 360.0), horizontal_res=10.0,
        ),
        max_distance=15.0, mesh_prim_paths=["/World/ground"], debug_vis=False,
    )
    lidar_max_distance: float = 15.0
    lidar_n_channels: int = 3

    # Navigation
    goal_threshold: float = 0.5
    landing_speed_threshold: float = 1.0
    max_flight_height: float = 8.0
    min_flight_height: float = 0.05
    cruise_altitude: float = 1.5

    takeoff_altitude_tolerance: float = 0.3
    stabilize_duration: float = 2.0
    hover_position_tolerance: float = 0.5
    hover_duration: float = 2.0

    # Multi-waypoint
    static_start_pos: tuple = (-10.0, 0.0, 0.2)
    min_num_waypoints: int = 3
    max_num_waypoints: int = 5
    arena_x_range: tuple = (-8.0, 8.0)
    arena_y_range: tuple = (-6.0, 6.0)
    min_y_change: float = 2.5
    intermediate_waypoint_tolerance: float = 2.0
    final_waypoint_tolerance: float = 0.5
    waypoint_obstacle_clearance: float = 3.0

    # Obstacles — ALL are spawned at scene creation, but curriculum controls
    # how many are "active" (have collision enabled and are visible to LiDAR)
    obstacle_positions: list = [
        # Interior obstacles (introduced first in curriculum)
        (-5.0,  4.0, 0.50, 5.0),
        (-5.0, -4.0, 0.50, 5.0),
        ( 5.0,  4.0, 0.50, 5.0),
        ( 5.0, -4.0, 0.50, 5.0),
        # Boundary obstacles (introduced later)
        (-6.0,  10.0, 0.50, 5.0),
        ( 0.0,  10.0, 0.50, 5.0),
        ( 6.0,  10.0, 0.50, 5.0),
        (-6.0, -10.0, 0.50, 5.0),
        ( 0.0, -10.0, 0.50, 5.0),
        ( 6.0, -10.0, 0.50, 5.0),
        (-9.0, -7.0, 0.50, 5.0),
        (-9.0,  7.0, 0.50, 5.0),
        ( 9.0, -7.0, 0.50, 5.0),
        ( 9.0,  7.0, 0.50, 5.0),
    ]
    obstacle_collision_radius: float = 0.5
    obstacle_safe_distance: float = 1.5

    # Curriculum: fraction of total_timesteps at which all obstacles are active
    curriculum_start_frac: float = 0.05   # start adding obstacles at 5% of training
    curriculum_end_frac: float = 0.40     # all obstacles active by 40% of training

    # Reward scales (based on proven multi_waypoint_sac rewards + obstacle avoidance)
    takeoff_ascent_scale: float = 10.0
    takeoff_drift_penalty: float = -2.0
    stabilize_position_scale: float = 3.0
    stabilize_low_speed_scale: float = 2.0
    stabilize_altitude_scale: float = -2.0
    stabilize_ang_vel_scale: float = 1.0
    nav_xy_progress_scale: float = 5.0
    nav_velocity_align_scale: float = 10.0
    nav_lateral_penalty_scale: float = -1.0
    nav_altitude_scale: float = -2.0
    nav_stability_scale: float = 2.0
    nav_max_speed: float = 2.0
    nav_speed_penalty_scale: float = -3.0
    intermediate_waypoint_bonus: float = 150.0
    speed_carry_bonus_scale: float = 5.0
    waypoint_progress_scale: float = 2.0
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
    # Obstacle-specific rewards
    obstacle_proximity_penalty_scale: float = -8.0
    lidar_obstacle_penalty_scale: float = -5.0
    lidar_danger_distance: float = 2.5
    crash_penalty: float = -500.0


# ══════════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════════

class ObstacleCurriculumEnv(DirectRLEnv):
    """Single-level obstacle avoidance with curriculum training."""

    cfg: ObstacleCurriculumEnvCfg

    TAKEOFF = 0
    STABILIZE = 1
    NAVIGATE = 2
    HOVER = 3
    LAND = 4

    def __init__(self, cfg: ObstacleCurriculumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        max_wps = self.cfg.max_num_waypoints

        # Action buffers
        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Multi-waypoint storage
        self._waypoints_w = torch.zeros(self.num_envs, max_wps, 3, device=self.device)
        self._num_waypoints = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._current_wp_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._final_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._prev_xy_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_z_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_alt = torch.zeros(self.num_envs, device=self.device)

        # Phase state machine
        self._phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._stabilize_timer = torch.zeros(self.num_envs, device=self.device)
        self._hover_timer = torch.zeros(self.num_envs, device=self.device)

        # Robot physical properties
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Obstacle tensors
        self._obstacle_local = torch.tensor(
            [(ox, oy) for (ox, oy, r, h) in self.cfg.obstacle_positions],
            dtype=torch.float32, device=self.device,
        )
        self._obstacle_radii = torch.tensor(
            [r for (ox, oy, r, h) in self.cfg.obstacle_positions],
            dtype=torch.float32, device=self.device,
        )
        self._obstacle_heights = torch.tensor(
            [h for (ox, oy, r, h) in self.cfg.obstacle_positions],
            dtype=torch.float32, device=self.device,
        )
        self._num_obstacles = len(self.cfg.obstacle_positions)
        self._obstacle_pos_w = torch.zeros(
            self.num_envs, self._num_obstacles, 2, device=self.device
        )
        self._nearest_obs_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._nearest_obs_dist = torch.zeros(self.num_envs, device=self.device)

        # Curriculum: how many obstacles are active (starts at 0, grows over training)
        self._active_obstacles = 0  # updated by curriculum callback
        self._total_env_steps = 0

        # LiDAR ray direction cache for analytical augmentation
        self._lidar_ray_dirs = None
        self._lidar_horiz_indices = None

        # Episode logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "takeoff_ascent", "takeoff_drift",
                "stabilize_position", "stabilize_speed", "stabilize_altitude", "stabilize_ang_vel",
                "nav_xy_progress", "nav_velocity_align", "nav_lateral", "nav_altitude",
                "nav_stability", "nav_speed_penalty",
                "obstacle_proximity", "lidar_penalty", "crash_penalty",
                "waypoint_bonus", "speed_carry", "wp_progress",
                "hover_position", "hover_speed", "hover_altitude",
                "land_descent", "land_xy_stability", "land_drift", "land_precision",
                "land_descent_control", "land_controlled_descent", "land_altitude_penalty",
                "goal_bonus", "time_penalty", "ang_vel", "yaw_rate", "upright",
            ]
        }

        # Crash tracking by phase
        self._crash_phase_counts = {
            "TAKEOFF": 0, "STABILIZE": 0, "NAVIGATE": 0, "HOVER": 0, "LAND": 0,
        }

        self.set_debug_vis(self.cfg.debug_vis)

    # ── Scene Setup ──────────────────────────────────────────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._lidar = RayCaster(self.cfg.lidar)
        self.scene.sensors["lidar"] = self._lidar

        # Spawn ALL obstacles at scene creation (curriculum hides inactive ones via LiDAR masking)
        for i, (ox, oy, radius, height) in enumerate(self.cfg.obstacle_positions):
            trunk_cfg = sim_utils.CylinderCfg(
                radius=radius, height=height, axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.36, 0.25, 0.13)),
            )
            trunk_cfg.func(
                f"/World/envs/env_0/Tree_{i}_Trunk", trunk_cfg,
                translation=(ox, oy, height / 2.0),
            )
            canopy_height = 1.5
            canopy_cfg = sim_utils.CylinderCfg(
                radius=radius * 1.5, height=canopy_height, axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.12)),
            )
            canopy_cfg.func(
                f"/World/envs/env_0/Tree_{i}_Canopy", canopy_cfg,
                translation=(ox, oy, height + canopy_height / 2.0),
            )

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.sim.set_camera_view(eye=[0.0, -25.0, 10.0], target=[0.0, 0.0, 2.0])

    # ── Curriculum ───────────────────────────────────────────────────────────

    def set_curriculum_step(self, current_step: int, total_steps: int):
        """Called by CurriculumCallback to update active obstacle count."""
        frac = current_step / max(total_steps, 1)
        start = self.cfg.curriculum_start_frac
        end = self.cfg.curriculum_end_frac
        if frac < start:
            ratio = 0.0
        elif frac >= end:
            ratio = 1.0
        else:
            ratio = (frac - start) / (end - start)
        self._active_obstacles = int(ratio * self._num_obstacles)

    # ── Waypoint Generation ──────────────────────────────────────────────────

    def _generate_waypoints(self, env_ids: torch.Tensor):
        num_reset = len(env_ids)
        device = self.device
        max_wps = self.cfg.max_num_waypoints
        x_min, x_max = self.cfg.arena_x_range
        y_min, y_max = self.cfg.arena_y_range

        num_wps = torch.randint(
            self.cfg.min_num_waypoints, self.cfg.max_num_waypoints + 1,
            (num_reset,), device=device, dtype=torch.int32,
        )

        base_x = torch.linspace(x_min, x_max, max_wps, device=device).unsqueeze(0).expand(num_reset, -1)
        x_jitter = (torch.rand(num_reset, max_wps, device=device) - 0.5) * 2.0
        wp_x = (base_x + x_jitter).clamp(x_min, x_max)
        wp_x, _ = wp_x.sort(dim=1)

        wp_y = torch.zeros(num_reset, max_wps, device=device)
        wp_y[:, 0] = (torch.rand(num_reset, device=device) - 0.5) * (y_max - y_min) * 0.6
        for j in range(1, max_wps):
            direction = -torch.sign(wp_y[:, j - 1])
            direction = torch.where(direction == 0, torch.ones_like(direction), direction)
            magnitude = self.cfg.min_y_change + torch.rand(num_reset, device=device) * (
                abs(y_max) * 0.8 - self.cfg.min_y_change
            )
            wp_y[:, j] = (direction * magnitude).clamp(y_min, y_max)

        wp_z = torch.full((num_reset, max_wps), self.cfg.cruise_altitude, device=device)

        # Obstacle clearance nudging (only for active obstacles)
        n_active = self._active_obstacles
        if n_active > 0:
            obs_xy = self._obstacle_local[:n_active]
            obs_r = self._obstacle_radii[:n_active]
            clearance = self.cfg.waypoint_obstacle_clearance

            for wi in range(max_wps):
                wp_xy = torch.stack([wp_x[:, wi], wp_y[:, wi]], dim=1)
                delta = wp_xy.unsqueeze(1) - obs_xy.unsqueeze(0)
                dists = torch.linalg.norm(delta, dim=2)
                surface_dists = dists - obs_r.unsqueeze(0)
                min_surf_dist, min_idx = surface_dists.min(dim=1)
                too_close = min_surf_dist < clearance
                if too_close.any():
                    batch_idx = torch.arange(num_reset, device=device)
                    closest_delta = delta[batch_idx[too_close], min_idx[too_close]]
                    nudge_dir = closest_delta / (torch.linalg.norm(closest_delta, dim=1, keepdim=True) + 1e-6)
                    needed = clearance - min_surf_dist[too_close]
                    nudge = nudge_dir * (needed.unsqueeze(1) + 0.5)
                    wp_x[too_close, wi] = (wp_x[too_close, wi] + nudge[:, 0]).clamp(x_min, x_max)
                    wp_y[too_close, wi] = (wp_y[too_close, wi] + nudge[:, 1]).clamp(y_min, y_max)

        batch_idx = torch.arange(num_reset, device=device)
        waypoints_local = torch.stack([wp_x, wp_y, wp_z], dim=-1)

        for i in range(max_wps):
            mask = (i >= num_wps)
            if mask.any():
                final_wp = waypoints_local[batch_idx[mask], (num_wps[mask] - 1).long()]
                waypoints_local[mask, i] = final_wp

        self._num_waypoints[env_ids] = num_wps

        env_origins_xy = self._terrain.env_origins[env_ids, :2]
        waypoints_w = waypoints_local.clone()
        waypoints_w[:, :, 0] += env_origins_xy[:, 0:1]
        waypoints_w[:, :, 1] += env_origins_xy[:, 1:2]
        self._waypoints_w[env_ids] = waypoints_w

        final_wp_idx = (num_wps - 1).long()
        self._final_goal_pos_w[env_ids] = waypoints_w[batch_idx, final_wp_idx].clone()
        self._final_goal_pos_w[env_ids, 2] = 0.2

    def _update_current_goal(self, env_ids: torch.Tensor):
        batch_idx = torch.arange(len(env_ids), device=self.device)
        wp_idx = self._current_wp_idx[env_ids].long()
        self._goal_pos_w[env_ids] = self._waypoints_w[env_ids][batch_idx, wp_idx]

    def _check_waypoint_advancement(self, xy_dist: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        is_navigate = self._phase == self.NAVIGATE
        is_intermediate = self._current_wp_idx < (self._num_waypoints - 1)
        reached = is_navigate & is_intermediate & (xy_dist < self.cfg.intermediate_waypoint_tolerance)

        if reached.any():
            self._current_wp_idx[reached] += 1
            reached_ids = torch.where(reached)[0]
            self._update_current_goal(reached_ids)
            new_delta = self._goal_pos_w[reached_ids] - self._robot.data.root_pos_w[reached_ids]
            self._prev_xy_dist[reached_ids] = torch.linalg.norm(new_delta[:, :2], dim=1)

        return reached

    # ── LiDAR Analytical Augmentation ────────────────────────────────────────

    def _init_lidar_rays(self, n_rays: int):
        """Precompute LiDAR ray directions for analytical obstacle intersection."""
        pcfg = self.cfg.lidar.pattern_cfg
        el_deg = torch.linspace(
            pcfg.vertical_fov_range[0], pcfg.vertical_fov_range[1],
            pcfg.channels, device=self.device,
        )
        elevations = torch.deg2rad(el_deg)
        n_horiz = n_rays // max(pcfg.channels, 1)
        az_deg = torch.linspace(0.0, 360.0, n_horiz + 1, device=self.device)[:-1]
        azimuths = torch.deg2rad(az_deg)

        dirs = []
        for el in elevations:
            cos_el, sin_el = torch.cos(el), torch.sin(el)
            for az in azimuths:
                dirs.append(torch.stack([cos_el * torch.cos(az), cos_el * torch.sin(az), sin_el]))
        self._lidar_ray_dirs = torch.stack(dirs)
        # Indices of rays with small vertical component (horizontal-ish)
        # Include all rays that could hit a 5m tall cylinder from cruise altitude
        # At -45° elevation, a ray from 1.5m hits a surface at ~1.5m distance for 0m height
        # We want ALL non-straight-down rays to check against cylinders
        self._lidar_horiz_indices = torch.where(
            self._lidar_ray_dirs[:, 2].abs() < 0.75  # includes -45° (sin=-0.707) and 0° channels
        )[0]

    def _augment_lidar_with_obstacles(
        self, lidar_distances: torch.Tensor,
        root_pos: torch.Tensor, root_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Overlay analytical ray-cylinder intersections onto ground-only LiDAR.

        Unlike v2 which only processes horizontal rays, this processes ALL rays
        with elevation > -60°, so angled rays can also detect cylinders.
        """
        n_active = self._active_obstacles
        if n_active == 0:
            return lidar_distances

        N_rays = lidar_distances.shape[1]
        if self._lidar_ray_dirs is None:
            self._init_lidar_rays(N_rays)

        proc_idx = self._lidar_horiz_indices
        N_proc = len(proc_idx)
        if N_proc == 0:
            return lidar_distances

        N = self.num_envs
        proc_dirs = self._lidar_ray_dirs[proc_idx]

        # Extract yaw from quaternion for ray rotation
        qw, qx, qy, qz = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

        dx_l, dy_l, dz_l = proc_dirs[:, 0], proc_dirs[:, 1], proc_dirs[:, 2]
        dx_w = cos_y[:, None] * dx_l[None] - sin_y[:, None] * dy_l[None]
        dy_w = sin_y[:, None] * dx_l[None] + cos_y[:, None] * dy_l[None]
        dz_w = dz_l[None].expand(N, -1)

        ox = root_pos[:, 0:1].expand(-1, N_proc)
        oy = root_pos[:, 1:2].expand(-1, N_proc)
        oz = root_pos[:, 2:3].expand(-1, N_proc)

        min_t = torch.full((N, N_proc), self.cfg.lidar_max_distance, device=self.device)

        # Only check active obstacles
        for oi in range(n_active):
            cx = self._obstacle_pos_w[:, oi, 0:1]
            cy = self._obstacle_pos_w[:, oi, 1:2]
            r = self._obstacle_radii[oi]
            h = self._obstacle_heights[oi]

            rel_x = ox - cx
            rel_y = oy - cy
            a = dx_w ** 2 + dy_w ** 2
            b = 2 * (rel_x * dx_w + rel_y * dy_w)
            c_val = rel_x ** 2 + rel_y ** 2 - r ** 2
            disc = b ** 2 - 4 * a * c_val
            valid = (disc >= 0) & (a > 1e-8)
            sqrt_d = torch.sqrt(disc.clamp(min=0.0))

            for t_cand in [(-b - sqrt_d) / (2 * a + 1e-8), (-b + sqrt_d) / (2 * a + 1e-8)]:
                hit_z = oz + t_cand * dz_w
                ok = valid & (t_cand > 0.01) & (hit_z >= 0) & (hit_z <= h)
                min_t = torch.where(ok & (t_cand < min_t), t_cand, min_t)

        result = lidar_distances.clone()
        result[:, proc_idx] = torch.min(lidar_distances[:, proc_idx], min_t)
        return result

    # ── Nearest Obstacle Computation ─────────────────────────────────────────

    def _compute_nearest_obstacle(self):
        """Compute nearest active obstacle distance and direction for each drone."""
        n_active = self._active_obstacles
        if n_active == 0:
            self._nearest_obs_dist.fill_(self.cfg.lidar_max_distance)
            self._nearest_obs_pos_w.zero_()
            return

        root_pos = self._robot.data.root_pos_w
        drone_xy = root_pos[:, :2]

        obs_xy = self._obstacle_pos_w[:, :n_active, :]  # (N, n_active, 2)
        delta = obs_xy - drone_xy.unsqueeze(1)  # (N, n_active, 2)
        dists = torch.linalg.norm(delta, dim=2)  # (N, n_active)
        surface_dists = dists - self._obstacle_radii[:n_active].unsqueeze(0)

        min_dist, min_idx = surface_dists.min(dim=1)
        self._nearest_obs_dist = min_dist.clamp(min=0.0)

        batch_idx = torch.arange(self.num_envs, device=self.device)
        nearest_xy = obs_xy[batch_idx, min_idx]
        self._nearest_obs_pos_w[:, :2] = nearest_xy
        self._nearest_obs_pos_w[:, 2] = 0.0

    # ── Physics ──────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        vx_cmd = self._actions[:, 0] * self.cfg.max_velocity_xy
        vy_cmd = self._actions[:, 1] * self.cfg.max_velocity_xy
        vz_cmd = self._actions[:, 2] * self.cfg.max_velocity_z
        yaw_rate_cmd = self._actions[:, 3] * self.cfg.max_yaw_rate

        vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        gravity_b = self._robot.data.projected_gravity_b

        vx_err = vx_cmd - vel_b[:, 0]
        vy_err = vy_cmd - vel_b[:, 1]
        vz_err = vz_cmd - vel_b[:, 2]

        desired_roll = (self.cfg.pid_vel_kp * vy_err).clamp(
            -self.cfg.pid_max_tilt, self.cfg.pid_max_tilt
        )
        desired_pitch = (-self.cfg.pid_vel_kp * vx_err).clamp(
            -self.cfg.pid_max_tilt, self.cfg.pid_max_tilt
        )

        current_roll = gravity_b[:, 1]
        current_pitch = -gravity_b[:, 0]

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

        hover_thrust = self._robot_weight
        thrust = hover_thrust * (1.0 + self.cfg.pid_vz_kp * vz_err)
        max_thrust = self.cfg.thrust_to_weight * self._robot_weight
        thrust = thrust.clamp(0.0, max_thrust)

        self._thrust[:, 0, 2] = thrust
        self._moment[:, 0, 0] = self.cfg.moment_scale * self._robot_weight * roll_torque
        self._moment[:, 0, 1] = self.cfg.moment_scale * self._robot_weight * pitch_torque
        self._moment[:, 0, 2] = self.cfg.moment_scale * self._robot_weight * yaw_torque

    def _apply_action(self):
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )

    # ── Observations ─────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """Observations: LiDAR(108) + state(21) = 129.

        State: lin_vel_b(3) + ang_vel_b(3) + gravity_b(3) + goal_pos_b(3) + phase(5)
               + nearest_obs_dir(2) + nearest_obs_dist(1) + wp_progress(1)
        """
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w

        goal_pos_b, _ = subtract_frame_transforms(root_pos, root_quat, self._goal_pos_w)

        # LiDAR distances (ground-only from RayCaster)
        lidar_hits = self._lidar.data.ray_hits_w
        lidar_pos = self._lidar.data.pos_w.unsqueeze(1)
        lidar_distances = torch.linalg.norm(lidar_hits - lidar_pos, dim=-1)

        # Augment with analytical ray-cylinder intersections for obstacles
        lidar_distances = self._augment_lidar_with_obstacles(lidar_distances, root_pos, root_quat)

        lidar_normalized = torch.clamp(lidar_distances / self.cfg.lidar_max_distance, 0.0, 1.0)

        # Phase one-hot
        phase_one_hot = torch.nn.functional.one_hot(self._phase.long(), num_classes=5).float()

        # Nearest obstacle info
        self._compute_nearest_obstacle()
        delta_to_obs = self._nearest_obs_pos_w[:, :2] - root_pos[:, :2]
        obs_dist_norm = (self._nearest_obs_dist / self.cfg.lidar_max_distance).clamp(0, 1).unsqueeze(1)
        obs_dir = delta_to_obs / (torch.linalg.norm(delta_to_obs, dim=1, keepdim=True) + 1e-6)
        # If no active obstacles, zero out direction
        if self._active_obstacles == 0:
            obs_dir = torch.zeros_like(obs_dir)
            obs_dist_norm = torch.ones_like(obs_dist_norm)

        # Waypoint progress fraction
        wp_progress = (self._current_wp_idx.float() / self._num_waypoints.float().clamp(min=1)).unsqueeze(1)

        obs = torch.cat([
            lidar_normalized,                       # 108
            self._robot.data.root_lin_vel_b,       # 3
            self._robot.data.root_ang_vel_b,       # 3
            self._robot.data.projected_gravity_b,   # 3
            goal_pos_b,                             # 3
            phase_one_hot,                          # 5
            obs_dir,                                # 2
            obs_dist_norm,                          # 1
            wp_progress,                            # 1
        ], dim=-1)

        return {"policy": obs}

    # ── Rewards ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """5-phase reward with multi-waypoint navigation + obstacle avoidance."""
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

        is_at_final_wp = self._current_wp_idx >= (self._num_waypoints - 1)

        phase = self._phase.clone()
        is_takeoff = phase == self.TAKEOFF
        is_stabilize = phase == self.STABILIZE
        is_navigate = phase == self.NAVIGATE
        is_hover = phase == self.HOVER
        is_land = phase == self.LAND

        # ── Phase 0: TAKEOFF ─────────────────────────────────────────────────
        alt_progress = altitude - self._prev_alt
        self._prev_alt = altitude.clone()
        takeoff_ascent = torch.where(
            is_takeoff, alt_progress * self.cfg.takeoff_ascent_scale,
            torch.zeros_like(alt_progress),
        )
        takeoff_drift = torch.where(
            is_takeoff, horizontal_speed * self.cfg.takeoff_drift_penalty * dt,
            torch.zeros_like(horizontal_speed),
        )

        # ── Phase 1: STABILIZE ───────────────────────────────────────────────
        stabilize_position = torch.where(
            is_stabilize, torch.exp(-start_xy_dist) * self.cfg.stabilize_position_scale * dt,
            torch.zeros_like(start_xy_dist),
        )
        stabilize_speed = torch.where(
            is_stabilize, torch.exp(-speed) * self.cfg.stabilize_low_speed_scale * dt,
            torch.zeros_like(speed),
        )
        stabilize_altitude = torch.where(
            is_stabilize, alt_error * self.cfg.stabilize_altitude_scale * dt,
            torch.zeros_like(alt_error),
        )
        stabilize_ang_vel = torch.where(
            is_stabilize, torch.exp(-ang_vel_magnitude) * self.cfg.stabilize_ang_vel_scale * dt,
            torch.zeros_like(ang_vel_magnitude),
        )

        # ── Phase 2: NAVIGATE ────────────────────────────────────────────────
        xy_progress = self._prev_xy_dist - xy_dist
        self._prev_xy_dist = xy_dist.clone()
        nav_xy_progress = torch.where(
            is_navigate, xy_progress * self.cfg.nav_xy_progress_scale,
            torch.zeros_like(xy_progress),
        )

        goal_dir_xy = delta[:, :2] / (xy_dist.unsqueeze(1) + 1e-6)
        vel_toward_goal = torch.sum(lin_vel_w[:, :2] * goal_dir_xy, dim=1)
        nav_velocity_align = torch.where(
            is_navigate,
            torch.clamp(vel_toward_goal, min=0.0) * self.cfg.nav_velocity_align_scale * dt,
            torch.zeros_like(vel_toward_goal),
        )

        lateral_scale = torch.where(
            is_at_final_wp,
            torch.full_like(xy_dist, self.cfg.nav_lateral_penalty_scale),
            torch.full_like(xy_dist, self.cfg.nav_lateral_penalty_scale * 0.3),
        )
        vel_lateral = lin_vel_w[:, :2] - vel_toward_goal.unsqueeze(1) * goal_dir_xy
        lateral_speed = torch.linalg.norm(vel_lateral, dim=1)
        nav_lateral_penalty = torch.where(
            is_navigate, lateral_speed * lateral_scale * dt,
            torch.zeros_like(lateral_speed),
        )

        nav_altitude = torch.where(
            is_navigate, alt_error * self.cfg.nav_altitude_scale * dt,
            torch.zeros_like(alt_error),
        )

        gravity_z = self._robot.data.projected_gravity_b[:, 2]
        nav_stability = torch.where(
            is_navigate,
            (torch.exp(-ang_vel_magnitude) + (-gravity_z)) * self.cfg.nav_stability_scale * dt,
            torch.zeros_like(ang_vel_magnitude),
        )

        excess_speed = torch.clamp(speed - self.cfg.nav_max_speed, min=0.0)
        nav_speed_penalty = torch.where(
            is_navigate, excess_speed * self.cfg.nav_speed_penalty_scale * dt,
            torch.zeros_like(excess_speed),
        )

        # ── Obstacle avoidance rewards (NAVIGATE phase only) ─────────────────
        n_active = self._active_obstacles
        obstacle_proximity = torch.zeros(self.num_envs, device=self.device)
        lidar_penalty = torch.zeros(self.num_envs, device=self.device)
        crash_penalty_reward = torch.zeros(self.num_envs, device=self.device)

        if n_active > 0:
            drone_xy = root_pos[:, :2]
            drone_z = root_pos[:, 2]
            obs_xy = self._obstacle_pos_w[:, :n_active, :]
            obs_r = self._obstacle_radii[:n_active]
            obs_h = self._obstacle_heights[:n_active]

            delta_obs = obs_xy - drone_xy.unsqueeze(1)
            dists_center = torch.linalg.norm(delta_obs, dim=2)
            surface_dists = dists_center - obs_r.unsqueeze(0)

            # Only penalize when drone is at obstacle height
            below_height = drone_z.unsqueeze(1) < obs_h.unsqueeze(0)
            effective_dists = torch.where(below_height, surface_dists, torch.full_like(surface_dists, 100.0))
            min_surface_dist = effective_dists.min(dim=1).values

            # Proximity penalty: exponential decay within safe distance
            proximity_active = min_surface_dist < self.cfg.obstacle_safe_distance
            obstacle_proximity = torch.where(
                is_navigate & proximity_active,
                torch.exp(-min_surface_dist) * self.cfg.obstacle_proximity_penalty_scale * dt,
                torch.zeros_like(min_surface_dist),
            )

            # LiDAR-based penalty: detects obstacles from any direction via LiDAR readings
            lidar_hits = self._lidar.data.ray_hits_w
            lidar_pos = self._lidar.data.pos_w.unsqueeze(1)
            lidar_dists_raw = torch.linalg.norm(lidar_hits - lidar_pos, dim=-1)
            lidar_dists_aug = self._augment_lidar_with_obstacles(lidar_dists_raw, root_pos, self._robot.data.root_quat_w)

            # Focus on horizontal rays (top 1/3 of ray indices = 0° elevation channel)
            n_rays = lidar_dists_aug.shape[1]
            n_per_ch = n_rays // self.cfg.lidar_n_channels
            horiz_rays = lidar_dists_aug[:, 2 * n_per_ch:]  # last channel = 0° elevation
            min_horiz = horiz_rays.min(dim=1).values

            lidar_danger = min_horiz < self.cfg.lidar_danger_distance
            lidar_closeness = (1.0 - min_horiz / self.cfg.lidar_danger_distance).clamp(0, 1)
            lidar_penalty = torch.where(
                is_navigate & lidar_danger,
                lidar_closeness * self.cfg.lidar_obstacle_penalty_scale * dt,
                torch.zeros_like(min_horiz),
            )

            # Crash penalty
            crash_dist = effective_dists.min(dim=1).values
            is_crashing = crash_dist < self.cfg.obstacle_collision_radius
            crash_penalty_reward = torch.where(
                is_crashing,
                torch.full_like(crash_dist, self.cfg.crash_penalty),
                torch.zeros_like(crash_dist),
            )

        # ── Waypoint advancement ─────────────────────────────────────────────
        just_advanced = self._check_waypoint_advancement(xy_dist, speed)
        waypoint_bonus = just_advanced.float() * self.cfg.intermediate_waypoint_bonus
        speed_carry = torch.where(
            just_advanced,
            torch.clamp(speed, min=0.0, max=self.cfg.nav_max_speed) * self.cfg.speed_carry_bonus_scale,
            torch.zeros_like(speed),
        )
        fraction_done = self._current_wp_idx.float() / self._num_waypoints.float().clamp(min=1)
        wp_progress_reward = torch.where(
            is_navigate, fraction_done * self.cfg.waypoint_progress_scale * dt,
            torch.zeros_like(fraction_done),
        )

        # ── Phase 3: HOVER ───────────────────────────────────────────────────
        hover_position = torch.where(
            is_hover, torch.exp(-xy_dist) * self.cfg.hover_position_scale * dt,
            torch.zeros_like(xy_dist),
        )
        hover_speed = torch.where(
            is_hover, torch.exp(-speed) * self.cfg.hover_low_speed_scale * dt,
            torch.zeros_like(speed),
        )
        hover_altitude = torch.where(
            is_hover, alt_error * self.cfg.hover_altitude_scale * dt,
            torch.zeros_like(alt_error),
        )

        # ── Phase 4: LAND ───────────────────────────────────────────────────
        descent_progress = self._prev_z_dist - z_dist
        self._prev_z_dist = z_dist.clone()
        land_descent = torch.where(
            is_land, descent_progress * self.cfg.land_descent_scale,
            torch.zeros_like(descent_progress),
        )
        land_xy_stability = torch.where(
            is_land, torch.exp(-xy_dist) * self.cfg.land_xy_stability_scale * dt,
            torch.zeros_like(xy_dist),
        )
        land_drift = torch.where(
            is_land, horizontal_speed * self.cfg.land_drift_penalty * dt,
            torch.zeros_like(horizontal_speed),
        )
        land_precision = torch.where(
            is_land, torch.exp(-dist_3d * 5.0) * self.cfg.land_precision_scale * dt,
            torch.zeros_like(dist_3d),
        )
        vertical_speed = torch.abs(lin_vel_w[:, 2])
        excess_descent = torch.clamp(vertical_speed - self.cfg.land_max_descent_speed, min=0.0)
        land_descent_control = torch.where(
            is_land, excess_descent * self.cfg.land_descent_speed_penalty * dt,
            torch.zeros_like(excess_descent),
        )
        descent_vel = -lin_vel_w[:, 2]
        ideal_descent = torch.exp(-((descent_vel - 0.35) ** 2) / 0.05)
        land_controlled_descent = torch.where(
            is_land, ideal_descent * self.cfg.land_controlled_descent_scale * dt,
            torch.zeros_like(ideal_descent),
        )
        land_altitude_penalty = torch.where(
            is_land, z_dist * self.cfg.land_altitude_penalty_scale * dt,
            torch.zeros_like(z_dist),
        )

        # Goal reached
        final_delta = self._final_goal_pos_w - root_pos
        final_dist_3d = torch.linalg.norm(final_delta, dim=1)
        goal_reached = is_land & (final_dist_3d < self.cfg.goal_threshold) & (speed < self.cfg.landing_speed_threshold)
        goal_bonus = goal_reached.float() * self.cfg.goal_reached_bonus

        # ── All-phase rewards ────────────────────────────────────────────────
        time_penalty = torch.full_like(dist_3d, self.cfg.time_penalty * dt)
        ang_vel = torch.sum(torch.square(ang_vel_vec), dim=1)
        orient_scale = torch.ones_like(dist_3d)
        orient_scale = torch.where(is_navigate, 0.3 * orient_scale, orient_scale)
        orient_scale = torch.where(is_hover | is_land, 1.5 * orient_scale, orient_scale)
        ang_vel_penalty = ang_vel * self.cfg.ang_vel_reward_scale * dt * orient_scale
        yaw_rate = torch.abs(ang_vel_vec[:, 2])
        yaw_penalty = yaw_rate * self.cfg.yaw_rate_penalty_scale * dt
        upright_reward = -gravity_z * self.cfg.upright_reward_scale * dt * orient_scale

        # ── Phase transitions ────────────────────────────────────────────────
        to_stabilize = is_takeoff & (alt_error < self.cfg.takeoff_altitude_tolerance)
        self._phase[to_stabilize] = self.STABILIZE
        self._stabilize_timer[to_stabilize] = 0.0

        self._stabilize_timer[self._phase == self.STABILIZE] += dt
        to_navigate = (self._phase == self.STABILIZE) & (self._stabilize_timer >= self.cfg.stabilize_duration)
        self._phase[to_navigate] = self.NAVIGATE

        current_delta = self._goal_pos_w - root_pos
        current_xy_dist = torch.linalg.norm(current_delta[:, :2], dim=1)
        altitude_now = self._robot.data.root_pos_w[:, 2]
        to_hover = (
            (self._phase == self.NAVIGATE)
            & (self._current_wp_idx >= (self._num_waypoints - 1))
            & (current_xy_dist < self.cfg.final_waypoint_tolerance)
            & (speed < 1.5)
            & (altitude_now > 1.0)
        )
        self._phase[to_hover] = self.HOVER
        self._hover_timer[to_hover] = 0.0

        self._hover_timer[self._phase == self.HOVER] += dt
        to_land = (
            (self._phase == self.HOVER)
            & (self._hover_timer >= self.cfg.hover_duration)
            & (current_xy_dist < self.cfg.final_waypoint_tolerance)
        )
        self._phase[to_land] = self.LAND
        self._goal_pos_w[to_land] = self._final_goal_pos_w[to_land]
        if to_land.any():
            land_delta = self._final_goal_pos_w[to_land] - root_pos[to_land]
            self._prev_z_dist[to_land] = torch.abs(land_delta[:, 2])

        # ── Total ────────────────────────────────────────────────────────────
        reward = (
            takeoff_ascent + takeoff_drift
            + stabilize_position + stabilize_speed + stabilize_altitude + stabilize_ang_vel
            + nav_xy_progress + nav_velocity_align + nav_lateral_penalty + nav_altitude
            + nav_stability + nav_speed_penalty
            + obstacle_proximity + lidar_penalty + crash_penalty_reward
            + waypoint_bonus + speed_carry + wp_progress_reward
            + hover_position + hover_speed + hover_altitude
            + land_descent + land_xy_stability + land_drift + land_precision
            + land_descent_control + land_controlled_descent + land_altitude_penalty + goal_bonus
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
        self._episode_sums["obstacle_proximity"] += obstacle_proximity
        self._episode_sums["lidar_penalty"] += lidar_penalty
        self._episode_sums["crash_penalty"] += crash_penalty_reward
        self._episode_sums["waypoint_bonus"] += waypoint_bonus
        self._episode_sums["speed_carry"] += speed_carry
        self._episode_sums["wp_progress"] += wp_progress_reward
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

    # ── Terminations ─────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        too_low = (
            (self._robot.data.root_pos_w[:, 2] < self.cfg.min_flight_height)
            & (self._phase != self.TAKEOFF)
            & (self._phase != self.LAND)
        )
        too_high = self._robot.data.root_pos_w[:, 2] > self.cfg.max_flight_height
        flipped = self._robot.data.projected_gravity_b[:, 2] > 0.7

        # Obstacle collision
        hit_obstacle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        n_active = self._active_obstacles
        if n_active > 0:
            root_pos = self._robot.data.root_pos_w
            drone_xy = root_pos[:, :2]
            drone_z = root_pos[:, 2]

            obs_xy = self._obstacle_pos_w[:, :n_active, :]
            obs_r = self._obstacle_radii[:n_active]
            obs_h = self._obstacle_heights[:n_active]

            delta_obs = obs_xy - drone_xy.unsqueeze(1)
            dists_center = torch.linalg.norm(delta_obs, dim=2)
            surface_dists = dists_center - obs_r.unsqueeze(0)

            below_height = drone_z.unsqueeze(1) < obs_h.unsqueeze(0)
            effective_dists = torch.where(below_height, surface_dists, torch.full_like(surface_dists, 100.0))
            min_dist = effective_dists.min(dim=1).values
            hit_obstacle = min_dist < self.cfg.obstacle_collision_radius

        # Goal reached
        goal_dist = torch.linalg.norm(self._final_goal_pos_w - self._robot.data.root_pos_w, dim=1)
        speed = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        goal_reached = (
            (self._phase == self.LAND) & (goal_dist < self.cfg.goal_threshold)
            & (speed < self.cfg.landing_speed_threshold)
        )

        touched_ground = (
            ((self._phase == self.HOVER) | (self._phase == self.LAND))
            & (self._robot.data.root_pos_w[:, 2] < 0.05)
        )

        died = too_low | too_high | flipped | hit_obstacle

        # Track crash phase
        if died.any():
            phase_names = ["TAKEOFF", "STABILIZE", "NAVIGATE", "HOVER", "LAND"]
            for idx in torch.where(died)[0]:
                p = self._phase[idx].item()
                if p < len(phase_names):
                    self._crash_phase_counts[phase_names[p]] += 1

        return died, time_out | goal_reached | touched_ground

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance = torch.linalg.norm(
            self._final_goal_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        wp_completion_rate = self._current_wp_idx[env_ids].float() / self._num_waypoints[env_ids].float().clamp(min=1)

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_distance.item()
        self.extras["log"]["Metrics/final_phase_mean"] = self._phase[env_ids].float().mean().item()
        self.extras["log"]["Metrics/waypoint_completion_rate"] = wp_completion_rate.mean().item()
        self.extras["log"]["Metrics/avg_waypoints_reached"] = self._current_wp_idx[env_ids].float().mean().item()
        self.extras["log"]["Metrics/active_obstacles"] = float(self._active_obstacles)

        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0

        # Generate waypoints
        self._generate_waypoints(env_ids)

        # Set start position
        num_reset = len(env_ids)
        start_local = torch.tensor(self.cfg.static_start_pos, dtype=torch.float32, device=self.device)
        start_pos = start_local.unsqueeze(0).expand(num_reset, -1).clone()
        start_pos[:, :2] += self._terrain.env_origins[env_ids, :2]
        self._start_pos_w[env_ids] = start_pos

        # Initialize waypoint tracking
        self._current_wp_idx[env_ids] = 0
        self._update_current_goal(env_ids)

        delta = self._goal_pos_w[env_ids] - start_pos
        self._prev_xy_dist[env_ids] = torch.linalg.norm(delta[:, :2], dim=1)
        self._prev_z_dist[env_ids] = torch.abs(delta[:, 2])
        self._prev_alt[env_ids] = start_pos[:, 2]

        # Reset phase state
        self._phase[env_ids] = self.TAKEOFF
        self._stabilize_timer[env_ids] = 0.0
        self._hover_timer[env_ids] = 0.0

        # Compute obstacle world positions
        env_origins_xy = self._terrain.env_origins[env_ids, :2]
        for oi in range(self._num_obstacles):
            self._obstacle_pos_w[env_ids, oi, 0] = self._obstacle_local[oi, 0] + env_origins_xy[:, 0]
            self._obstacle_pos_w[env_ids, oi, 1] = self._obstacle_local[oi, 1] + env_origins_xy[:, 1]

        # Set robot state
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = start_pos
        default_root_state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ── Debug Visualization ──────────────────────────────────────────────────

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.5, 0.5, 0.5)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "waypoint_visualizers"):
                self.waypoint_visualizers = []
                for wp_idx in range(self.cfg.max_num_waypoints):
                    wp_marker_cfg = CUBOID_MARKER_CFG.copy()
                    wp_marker_cfg.markers["cuboid"].size = (0.3, 0.3, 0.3)
                    wp_marker_cfg.prim_path = f"/Visuals/Command/waypoint_{wp_idx}"
                    self.waypoint_visualizers.append(VisualizationMarkers(wp_marker_cfg))

            self.goal_pos_visualizer.set_visibility(True)
            for viz in self.waypoint_visualizers:
                viz.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "waypoint_visualizers"):
                for viz in self.waypoint_visualizers:
                    viz.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._goal_pos_w)
        for wp_idx in range(self.cfg.max_num_waypoints):
            wp_positions = self._waypoints_w[:, wp_idx, :]
            valid = wp_idx < self._num_waypoints
            viz_pos = torch.where(
                valid.unsqueeze(1).expand(-1, 3),
                wp_positions,
                torch.full_like(wp_positions, -1000.0),
            )
            self.waypoint_visualizers[wp_idx].visualize(viz_pos)


# ══════════════════════════════════════════════════════════════════════════════
# Gym Registration
# ══════════════════════════════════════════════════════════════════════════════

gym.register(
    id="Isaac-CrazyflieObstacleCurriculum-SAC-v0",
    entry_point=f"{__name__}:ObstacleCurriculumEnv",
    disable_env_checker=True,
    kwargs={"cfg": ObstacleCurriculumEnvCfg()},
)


# ══════════════════════════════════════════════════════════════════════════════
# Curriculum Callback
# ══════════════════════════════════════════════════════════════════════════════

class CurriculumCallback(BaseCallback):
    """Updates obstacle count during training based on curriculum schedule."""

    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self._last_active = -1

    def _on_step(self) -> bool:
        env = self.training_env.unwrapped
        env.set_curriculum_step(self.num_timesteps, self.total_timesteps)

        current = env._active_obstacles
        if current != self._last_active:
            print(f"[CURRICULUM] Step {self.num_timesteps}/{self.total_timesteps}: "
                  f"Active obstacles: {current}/{env._num_obstacles}")
            self._last_active = current
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Train / Play / Eval
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def train():
    env_cfg = ObstacleCurriculumEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    log_root = os.path.join(SCRIPT_DIR, "logs", "sac_curriculum", "crazyflie_obstacle_curriculum_sac")
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Logging to: {log_dir}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Total timesteps: {args_cli.total_timesteps}")
    print(f"[INFO] Obstacles: {len(env_cfg.obstacle_positions)} (curriculum: "
          f"{env_cfg.curriculum_start_frac*100:.0f}%-{env_cfg.curriculum_end_frac*100:.0f}% of training)")
    print(f"[INFO] LiDAR: {env_cfg.lidar_n_channels}ch × 35 horiz = 105 rays, 10° resolution")
    print(f"[INFO] Architecture: single-level SAC with CNN feature extractor")

    env = gym.make("Isaac-CrazyflieObstacleCurriculum-SAC-v0", cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    if args_cli.checkpoint:
        print(f"[INFO] Resuming from: {args_cli.checkpoint}")
        agent = SAC.load(args_cli.checkpoint, env=env, device="cuda:0")
    else:
        agent = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.98,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
            policy_kwargs=dict(
                features_extractor_class=LidarCNNExtractor,
                features_extractor_kwargs=dict(
                    lidar_dim=105,
                    cnn_out_dim=32,
                    features_dim=128,
                    lidar_channels=env_cfg.lidar_n_channels,
                ),
                net_arch=[256, 128],
            ),
            verbose=1,
            seed=args_cli.seed,
            device="cuda:0",
        )

    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args_cli.num_envs, 1),
        save_path=log_dir,
        name_prefix="sac_model",
    )
    curriculum_callback = CurriculumCallback(total_timesteps=args_cli.total_timesteps)

    start_time = time.time()
    agent.learn(
        total_timesteps=args_cli.total_timesteps,
        callback=[checkpoint_callback, curriculum_callback],
        progress_bar=True,
    )
    elapsed = time.time() - start_time

    final_path = os.path.join(log_dir, "sac_final")
    agent.save(final_path)

    print(f"\n[INFO] Training complete! Duration: {elapsed:.1f}s")
    print(f"[INFO] Final model saved to: {final_path}.zip")
    print(f"[INFO] Logs saved to: {log_dir}")
    env.close()


def play():
    if not args_cli.checkpoint:
        print("[ERROR] --checkpoint is required for play mode.")
        return

    env_cfg = ObstacleCurriculumEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs != 256 else 1
    env_cfg.episode_length_s = 40.0
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    env = gym.make("Isaac-CrazyflieObstacleCurriculum-SAC-v0", cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    # Force all obstacles active for play/eval
    env.unwrapped._active_obstacles = env.unwrapped._num_obstacles

    agent = SAC.load(args_cli.checkpoint, env=env, device="cuda:0")
    obs = env.reset()

    episode_count = 0
    total_reward = 0.0
    step_count = 0
    playback_dt = env_cfg.sim.dt * env_cfg.decimation * 5.0

    print(f"\n{'='*60}")
    print(f"[INFO] Obstacle SAC (curriculum) policy loaded!")
    print(f"[INFO] Active obstacles: {env.unwrapped._active_obstacles}")
    print(f"[INFO] Press Ctrl+C to stop.")
    print(f"{'='*60}\n")

    try:
        while simulation_app.is_running():
            step_start = time.time()
            actions, _ = agent.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            total_reward += rewards.sum().item()
            step_count += 1

            if step_count % 50 == 0:
                unwrapped = env.unwrapped
                pos = unwrapped._robot.data.root_pos_w[0].cpu().numpy()
                phase = unwrapped._phase[0].item()
                wp_idx = unwrapped._current_wp_idx[0].item()
                num_wps = unwrapped._num_waypoints[0].item()
                phase_names = ["TAKEOFF", "STABILIZE", "NAVIGATE", "HOVER", "LAND"]
                print(f"  Step {step_count} | {phase_names[phase]} | WP {wp_idx}/{num_wps} | "
                      f"Pos ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}) | R={rewards[0].item():.3f}")

            if dones.any():
                episode_count += dones.sum().item()
                print(f"  === Episode {int(episode_count)} done (step {step_count}) ===")

            elapsed = time.time() - step_start
            sleep_time = playback_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    avg_reward = total_reward / max(episode_count, 1)
    print(f"\n[RESULT] Average reward: {avg_reward:.2f} over {int(episode_count)} episodes")
    env.close()


def evaluate():
    if not args_cli.checkpoint:
        print("[ERROR] --checkpoint is required for eval mode.")
        return

    num_eval_envs = min(args_cli.num_envs, 16) if args_cli.num_envs == 256 else args_cli.num_envs

    env_cfg = ObstacleCurriculumEnvCfg()
    env_cfg.scene.num_envs = num_eval_envs
    env_cfg.episode_length_s = 40.0
    env_cfg.debug_vis = False
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    env = gym.make("Isaac-CrazyflieObstacleCurriculum-SAC-v0", cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    # Force all obstacles active for eval
    env.unwrapped._active_obstacles = env.unwrapped._num_obstacles

    agent = SAC.load(args_cli.checkpoint, env=env, device="cuda:0")
    obs = env.reset()

    total_episodes = 0
    episode_rewards = []
    final_distances = []
    episode_lengths = []
    crashes = 0
    goal_reached_count = 0
    wp_completion_rates = []

    env_rewards = np.zeros(num_eval_envs)
    env_steps = np.zeros(num_eval_envs, dtype=int)
    step_dt = env_cfg.sim.dt * env_cfg.decimation

    # Reset crash phase tracking
    env.unwrapped._crash_phase_counts = {
        "TAKEOFF": 0, "STABILIZE": 0, "NAVIGATE": 0, "HOVER": 0, "LAND": 0,
    }

    print(f"\n{'='*60}")
    print(f"EVALUATION: Obstacle SAC (Curriculum) | {args_cli.num_episodes} episodes")
    print(f"Checkpoint: {os.path.basename(args_cli.checkpoint)}")
    print(f"Active obstacles: {env.unwrapped._active_obstacles}")
    print(f"{'='*60}\n")

    while total_episodes < args_cli.num_episodes and simulation_app.is_running():
        actions, _ = agent.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy().flatten()
            dones_np = dones.cpu().numpy().flatten()
        else:
            rewards_np = np.array(rewards).flatten()
            dones_np = np.array(dones).flatten()

        env_rewards += rewards_np
        env_steps += 1

        for i in range(num_eval_envs):
            if dones_np[i] and total_episodes < args_cli.num_episodes:
                total_episodes += 1
                episode_rewards.append(env_rewards[i])
                episode_lengths.append(env_steps[i] * step_dt)

                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'extras'):
                    extras = env.unwrapped.extras.get("log", {})
                    dist = extras.get("Metrics/final_distance_to_goal", -1)
                    died_count = extras.get("Episode_Termination/died", 0)
                    wp_rate = extras.get("Metrics/waypoint_completion_rate", 0)
                    final_distances.append(dist)
                    wp_completion_rates.append(wp_rate)
                    if died_count > 0:
                        crashes += 1
                    if dist >= 0 and dist < env_cfg.goal_threshold:
                        goal_reached_count += 1

                if total_episodes % 10 == 0 or total_episodes == args_cli.num_episodes:
                    print(f"  Episodes: {total_episodes}/{args_cli.num_episodes}")

                env_rewards[i] = 0.0
                env_steps[i] = 0

    n = len(episode_rewards)
    crash_counts = env.unwrapped._crash_phase_counts

    print(f"\n{'='*60}")
    print(f"RESULTS -- Obstacle SAC Curriculum ({n} episodes)")
    print(f"{'='*60}")
    print(f"  Success Rate:           {goal_reached_count/n*100:5.1f}%  ({goal_reached_count}/{n})")
    print(f"  Crash Rate:             {crashes/n*100:5.1f}%  ({crashes}/{n})")
    print(f"  Crashes by Phase:")
    for phase_name in ["TAKEOFF", "STABILIZE", "NAVIGATE", "HOVER", "LAND"]:
        cnt = crash_counts.get(phase_name, 0)
        if cnt > 0:
            print(f"    {phase_name:12s}:  {cnt:6d}")
    if wp_completion_rates:
        print(f"  Waypoint Completion:    {np.mean(wp_completion_rates)*100:5.1f}%")
    if final_distances:
        print(f"  Avg Final Distance:   {np.mean(final_distances):7.3f} m")
        print(f"  Min Final Distance:   {np.min(final_distances):7.3f} m")
    print(f"  Avg Reward:             {np.mean(episode_rewards):8.2f}  (+/- {np.std(episode_rewards):.2f})")
    print(f"  Avg Episode Length:     {np.mean(episode_lengths):6.2f} s")
    print(f"{'='*60}")

    env.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" or True:
    if args_cli.mode == "train":
        train()
    elif args_cli.mode == "play":
        play()
    elif args_cli.mode == "eval":
        evaluate()

    simulation_app.close()
