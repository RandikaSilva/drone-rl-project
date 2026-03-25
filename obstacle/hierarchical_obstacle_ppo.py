"""Hierarchical Crazyflie obstacle avoidance + multi-waypoint navigation with PPO (RSL-RL).

Two-level control architecture:
    HIGH-LEVEL (trained):  LiDAR 450 rays + state → local sub-goal (3D offset)
    LOW-LEVEL  (frozen):   Sub-goal → velocity commands (vx, vy, vz, yaw_rate)
    PID        (fixed):    Velocity commands → thrust/torques

The frozen low-level is a pre-trained multi-waypoint SAC policy loaded from checkpoint.
During NAVIGATE phase, the high-level picks obstacle-free sub-goals that the low-level tracks.
During other phases (TAKEOFF/STABILIZE/HOVER/LAND), the low-level runs directly with the real goal.

Decision frequencies:
    High-level:  10 Hz  (one sub-goal per env.step)
    Low-level:   50 Hz  (5 decisions per env.step, every 2 sim steps)
    PID:        100 Hz  (every sim step)

Usage (from IsaacLab directory):
    cd ~/projects/isaac/IsaacLab
    source ~/projects/isaac/env_isaaclab/bin/activate

    # Train high-level:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_ppo.py \
        --mode train --num_envs 512 --max_iterations 3000 --headless

    # Play:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_ppo.py \
        --mode play --checkpoint /path/to/model_2999.pt

    # Eval:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_ppo.py \
        --mode eval --checkpoint /path/to/model_2999.pt --num_episodes 50
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

# ── Argument Parser ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hierarchical Crazyflie Obstacle Nav — PPO (RSL-RL)")
parser.add_argument("--mode", type=str, default="train", choices=["train", "play", "eval"])
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None, help="High-level checkpoint (play/eval/resume).")
parser.add_argument("--ll_checkpoint", type=str, default=None,
                    help="Low-level SAC checkpoint. Defaults to latest multi_waypoint_sac final.")
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--min_waypoints", type=int, default=3)
parser.add_argument("--max_waypoints", type=int, default=5)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# ── Launch Sim ───────────────────────────────────────────────────────────────
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
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.markers import CUBOID_MARKER_CFG

try:
    from isaacsim.util.debug_draw import _debug_draw
except (ImportError, ModuleNotFoundError):
    try:
        from omni.isaac.debug_draw import _debug_draw
    except (ImportError, ModuleNotFoundError):
        _debug_draw = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default low-level checkpoint (latest multi_waypoint SAC)
DEFAULT_LL_CHECKPOINT = os.path.join(
    SCRIPT_DIR, "..", "multi_waypoint", "logs", "sac",
    "crazyflie_multi_waypoint_nav_sac", "2026-03-08_16-11-44", "sac_final.zip"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Robot Configuration
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Environment Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@configclass
class HierarchicalObstacleNavEnvCfg(DirectRLEnvCfg):
    """Hierarchical obstacle avoidance: high-level at 10 Hz, low-level at 50 Hz."""

    episode_length_s = 40.0

    # decimation=2 matches multi_waypoint training exactly (50 Hz env.step)
    # High-level goal modifier updates every 5 env.steps internally (10 Hz)
    decimation = 2

    # High-level spaces
    # obs: lidar_449 + lin_vel_b(3) + ang_vel_b(3) + gravity_b(3) + goal_pos_b(3)
    #      + phase_one_hot(5) + nearest_obstacle(3) = 449 + 20 = 469
    # Note: horizontal_res=0.8 on (0,360) gives 449 rays in Isaac Lab (not 450)
    action_space = 3   # goal modifier: yaw_offset, speed_factor, altitude_offset
    observation_space = 469
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
        env_spacing=70.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = VISIBLE_CRAZYFLIE_CFG
    thrust_to_weight = 1.9
    moment_scale = 0.04

    # PID controller — MUST match low-level training values (2.0) or policy breaks
    max_velocity_xy: float = 2.0
    max_velocity_z: float = 1.0
    max_yaw_rate: float = 1.5
    pid_vel_kp: float = 0.25
    pid_att_kp: float = 6.0
    pid_att_kd: float = 1.0
    pid_vz_kp: float = 0.5
    pid_yaw_kp: float = 0.4
    pid_max_tilt: float = 0.5

    # ── High-Level LiDAR (450 horizontal rays at 10 Hz) ─────────────────────
    hl_lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),       # horizontal ring only
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=0.8,                   # 360 / 0.8 = 450 rays
        ),
        max_distance=20.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    hl_lidar_max_distance: float = 20.0

    # ── Low-Level LiDAR (9 rays, matching multi_waypoint_sac training) ───────
    ll_lidar = RayCasterCfg(
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
    ll_lidar_max_distance: float = 15.0

    # ── Goal Modifier (high-level adjusts direction/speed toward real goal) ──
    max_yaw_offset: float = 1.047  # radians (~60°) — max angular deviation from direct path
    speed_factor_range: tuple = (0.3, 1.5)  # multiplier on goal distance for low-level
    max_altitude_offset: float = 1.0  # meters — vertical adjustment

    # ── Navigation ───────────────────────────────────────────────────────────
    goal_threshold: float = 0.5
    landing_speed_threshold: float = 1.0
    max_flight_height: float = 8.0
    min_flight_height: float = 0.05
    cruise_altitude: float = 1.5

    takeoff_altitude_tolerance: float = 0.3
    stabilize_duration: float = 2.0
    hover_position_tolerance: float = 0.5
    hover_duration: float = 2.0

    # ── Multi-Waypoint ───────────────────────────────────────────────────────
    static_start_pos: tuple = (-7.0, 0.0, 0.2)
    min_num_waypoints: int = 3
    max_num_waypoints: int = 5
    arena_x_range: tuple = (-8.0, 8.0)
    arena_y_range: tuple = (-8.0, 8.0)
    min_y_change: float = 2.5
    intermediate_waypoint_tolerance: float = 2.0
    final_waypoint_tolerance: float = 0.5
    waypoint_obstacle_clearance: float = 3.0

    # ── Obstacles ────────────────────────────────────────────────────────────
    obstacle_positions: list = [
        # Top wall
        (-6.0, 10.0, 0.50, 5.0),
        ( 0.0, 10.0, 0.50, 5.0),
        ( 6.0, 10.0, 0.50, 5.0),
        # Bottom wall
        (-6.0, -10.0, 0.50, 5.0),
        ( 0.0, -10.0, 0.50, 5.0),
        ( 6.0, -10.0, 0.50, 5.0),
        # Left wall
        (-9.0, -7.0, 0.50, 5.0),
        (-9.0,  7.0, 0.50, 5.0),
        # Right wall
        ( 9.0, -7.0, 0.50, 5.0),
        ( 9.0,  7.0, 0.50, 5.0),
        # Interior
        (-5.0,  4.0, 0.50, 5.0),
        (-5.0, -4.0, 0.50, 5.0),
        ( 5.0,  4.0, 0.50, 5.0),
        ( 5.0, -4.0, 0.50, 5.0),
    ]
    obstacle_collision_radius: float = 0.5
    obstacle_safe_distance: float = 1.5

    # ── High-Level Reward Scales ─────────────────────────────────────────────
    # NAVIGATE phase (high-level is active)
    nav_xy_progress_scale: float = 5.0
    nav_velocity_align_scale: float = 6.0
    nav_altitude_scale: float = -1.5
    obstacle_proximity_penalty_scale: float = -25.0
    lidar_obstacle_penalty_scale: float = -25.0
    lidar_danger_distance: float = 3.5
    subgoal_reachability_scale: float = 5.0
    subgoal_magnitude_penalty: float = -0.3
    crash_penalty: float = -500.0
    intermediate_waypoint_bonus: float = 200.0
    speed_carry_bonus_scale: float = 5.0
    waypoint_progress_scale: float = 3.0
    nav_max_speed: float = 1.5
    nav_speed_penalty_scale: float = -3.0
    nav_lateral_penalty_scale: float = -1.0            # penalise lateral drift
    nav_stability_scale: float = 2.0                   # reward low ang-vel + upright
    # Non-NAVIGATE phases (low-level runs directly)
    takeoff_ascent_scale: float = 20.0
    takeoff_altitude_reward_scale: float = 5.0
    takeoff_drift_penalty: float = -2.0
    stabilize_position_scale: float = 3.0
    stabilize_low_speed_scale: float = 2.0
    stabilize_altitude_scale: float = -2.0
    stabilize_ang_vel_scale: float = 1.0
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
    goal_reached_bonus: float = 1000.0
    # All phases
    time_penalty: float = -1.5
    ang_vel_reward_scale: float = -0.5
    yaw_rate_penalty_scale: float = -1.0
    upright_reward_scale: float = 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalObstacleNavEnv(DirectRLEnv):
    """Hierarchical obstacle avoidance: high-level sub-goal planner + frozen low-level tracker.

    env.step() runs at 10 Hz (decimation=10).
    Inside _apply_action(), the low-level policy runs every 2 sim steps (50 Hz)
    and the PID controller runs every sim step (100 Hz).
    """

    cfg: HierarchicalObstacleNavEnvCfg

    TAKEOFF = 0
    STABILIZE = 1
    NAVIGATE = 2
    HOVER = 3
    LAND = 4

    def __init__(self, cfg: HierarchicalObstacleNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        max_wps = self.cfg.max_num_waypoints

        # ── Buffers ──────────────────────────────────────────────────────────
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

        # High-level goal modifier (yaw_offset, speed_factor, alt_offset)
        self._hl_yaw_offset = torch.zeros(self.num_envs, device=self.device)
        self._hl_speed_factor = torch.ones(self.num_envs, device=self.device)
        self._hl_alt_offset = torch.zeros(self.num_envs, device=self.device)

        # Low-level velocity commands (updated every 2 sim steps in _apply_action)
        self._ll_vx_cmd = torch.zeros(self.num_envs, device=self.device)
        self._ll_vy_cmd = torch.zeros(self.num_envs, device=self.device)
        self._ll_vz_cmd = torch.zeros(self.num_envs, device=self.device)
        self._ll_yaw_rate_cmd = torch.zeros(self.num_envs, device=self.device)

        # High-level update counter (updates every 5 env.steps = 10 Hz)
        self._hl_step_counter = 0

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

        # LiDAR ray direction cache
        self._hl_lidar_ray_dirs = None
        self._hl_lidar_horiz_indices = None
        self._min_horiz_lidar_dist = torch.full(
            (self.num_envs,), self.cfg.hl_lidar_max_distance, device=self.device
        )

        # Debug draw
        self._draw = None

        # ── Load frozen low-level SAC policy ─────────────────────────────────
        ll_path = args_cli.ll_checkpoint or DEFAULT_LL_CHECKPOINT
        ll_path = os.path.abspath(ll_path)
        print(f"[INFO] Loading frozen low-level SAC from: {ll_path}")
        ll_agent = SAC.load(ll_path, device=self.device)
        # Extract the actor network for fast GPU inference
        self._ll_actor = ll_agent.policy.actor
        self._ll_actor.eval()
        for param in self._ll_actor.parameters():
            param.requires_grad = False
        # Also keep the observation normalizer if present
        self._ll_obs_normalize = None
        if hasattr(ll_agent.policy, 'normalize_observations') and ll_agent.policy.normalize_observations:
            self._ll_obs_normalize = ll_agent.policy.obs_rms
        print(f"[INFO] Low-level policy frozen ({sum(p.numel() for p in self._ll_actor.parameters())} params)")

        # Episode logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "takeoff_ascent", "takeoff_alt_reward", "takeoff_drift",
                "stabilize_position", "stabilize_speed", "stabilize_altitude", "stabilize_ang_vel",
                "nav_xy_progress", "nav_velocity_align", "nav_altitude",
                "nav_speed_penalty", "nav_lateral", "nav_stability",
                "obstacle_proximity", "lidar_penalty",
                "subgoal_reachability", "subgoal_magnitude",
                "waypoint_bonus", "speed_carry", "wp_progress",
                "hover_position", "hover_speed", "hover_altitude",
                "land_descent", "land_xy_stability", "land_drift", "land_precision",
                "land_descent_control", "land_controlled_descent", "land_altitude_penalty",
                "goal_bonus", "crash_penalty", "time_penalty", "ang_vel", "yaw_rate", "upright",
            ]
        }

        self.set_debug_vis(self.cfg.debug_vis)

    # ── Scene Setup ──────────────────────────────────────────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Two LiDAR sensors
        self._hl_lidar = RayCaster(self.cfg.hl_lidar)
        self.scene.sensors["hl_lidar"] = self._hl_lidar

        self._ll_lidar = RayCaster(self.cfg.ll_lidar)
        self.scene.sensors["ll_lidar"] = self._ll_lidar

        # Spawn tree obstacles
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

        sun_cfg = sim_utils.DistantLightCfg(intensity=2500.0, color=(0.9, 0.85, 0.7), angle=0.53)
        sun_cfg.func("/World/Sun", sun_cfg, translation=(0, 0, 20))
        sky_cfg = sim_utils.DomeLightCfg(
            intensity=800.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        sky_cfg.func("/World/SkyDome", sky_cfg)
        self.sim.set_camera_view(eye=[0.0, -25.0, 10.0], target=[0.0, 0.0, 2.0])

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

        # Obstacle clearance nudging
        obs_xy = self._obstacle_local
        obs_r = self._obstacle_radii
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

    # ── LiDAR Obstacle Augmentation ──────────────────────────────────────────

    def _init_hl_lidar_rays(self, n_rays: int):
        pcfg = self.cfg.hl_lidar.pattern_cfg
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
        self._hl_lidar_ray_dirs = torch.stack(dirs)
        self._hl_lidar_horiz_indices = torch.where(
            self._hl_lidar_ray_dirs[:, 2].abs() < 0.1
        )[0]

    def _augment_hl_lidar_with_obstacles(
        self, lidar_distances: torch.Tensor,
        root_pos: torch.Tensor, root_quat: torch.Tensor,
    ) -> torch.Tensor:
        N_rays = lidar_distances.shape[1]
        if self._hl_lidar_ray_dirs is None:
            self._init_hl_lidar_rays(N_rays)

        horiz_idx = self._hl_lidar_horiz_indices
        N_proc = len(horiz_idx)
        if N_proc == 0:
            return lidar_distances

        N = self.num_envs
        proc_dirs = self._hl_lidar_ray_dirs[horiz_idx]

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

        min_t = torch.full((N, N_proc), self.cfg.hl_lidar_max_distance, device=self.device)

        for oi in range(self._num_obstacles):
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
        result[:, horiz_idx] = torch.min(lidar_distances[:, horiz_idx], min_t)
        return result

    # ── Low-Level Policy Inference ───────────────────────────────────────────

    def _build_low_level_obs(self, use_modifier_mask: torch.Tensor) -> torch.Tensor:
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w

        real_goal_b, _ = subtract_frame_transforms(root_pos, root_quat, self._goal_pos_w)

        # Apply goal modifier: rotate XY direction by yaw_offset, scale distance, adjust Z
        goal_xy = real_goal_b[:, :2]
        cos_off = torch.cos(self._hl_yaw_offset)
        sin_off = torch.sin(self._hl_yaw_offset)
        rotated_x = cos_off * goal_xy[:, 0] - sin_off * goal_xy[:, 1]
        rotated_y = sin_off * goal_xy[:, 0] + cos_off * goal_xy[:, 1]
        modified_goal_b = torch.stack([
            rotated_x * self._hl_speed_factor,
            rotated_y * self._hl_speed_factor,
            real_goal_b[:, 2] + self._hl_alt_offset,
        ], dim=1)

        goal_for_ll = torch.where(
            use_modifier_mask.unsqueeze(1).expand(-1, 3),
            modified_goal_b,
            real_goal_b,
        )

        ll_lidar_hits = self._ll_lidar.data.ray_hits_w
        ll_lidar_pos = self._ll_lidar.data.pos_w.unsqueeze(1)
        ll_lidar_dists = torch.linalg.norm(ll_lidar_hits - ll_lidar_pos, dim=-1)
        ll_lidar_norm = (ll_lidar_dists / self.cfg.ll_lidar_max_distance).clamp(0, 1)

        phase_one_hot = torch.nn.functional.one_hot(self._phase.long(), 5).float()

        return torch.cat([
            self._robot.data.root_lin_vel_b,       # 3
            self._robot.data.root_ang_vel_b,       # 3
            self._robot.data.projected_gravity_b,   # 3
            goal_for_ll,                            # 3
            ll_lidar_norm,                          # 9
            phase_one_hot,                          # 5
        ], dim=-1)  # Total: 26

    def _run_low_level_policy(self):
        is_navigate = (self._phase == self.NAVIGATE)
        ll_obs = self._build_low_level_obs(use_modifier_mask=is_navigate)

        with torch.no_grad():
            ll_actions = self._ll_actor(ll_obs, deterministic=True)
            ll_actions = ll_actions.clamp(-1.0, 1.0)

        self._ll_vx_cmd = ll_actions[:, 0] * self.cfg.max_velocity_xy
        self._ll_vy_cmd = ll_actions[:, 1] * self.cfg.max_velocity_xy
        self._ll_vz_cmd = ll_actions[:, 2] * self.cfg.max_velocity_z
        self._ll_yaw_rate_cmd = ll_actions[:, 3] * self.cfg.max_yaw_rate

    def _run_pid(self):
        vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        gravity_b = self._robot.data.projected_gravity_b

        vx_err = self._ll_vx_cmd - vel_b[:, 0]
        vy_err = self._ll_vy_cmd - vel_b[:, 1]
        vz_err = self._ll_vz_cmd - vel_b[:, 2]

        desired_roll = (self.cfg.pid_vel_kp * vy_err).clamp(-self.cfg.pid_max_tilt, self.cfg.pid_max_tilt)
        desired_pitch = (-self.cfg.pid_vel_kp * vx_err).clamp(-self.cfg.pid_max_tilt, self.cfg.pid_max_tilt)

        current_roll = gravity_b[:, 1]
        current_pitch = -gravity_b[:, 0]

        roll_torque = (
            self.cfg.pid_att_kp * (desired_roll - current_roll) - self.cfg.pid_att_kd * ang_vel_b[:, 0]
        ).clamp(-1.0, 1.0)
        pitch_torque = (
            self.cfg.pid_att_kp * (desired_pitch - current_pitch) - self.cfg.pid_att_kd * ang_vel_b[:, 1]
        ).clamp(-1.0, 1.0)
        yaw_torque = (
            self.cfg.pid_yaw_kp * (self._ll_yaw_rate_cmd - ang_vel_b[:, 2])
        ).clamp(-1.0, 1.0)

        hover_thrust = self._robot_weight
        thrust = hover_thrust * (1.0 + self.cfg.pid_vz_kp * vz_err)
        max_thrust = self.cfg.thrust_to_weight * self._robot_weight
        thrust = thrust.clamp(0.0, max_thrust)

        self._thrust[:, 0, 2] = thrust
        self._moment[:, 0, 0] = self.cfg.moment_scale * self._robot_weight * roll_torque
        self._moment[:, 0, 1] = self.cfg.moment_scale * self._robot_weight * pitch_torque
        self._moment[:, 0, 2] = self.cfg.moment_scale * self._robot_weight * yaw_torque

    # ── Physics Step (dual-rate control) ─────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        """Called once per env.step() at 50 Hz (decimation=2, matching multi_waypoint).

        High-level goal modifier updates every 5 env.steps (10 Hz).
        Low-level policy runs every env.step (50 Hz) — same as multi_waypoint training.
        PID computed here (same pattern as multi_waypoint).
        """
        # Update high-level goal modifier every 5 steps (10 Hz)
        self._hl_step_counter += 1
        if self._hl_step_counter >= 5:
            self._hl_step_counter = 0
            a = actions.clone().clamp(-1.0, 1.0)
            self._hl_yaw_offset = a[:, 0] * self.cfg.max_yaw_offset
            lo, hi = self.cfg.speed_factor_range
            self._hl_speed_factor = lo + (a[:, 1] + 1.0) * 0.5 * (hi - lo)
            self._hl_alt_offset = a[:, 2] * self.cfg.max_altitude_offset

        # Run frozen low-level at 50 Hz (every env.step, matching training)
        self._run_low_level_policy()

        # PID (same as multi_waypoint _pre_physics_step)
        self._run_pid()

    def _apply_action(self):
        """Just apply forces — same pattern as multi_waypoint."""
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )

    # ── Observations (high-level) ────────────────────────────────────────────

    def _get_observations(self) -> dict:
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w

        goal_pos_b, _ = subtract_frame_transforms(root_pos, root_quat, self._goal_pos_w)

        # High-level LiDAR
        hl_lidar_hits = self._hl_lidar.data.ray_hits_w
        hl_lidar_pos = self._hl_lidar.data.pos_w.unsqueeze(1)
        hl_lidar_dists = torch.linalg.norm(hl_lidar_hits - hl_lidar_pos, dim=-1)
        hl_lidar_dists = self._augment_hl_lidar_with_obstacles(hl_lidar_dists, root_pos, root_quat)
        hl_lidar_norm = (hl_lidar_dists / self.cfg.hl_lidar_max_distance).clamp(0.0, 1.0)

        self._last_hl_lidar_distances = hl_lidar_dists
        if self._hl_lidar_horiz_indices is not None and len(self._hl_lidar_horiz_indices) > 0:
            horiz_dists = hl_lidar_dists[:, self._hl_lidar_horiz_indices]
            self._min_horiz_lidar_dist = horiz_dists.min(dim=1).values

        phase_one_hot = torch.nn.functional.one_hot(self._phase.long(), 5).float()

        # Nearest obstacle in body frame
        drone_xy = root_pos[:, :2].unsqueeze(1)
        obs_delta_w = self._obstacle_pos_w - drone_xy
        obs_center_dists = torch.linalg.norm(obs_delta_w, dim=2)
        obs_surface_dists = obs_center_dists - self._obstacle_radii.unsqueeze(0)
        min_idx = obs_surface_dists.argmin(dim=1)
        batch_idx = torch.arange(self.num_envs, device=self.device)

        nearest_delta_w = obs_delta_w[batch_idx, min_idx]
        nearest_dist = obs_surface_dists[batch_idx, min_idx]

        qw, qx, qy, qz = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        cos_yaw, sin_yaw = torch.cos(-yaw), torch.sin(-yaw)

        nearest_body_x = cos_yaw * nearest_delta_w[:, 0] - sin_yaw * nearest_delta_w[:, 1]
        nearest_body_y = sin_yaw * nearest_delta_w[:, 0] + cos_yaw * nearest_delta_w[:, 1]
        nearest_body_xy = torch.stack([nearest_body_x, nearest_body_y], dim=1)
        body_dir_norm = torch.linalg.norm(nearest_body_xy, dim=1, keepdim=True).clamp(min=1e-6)
        nearest_body_dir = nearest_body_xy / body_dir_norm
        nearest_dist_normalized = (nearest_dist / self.cfg.obstacle_safe_distance).clamp(0.0, 1.0).unsqueeze(1)

        nearest_obs_xy = self._obstacle_pos_w[batch_idx, min_idx]
        self._nearest_obs_pos_w[:, 0] = nearest_obs_xy[:, 0]
        self._nearest_obs_pos_w[:, 1] = nearest_obs_xy[:, 1]
        self._nearest_obs_pos_w[:, 2] = root_pos[:, 2]
        self._nearest_obs_dist = nearest_dist

        obs = torch.cat([
            hl_lidar_norm,                          # 449
            self._robot.data.root_lin_vel_b,        # 3
            self._robot.data.root_ang_vel_b,        # 3
            self._robot.data.projected_gravity_b,    # 3
            goal_pos_b,                              # 3
            phase_one_hot,                           # 5
            nearest_body_dir,                        # 2
            nearest_dist_normalized,                 # 1
        ], dim=-1)  # Total: 469
        return {"policy": obs}

    # ── Rewards ──────────────────────────────────────────────────────────────

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

        is_at_final_wp = self._current_wp_idx >= (self._num_waypoints - 1)

        phase = self._phase.clone()
        is_takeoff = phase == self.TAKEOFF
        is_stabilize = phase == self.STABILIZE
        is_navigate = phase == self.NAVIGATE
        is_hover = phase == self.HOVER
        is_land = phase == self.LAND

        # ── Phase 0: TAKEOFF
        alt_progress = altitude - self._prev_alt
        self._prev_alt = altitude.clone()
        takeoff_ascent = torch.where(is_takeoff, alt_progress * self.cfg.takeoff_ascent_scale, torch.zeros_like(alt_progress))
        alt_fraction = (altitude / self.cfg.cruise_altitude).clamp(0.0, 1.0)
        takeoff_alt_reward = torch.where(is_takeoff, alt_fraction * self.cfg.takeoff_altitude_reward_scale * dt, torch.zeros_like(altitude))
        takeoff_drift = torch.where(is_takeoff, horizontal_speed * self.cfg.takeoff_drift_penalty * dt, torch.zeros_like(horizontal_speed))

        # ── Phase 1: STABILIZE
        stabilize_position = torch.where(is_stabilize, torch.exp(-start_xy_dist) * self.cfg.stabilize_position_scale * dt, torch.zeros_like(start_xy_dist))
        stabilize_speed = torch.where(is_stabilize, torch.exp(-speed) * self.cfg.stabilize_low_speed_scale * dt, torch.zeros_like(speed))
        stabilize_altitude = torch.where(is_stabilize, alt_error * self.cfg.stabilize_altitude_scale * dt, torch.zeros_like(alt_error))
        stabilize_ang_vel = torch.where(is_stabilize, torch.exp(-ang_vel_magnitude) * self.cfg.stabilize_ang_vel_scale * dt, torch.zeros_like(ang_vel_magnitude))

        # ── Phase 2: NAVIGATE (high-level is active)
        xy_progress = self._prev_xy_dist - xy_dist
        self._prev_xy_dist = xy_dist.clone()
        nav_xy_progress = torch.where(is_navigate, xy_progress * self.cfg.nav_xy_progress_scale, torch.zeros_like(xy_progress))

        goal_dir_xy = delta[:, :2] / (xy_dist.unsqueeze(1) + 1e-6)
        vel_toward_goal = torch.sum(lin_vel_w[:, :2] * goal_dir_xy, dim=1)
        nav_velocity_align = torch.where(is_navigate, torch.clamp(vel_toward_goal, min=0.0) * self.cfg.nav_velocity_align_scale * dt, torch.zeros_like(vel_toward_goal))

        nav_altitude = torch.where(is_navigate, alt_error * self.cfg.nav_altitude_scale * dt, torch.zeros_like(alt_error))

        excess_speed = torch.clamp(speed - self.cfg.nav_max_speed, min=0.0)
        nav_speed_penalty = torch.where(is_navigate, excess_speed * self.cfg.nav_speed_penalty_scale * dt, torch.zeros_like(excess_speed))

        # Lateral penalty: reduced to 30% for intermediate waypoints (turns expected)
        lateral_scale = torch.where(
            is_at_final_wp,
            torch.full_like(xy_dist, self.cfg.nav_lateral_penalty_scale),
            torch.full_like(xy_dist, self.cfg.nav_lateral_penalty_scale * 0.3),
        )
        vel_lateral = lin_vel_w[:, :2] - vel_toward_goal.unsqueeze(1) * goal_dir_xy
        lateral_speed = torch.linalg.norm(vel_lateral, dim=1)
        nav_lateral_penalty = torch.where(is_navigate, lateral_speed * lateral_scale * dt, torch.zeros_like(lateral_speed))

        # Stability reward: low angular velocity + upright
        gravity_z_nav = self._robot.data.projected_gravity_b[:, 2]
        nav_stability = torch.where(
            is_navigate,
            (torch.exp(-ang_vel_magnitude) + (-gravity_z_nav)) * self.cfg.nav_stability_scale * dt,
            torch.zeros_like(ang_vel_magnitude),
        )

        # Obstacle proximity penalty
        drone_xy = root_pos[:, :2].unsqueeze(1)
        obstacle_xy = self._obstacle_pos_w
        obstacle_dists = torch.linalg.norm(drone_xy - obstacle_xy, dim=2)
        obstacle_surface_dists = obstacle_dists - self._obstacle_radii.unsqueeze(0)
        min_obstacle_dist, _ = obstacle_surface_dists.min(dim=1)
        proximity_raw = torch.exp(-min_obstacle_dist) * (min_obstacle_dist < self.cfg.obstacle_safe_distance).float()
        obstacle_proximity = torch.where(is_navigate, proximity_raw * self.cfg.obstacle_proximity_penalty_scale * dt, torch.zeros_like(proximity_raw))

        # LiDAR-based obstacle penalty
        lidar_dist = self._min_horiz_lidar_dist
        lidar_danger = (lidar_dist < self.cfg.lidar_danger_distance).float()
        lidar_closeness = (1.0 - lidar_dist / self.cfg.lidar_danger_distance).clamp(min=0.0)
        lidar_penalty = torch.where(is_navigate, lidar_danger * lidar_closeness * self.cfg.lidar_obstacle_penalty_scale * dt, torch.zeros_like(lidar_dist))

        # Goal-modifier quality rewards
        yaw_deviation = torch.abs(self._hl_yaw_offset)
        subgoal_magnitude = torch.where(is_navigate, yaw_deviation * self.cfg.subgoal_magnitude_penalty * dt, torch.zeros_like(yaw_deviation))

        subgoal_reachability = torch.where(
            is_navigate & (self._min_horiz_lidar_dist > self.cfg.lidar_danger_distance),
            torch.full_like(yaw_deviation, self.cfg.subgoal_reachability_scale * dt),
            torch.zeros_like(yaw_deviation),
        )

        # Multi-waypoint advancement
        just_advanced = self._check_waypoint_advancement(xy_dist, speed)
        waypoint_bonus = just_advanced.float() * self.cfg.intermediate_waypoint_bonus
        speed_carry = torch.where(
            just_advanced,
            torch.clamp(speed, min=0.0, max=self.cfg.nav_max_speed) * self.cfg.speed_carry_bonus_scale,
            torch.zeros_like(speed),
        )
        fraction_done = self._current_wp_idx.float() / self._num_waypoints.float().clamp(min=1)
        wp_progress_reward = torch.where(is_navigate, fraction_done * self.cfg.waypoint_progress_scale * dt, torch.zeros_like(fraction_done))

        # ── Phase 3: HOVER
        hover_position = torch.where(is_hover, torch.exp(-xy_dist) * self.cfg.hover_position_scale * dt, torch.zeros_like(xy_dist))
        hover_speed = torch.where(is_hover, torch.exp(-speed) * self.cfg.hover_low_speed_scale * dt, torch.zeros_like(speed))
        hover_altitude = torch.where(is_hover, alt_error * self.cfg.hover_altitude_scale * dt, torch.zeros_like(alt_error))

        # ── Phase 4: LAND
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

        final_delta = self._final_goal_pos_w - root_pos
        final_dist_3d = torch.linalg.norm(final_delta, dim=1)
        goal_reached = is_land & (final_dist_3d < self.cfg.goal_threshold) & (speed < self.cfg.landing_speed_threshold)
        goal_bonus = goal_reached.float() * self.cfg.goal_reached_bonus

        # Crash penalty
        drone_xy_r = root_pos[:, :2].unsqueeze(1)
        crash_obs_dists = torch.linalg.norm(drone_xy_r - self._obstacle_pos_w, dim=2)
        crash_surface_dists = crash_obs_dists - self._obstacle_radii.unsqueeze(0)
        min_crash_dist, min_crash_idx = crash_surface_dists.min(dim=1)
        crash_nearest_h = self._obstacle_heights[min_crash_idx]
        is_crashing = (min_crash_dist < self.cfg.obstacle_collision_radius) & (root_pos[:, 2] < crash_nearest_h)
        crash_penalty = is_crashing.float() * self.cfg.crash_penalty

        # ── All-phase rewards
        time_scale = torch.ones_like(dist_3d)
        time_scale = torch.where(is_takeoff, 0.2 * time_scale, time_scale)
        time_scale = torch.where(is_stabilize, 0.3 * time_scale, time_scale)
        time_penalty = time_scale * self.cfg.time_penalty * dt

        ang_vel = torch.sum(torch.square(ang_vel_vec), dim=1)
        orient_scale = torch.ones_like(dist_3d)
        orient_scale = torch.where(is_navigate, 0.3 * orient_scale, orient_scale)
        orient_scale = torch.where(is_hover | is_land, 1.5 * orient_scale, orient_scale)
        ang_vel_penalty = ang_vel * self.cfg.ang_vel_reward_scale * dt * orient_scale

        yaw_rate = torch.abs(ang_vel_vec[:, 2])
        yaw_penalty = yaw_rate * self.cfg.yaw_rate_penalty_scale * dt

        gravity_z = self._robot.data.projected_gravity_b[:, 2]
        upright_reward = -gravity_z * self.cfg.upright_reward_scale * dt * orient_scale

        # ── Phase transitions
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
            & (speed < 1.5) & (altitude_now > 1.0)
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

        # ── Total
        reward = (
            takeoff_ascent + takeoff_alt_reward + takeoff_drift
            + stabilize_position + stabilize_speed + stabilize_altitude + stabilize_ang_vel
            + nav_xy_progress + nav_velocity_align + nav_altitude + nav_speed_penalty
            + nav_lateral_penalty + nav_stability
            + obstacle_proximity + lidar_penalty + subgoal_reachability + subgoal_magnitude
            + waypoint_bonus + speed_carry + wp_progress_reward
            + hover_position + hover_speed + hover_altitude
            + land_descent + land_xy_stability + land_drift + land_precision
            + land_descent_control + land_controlled_descent + land_altitude_penalty + goal_bonus + crash_penalty
            + time_penalty + ang_vel_penalty + yaw_penalty + upright_reward
        )

        # Logging
        self._episode_sums["takeoff_ascent"] += takeoff_ascent
        self._episode_sums["takeoff_alt_reward"] += takeoff_alt_reward
        self._episode_sums["takeoff_drift"] += takeoff_drift
        self._episode_sums["stabilize_position"] += stabilize_position
        self._episode_sums["stabilize_speed"] += stabilize_speed
        self._episode_sums["stabilize_altitude"] += stabilize_altitude
        self._episode_sums["stabilize_ang_vel"] += stabilize_ang_vel
        self._episode_sums["nav_xy_progress"] += nav_xy_progress
        self._episode_sums["nav_velocity_align"] += nav_velocity_align
        self._episode_sums["nav_altitude"] += nav_altitude
        self._episode_sums["nav_speed_penalty"] += nav_speed_penalty
        self._episode_sums["nav_lateral"] += nav_lateral_penalty
        self._episode_sums["nav_stability"] += nav_stability
        self._episode_sums["obstacle_proximity"] += obstacle_proximity
        self._episode_sums["lidar_penalty"] += lidar_penalty
        self._episode_sums["subgoal_reachability"] += subgoal_reachability
        self._episode_sums["subgoal_magnitude"] += subgoal_magnitude
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
        self._episode_sums["crash_penalty"] += crash_penalty
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
        drone_xy = self._robot.data.root_pos_w[:, :2].unsqueeze(1)
        obs_dists = torch.linalg.norm(drone_xy - self._obstacle_pos_w, dim=2)
        obs_surface_dists = obs_dists - self._obstacle_radii.unsqueeze(0)
        min_obs_dist, min_obs_idx = obs_surface_dists.min(dim=1)
        nearest_height = self._obstacle_heights[min_obs_idx]
        hit_obstacle = (
            (min_obs_dist < self.cfg.obstacle_collision_radius)
            & (self._robot.data.root_pos_w[:, 2] < nearest_height)
        )

        goal_dist = torch.linalg.norm(self._final_goal_pos_w - self._robot.data.root_pos_w, dim=1)
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

        died = too_low | too_high | flipped | hit_obstacle
        return died, time_out | goal_reached | touched_ground

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

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
        died_mask = self.reset_terminated[env_ids]
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(died_mask).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_distance.item()
        self.extras["log"]["Metrics/final_phase_mean"] = self._phase[env_ids].float().mean().item()
        self.extras["log"]["Metrics/waypoint_completion_rate"] = wp_completion_rate.mean().item()
        self.extras["log"]["Metrics/avg_waypoints_reached"] = self._current_wp_idx[env_ids].float().mean().item()
        # Per-phase crash counts
        phase_names = ["TAKEOFF", "STABILIZE", "NAVIGATE", "HOVER", "LAND"]
        for pi, pname in enumerate(phase_names):
            phase_died = (died_mask & (self._phase[env_ids] == pi)).sum().item()
            self.extras["log"][f"Episode_Termination/died_{pname}"] = phase_died

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Generate waypoints & set start
        self._generate_waypoints(env_ids)

        num_reset = len(env_ids)
        start_local = torch.tensor(self.cfg.static_start_pos, dtype=torch.float32, device=self.device)
        start_pos = start_local.unsqueeze(0).expand(num_reset, -1).clone()
        start_pos[:, :2] += self._terrain.env_origins[env_ids, :2]
        self._start_pos_w[env_ids] = start_pos

        # Update obstacle world positions
        env_origins_xy = self._terrain.env_origins[env_ids, :2]
        self._obstacle_pos_w[env_ids] = self._obstacle_local.unsqueeze(0) + env_origins_xy.unsqueeze(1)

        # Initialize waypoint tracking
        self._current_wp_idx[env_ids] = 0
        self._update_current_goal(env_ids)

        delta = self._goal_pos_w[env_ids] - start_pos
        self._prev_xy_dist[env_ids] = torch.linalg.norm(delta[:, :2], dim=1)
        self._prev_z_dist[env_ids] = torch.abs(delta[:, 2])
        self._prev_alt[env_ids] = start_pos[:, 2]

        self._phase[env_ids] = self.TAKEOFF
        self._stabilize_timer[env_ids] = 0.0
        self._hover_timer[env_ids] = 0.0

        # Reset goal modifier and step counter
        self._hl_step_counter = 0
        self._hl_yaw_offset[env_ids] = 0.0
        self._hl_speed_factor[env_ids] = 1.0
        self._hl_alt_offset[env_ids] = 0.0
        self._ll_vx_cmd[env_ids] = 0.0
        self._ll_vy_cmd[env_ids] = 0.0
        self._ll_vz_cmd[env_ids] = 0.0
        self._ll_yaw_rate_cmd[env_ids] = 0.0

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

            if not hasattr(self, "subgoal_visualizer"):
                sg_marker_cfg = CUBOID_MARKER_CFG.copy()
                sg_marker_cfg.markers["cuboid"].size = (0.25, 0.25, 0.25)
                sg_marker_cfg.prim_path = "/Visuals/Command/subgoal"
                self.subgoal_visualizer = VisualizationMarkers(sg_marker_cfg)

            if self._draw is None and _debug_draw is not None:
                self._draw = _debug_draw.acquire_debug_draw_interface()

            self._lidar_close_thresh = 2.0
            self._lidar_mid_thresh = 6.0

            self.goal_pos_visualizer.set_visibility(True)
            for viz in self.waypoint_visualizers:
                viz.set_visibility(True)
            self.subgoal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "waypoint_visualizers"):
                for viz in self.waypoint_visualizers:
                    viz.set_visibility(False)
            if hasattr(self, "subgoal_visualizer"):
                self.subgoal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._goal_pos_w)

        for wp_idx in range(self.cfg.max_num_waypoints):
            wp_positions = self._waypoints_w[:, wp_idx, :]
            valid = wp_idx < self._num_waypoints
            viz_pos = torch.where(
                valid.unsqueeze(1).expand(-1, 3), wp_positions,
                torch.full_like(wp_positions, -1000.0),
            )
            self.waypoint_visualizers[wp_idx].visualize(viz_pos)

        # Visualize modified goal in world frame
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w

        real_goal_b, _ = subtract_frame_transforms(root_pos, root_quat, self._goal_pos_w)
        goal_xy = real_goal_b[:, :2]
        cos_off = torch.cos(self._hl_yaw_offset)
        sin_off = torch.sin(self._hl_yaw_offset)
        mod_x = (cos_off * goal_xy[:, 0] - sin_off * goal_xy[:, 1]) * self._hl_speed_factor
        mod_y = (sin_off * goal_xy[:, 0] + cos_off * goal_xy[:, 1]) * self._hl_speed_factor
        mod_z = real_goal_b[:, 2] + self._hl_alt_offset

        qw, qx, qy, qz = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        sg_w_x = cos_yaw * mod_x - sin_yaw * mod_y + root_pos[:, 0]
        sg_w_y = sin_yaw * mod_x + cos_yaw * mod_y + root_pos[:, 1]
        sg_w_z = mod_z + root_pos[:, 2]
        sg_world = torch.stack([sg_w_x, sg_w_y, sg_w_z], dim=1)
        self.subgoal_visualizer.visualize(sg_world)

        # LiDAR debug draw (env 0 only)
        if self._draw is None or self._hl_lidar_ray_dirs is None:
            return
        if not hasattr(self, "_last_hl_lidar_distances"):
            return

        self._draw.clear_lines()
        self._draw.clear_points()

        rp = self._robot.data.root_pos_w[0].cpu()
        rq = self._robot.data.root_quat_w[0].cpu()
        ld = self._last_hl_lidar_distances[0].cpu()

        n_rays = self._hl_lidar_ray_dirs.shape[0]
        ray_dirs = self._hl_lidar_ray_dirs.cpu()

        q0, q1, q2, q3 = float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])
        y = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        cy, sy = math.cos(y), math.sin(y)
        ox, oy, oz = float(rp[0]), float(rp[1]), float(rp[2])

        starts, ends, colors, widths = [], [], [], []
        for ri in range(0, n_rays, 5):
            dx_l, dy_l, dz_l = float(ray_dirs[ri, 0]), float(ray_dirs[ri, 1]), float(ray_dirs[ri, 2])
            dx_w = cy * dx_l - sy * dy_l
            dy_w = sy * dx_l + cy * dy_l
            dist = min(float(ld[ri]), self.cfg.hl_lidar_max_distance)

            ex = ox + dx_w * dist
            ey = oy + dy_w * dist
            ez = oz + dz_l * dist

            if dist < self._lidar_close_thresh:
                color = (1.0, 0.0, 0.0, 0.9)
            elif dist < self._lidar_mid_thresh:
                color = (1.0, 1.0, 0.0, 0.7)
            else:
                color = (0.0, 1.0, 0.0, 0.5)

            starts.append((ox, oy, oz))
            ends.append((ex, ey, ez))
            colors.append(color)
            widths.append(1.5)

        if starts:
            self._draw.draw_lines(starts, ends, colors, widths)


# ═══════════════════════════════════════════════════════════════════════════════
# PPO Runner Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@configclass
class HierarchicalObstaclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 3000
    save_interval = 100
    experiment_name = "crazyflie_hierarchical_obstacle_ppo"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],   # larger to handle 469-dim obs with LiDAR
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Gym Registration
# ═══════════════════════════════════════════════════════════════════════════════

gym.register(
    id="Isaac-HierarchicalObstacle-PPO-v0",
    entry_point=f"{__name__}:HierarchicalObstacleNavEnv",
    disable_env_checker=True,
    kwargs={"cfg": HierarchicalObstacleNavEnvCfg()},
)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Camera Setup (for play mode)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_cameras(env_unwrapped):
    from pxr import UsdGeom, Sdf, Gf
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    cam1_path = "/World/Cameras/FollowCam"
    cam1 = UsdGeom.Camera.Define(stage, Sdf.Path(cam1_path))
    cam1.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000.0))

    cam2_path = "/World/Cameras/TopDownCam"
    cam2 = UsdGeom.Camera.Define(stage, Sdf.Path(cam2_path))
    cam2.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000.0))

    cam3_path = "/World/Cameras/SideCam"
    cam3 = UsdGeom.Camera.Define(stage, Sdf.Path(cam3_path))
    cam3.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000.0))

    from omni.kit.viewport.utility import get_active_viewport, create_viewport_window

    vp_follow = create_viewport_window("Follow Cam", width=640, height=480)
    vp_follow.viewport_api.camera_path = cam1_path
    vp_topdown = create_viewport_window("Top-Down View", width=640, height=480)
    vp_topdown.viewport_api.camera_path = cam2_path
    vp_side = create_viewport_window("Side View", width=640, height=480)
    vp_side.viewport_api.camera_path = cam3_path

    print("[INFO] 4 camera views created (Main=FREE, Follow, Top-Down, Side)")
    return {"follow": cam1_path, "topdown": cam2_path, "side": cam3_path, "stage": stage}


def update_cameras(cameras, drone_pos):
    from pxr import UsdGeom, Gf
    stage = cameras["stage"]
    x, y, z = float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])

    for cam_key, eye_fn in [
        ("follow", lambda: Gf.Vec3d(x - 8.0, y - 5.0, z + 5.0)),
        ("topdown", lambda: Gf.Vec3d(x - 5.0, y - 5.0, 20.0)),
        ("side", lambda: Gf.Vec3d(x, y - 15.0, z + 3.0)),
    ]:
        cam = UsdGeom.Camera.Get(stage, cameras[cam_key])
        xform = UsdGeom.Xformable(cam.GetPrim())
        xform.ClearXformOpOrder()
        eye = eye_fn()
        target = Gf.Vec3d(x, y, z)
        up = Gf.Vec3d(0, 0, 1) if cam_key != "topdown" else Gf.Vec3d(0, 1, 0)
        mat = Gf.Matrix4d()
        mat.SetLookAt(eye, target, up)
        xform.AddTransformOp().Set(mat.GetInverse())


# ═══════════════════════════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def train():
    env_cfg = HierarchicalObstacleNavEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    agent_cfg = HierarchicalObstaclePPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    log_root = os.path.join(SCRIPT_DIR, "logs", "hierarchical_ppo", agent_cfg.experiment_name)
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    ll_path = args_cli.ll_checkpoint or DEFAULT_LL_CHECKPOINT
    print(f"[INFO] Logging to: {log_dir}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Max iterations: {agent_cfg.max_iterations}")
    print(f"[INFO] Low-level checkpoint: {os.path.abspath(ll_path)}")
    print(f"[INFO] Waypoints: {env_cfg.min_num_waypoints}-{env_cfg.max_num_waypoints}")
    print(f"[INFO] Obstacles: {len(env_cfg.obstacle_positions)} trees")
    print(f"[INFO] High-level: 10 Hz, {env_cfg.observation_space - 20}-ray LiDAR, action=3D sub-goal")
    print(f"[INFO] Low-level: 50 Hz (frozen), PID: 100 Hz")

    env = gym.make(
        "Isaac-HierarchicalObstacle-PPO-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0"
    )

    if args_cli.checkpoint:
        print(f"[INFO] Resuming high-level from: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)

    start_time = time.time()
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    elapsed = time.time() - start_time

    print(f"\n[INFO] Training complete! Duration: {elapsed:.1f}s")
    print(f"[INFO] Logs saved to: {log_dir}")
    env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Play
# ═══════════════════════════════════════════════════════════════════════════════

def play():
    assert args_cli.checkpoint is not None, "Must provide --checkpoint for play mode."

    env_cfg = HierarchicalObstacleNavEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 40.0
    env_cfg.debug_vis = True
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    env = gym.make("Isaac-HierarchicalObstacle-PPO-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = HierarchicalObstaclePPORunnerCfg()
    agent_cfg.max_iterations = 0

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    obs = env.get_observations()

    cameras = setup_cameras(env.unwrapped)

    episode_count = 0
    total_reward = 0.0
    step_count = 0
    playback_dt = env_cfg.sim.dt * env_cfg.decimation * 3.0  # 3x slower

    print(f"\n{'='*60}")
    print(f"[INFO] Hierarchical PPO policy loaded!")
    print(f"[INFO] High-level: 10 Hz sub-goal planner")
    print(f"[INFO] Low-level: frozen multi-waypoint SAC (50 Hz)")
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

            drone_pos = env.unwrapped._robot.data.root_pos_w[0].cpu().numpy()
            update_cameras(cameras, drone_pos)

            if step_count % 20 == 0:
                unwrapped = env.unwrapped
                pos = unwrapped._robot.data.root_pos_w[0].cpu().numpy()
                phase = unwrapped._phase[0].item()
                wp_idx = unwrapped._current_wp_idx[0].item()
                num_wps = unwrapped._num_waypoints[0].item()
                yaw_off = unwrapped._hl_yaw_offset[0].item()
                spd_fac = unwrapped._hl_speed_factor[0].item()
                alt_off = unwrapped._hl_alt_offset[0].item()
                phase_names = ["TAKEOFF", "STABILIZE", "NAVIGATE", "HOVER", "LAND"]
                print(f"  Step {step_count} | {phase_names[phase]} | WP {wp_idx}/{num_wps} | "
                      f"Pos ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}) | "
                      f"Yaw={yaw_off:.2f} Spd={spd_fac:.2f} Alt={alt_off:.2f} | R={rewards[0].item():.3f}")

            if dones.any():
                episode_count += dones.sum().item()
                print(f"  === Episode {int(episode_count)} done ===")

            elapsed = time.time() - step_start
            sleep_time = playback_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    avg_reward = total_reward / max(episode_count, 1)
    print(f"\n[RESULT] Avg reward: {avg_reward:.2f} over {int(episode_count)} episodes")
    env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Eval
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate():
    assert args_cli.checkpoint is not None, "Must provide --checkpoint for eval mode."

    num_eval_envs = min(args_cli.num_envs, 16) if args_cli.num_envs == 512 else args_cli.num_envs

    env_cfg = HierarchicalObstacleNavEnvCfg()
    env_cfg.scene.num_envs = num_eval_envs
    env_cfg.episode_length_s = 40.0
    env_cfg.debug_vis = False
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    env = gym.make("Isaac-HierarchicalObstacle-PPO-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = HierarchicalObstaclePPORunnerCfg()
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
    wp_completion_rates = []
    phase_crashes = {"TAKEOFF": 0, "STABILIZE": 0, "NAVIGATE": 0, "HOVER": 0, "LAND": 0}
    env_rewards = np.zeros(num_eval_envs)
    env_steps = np.zeros(num_eval_envs, dtype=int)
    step_dt = env_cfg.sim.dt * env_cfg.decimation

    print(f"\n{'='*60}")
    print(f"EVALUATION: Hierarchical PPO | {args_cli.num_episodes} episodes")
    print(f"HL Checkpoint: {os.path.basename(args_cli.checkpoint)}")
    print(f"LL Checkpoint: {os.path.basename(args_cli.ll_checkpoint or DEFAULT_LL_CHECKPOINT)}")
    print(f"{'='*60}\n")

    while total_episodes < args_cli.num_episodes and simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)

        rewards_np = rewards.cpu().numpy().flatten()
        dones_np = dones.cpu().numpy().flatten()

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
                        for pname in phase_crashes:
                            phase_crashes[pname] += extras.get(f"Episode_Termination/died_{pname}", 0)
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
    avg_wp_rate = np.mean(wp_completion_rates) * 100 if wp_completion_rates else 0

    valid_distances = [d for d in final_distances if d >= 0]
    avg_distance = np.mean(valid_distances) if valid_distances else -1
    min_distance = np.min(valid_distances) if valid_distances else -1

    print(f"\n{'='*60}")
    print(f"RESULTS -- Hierarchical PPO ({n} episodes)")
    print(f"{'='*60}")
    print(f"  Success Rate:           {success_rate:6.1f}%  ({goal_reached_count}/{n})")
    print(f"  Crash Rate:             {crash_rate:6.1f}%  ({crashes}/{n})")
    print(f"  Crashes by Phase:")
    for pname, count in phase_crashes.items():
        print(f"    {pname:12s}:     {count:4d}")
    print(f"  Waypoint Completion:    {avg_wp_rate:6.1f}%")
    print(f"  Avg Final Distance:     {avg_distance:6.3f} m")
    print(f"  Min Final Distance:     {min_distance:6.3f} m")
    print(f"  Avg Reward:             {avg_reward:8.2f}  (+/- {std_reward:.2f})")
    print(f"  Avg Episode Length:      {avg_length:5.2f} s")
    print(f"{'='*60}\n")
    env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if args_cli.mode == "train":
        train()
    elif args_cli.mode == "play":
        play()
    elif args_cli.mode == "eval":
        evaluate()
    simulation_app.close()
