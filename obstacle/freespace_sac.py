"""Free-space detection Crazyflie navigation with SAC — matching drone_freespace_sim.py environment.

Comparison file for hierarchical_obstacle_sac_v10.py.
Same Crazyflie drone config, kinematic control, LiDAR, phase state machine — but
uses the SAME forest environment as drone_freespace_sim.py:
    - 40 trees in 20m x 20m area
    - Tree radius 0.1-0.3m, height 4.0m
    - Clear zones at start (-9, 0) and goal (+9, 0)
    - Fixed seed (42) for reproducible layout

This allows direct comparison between:
    - drone_freespace_sim.py (rule-based free-space detection)
    - This file (RL-trained SAC with same forest)
    - hierarchical_obstacle_sac_v10.py (RL with dense 35-obstacle forest)

Two-level control architecture (same as v10):
    HIGH-LEVEL (trained):  LiDAR 1347 rays (3 vert ch x 449 horiz) + state -> goal modifier
    LOW-LEVEL  (deterministic): Goal -> proportional velocity commands (vx, vy, vz, yaw_rate)

Usage (from IsaacLab directory):
    cd ~/projects/isaac/IsaacLab
    source ~/projects/isaac/env_isaaclab/bin/activate

    # Train:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/freespace_sac.py \
        --mode train --num_envs 256 --total_timesteps 10000000 --headless

    # Play:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/freespace_sac.py \
        --mode play --checkpoint /path/to/hl_sac_final.zip

    # Eval:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/freespace_sac.py \
        --mode eval --checkpoint /path/to/hl_sac_final.zip --num_episodes 50
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

# -- Argument Parser ----------------------------------------------------------
parser = argparse.ArgumentParser(description="Free-space Forest Crazyflie Nav -- SAC")
parser.add_argument("--mode", type=str, default="train", choices=["train", "play", "eval"])
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--total_timesteps", type=int, default=10000000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None, help="High-level checkpoint (play/eval/resume).")
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--min_waypoints", type=int, default=3)
parser.add_argument("--max_waypoints", type=int, default=5)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -- Launch Sim ---------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Imports (after sim launch) -----------------------------------------------
import gymnasium as gym
import time
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
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


# ==============================================================================
# 1D CNN Feature Extractor for LiDAR + State
# ==============================================================================

class LidarCNNExtractor(BaseFeaturesExtractor):
    """1D CNN for multi-channel LiDAR + MLP for state, fused together.

    Observation layout: [lidar_1347, lin_vel_b(3), ang_vel_b(3), gravity_b(3),
                         goal_pos_b(3), phase_one_hot(5), nearest_obstacle(3)]
    State dims = 20 (same as v10).
    """

    def __init__(self, observation_space, features_dim=128, lidar_dim=None,
                 cnn_out_dim=32, lidar_channels=3):
        if lidar_dim is None:
            lidar_dim = observation_space.shape[0] - 20
        super().__init__(observation_space, features_dim)

        self.lidar_dim = lidar_dim
        self.state_dim = observation_space.shape[0] - lidar_dim
        self.lidar_channels = lidar_channels
        self.rays_per_channel = lidar_dim // lidar_channels

        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(lidar_channels, 16, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, cnn_out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
        )

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


# ==============================================================================
# Robot Configuration (identical to v10 -- 10x scaled Crazyflie)
# ==============================================================================

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


# ==============================================================================
# Environment Configuration
# ==============================================================================

@configclass
class FreespaceForestNavEnvCfg(DirectRLEnvCfg):
    """Forest environment matching drone_freespace_sim.py layout.

    40 trees in 20x20m area, tree radius 0.1-0.3m, height 4.0m.
    Same Crazyflie + kinematic control + LiDAR as v10.
    """

    episode_length_s = 180.0
    decimation = 2  # 50 Hz env.step

    # obs: lidar_1347 + lin_vel_b(3) + ang_vel_b(3) + gravity_b(3) + goal_pos_b(3)
    #      + phase_one_hot(5) + nearest_obstacle(3) = 1347 + 20 = 1367
    action_space = 3   # goal modifier: yaw_offset, speed_factor, altitude_offset
    observation_space = 1367
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
        num_envs=256,
        env_spacing=70.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = VISIBLE_CRAZYFLIE_CFG
    thrust_to_weight = 1.9
    moment_scale = 0.04

    # PID controller (identical to v10)
    max_velocity_xy: float = 1.5
    max_velocity_z: float = 0.8
    max_yaw_rate: float = 1.0
    pid_vel_kp: float = 0.15
    pid_att_kp: float = 5.0
    pid_att_kd: float = 1.5
    pid_vz_kp: float = 0.4
    pid_yaw_kp: float = 0.3
    pid_max_tilt: float = 0.4

    # Deterministic LL proportional controller gains (identical to v10)
    ll_gain_xy: float = 1.5
    ll_gain_z: float = 0.8
    ll_gain_yaw: float = 0.5

    drag_tilt_coeff: float = 0.06

    # Movement noise -- Ornstein-Uhlenbeck (identical to v10)
    noise_theta: float = 2.0
    noise_vel_sigma: float = 0.005
    noise_att_sigma: float = 0.001
    noise_yaw_sigma: float = 0.001
    noise_vel_max: float = 0.003
    noise_att_max: float = 0.002
    noise_yaw_max: float = 0.001

    # Domain Randomization (identical to v10)
    dr_wind_enabled: bool = True
    dr_wind_max_speed: float = 0.8
    dr_wind_theta: float = 0.5
    dr_wind_sigma: float = 0.3

    dr_vel_scale_enabled: bool = True
    dr_vel_scale_range: tuple = (0.85, 1.15)

    dr_drag_enabled: bool = True
    dr_drag_range: tuple = (0.03, 0.10)

    dr_lidar_noise_enabled: bool = True
    dr_lidar_noise_std: float = 0.05

    dr_obs_noise_enabled: bool = True
    dr_obs_vel_noise_std: float = 0.05
    dr_obs_angvel_noise_std: float = 0.03

    # LiDAR (identical to v10)
    hl_lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=3,
            vertical_fov_range=(-45.0, 0.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=0.8,
        ),
        max_distance=20.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    hl_lidar_max_distance: float = 20.0
    hl_lidar_n_channels: int = 3

    # Goal Modifier (same as v10)
    max_yaw_offset: float = 1.0
    speed_factor_range: tuple = (0.0, 1.2)
    max_altitude_offset: float = 0.8
    goal_modifier_smoothing: float = 0.4

    # Velocity-projection safety (same as v10)
    reactive_safety_distance: float = 2.0
    reactive_safety_push_gain: float = 0.8
    predictive_avoidance_lookahead: float = 1.5
    predictive_avoidance_trigger: float = 2.0

    # Navigation
    goal_threshold: float = 6.0
    landing_speed_threshold: float = 2.5
    max_flight_height: float = 8.0
    min_flight_height: float = 0.05
    cruise_altitude: float = 1.5  # matches drone_freespace_sim.py DRONE_ALTITUDE

    takeoff_altitude_tolerance: float = 0.3
    stabilize_duration: float = 0.5
    hover_position_tolerance: float = 5.0
    hover_duration: float = 0.5

    # Multi-Waypoint -- wider arena matching 20x20m forest
    static_start_pos: tuple = (-9.0, 0.0, 0.2)  # matches drone_freespace_sim start
    min_num_waypoints: int = 2
    max_num_waypoints: int = 3
    arena_x_range: tuple = (-8.0, 8.0)
    arena_y_range: tuple = (-8.0, 8.0)  # wider to match 20m area
    min_y_change: float = 2.5
    intermediate_waypoint_tolerance: float = 6.0
    final_waypoint_tolerance: float = 5.0
    waypoint_obstacle_clearance: float = 1.5  # smaller trees need less clearance

    # -- Obstacles: 40 trees matching drone_freespace_sim.py layout --
    # Generated with seed=42, 20x20m area, radius 0.1-0.3m, height 4m
    # Clear zones at (-9, 0, 2m) and (+9, 0, 2m)
    obstacle_positions: list = [
        (  2.79,  -9.50, 0.16, 4.0),
        ( -5.54,   4.73, 0.24, 4.0),
        (  7.84,  -8.26, 0.18, 4.0),
        ( -9.40,  -5.63, 0.20, 4.0),
        ( -9.47,  -6.02, 0.23, 4.0),
        (  0.90,  -5.59, 0.22, 4.0),
        (  6.19,  -9.87, 0.26, 4.0),
        (  3.96,  -3.19, 0.13, 4.0),
        (  9.14,  -3.27, 0.12, 4.0),
        ( -8.07,   6.95, 0.22, 4.0),
        (  6.14,   4.59, 0.21, 4.0),
        (  9.46,  -2.43, 0.21, 4.0),
        (  6.59,   2.37, 0.27, 4.0),
        (  1.55,   4.09, 0.11, 4.0),
        ( -5.44,  -4.21, 0.12, 4.0),
        ( -5.34,  -7.98, 0.16, 4.0),
        (  2.71,  -2.70, 0.17, 4.0),
        ( -5.81,  -4.66, 0.29, 4.0),
        (  2.96,   2.18, 0.13, 4.0),
        (  4.58,  -6.73, 0.18, 4.0),
        (  9.79,   2.80, 0.21, 4.0),
        (  3.69,   6.86, 0.26, 4.0),
        ( -5.42,  -9.36, 0.16, 4.0),
        ( -4.65,  -5.78, 0.29, 4.0),
        (  7.53,  -3.71, 0.23, 4.0),
        ( -2.09,   8.29, 0.19, 4.0),
        ( -4.70,  -5.07, 0.21, 4.0),
        ( -4.75,   1.69, 0.28, 4.0),
        ( -2.01,  -5.61, 0.30, 4.0),
        (  0.19,  -8.18, 0.11, 4.0),
        ( -7.81,   2.55, 0.26, 4.0),
        ( -1.56,  -8.73, 0.18, 4.0),
        (  9.42,   7.22, 0.10, 4.0),
        (  4.41,   3.63, 0.21, 4.0),
        ( -4.66,   2.82, 0.12, 4.0),
        ( -1.30,  -0.93, 0.29, 4.0),
        (  7.52,  -4.73, 0.20, 4.0),
        ( -6.43,   8.25, 0.27, 4.0),
        ( -4.03,   2.78, 0.22, 4.0),
        ( -6.94,   5.25, 0.21, 4.0),
    ]
    obstacle_collision_radius: float = 0.25  # matches DRONE_PHYS_RADIUS from freespace_sim
    obstacle_wall_radius: float = 0.40       # wall push radius (margin over detection)
    obstacle_safe_distance: float = 1.5      # smaller trees = smaller safe zone

    # Stagnation / auto-advance (same as v10)
    stagnation_timeout: float = 6.0
    stagnation_penalty_scale: float = -8.0
    escape_tangential_speed: float = 1.5
    auto_advance_timeout: float = 8.0
    final_wp_force_hover_timeout: float = 12.0

    # Reward Scales (same structure as v10)
    nav_xy_progress_scale: float = 5.0
    nav_velocity_align_scale: float = 2.0
    nav_altitude_scale: float = -1.5
    obstacle_proximity_penalty_scale: float = -50.0
    lidar_obstacle_penalty_scale: float = -40.0
    lidar_danger_distance: float = 4.0
    subgoal_reachability_scale: float = 5.0
    subgoal_magnitude_penalty: float = -0.2
    crash_penalty: float = -900.0
    intermediate_waypoint_bonus: float = 200.0
    speed_carry_bonus_scale: float = 5.0
    waypoint_progress_scale: float = 3.0
    nav_max_speed: float = 1.5
    nav_speed_penalty_scale: float = -3.0
    nav_lateral_penalty_scale: float = -1.0
    nav_stability_scale: float = 2.0
    obstacle_clearance_bonus_scale: float = 3.0

    # Non-NAVIGATE phases
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
    ang_vel_reward_scale: float = -1.0
    yaw_rate_penalty_scale: float = -1.5
    upright_reward_scale: float = 4.0


# ==============================================================================
# Environment
# ==============================================================================

class FreespaceForestNavEnv(DirectRLEnv):
    """Forest navigation matching drone_freespace_sim.py environment.

    Same phase state machine as v10:
        TAKEOFF -> STABILIZE -> NAVIGATE -> HOVER -> LAND

    40 trees (radius 0.1-0.3m, height 4m) in 20x20m area.
    """

    cfg: FreespaceForestNavEnvCfg

    TAKEOFF = 0
    STABILIZE = 1
    NAVIGATE = 2
    HOVER = 3
    LAND = 4

    def __init__(self, cfg: FreespaceForestNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        max_wps = self.cfg.max_num_waypoints

        # -- Buffers --
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._waypoints_w = torch.zeros(self.num_envs, max_wps, 3, device=self.device)
        self._num_waypoints = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._current_wp_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._final_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_xy_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_z_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_alt = torch.zeros(self.num_envs, device=self.device)

        self._phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._stabilize_timer = torch.zeros(self.num_envs, device=self.device)
        self._hover_timer = torch.zeros(self.num_envs, device=self.device)

        self._hl_yaw_offset = torch.zeros(self.num_envs, device=self.device)
        self._hl_speed_factor = torch.ones(self.num_envs, device=self.device)
        self._hl_alt_offset = torch.zeros(self.num_envs, device=self.device)

        self._ll_vx_cmd = torch.zeros(self.num_envs, device=self.device)
        self._ll_vy_cmd = torch.zeros(self.num_envs, device=self.device)
        self._ll_vz_cmd = torch.zeros(self.num_envs, device=self.device)
        self._ll_yaw_rate_cmd = torch.zeros(self.num_envs, device=self.device)

        self._tracked_yaw = torch.zeros(self.num_envs, device=self.device)
        self._desired_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._actual_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._smooth_vx = torch.zeros(self.num_envs, device=self.device)
        self._smooth_vy = torch.zeros(self.num_envs, device=self.device)
        self._smooth_vz = torch.zeros(self.num_envs, device=self.device)
        self._smooth_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self._smoothed_accel_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._noise_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._noise_att = torch.zeros(self.num_envs, 2, device=self.device)
        self._noise_yaw = torch.zeros(self.num_envs, device=self.device)

        self._dr_wind = torch.zeros(self.num_envs, 3, device=self.device)
        self._dr_vel_scale = torch.ones(self.num_envs, device=self.device)
        self._dr_drag_coeff = torch.full((self.num_envs,), self.cfg.drag_tilt_coeff, device=self.device)

        self._last_wp_advance_time = torch.zeros(self.num_envs, device=self.device)
        self._episode_time = torch.zeros(self.num_envs, device=self.device)

        self._per_env_final_dist = torch.zeros(self.num_envs, device=self.device)
        self._per_env_died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._per_env_wp_rate = torch.zeros(self.num_envs, device=self.device)
        self._per_env_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._hl_step_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Robot physical properties
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Obstacle tensors (same structure as v10)
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

        self._draw = None

        print(f"[INFO] Free-space forest SAC: {self._num_obstacles} trees (matching drone_freespace_sim.py)")
        print(f"[INFO]   Tree radii: {self._obstacle_radii.min().item():.2f}-{self._obstacle_radii.max().item():.2f}m, height: 4.0m")
        print(f"[INFO]   Arena: 20x20m, start: {self.cfg.static_start_pos}")
        dr_features = []
        if self.cfg.dr_wind_enabled: dr_features.append(f"wind(max={self.cfg.dr_wind_max_speed}m/s)")
        if self.cfg.dr_vel_scale_enabled: dr_features.append(f"vel_scale{self.cfg.dr_vel_scale_range}")
        if self.cfg.dr_drag_enabled: dr_features.append(f"drag{self.cfg.dr_drag_range}")
        if self.cfg.dr_lidar_noise_enabled: dr_features.append(f"lidar_noise(std={self.cfg.dr_lidar_noise_std})")
        if self.cfg.dr_obs_noise_enabled: dr_features.append(f"obs_noise(vel={self.cfg.dr_obs_vel_noise_std},angvel={self.cfg.dr_obs_angvel_noise_std})")
        print(f"[INFO]   DR: {', '.join(dr_features)}")

        # Episode logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "takeoff_ascent", "takeoff_alt_reward", "takeoff_drift",
                "stabilize_position", "stabilize_speed", "stabilize_altitude", "stabilize_ang_vel",
                "nav_xy_progress", "nav_velocity_align", "nav_altitude",
                "nav_speed_penalty", "nav_lateral", "nav_stability",
                "obstacle_proximity", "lidar_penalty", "obstacle_clearance", "stagnation",
                "subgoal_reachability", "subgoal_magnitude",
                "waypoint_bonus", "speed_carry", "wp_progress",
                "hover_position", "hover_speed", "hover_altitude",
                "land_descent", "land_xy_stability", "land_drift", "land_precision",
                "land_descent_control", "land_controlled_descent", "land_altitude_penalty",
                "goal_bonus", "crash_penalty", "time_penalty", "ang_vel", "yaw_rate", "upright",
            ]
        }

        self.set_debug_vis(self.cfg.debug_vis)

    # -- Scene Setup --

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._hl_lidar = RayCaster(self.cfg.hl_lidar)
        self.scene.sensors["hl_lidar"] = self._hl_lidar

        # Spawn trees matching drone_freespace_sim.py (simple brown cylinders)
        trunk_colors = [
            (0.35, 0.25, 0.15),  # matches drone_freespace_sim tree color
            (0.30, 0.20, 0.10),
            (0.40, 0.28, 0.14),
            (0.33, 0.22, 0.12),
            (0.38, 0.26, 0.16),
        ]
        for i, (ox, oy, radius, height) in enumerate(self.cfg.obstacle_positions):
            t_color = trunk_colors[i % len(trunk_colors)]
            trunk_cfg = sim_utils.CylinderCfg(
                radius=radius, height=height, axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=t_color),
            )
            trunk_cfg.func(
                f"/World/envs/env_0/Tree_{i}", trunk_cfg,
                translation=(ox, oy, height / 2.0),
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

    # -- Waypoint Generation (with obstacle clearance) --

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

        time_since_advance = self._episode_time - self._last_wp_advance_time
        stuck_timeout = is_navigate & is_intermediate & (time_since_advance > self.cfg.auto_advance_timeout)
        reached = reached | stuck_timeout

        if reached.any():
            self._current_wp_idx[reached] += 1
            reached_ids = torch.where(reached)[0]
            self._update_current_goal(reached_ids)
            new_delta = self._goal_pos_w[reached_ids] - self._robot.data.root_pos_w[reached_ids]
            self._prev_xy_dist[reached_ids] = torch.linalg.norm(new_delta[:, :2], dim=1)
            self._last_wp_advance_time[reached] = self._episode_time[reached]

        return reached

    # -- LiDAR Obstacle Augmentation (same as v10) --

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
            self._hl_lidar_ray_dirs[:, 2].abs() < 0.45
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

    # -- Low-Level Policy --

    def _compute_goal_for_ll(self) -> torch.Tensor:
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w
        real_goal_b, _ = subtract_frame_transforms(root_pos, root_quat, self._goal_pos_w)

        is_navigate = (self._phase == self.NAVIGATE)

        goal_xy = real_goal_b[:, :2]
        yaw_off = self._hl_yaw_offset
        cos_off = torch.cos(yaw_off)
        sin_off = torch.sin(yaw_off)
        rotated_x = cos_off * goal_xy[:, 0] - sin_off * goal_xy[:, 1]
        rotated_y = sin_off * goal_xy[:, 0] + cos_off * goal_xy[:, 1]
        modified_goal_b = torch.stack([
            rotated_x, rotated_y,
            real_goal_b[:, 2] + self._hl_alt_offset,
        ], dim=1)

        goal_for_ll = torch.where(
            is_navigate.unsqueeze(1).expand(-1, 3),
            modified_goal_b, real_goal_b,
        )

        speed_scale = torch.where(is_navigate, self._hl_speed_factor, torch.ones_like(self._hl_speed_factor))
        goal_for_ll = torch.cat([
            goal_for_ll[:, :2] * speed_scale.unsqueeze(1),
            goal_for_ll[:, 2:3],
        ], dim=1)

        return goal_for_ll

    def _run_low_level_policy(self):
        goal_b = self._compute_goal_for_ll()
        phase = self._phase

        K_xy = self.cfg.ll_gain_xy
        K_z = self.cfg.ll_gain_z
        K_yaw = self.cfg.ll_gain_yaw

        vx_cmd = K_xy * goal_b[:, 0]
        vy_cmd = K_xy * goal_b[:, 1]
        vz_cmd = K_z * goal_b[:, 2]
        yaw_to_goal = torch.atan2(goal_b[:, 1], goal_b[:, 0])
        yaw_rate_cmd = K_yaw * yaw_to_goal

        takeoff = (phase == self.TAKEOFF)
        if takeoff.any():
            alt_err = self.cfg.cruise_altitude - self._robot.data.root_pos_w[takeoff, 2]
            vx_cmd[takeoff] = 0.0
            vy_cmd[takeoff] = 0.0
            vz_cmd[takeoff] = K_z * alt_err
            yaw_rate_cmd[takeoff] = 0.0

        hold = (phase == self.STABILIZE) | (phase == self.HOVER)
        if hold.any():
            vx_cmd[hold] = K_xy * goal_b[hold, 0] * 0.3
            vy_cmd[hold] = K_xy * goal_b[hold, 1] * 0.3
            vz_cmd[hold] = K_z * goal_b[hold, 2]
            yaw_rate_cmd[hold] = 0.0

        land = (phase == self.LAND)
        if land.any():
            vx_cmd[land] = 0.0
            vy_cmd[land] = 0.0
            vz_cmd[land] = -0.5
            yaw_rate_cmd[land] = 0.0

        self._ll_vx_cmd = vx_cmd.clamp(-self.cfg.max_velocity_xy, self.cfg.max_velocity_xy)
        self._ll_vy_cmd = vy_cmd.clamp(-self.cfg.max_velocity_xy, self.cfg.max_velocity_xy)
        self._ll_vz_cmd = vz_cmd.clamp(-self.cfg.max_velocity_z, self.cfg.max_velocity_z)
        self._ll_yaw_rate_cmd = yaw_rate_cmd.clamp(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)

        # Velocity-projection safety (same as v10)
        nav_mask = (phase == self.NAVIGATE)
        if nav_mask.any():
            drone_pos = self._robot.data.root_pos_w
            drone_xy = drone_pos[:, :2].unsqueeze(1)
            obs_delta = drone_xy - self._obstacle_pos_w
            obs_dists = torch.linalg.norm(obs_delta, dim=2)
            obs_surface = obs_dists - self._obstacle_radii.unsqueeze(0)

            vel_w_actual = self._robot.data.root_lin_vel_w[:, :2]
            lookahead = self.cfg.predictive_avoidance_lookahead
            predicted_xy = drone_pos[:, :2] + vel_w_actual * lookahead
            pred_delta = predicted_xy.unsqueeze(1) - self._obstacle_pos_w
            pred_dists = torch.linalg.norm(pred_delta, dim=2)
            pred_surface = pred_dists - self._obstacle_radii.unsqueeze(0)

            effective_surface = torch.min(obs_surface, pred_surface)

            safety_d = self.cfg.reactive_safety_distance
            min_eff_surf = effective_surface.min(dim=1).values
            apply_mask = nav_mask & (min_eff_surf < safety_d)

            if apply_mask.any():
                qw = self._robot.data.root_quat_w[:, 0]
                qx = self._robot.data.root_quat_w[:, 1]
                qy = self._robot.data.root_quat_w[:, 2]
                qz = self._robot.data.root_quat_w[:, 3]
                yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                cos_yaw = torch.cos(yaw)
                sin_yaw = torch.sin(yaw)
                vx_w = cos_yaw * self._ll_vx_cmd - sin_yaw * self._ll_vy_cmd
                vy_w = sin_yaw * self._ll_vx_cmd + cos_yaw * self._ll_vy_cmd
                vel_w = torch.stack([vx_w, vy_w], dim=1)

                strength_all = (1.0 - effective_surface / safety_d).clamp(0.0, 1.0)
                away_dirs = obs_delta / (obs_dists.unsqueeze(2) + 1e-6)

                approach_speeds = (vel_w.unsqueeze(1) * away_dirs).sum(dim=2)

                total_correction = torch.zeros_like(vel_w)
                in_range = effective_surface < safety_d

                approaching = approach_speeds < 0
                remove_mask_obs = in_range & approaching
                remove_frac = strength_all
                correction_mag = (-approach_speeds * remove_frac).clamp(min=0.0)
                correction_mag = correction_mag * remove_mask_obs.float()
                correction_vecs = away_dirs * correction_mag.unsqueeze(2)
                total_correction += correction_vecs.sum(dim=1)

                push_mag = strength_all * self.cfg.reactive_safety_push_gain * in_range.float()
                push_vecs = away_dirs * push_mag.unsqueeze(2)
                total_correction += push_vecs.sum(dim=1)

                # Anti-stuck escape (same as v10)
                num_threatening = in_range.float().sum(dim=1)
                correction_mag_total = torch.linalg.norm(total_correction, dim=1)
                is_stuck = apply_mask & (num_threatening >= 2) & (correction_mag_total < 0.3)
                if is_stuck.any():
                    goal_delta_w = self._goal_pos_w[is_stuck, :2] - drone_pos[is_stuck, :2]
                    goal_dir = goal_delta_w / (torch.linalg.norm(goal_delta_w, dim=1, keepdim=True) + 1e-6)
                    perp_dir = torch.stack([-goal_dir[:, 1], goal_dir[:, 0]], dim=1)
                    nearest_obs_idx = effective_surface[is_stuck].argmin(dim=1)
                    stuck_indices = torch.where(is_stuck)[0]
                    nearest_away = away_dirs[stuck_indices, nearest_obs_idx]
                    side_dot = (perp_dir * nearest_away).sum(dim=1, keepdim=True)
                    perp_dir = perp_dir * side_dot.sign()
                    stag_time = (self._episode_time[is_stuck] - self._last_wp_advance_time[is_stuck]).clamp(min=0.0)
                    escape_strength = (stag_time / self.cfg.stagnation_timeout).clamp(0.0, 2.0)
                    escape_vel = perp_dir * escape_strength.unsqueeze(1) * self.cfg.escape_tangential_speed
                    total_correction[is_stuck] += escape_vel

                vel_w[apply_mask] += total_correction[apply_mask]

                cos_neg_yaw = torch.cos(-yaw)
                sin_neg_yaw = torch.sin(-yaw)
                new_vx_b = cos_neg_yaw * vel_w[:, 0] - sin_neg_yaw * vel_w[:, 1]
                new_vy_b = sin_neg_yaw * vel_w[:, 0] + cos_neg_yaw * vel_w[:, 1]

                self._ll_vx_cmd[apply_mask] = new_vx_b[apply_mask].clamp(
                    -self.cfg.max_velocity_xy, self.cfg.max_velocity_xy)
                self._ll_vy_cmd[apply_mask] = new_vy_b[apply_mask].clamp(
                    -self.cfg.max_velocity_xy, self.cfg.max_velocity_xy)

    def _euler_to_quat(self, roll, pitch, yaw):
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return torch.stack([w, x, y, z], dim=-1)

    def _step_ou_noise(self, dt: float):
        theta = self.cfg.noise_theta
        sqrt_dt = dt ** 0.5

        self._noise_vel += (-theta * self._noise_vel * dt
                            + self.cfg.noise_vel_sigma * sqrt_dt * torch.randn_like(self._noise_vel))
        self._noise_vel.clamp_(-self.cfg.noise_vel_max, self.cfg.noise_vel_max)

        self._noise_att += (-theta * self._noise_att * dt
                            + self.cfg.noise_att_sigma * sqrt_dt * torch.randn_like(self._noise_att))
        self._noise_att.clamp_(-self.cfg.noise_att_max, self.cfg.noise_att_max)

        self._noise_yaw += (-theta * self._noise_yaw * dt
                            + self.cfg.noise_yaw_sigma * sqrt_dt * torch.randn(self.num_envs, device=self.device))
        self._noise_yaw.clamp_(-self.cfg.noise_yaw_max, self.cfg.noise_yaw_max)

        if self.cfg.dr_wind_enabled:
            self._dr_wind += (-self.cfg.dr_wind_theta * self._dr_wind * dt
                              + self.cfg.dr_wind_sigma * sqrt_dt * torch.randn_like(self._dr_wind))
            self._dr_wind.clamp_(-self.cfg.dr_wind_max_speed, self.cfg.dr_wind_max_speed)
            self._dr_wind[:, 2].mul_(0.3)

    def _kinematic_update(self):
        dt = self.sim.cfg.dt
        N = self.num_envs

        self._step_ou_noise(dt)

        cmd_alpha = 0.06
        self._smooth_vx = (1 - cmd_alpha) * self._smooth_vx + cmd_alpha * self._ll_vx_cmd
        self._smooth_vy = (1 - cmd_alpha) * self._smooth_vy + cmd_alpha * self._ll_vy_cmd
        self._smooth_vz = (1 - cmd_alpha) * self._smooth_vz + cmd_alpha * self._ll_vz_cmd
        self._smooth_yaw_rate = (1 - cmd_alpha) * self._smooth_yaw_rate + cmd_alpha * self._ll_yaw_rate_cmd

        if self.cfg.dr_vel_scale_enabled:
            eff_vx = self._smooth_vx * self._dr_vel_scale
            eff_vy = self._smooth_vy * self._dr_vel_scale
            eff_vz = self._smooth_vz * self._dr_vel_scale
        else:
            eff_vx = self._smooth_vx
            eff_vy = self._smooth_vy
            eff_vz = self._smooth_vz

        noisy_vx = eff_vx + self._noise_vel[:, 0]
        noisy_vy = eff_vy + self._noise_vel[:, 1]
        noisy_vz = eff_vz + self._noise_vel[:, 2] * 0.5

        cy = torch.cos(self._tracked_yaw)
        sy = torch.sin(self._tracked_yaw)
        vel_w_x = cy * noisy_vx - sy * noisy_vy
        vel_w_y = sy * noisy_vx + cy * noisy_vy
        vel_w_z = noisy_vz

        if self.cfg.dr_wind_enabled:
            vel_w_x = vel_w_x + self._dr_wind[:, 0]
            vel_w_y = vel_w_y + self._dr_wind[:, 1]
            vel_w_z = vel_w_z + self._dr_wind[:, 2]
        self._desired_vel_w[:, 0] = vel_w_x
        self._desired_vel_w[:, 1] = vel_w_y
        self._desired_vel_w[:, 2] = vel_w_z

        pos = self._robot.data.root_pos_w.clone()
        pos[:, 0] += vel_w_x * dt
        pos[:, 1] += vel_w_y * dt
        pos[:, 2] += vel_w_z * dt
        pos[:, 2].clamp_(min=0.05, max=self.cfg.max_flight_height)

        # Hard collision wall (same as v10)
        col_radius = self.cfg.obstacle_wall_radius
        for _collision_pass in range(2):
            drone_xy_new = pos[:, :2].unsqueeze(1)
            drone_alt_new = pos[:, 2]
            obs_delta_col = drone_xy_new - self._obstacle_pos_w
            obs_center_dists_col = torch.linalg.norm(obs_delta_col, dim=2)
            obs_surface_dists_col = obs_center_dists_col - self._obstacle_radii.unsqueeze(0)
            is_tall_col = drone_alt_new.unsqueeze(1) < self._obstacle_heights.unsqueeze(0)
            penetrating = (obs_surface_dists_col < col_radius) & is_tall_col
            pen_surface = torch.where(penetrating, obs_surface_dists_col, torch.full_like(obs_surface_dists_col, 100.0))
            min_pen_dist, min_pen_idx = pen_surface.min(dim=1)
            fix_mask = min_pen_dist < col_radius
            if not fix_mask.any():
                break
            fix_idx = torch.where(fix_mask)[0]
            fix_oi = min_pen_idx[fix_mask]
            fix_delta = obs_delta_col[fix_idx, fix_oi]
            fix_dist = obs_center_dists_col[fix_idx, fix_oi].clamp(min=1e-6)
            away_dir = fix_delta / fix_dist.unsqueeze(1)
            required_dist = self._obstacle_radii[fix_oi] + col_radius + 0.05
            obs_xy = self._obstacle_pos_w[fix_idx, fix_oi]
            new_xy = obs_xy + away_dir * required_dist.unsqueeze(1)
            pos[fix_mask, 0] = new_xy[:, 0]
            pos[fix_mask, 1] = new_xy[:, 1]
            approach = -(vel_w_x[fix_mask] * away_dir[:, 0] + vel_w_y[fix_mask] * away_dir[:, 1])
            approach = approach.clamp(min=0.0)
            vel_w_x[fix_mask] = vel_w_x[fix_mask] + away_dir[:, 0] * approach
            vel_w_y[fix_mask] = vel_w_y[fix_mask] + away_dir[:, 1] * approach
            cy_fix = torch.cos(self._tracked_yaw[fix_mask])
            sy_fix = torch.sin(self._tracked_yaw[fix_mask])
            away_body_x = cy_fix * away_dir[:, 0] + sy_fix * away_dir[:, 1]
            away_body_y = -sy_fix * away_dir[:, 0] + cy_fix * away_dir[:, 1]
            smooth_approach = -(self._smooth_vx[fix_mask] * away_body_x + self._smooth_vy[fix_mask] * away_body_y)
            smooth_approach = smooth_approach.clamp(min=0.0)
            self._smooth_vx[fix_mask] = self._smooth_vx[fix_mask] + away_body_x * smooth_approach
            self._smooth_vy[fix_mask] = self._smooth_vy[fix_mask] + away_body_y * smooth_approach

        self._tracked_yaw += (self._smooth_yaw_rate + self._noise_yaw) * dt

        g = 9.81
        max_tilt = self.cfg.pid_max_tilt

        raw_accel_w = (self._desired_vel_w - self._prev_vel_w) / dt
        self._prev_vel_w = self._desired_vel_w.clone()
        accel_alpha = 0.15
        self._smoothed_accel_w = (1 - accel_alpha) * self._smoothed_accel_w + accel_alpha * raw_accel_w

        accel_body_x = cy * self._smoothed_accel_w[:, 0] + sy * self._smoothed_accel_w[:, 1]
        accel_body_y = -sy * self._smoothed_accel_w[:, 0] + cy * self._smoothed_accel_w[:, 1]

        vel_body_x = cy * vel_w_x + sy * vel_w_y
        vel_body_y = -sy * vel_w_x + cy * vel_w_y

        k_drag = self._dr_drag_coeff if self.cfg.dr_drag_enabled else torch.full((self.num_envs,), self.cfg.drag_tilt_coeff, device=self.device)
        pitch = (accel_body_x / g + vel_body_x * k_drag + self._noise_att[:, 0]).clamp(-max_tilt, max_tilt)
        roll = (-accel_body_y / g - vel_body_y * k_drag + self._noise_att[:, 1]).clamp(-max_tilt, max_tilt)
        quat = self._euler_to_quat(roll, pitch, self._tracked_yaw)

        root_pose = torch.cat([pos, quat], dim=-1)
        vel_state = torch.zeros(N, 6, device=self.device)
        vel_state[:, 0] = vel_w_x
        vel_state[:, 1] = vel_w_y
        vel_state[:, 2] = vel_w_z
        vel_state[:, 5] = self._smooth_yaw_rate
        self._robot.write_root_pose_to_sim(root_pose)
        self._robot.write_root_velocity_to_sim(vel_state)

    # -- Physics Step --

    def _pre_physics_step(self, actions: torch.Tensor):
        self._hl_step_counter += 1
        update_mask = self._hl_step_counter >= 5
        if update_mask.any():
            self._hl_step_counter[update_mask] = 0
            a = actions.clone().clamp(-1.0, 1.0)
            lo, hi = self.cfg.speed_factor_range
            target_yaw = a[:, 0] * self.cfg.max_yaw_offset
            target_spd = lo + (a[:, 1] + 1.0) * 0.5 * (hi - lo)
            target_alt = a[:, 2] * self.cfg.max_altitude_offset
            alpha = self.cfg.goal_modifier_smoothing
            self._hl_yaw_offset[update_mask] = ((1 - alpha) * self._hl_yaw_offset + alpha * target_yaw)[update_mask]
            self._hl_speed_factor[update_mask] = ((1 - alpha) * self._hl_speed_factor + alpha * target_spd)[update_mask]
            self._hl_alt_offset[update_mask] = ((1 - alpha) * self._hl_alt_offset + alpha * target_alt)[update_mask]

        self._episode_time += self.step_dt
        self._run_low_level_policy()

    def _apply_action(self):
        self._kinematic_update()
        self._thrust[:, 0, 2] = self._robot_weight
        self._moment[:] = 0
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )

    # -- Observations (same as v10 with obstacle info) --

    def _get_observations(self) -> dict:
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w

        goal_pos_b, _ = subtract_frame_transforms(root_pos, root_quat, self._goal_pos_w)

        hl_lidar_hits = self._hl_lidar.data.ray_hits_w
        hl_lidar_pos = self._hl_lidar.data.pos_w.unsqueeze(1)
        hl_lidar_dists = torch.linalg.norm(hl_lidar_hits - hl_lidar_pos, dim=-1)
        hl_lidar_dists = self._augment_hl_lidar_with_obstacles(hl_lidar_dists, root_pos, root_quat)
        hl_lidar_norm = (hl_lidar_dists / self.cfg.hl_lidar_max_distance).clamp(0.0, 1.0)

        if self.cfg.dr_lidar_noise_enabled:
            hl_lidar_norm = (hl_lidar_norm + self.cfg.dr_lidar_noise_std
                             * torch.randn_like(hl_lidar_norm)).clamp(0.0, 1.0)

        self._last_hl_lidar_distances = hl_lidar_dists
        if self._hl_lidar_horiz_indices is not None and len(self._hl_lidar_horiz_indices) > 0:
            horiz_dists = hl_lidar_dists[:, self._hl_lidar_horiz_indices]
            self._min_horiz_lidar_dist = horiz_dists.min(dim=1).values

        phase_one_hot = torch.nn.functional.one_hot(self._phase.long(), 5).float()

        # Nearest obstacle in body frame (same as v10)
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

        lin_vel_b_obs = self._robot.data.root_lin_vel_b
        ang_vel_b_obs = self._robot.data.root_ang_vel_b
        if self.cfg.dr_obs_noise_enabled:
            lin_vel_b_obs = lin_vel_b_obs + self.cfg.dr_obs_vel_noise_std * torch.randn_like(lin_vel_b_obs)
            ang_vel_b_obs = ang_vel_b_obs + self.cfg.dr_obs_angvel_noise_std * torch.randn_like(ang_vel_b_obs)

        obs = torch.cat([
            hl_lidar_norm,                          # 1347
            lin_vel_b_obs,                          # 3
            ang_vel_b_obs,                          # 3
            self._robot.data.projected_gravity_b,   # 3
            goal_pos_b,                             # 3
            phase_one_hot,                          # 5
            nearest_body_dir,                       # 2
            nearest_dist_normalized,                # 1
        ], dim=-1)  # Total: 1367
        return {"policy": obs}

    # -- Rewards (same structure as v10) --

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

        # TAKEOFF
        alt_progress = altitude - self._prev_alt
        self._prev_alt = altitude.clone()
        takeoff_ascent = torch.where(is_takeoff, alt_progress * self.cfg.takeoff_ascent_scale, torch.zeros_like(alt_progress))
        alt_fraction = (altitude / self.cfg.cruise_altitude).clamp(0.0, 1.0)
        takeoff_alt_reward = torch.where(is_takeoff, alt_fraction * self.cfg.takeoff_altitude_reward_scale * dt, torch.zeros_like(altitude))
        takeoff_drift = torch.where(is_takeoff, horizontal_speed * self.cfg.takeoff_drift_penalty * dt, torch.zeros_like(horizontal_speed))

        # STABILIZE
        stabilize_position = torch.where(is_stabilize, torch.exp(-start_xy_dist) * self.cfg.stabilize_position_scale * dt, torch.zeros_like(start_xy_dist))
        stabilize_speed = torch.where(is_stabilize, torch.exp(-speed) * self.cfg.stabilize_low_speed_scale * dt, torch.zeros_like(speed))
        stabilize_altitude = torch.where(is_stabilize, alt_error * self.cfg.stabilize_altitude_scale * dt, torch.zeros_like(alt_error))
        stabilize_ang_vel = torch.where(is_stabilize, torch.exp(-ang_vel_magnitude) * self.cfg.stabilize_ang_vel_scale * dt, torch.zeros_like(ang_vel_magnitude))

        # NAVIGATE
        xy_progress = self._prev_xy_dist - xy_dist
        self._prev_xy_dist = xy_dist.clone()
        nav_xy_progress = torch.where(is_navigate, xy_progress * self.cfg.nav_xy_progress_scale, torch.zeros_like(xy_progress))

        goal_dir_xy = delta[:, :2] / (xy_dist.unsqueeze(1) + 1e-6)
        vel_toward_goal = torch.sum(lin_vel_w[:, :2] * goal_dir_xy, dim=1)
        nav_velocity_align = torch.where(is_navigate, torch.clamp(vel_toward_goal, min=0.0) * self.cfg.nav_velocity_align_scale * dt, torch.zeros_like(vel_toward_goal))
        nav_altitude = torch.where(is_navigate, alt_error * self.cfg.nav_altitude_scale * dt, torch.zeros_like(alt_error))

        excess_speed = torch.clamp(speed - self.cfg.nav_max_speed, min=0.0)
        nav_speed_penalty = torch.where(is_navigate, excess_speed * self.cfg.nav_speed_penalty_scale * dt, torch.zeros_like(excess_speed))

        lateral_scale = torch.where(
            is_at_final_wp,
            torch.full_like(xy_dist, self.cfg.nav_lateral_penalty_scale),
            torch.full_like(xy_dist, self.cfg.nav_lateral_penalty_scale * 0.3),
        )
        vel_lateral = lin_vel_w[:, :2] - vel_toward_goal.unsqueeze(1) * goal_dir_xy
        lateral_speed = torch.linalg.norm(vel_lateral, dim=1)
        nav_lateral_penalty = torch.where(is_navigate, lateral_speed * lateral_scale * dt, torch.zeros_like(lateral_speed))

        gravity_z_nav = self._robot.data.projected_gravity_b[:, 2]
        nav_stability = torch.where(
            is_navigate,
            (torch.exp(-ang_vel_magnitude) + (-gravity_z_nav)) * self.cfg.nav_stability_scale * dt,
            torch.zeros_like(ang_vel_magnitude),
        )

        # Obstacle rewards (same as v10)
        drone_xy = root_pos[:, :2].unsqueeze(1)
        obstacle_xy = self._obstacle_pos_w
        obstacle_dists = torch.linalg.norm(drone_xy - obstacle_xy, dim=2)
        obstacle_surface_dists = obstacle_dists - self._obstacle_radii.unsqueeze(0)
        min_obstacle_dist, _ = obstacle_surface_dists.min(dim=1)
        proximity_raw = torch.exp(-min_obstacle_dist) * (min_obstacle_dist < self.cfg.obstacle_safe_distance).float()
        obstacle_proximity = torch.where(is_navigate, proximity_raw * self.cfg.obstacle_proximity_penalty_scale * dt, torch.zeros_like(proximity_raw))

        is_clear = (min_obstacle_dist > self.cfg.obstacle_safe_distance).float()
        obstacle_clearance = torch.where(is_navigate, is_clear * self.cfg.obstacle_clearance_bonus_scale * dt, torch.zeros_like(is_clear))

        time_since_advance = self._episode_time - self._last_wp_advance_time
        stagnation_excess = (time_since_advance - self.cfg.stagnation_timeout).clamp(min=0.0)
        stagnation_penalty = torch.where(
            is_navigate, stagnation_excess * self.cfg.stagnation_penalty_scale * dt, torch.zeros_like(stagnation_excess)
        )

        lidar_dist = self._min_horiz_lidar_dist
        lidar_danger = (lidar_dist < self.cfg.lidar_danger_distance).float()
        lidar_closeness = (1.0 - lidar_dist / self.cfg.lidar_danger_distance).clamp(min=0.0)
        lidar_penalty = torch.where(is_navigate, lidar_danger * lidar_closeness * self.cfg.lidar_obstacle_penalty_scale * dt, torch.zeros_like(lidar_dist))

        yaw_deviation = torch.abs(self._hl_yaw_offset)
        subgoal_magnitude = torch.where(is_navigate, yaw_deviation * self.cfg.subgoal_magnitude_penalty * dt, torch.zeros_like(yaw_deviation))

        subgoal_reachability = torch.where(
            is_navigate & (self._min_horiz_lidar_dist > self.cfg.lidar_danger_distance),
            torch.full_like(yaw_deviation, self.cfg.subgoal_reachability_scale * dt),
            torch.zeros_like(yaw_deviation),
        )

        just_advanced = self._check_waypoint_advancement(xy_dist, speed)
        waypoint_bonus = just_advanced.float() * self.cfg.intermediate_waypoint_bonus
        speed_carry = torch.where(
            just_advanced,
            torch.clamp(speed, min=0.0, max=self.cfg.nav_max_speed) * self.cfg.speed_carry_bonus_scale,
            torch.zeros_like(speed),
        )
        fraction_done = self._current_wp_idx.float() / self._num_waypoints.float().clamp(min=1)
        wp_progress_reward = torch.where(is_navigate, fraction_done * self.cfg.waypoint_progress_scale * dt, torch.zeros_like(fraction_done))

        # HOVER
        hover_position = torch.where(is_hover, torch.exp(-xy_dist) * self.cfg.hover_position_scale * dt, torch.zeros_like(xy_dist))
        hover_speed = torch.where(is_hover, torch.exp(-speed) * self.cfg.hover_low_speed_scale * dt, torch.zeros_like(speed))
        hover_altitude = torch.where(is_hover, alt_error * self.cfg.hover_altitude_scale * dt, torch.zeros_like(alt_error))

        # LAND
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

        # All-phase
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

        # Phase transitions
        to_stabilize = is_takeoff & (alt_error < self.cfg.takeoff_altitude_tolerance)
        self._phase[to_stabilize] = self.STABILIZE
        self._stabilize_timer[to_stabilize] = 0.0

        self._stabilize_timer[self._phase == self.STABILIZE] += dt
        to_navigate = (self._phase == self.STABILIZE) & (self._stabilize_timer >= self.cfg.stabilize_duration)
        self._phase[to_navigate] = self.NAVIGATE

        current_delta = self._goal_pos_w - root_pos
        current_xy_dist = torch.linalg.norm(current_delta[:, :2], dim=1)
        altitude_now = self._robot.data.root_pos_w[:, 2]
        is_at_final_wp_nav = (self._phase == self.NAVIGATE) & (self._current_wp_idx >= (self._num_waypoints - 1))
        time_at_final = self._episode_time - self._last_wp_advance_time
        force_hover = is_at_final_wp_nav & (time_at_final > self.cfg.final_wp_force_hover_timeout)
        normal_hover = is_at_final_wp_nav & (current_xy_dist < self.cfg.final_waypoint_tolerance)
        to_hover = (normal_hover | force_hover) & (speed < 3.0) & (altitude_now > 0.3)
        force_hover_landing = to_hover & force_hover & ~normal_hover
        if force_hover_landing.any():
            self._final_goal_pos_w[force_hover_landing, 0] = root_pos[force_hover_landing, 0]
            self._final_goal_pos_w[force_hover_landing, 1] = root_pos[force_hover_landing, 1]
            self._final_goal_pos_w[force_hover_landing, 2] = 0.2
        self._phase[to_hover] = self.HOVER
        self._hover_timer[to_hover] = 0.0

        self._hover_timer[self._phase == self.HOVER] += dt
        to_land = (self._phase == self.HOVER) & (self._hover_timer >= self.cfg.hover_duration)
        self._phase[to_land] = self.LAND
        self._goal_pos_w[to_land] = self._final_goal_pos_w[to_land]
        if to_land.any():
            land_delta = self._final_goal_pos_w[to_land] - root_pos[to_land]
            self._prev_z_dist[to_land] = torch.abs(land_delta[:, 2])

        reward = (
            takeoff_ascent + takeoff_alt_reward + takeoff_drift
            + stabilize_position + stabilize_speed + stabilize_altitude + stabilize_ang_vel
            + nav_xy_progress + nav_velocity_align + nav_altitude + nav_speed_penalty
            + nav_lateral_penalty + nav_stability
            + obstacle_proximity + obstacle_clearance + stagnation_penalty + lidar_penalty + subgoal_reachability + subgoal_magnitude
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
        self._episode_sums["obstacle_clearance"] += obstacle_clearance
        self._episode_sums["stagnation"] += stagnation_penalty
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

    # -- Terminations --

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

        goal_reached = goal_reached | ((self._phase == self.LAND) & touched_ground)

        died = too_low | too_high | flipped | hit_obstacle

        self._per_env_goal_reached = goal_reached
        self._crash_too_low = too_low
        self._crash_too_high = too_high
        self._crash_flipped = flipped
        self._crash_hit_obstacle = hit_obstacle

        return died, time_out | goal_reached | touched_ground

    # -- Reset --

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        per_env_dists = torch.linalg.norm(
            self._final_goal_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )
        final_distance = per_env_dists.mean()
        wp_completion_rate = self._current_wp_idx[env_ids].float() / self._num_waypoints[env_ids].float().clamp(min=1)

        self._per_env_final_dist[env_ids] = per_env_dists
        self._per_env_died[env_ids] = self.reset_terminated[env_ids]
        self._per_env_wp_rate[env_ids] = wp_completion_rate
        if hasattr(self, '_per_env_goal_reached'):
            self._per_env_success[env_ids] = self._per_env_goal_reached[env_ids]
        else:
            self._per_env_success[env_ids] = ~self.reset_terminated[env_ids] & ~self.reset_time_outs[env_ids]

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

        phase_names = ["TAKEOFF", "STABILIZE", "NAVIGATE", "HOVER", "LAND"]
        for pi, pname in enumerate(phase_names):
            phase_died = (died_mask & (self._phase[env_ids] == pi)).sum().item()
            self.extras["log"][f"Episode_Termination/died_{pname}"] = phase_died

        cause_names = ["hit_obstacle", "too_low", "too_high", "flipped"]
        cause_tensors = [
            getattr(self, "_crash_hit_obstacle", None),
            getattr(self, "_crash_too_low", None),
            getattr(self, "_crash_too_high", None),
            getattr(self, "_crash_flipped", None),
        ]
        for cname, ctensor in zip(cause_names, cause_tensors):
            if ctensor is not None:
                count = (died_mask & ctensor[env_ids]).sum().item()
            else:
                count = 0
            self.extras["log"][f"Episode_Termination/cause_{cname}"] = count

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._generate_waypoints(env_ids)

        num_reset = len(env_ids)
        start_local = torch.tensor(self.cfg.static_start_pos, dtype=torch.float32, device=self.device)
        start_pos = start_local.unsqueeze(0).expand(num_reset, -1).clone()
        start_pos[:, :2] += self._terrain.env_origins[env_ids, :2]
        self._start_pos_w[env_ids] = start_pos

        # Update obstacle world positions
        env_origins_xy = self._terrain.env_origins[env_ids, :2]
        self._obstacle_pos_w[env_ids] = self._obstacle_local.unsqueeze(0) + env_origins_xy.unsqueeze(1)

        self._current_wp_idx[env_ids] = 0
        self._update_current_goal(env_ids)

        delta = self._goal_pos_w[env_ids] - start_pos
        self._prev_xy_dist[env_ids] = torch.linalg.norm(delta[:, :2], dim=1)
        self._prev_z_dist[env_ids] = torch.abs(delta[:, 2])
        self._prev_alt[env_ids] = start_pos[:, 2]

        self._phase[env_ids] = self.TAKEOFF
        self._stabilize_timer[env_ids] = 0.0
        self._hover_timer[env_ids] = 0.0

        self._hl_step_counter[env_ids] = 0
        self._hl_yaw_offset[env_ids] = 0.0
        self._hl_speed_factor[env_ids] = 1.0
        self._hl_alt_offset[env_ids] = 0.0
        self._ll_vx_cmd[env_ids] = 0.0
        self._ll_vy_cmd[env_ids] = 0.0
        self._ll_vz_cmd[env_ids] = 0.0
        self._ll_yaw_rate_cmd[env_ids] = 0.0
        self._tracked_yaw[env_ids] = 0.0
        self._desired_vel_w[env_ids] = 0.0
        self._actual_vel_w[env_ids] = 0.0
        self._prev_vel_w[env_ids] = 0.0
        self._smooth_vx[env_ids] = 0.0
        self._smooth_vy[env_ids] = 0.0
        self._smooth_vz[env_ids] = 0.0
        self._smooth_yaw_rate[env_ids] = 0.0
        self._smoothed_accel_w[env_ids] = 0.0
        self._noise_vel[env_ids] = 0.0
        self._noise_att[env_ids] = 0.0
        self._noise_yaw[env_ids] = 0.0

        self._dr_wind[env_ids] = 0.0
        if self.cfg.dr_vel_scale_enabled:
            lo, hi = self.cfg.dr_vel_scale_range
            self._dr_vel_scale[env_ids] = lo + (hi - lo) * torch.rand(num_reset, device=self.device)
        if self.cfg.dr_drag_enabled:
            lo, hi = self.cfg.dr_drag_range
            self._dr_drag_coeff[env_ids] = lo + (hi - lo) * torch.rand(num_reset, device=self.device)

        self._last_wp_advance_time[env_ids] = 0.0
        self._episode_time[env_ids] = 0.0

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = start_pos
        default_root_state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # -- Debug Visualization --

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
                valid.unsqueeze(1).expand(-1, 3),
                wp_positions,
                torch.full_like(wp_positions, -1000.0),
            )
            self.waypoint_visualizers[wp_idx].visualize(viz_pos)

        self.subgoal_visualizer.visualize(
            torch.full((self.num_envs, 3), -1000.0, device=self.device)
        )

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


# ==============================================================================
# Gym Registration
# ==============================================================================

gym.register(
    id="Isaac-FreespaceForest-SAC-v1",
    entry_point=f"{__name__}:FreespaceForestNavEnv",
    disable_env_checker=True,
    kwargs={"cfg": FreespaceForestNavEnvCfg()},
)


# ==============================================================================
# Multi-Camera Setup
# ==============================================================================

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


# ==============================================================================
# Train
# ==============================================================================

def train():
    env_cfg = FreespaceForestNavEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    log_root = os.path.join(SCRIPT_DIR, "logs", "freespace_forest_sac", "crazyflie_freespace_forest_sac")
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Logging to: {log_dir}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Total timesteps: {args_cli.total_timesteps}")
    print(f"[INFO] Waypoints: {env_cfg.min_num_waypoints}-{env_cfg.max_num_waypoints}")
    print(f"[INFO] FOREST: {len(env_cfg.obstacle_positions)} trees (matching drone_freespace_sim.py)")
    print(f"[INFO] High-level: 10 Hz, {env_cfg.observation_space - 20}-ray LiDAR ({env_cfg.hl_lidar_n_channels}ch), action=goal modifier")
    print(f"[INFO] Low-level: 50 Hz (deterministic), PID: 100 Hz")

    env = gym.make("Isaac-FreespaceForest-SAC-v1", cfg=env_cfg)
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
                    cnn_out_dim=32,
                    features_dim=128,
                    lidar_channels=env_cfg.hl_lidar_n_channels,
                ),
                net_arch=[512, 256],
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
        name_prefix="hl_sac_model",
    )

    start_time = time.time()
    agent.learn(total_timesteps=args_cli.total_timesteps, callback=checkpoint_callback, progress_bar=True)
    elapsed = time.time() - start_time

    final_path = os.path.join(log_dir, "hl_sac_final")
    agent.save(final_path)

    print(f"\n[INFO] Training complete! Duration: {elapsed:.1f}s")
    print(f"[INFO] Model saved to: {final_path}.zip")
    print(f"[INFO] Logs saved to: {log_dir}")
    env.close()


# ==============================================================================
# Play
# ==============================================================================

def play():
    env_cfg = FreespaceForestNavEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 180.0
    env_cfg.debug_vis = True
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    env_cfg.dr_wind_enabled = False
    env_cfg.dr_vel_scale_enabled = False
    env_cfg.dr_drag_enabled = False
    env_cfg.dr_lidar_noise_enabled = False
    env_cfg.dr_obs_noise_enabled = False
    env_cfg.goal_modifier_smoothing = 0.15

    env = gym.make("Isaac-FreespaceForest-SAC-v1", cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    agent = None
    if args_cli.checkpoint is not None:
        agent = SAC.load(args_cli.checkpoint, env=env, device="cuda:0")
    obs = env.reset()

    cameras = setup_cameras(env.unwrapped)

    episode_count = 0
    total_reward = 0.0
    step_count = 0
    playback_dt = env_cfg.sim.dt * env_cfg.decimation * 3.0

    print(f"\n{'='*60}")
    print(f"[INFO] Forest environment (40 trees, matching drone_freespace_sim.py)")
    if agent:
        print(f"[INFO] High-level: trained SAC policy")
    else:
        print(f"[INFO] High-level: NONE (zero actions = neutral goal modifiers)")
    print(f"[INFO] Low-level: proportional controller (gains: xy={env_cfg.ll_gain_xy}, z={env_cfg.ll_gain_z}, yaw={env_cfg.ll_gain_yaw})")
    print(f"[INFO] Press Ctrl+C to stop.")
    print(f"{'='*60}\n")

    zero_actions = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)

    try:
        while simulation_app.is_running():
            step_start = time.time()
            if agent:
                actions, _ = agent.predict(obs, deterministic=True)
            else:
                actions = zero_actions
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
                rq = unwrapped._robot.data.root_quat_w[0].cpu()
                pitch_approx = math.asin(max(-1, min(1, 2*(rq[0]*rq[2] - rq[3]*rq[1]).item())))
                roll_approx = math.atan2(2*(rq[0]*rq[1] + rq[2]*rq[3]).item(),
                                         1 - 2*(rq[1]**2 + rq[2]**2).item())
                print(f"  Step {step_count} | {phase_names[phase]} | WP {wp_idx}/{num_wps} | "
                      f"Pos ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}) | "
                      f"P={math.degrees(pitch_approx):+.1f}deg R={math.degrees(roll_approx):+.1f}deg | "
                      f"HL: yaw={yaw_off:+.2f} spd={spd_fac:.2f} alt={alt_off:+.2f} | "
                      f"Rew={rewards[0].item():.3f}")

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


# ==============================================================================
# Eval
# ==============================================================================

def evaluate():
    assert args_cli.checkpoint is not None, "Must provide --checkpoint for eval mode."

    env_cfg = FreespaceForestNavEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 180.0
    env_cfg.debug_vis = False
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    env_cfg.dr_wind_enabled = False
    env_cfg.dr_vel_scale_enabled = False
    env_cfg.dr_drag_enabled = False
    env_cfg.dr_lidar_noise_enabled = False
    env_cfg.dr_obs_noise_enabled = False
    env_cfg.goal_modifier_smoothing = 0.15

    env = gym.make("Isaac-FreespaceForest-SAC-v1", cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)
    agent = SAC.load(args_cli.checkpoint, env=env, device="cuda:0")
    obs = env.reset()

    env.unwrapped.episode_length_buf[:] = 0

    total_episodes = 0
    episode_rewards = []
    final_distances = []
    episode_lengths = []
    crashes = 0
    goal_reached_count = 0
    wp_completion_rates = []
    phase_crashes = {"TAKEOFF": 0, "STABILIZE": 0, "NAVIGATE": 0, "HOVER": 0, "LAND": 0}
    cause_crashes = {"hit_obstacle": 0, "too_low": 0, "too_high": 0, "flipped": 0}
    crash_details = []
    num_eval_envs = args_cli.num_envs
    env_rewards = np.zeros(num_eval_envs)
    env_steps = np.zeros(num_eval_envs, dtype=int)
    step_dt = env_cfg.sim.dt * env_cfg.decimation

    print(f"\n{'='*60}")
    print(f"EVALUATION: Free-space Forest SAC | {args_cli.num_episodes} episodes")
    print(f"Checkpoint: {os.path.basename(args_cli.checkpoint)}")
    print(f"Forest: {len(env_cfg.obstacle_positions)} trees (matching drone_freespace_sim.py)")
    print(f"LL: deterministic proportional controller")
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
                    raw_env = env.unwrapped
                    if hasattr(raw_env, '_per_env_final_dist'):
                        dist = raw_env._per_env_final_dist[i].item()
                        died_flag = raw_env._per_env_died[i].item()
                        wp_rate = raw_env._per_env_wp_rate[i].item()
                        success_flag = raw_env._per_env_success[i].item() if hasattr(raw_env, '_per_env_success') else False
                    else:
                        dist = extras.get("Metrics/final_distance_to_goal", -1)
                        died_flag = extras.get("Episode_Termination/died", 0) > 0
                        wp_rate = extras.get("Metrics/waypoint_completion_rate", 0)
                        success_flag = False
                    final_distances.append(dist)
                    wp_completion_rates.append(wp_rate)
                    if died_flag:
                        crashes += 1
                        for pname in phase_crashes:
                            phase_crashes[pname] += extras.get(f"Episode_Termination/died_{pname}", 0)
                        for cname in cause_crashes:
                            cause_crashes[cname] += extras.get(f"Episode_Termination/cause_{cname}", 0)
                        ep_phase = next((p for p in phase_crashes if extras.get(f"Episode_Termination/died_{p}", 0) > 0), None)
                        ep_cause = next((c for c in cause_crashes if extras.get(f"Episode_Termination/cause_{c}", 0) > 0), None)
                        if ep_phase and ep_cause:
                            crash_details.append((ep_phase, ep_cause))
                    if success_flag or (dist >= 0 and dist < env_cfg.goal_threshold):
                        goal_reached_count += 1

                if total_episodes % 10 == 0 or total_episodes == args_cli.num_episodes:
                    print(f"  Episodes: {total_episodes}/{args_cli.num_episodes}")

                env_rewards[i] = 0.0
                env_steps[i] = 0

    n = len(episode_rewards)
    if n == 0:
        print("No episodes completed!", flush=True)
        env.close()
        return

    try:
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        success_rate = (goal_reached_count / n) * 100
        crash_rate = (crashes / n) * 100
        avg_wp_rate = np.mean(wp_completion_rates) * 100 if wp_completion_rates else 0

        valid_distances = [d for d in final_distances if d >= 0]
        avg_distance = np.mean(valid_distances) if valid_distances else -1
        min_distance = np.min(valid_distances) if valid_distances else -1

        print(f"\n{'='*60}", flush=True)
        print(f"RESULTS -- Free-space Forest SAC ({n} episodes)", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Success Rate:           {success_rate:6.1f}%  ({goal_reached_count}/{n})", flush=True)
        print(f"  Crash Rate:             {crash_rate:6.1f}%  ({crashes}/{n})", flush=True)
        print(f"  Crashes by Phase:", flush=True)
        for pname, count in phase_crashes.items():
            print(f"    {pname:12s}:     {count:4d}", flush=True)
        print(f"  Crashes by Cause:", flush=True)
        for cname, count in cause_crashes.items():
            print(f"    {cname:16s}: {count:4d}", flush=True)
        if crash_details:
            print(f"  Crash Details (Phase + Cause):", flush=True)
            for phase, cause in crash_details:
                print(f"    - {phase} -> {cause}", flush=True)
        print(f"  Waypoint Completion:    {avg_wp_rate:6.1f}%", flush=True)
        print(f"  Avg Final Distance:     {avg_distance:6.3f} m", flush=True)
        print(f"  Min Final Distance:     {min_distance:6.3f} m", flush=True)
        print(f"  Avg Reward:             {avg_reward:8.2f}  (+/- {std_reward:.2f})", flush=True)
        print(f"  Avg Episode Length:      {avg_length:5.2f} s", flush=True)
        print(f"{'='*60}\n", flush=True)
    except Exception as e:
        print(f"[ERROR] Results printing failed: {e}", flush=True)
        import traceback; traceback.print_exc()
    env.close()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    if args_cli.mode == "train":
        train()
    elif args_cli.mode == "play":
        play()
    elif args_cli.mode == "eval":
        evaluate()
    simulation_app.close()
