"""Hierarchical Crazyflie obstacle avoidance + multi-waypoint navigation with SAC.
VERSION 5 — Improved obstacle avoidance over v4.

Evaluation Results (50 episodes, 19-tree dense obstacle field, 3-5 zigzag waypoints):
    ---------------------------------------------------------------
    Metric                      v4              v5          Change
    ---------------------------------------------------------------
    Success Rate              20.0% (10/50)   74.0% (37/50)   +54%
    Crash Rate                76.0% (38/50)   18.0%  (9/50)   -58%
    Waypoint Completion       43.9%           68.0%            +24%
    Avg Final Distance         8.437 m         2.665 m        -5.8m
    Avg Reward               275.88          1157.62          +882
    Avg Episode Length         5.53 s         11.77 s         +6.2s
    ---------------------------------------------------------------

Key changes from v4:
    1. Faster EMA smoothing (alpha 0.15 -> 0.4) — quicker reaction to obstacles
    2. Wider yaw offset range (+-0.5 -> +-1.0 rad) — bigger avoidance maneuvers
    3. Wider speed factor range ([0.6,1.0] -> [0.0,1.2]) — can stop near obstacles
    4. Reduced velocity_align reward (6.0 -> 2.0) — less goal-rushing
    5. Stronger obstacle penalties, earlier trigger (1.5m -> 2.5m safe dist)
    6. Added clearance bonus — reward for maintaining distance from obstacles
    7. Increased crash penalty (-500 -> -800)
    8. Larger network (net_arch [256,128] -> [512,256]) for complex avoidance
    9. Longer training default (16M -> 24M steps)

Two-level control architecture:
    HIGH-LEVEL (trained):  LiDAR 1347 rays (3 vert ch x 449 horiz) + state -> goal modifier
    LOW-LEVEL  (deterministic): Goal -> proportional velocity commands (vx, vy, vz, yaw_rate)

Usage (from IsaacLab directory):
    cd ~/projects/isaac/IsaacLab
    source ~/projects/isaac/env_isaaclab/bin/activate

    # Train high-level:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v5.py \
        --mode train --num_envs 256 --total_timesteps 24000000 --headless

    # Play:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v5.py \
        --mode play --checkpoint /path/to/hl_sac_final.zip

    # Eval:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v5.py \
        --mode eval --checkpoint /path/to/hl_sac_final.zip --num_episodes 50
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

# ── Argument Parser ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hierarchical Crazyflie Obstacle Nav — SAC v5")
parser.add_argument("--mode", type=str, default="train", choices=["train", "play", "eval"])
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--total_timesteps", type=int, default=24000000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None, help="High-level checkpoint (play/eval/resume).")
## --ll_checkpoint removed in v3 — deterministic controller, no checkpoint needed
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--min_waypoints", type=int, default=3)
parser.add_argument("--max_waypoints", type=int, default=5)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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

# v5: Improved obstacle avoidance — faster reactions, wider action range, better rewards


# ═══════════════════════════════════════════════════════════════════════════════
# 1D CNN Feature Extractor for LiDAR + State
# ═══════════════════════════════════════════════════════════════════════════════

class LidarCNNExtractor(BaseFeaturesExtractor):
    """1D CNN for multi-channel LiDAR + MLP for state, fused together.

    Observation layout: [lidar_1347, lin_vel_b(3), ang_vel_b(3), gravity_b(3),
                         goal_pos_b(3), phase_one_hot(5), nearest_obstacle(3)]

    LiDAR is reshaped to (batch, n_channels, rays_per_channel) so each vertical
    elevation angle becomes a Conv1d input channel (like RGB in images).
    """

    def __init__(self, observation_space, features_dim=128, lidar_dim=None,
                 cnn_out_dim=32, lidar_channels=3):
        # Default: everything except the 20 state dims is LiDAR
        if lidar_dim is None:
            lidar_dim = observation_space.shape[0] - 20
        super().__init__(observation_space, features_dim)

        self.lidar_dim = lidar_dim
        self.state_dim = observation_space.shape[0] - lidar_dim
        self.lidar_channels = lidar_channels
        self.rays_per_channel = lidar_dim // lidar_channels

        # 1D CNN for LiDAR: (batch, n_ch, 449) -> (batch, cnn_out_dim)
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(lidar_channels, 16, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, cnn_out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (cnn_out_dim, 1)
            nn.Flatten(),             # -> (cnn_out_dim,)
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

        # Reshape: (batch, 1347) -> (batch, 3, 449)
        lidar = lidar_flat.view(-1, self.lidar_channels, self.rays_per_channel)

        lidar_features = self.lidar_cnn(lidar)
        state_features = self.state_mlp(state)

        return self.fusion(torch.cat([lidar_features, state_features], dim=1))


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
    """v5: Improved obstacle avoidance. Faster reactions, wider action range, better rewards."""

    episode_length_s = 40.0

    # decimation=2 matches multi_waypoint training exactly (50 Hz env.step)
    # High-level goal modifier updates every 5 env.steps internally (10 Hz)
    decimation = 2

    # High-level spaces
    # obs: lidar_1347 + lin_vel_b(3) + ang_vel_b(3) + gravity_b(3) + goal_pos_b(3)
    #      + phase_one_hot(5) + nearest_obstacle(3) = 1347 + 20 = 1367
    # 3 vertical channels × 449 horiz rays = 1347 (horizontal_res=0.8 on 0-360°)
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

    # PID controller — conservative gains for stability
    max_velocity_xy: float = 1.5   # m/s — fast enough to need visible tilt
    max_velocity_z: float = 0.8    # m/s — conservative vertical speed
    max_yaw_rate: float = 1.0      # rad/s
    pid_vel_kp: float = 0.15       # velocity error → desired tilt (lower = gentler)
    pid_att_kp: float = 5.0        # attitude error → torque
    pid_att_kd: float = 1.5        # angular velocity damping (higher = more damped)
    pid_vz_kp: float = 0.4
    pid_yaw_kp: float = 0.3
    pid_max_tilt: float = 0.4      # hard pitch/roll limit (~23°) — visible tilt from acceleration

    # Deterministic LL proportional controller gains
    ll_gain_xy: float = 1.5    # lateral: aggressive acceleration → more tilt
    ll_gain_z: float = 0.8     # vertical: velocity = gain * altitude_error (capped at max_velocity_z)
    ll_gain_yaw: float = 0.5   # yaw: rate = gain * angle_to_goal (capped at max_yaw_rate)

    # Steady-state drag tilt — drone tilts to overcome drag at constant velocity
    drag_tilt_coeff: float = 0.06  # rad per m/s (~3.4°/m/s → ~5° at 1.5 m/s)

    # Movement noise — Ornstein-Uhlenbeck random walk (smooth, correlated)
    noise_theta: float = 2.0      # mean-reversion rate (higher = faster return to zero)
    noise_vel_sigma: float = 0.005   # velocity walk intensity (m/s) — barely perceptible
    noise_att_sigma: float = 0.001   # attitude walk intensity (rad)
    noise_yaw_sigma: float = 0.001   # yaw rate walk intensity (rad/s)
    noise_vel_max: float = 0.003    # max velocity perturbation (m/s)
    noise_att_max: float = 0.002    # max attitude perturbation (rad ~0.1°)
    noise_yaw_max: float = 0.001   # max yaw rate perturbation (rad/s)

    # ── High-Level LiDAR (3 vertical channels × 449 horiz = 1347 rays) ──────
    hl_lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="base",  # lidar tilts with full vehicle pose (pitch/roll/yaw)
        pattern_cfg=patterns.LidarPatternCfg(
            channels=3,
            vertical_fov_range=(-45.0, 0.0),     # -45°, -22.5°, 0° elevation
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=0.8,                   # 449 rays per channel
        ),
        max_distance=20.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    hl_lidar_max_distance: float = 20.0
    hl_lidar_n_channels: int = 3  # must match channels above

    # LL lidar removed in v3/v4 — deterministic controller doesn't need it

    # ── Goal Modifier (high-level adjusts direction/speed toward real goal) ──
    # v5: Wider ranges for better obstacle avoidance
    max_yaw_offset: float = 1.0    # radians (~57°) — v5: wider for bigger avoidance maneuvers (was 0.5)
    speed_factor_range: tuple = (0.0, 1.2)  # v5: can stop (0.0) or go slightly faster (1.2) (was 0.6-1.0)
    max_altitude_offset: float = 0.8  # meters — allows fly-over of 5m obstacles (cruise 1.5 + 0.8 = 2.3m)
    goal_modifier_smoothing: float = 0.4  # v5: faster EMA for quicker obstacle reaction (was 0.15)

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
    static_start_pos: tuple = (-10.0, 0.0, 0.2)
    min_num_waypoints: int = 3
    max_num_waypoints: int = 5
    arena_x_range: tuple = (-8.0, 8.0)
    arena_y_range: tuple = (-6.0, 6.0)
    min_y_change: float = 2.5
    intermediate_waypoint_tolerance: float = 2.0
    final_waypoint_tolerance: float = 0.5
    waypoint_obstacle_clearance: float = 3.0

    # ── Obstacles (v4: 19 trees — denser interior) ─────────────────────────
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
        # Interior (original 4)
        (-5.0,  4.0, 0.50, 5.0),
        (-5.0, -4.0, 0.50, 5.0),
        ( 5.0,  4.0, 0.50, 5.0),
        ( 5.0, -4.0, 0.50, 5.0),
        # v4: 5 additional interior trees (varied sizes for visual richness)
        ( 0.0,  0.0, 0.60, 6.0),   # center — tall wide tree, forces splits
        (-2.5,  2.5, 0.40, 4.0),   # upper-left corridor — shorter narrow tree
        ( 2.5, -2.5, 0.40, 4.0),   # lower-right corridor — shorter narrow tree
        (-7.0,  0.0, 0.45, 4.5),   # left midfield — blocks direct left path
        ( 3.0,  3.5, 0.55, 5.5),   # upper-right interior — tall medium tree
    ]
    obstacle_collision_radius: float = 0.5
    obstacle_safe_distance: float = 2.5  # v5: earlier avoidance trigger (was 1.5)

    # ── High-Level Reward Scales ─────────────────────────────────────────────
    # NAVIGATE phase (high-level is active)
    nav_xy_progress_scale: float = 5.0       # progress toward waypoint
    nav_velocity_align_scale: float = 2.0    # v5: reduced from 6.0 — less goal-rushing
    nav_altitude_scale: float = -1.5
    obstacle_proximity_penalty_scale: float = -50.0   # v5: doubled from -25 — stronger avoidance
    lidar_obstacle_penalty_scale: float = -40.0       # v5: increased from -25
    lidar_danger_distance: float = 4.0                # v5: wider danger zone (was 3.5)
    subgoal_reachability_scale: float = 5.0           # reward for safe sub-goals
    subgoal_magnitude_penalty: float = -0.2           # v5: reduced from -0.3 — allow bigger turns
    crash_penalty: float = -800.0                     # v5: increased from -500
    intermediate_waypoint_bonus: float = 200.0        # reaching waypoints
    speed_carry_bonus_scale: float = 5.0
    waypoint_progress_scale: float = 3.0
    nav_max_speed: float = 1.5                        # max speed cap
    nav_speed_penalty_scale: float = -3.0             # speed cap penalty
    nav_lateral_penalty_scale: float = -1.0            # penalise lateral drift
    nav_stability_scale: float = 2.0                   # reward low ang-vel + upright
    # v5: NEW — reward for maintaining clearance from obstacles
    obstacle_clearance_bonus_scale: float = 3.0       # bonus for staying > safe_distance from all obstacles
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
    ang_vel_reward_scale: float = -1.0   # increased from -0.5 — penalise spinning harder
    yaw_rate_penalty_scale: float = -1.5  # increased from -1.0 — yaw spin contributes to rolling
    upright_reward_scale: float = 4.0    # increased from 2.0 — freespace sim is always upright


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

        # Low-level velocity commands
        self._ll_vx_cmd = torch.zeros(self.num_envs, device=self.device)
        self._ll_vy_cmd = torch.zeros(self.num_envs, device=self.device)
        self._ll_vz_cmd = torch.zeros(self.num_envs, device=self.device)
        self._ll_yaw_rate_cmd = torch.zeros(self.num_envs, device=self.device)

        # v3/v4: Kinematic control state
        self._tracked_yaw = torch.zeros(self.num_envs, device=self.device)
        self._desired_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        # Ornstein-Uhlenbeck random walk states (smooth, correlated noise)
        self._noise_vel = torch.zeros(self.num_envs, 3, device=self.device)   # vx, vy, vz walk
        self._noise_att = torch.zeros(self.num_envs, 2, device=self.device)   # pitch, roll walk
        self._noise_yaw = torch.zeros(self.num_envs, device=self.device)      # yaw rate walk

        # High-level update counter (updates every 5 env.steps = 10 Hz).
        # FIX v2: per-environment tensor instead of a global scalar so that
        # resetting one environment does not desynchronise all others.
        self._hl_step_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

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

        # ── v4: No checkpoint — deterministic proportional controller ─────────
        print("[INFO] v5: 19 trees (improved avoidance). Low-level: deterministic proportional controller (no checkpoint)")
        print(f"[INFO]   gains: xy={self.cfg.ll_gain_xy}, z={self.cfg.ll_gain_z}, yaw={self.cfg.ll_gain_yaw}")
        print(f"[INFO]   pid_max_tilt={self.cfg.pid_max_tilt:.2f} rad ({self.cfg.pid_max_tilt * 180 / 3.14159:.1f}°)")

        # Episode logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "takeoff_ascent", "takeoff_alt_reward", "takeoff_drift",
                "stabilize_position", "stabilize_speed", "stabilize_altitude", "stabilize_ang_vel",
                "nav_xy_progress", "nav_velocity_align", "nav_altitude",
                "nav_speed_penalty", "nav_lateral", "nav_stability",
                "obstacle_proximity", "lidar_penalty", "obstacle_clearance",
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

        # HL LiDAR only — LL is deterministic, no lidar needed
        self._hl_lidar = RayCaster(self.cfg.hl_lidar)
        self.scene.sensors["hl_lidar"] = self._hl_lidar

        # Spawn tree obstacles (v4: varied trunk/canopy colors for visual richness)
        trunk_colors = [
            (0.36, 0.25, 0.13),  # brown
            (0.30, 0.20, 0.10),  # dark brown
            (0.42, 0.30, 0.15),  # light brown
            (0.35, 0.22, 0.08),  # reddish brown
            (0.38, 0.28, 0.16),  # warm brown
        ]
        canopy_colors = [
            (0.15, 0.40, 0.12),  # green
            (0.10, 0.35, 0.08),  # dark green
            (0.20, 0.45, 0.15),  # light green
            (0.12, 0.30, 0.10),  # forest green
            (0.18, 0.42, 0.18),  # bright green
        ]
        for i, (ox, oy, radius, height) in enumerate(self.cfg.obstacle_positions):
            t_color = trunk_colors[i % len(trunk_colors)]
            c_color = canopy_colors[i % len(canopy_colors)]
            trunk_cfg = sim_utils.CylinderCfg(
                radius=radius, height=height, axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=t_color),
            )
            trunk_cfg.func(
                f"/World/envs/env_0/Tree_{i}_Trunk", trunk_cfg,
                translation=(ox, oy, height / 2.0),
            )
            canopy_height = 1.2 + (i % 3) * 0.3  # varied canopy heights: 1.2, 1.5, 1.8
            canopy_cfg = sim_utils.CylinderCfg(
                radius=radius * (1.3 + (i % 4) * 0.15),  # varied canopy spread: 1.3x to 1.75x
                height=canopy_height, axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c_color),
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

    # ── Waypoint Generation (same as obstacle_sac.py) ────────────────────────

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
        """Precompute high-level LiDAR ray directions (3ch × 449 = 1347 rays)."""
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
        """Overlay analytical ray-cylinder intersections onto ground-only LiDAR."""
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

    def _compute_goal_for_ll(self) -> torch.Tensor:
        """Compute goal vector in body frame with HL modifiers applied during NAVIGATE."""
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w

        real_goal_b, _ = subtract_frame_transforms(root_pos, root_quat, self._goal_pos_w)

        is_navigate = (self._phase == self.NAVIGATE)

        # Apply goal modifier: rotate XY direction by yaw_offset, adjust Z
        goal_xy = real_goal_b[:, :2]
        yaw_off = self._hl_yaw_offset
        cos_off = torch.cos(yaw_off)
        sin_off = torch.sin(yaw_off)
        rotated_x = cos_off * goal_xy[:, 0] - sin_off * goal_xy[:, 1]
        rotated_y = sin_off * goal_xy[:, 0] + cos_off * goal_xy[:, 1]
        modified_goal_b = torch.stack([
            rotated_x,
            rotated_y,
            real_goal_b[:, 2] + self._hl_alt_offset,
        ], dim=1)

        # Select: modified goal in NAVIGATE, real goal in other phases
        goal_for_ll = torch.where(
            is_navigate.unsqueeze(1).expand(-1, 3),
            modified_goal_b,
            real_goal_b,
        )

        # Apply speed_factor as goal distance scaling (NAVIGATE only)
        speed_scale = torch.where(
            is_navigate,
            self._hl_speed_factor,
            torch.ones_like(self._hl_speed_factor),
        )
        goal_for_ll = torch.cat([
            goal_for_ll[:, :2] * speed_scale.unsqueeze(1),
            goal_for_ll[:, 2:3],
        ], dim=1)

        return goal_for_ll

    def _run_low_level_policy(self):
        """Deterministic proportional controller — replaces frozen LL checkpoint.

        Computes velocity commands proportional to goal direction/distance,
        with phase-appropriate behavior:
            TAKEOFF:   ascend only (no lateral movement)
            STABILIZE: hold position (gentle corrections)
            NAVIGATE:  fly toward (HL-modified) goal
            HOVER:     hold position (gentle corrections)
            LAND:      descend only
        """
        goal_b = self._compute_goal_for_ll()
        phase = self._phase

        K_xy = self.cfg.ll_gain_xy
        K_z = self.cfg.ll_gain_z
        K_yaw = self.cfg.ll_gain_yaw

        # Default: proportional control toward goal (used in NAVIGATE)
        vx_cmd = K_xy * goal_b[:, 0]
        vy_cmd = K_xy * goal_b[:, 1]
        vz_cmd = K_z * goal_b[:, 2]
        yaw_to_goal = torch.atan2(goal_b[:, 1], goal_b[:, 0])
        yaw_rate_cmd = K_yaw * yaw_to_goal

        # TAKEOFF: ascend only, no lateral drift
        takeoff = (phase == self.TAKEOFF)
        if takeoff.any():
            alt_err = self.cfg.cruise_altitude - self._robot.data.root_pos_w[takeoff, 2]
            vx_cmd[takeoff] = 0.0
            vy_cmd[takeoff] = 0.0
            vz_cmd[takeoff] = K_z * alt_err
            yaw_rate_cmd[takeoff] = 0.0

        # STABILIZE / HOVER: hold position with gentle corrections
        hold = (phase == self.STABILIZE) | (phase == self.HOVER)
        if hold.any():
            vx_cmd[hold] = K_xy * goal_b[hold, 0] * 0.3
            vy_cmd[hold] = K_xy * goal_b[hold, 1] * 0.3
            vz_cmd[hold] = K_z * goal_b[hold, 2]
            yaw_rate_cmd[hold] = 0.0

        # LAND: descend, no lateral
        land = (phase == self.LAND)
        if land.any():
            vx_cmd[land] = 0.0
            vy_cmd[land] = 0.0
            vz_cmd[land] = -0.5
            yaw_rate_cmd[land] = 0.0

        # Clamp to max velocities
        self._ll_vx_cmd = vx_cmd.clamp(-self.cfg.max_velocity_xy, self.cfg.max_velocity_xy)
        self._ll_vy_cmd = vy_cmd.clamp(-self.cfg.max_velocity_xy, self.cfg.max_velocity_xy)
        self._ll_vz_cmd = vz_cmd.clamp(-self.cfg.max_velocity_z, self.cfg.max_velocity_z)
        self._ll_yaw_rate_cmd = yaw_rate_cmd.clamp(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)

    def _euler_to_quat(self, roll, pitch, yaw):
        """Convert roll, pitch, yaw (N,) tensors to quaternion (N, 4) as [w, x, y, z]."""
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
        """Advance Ornstein-Uhlenbeck random walks — smooth, correlated perturbations."""
        theta = self.cfg.noise_theta
        sqrt_dt = dt ** 0.5

        # dx = -theta * x * dt + sigma * sqrt(dt) * dW
        self._noise_vel += (-theta * self._noise_vel * dt
                            + self.cfg.noise_vel_sigma * sqrt_dt
                            * torch.randn_like(self._noise_vel))
        self._noise_vel.clamp_(-self.cfg.noise_vel_max, self.cfg.noise_vel_max)

        self._noise_att += (-theta * self._noise_att * dt
                            + self.cfg.noise_att_sigma * sqrt_dt
                            * torch.randn_like(self._noise_att))
        self._noise_att.clamp_(-self.cfg.noise_att_max, self.cfg.noise_att_max)

        self._noise_yaw += (-theta * self._noise_yaw * dt
                            + self.cfg.noise_yaw_sigma * sqrt_dt
                            * torch.randn(self.num_envs, device=self.device))
        self._noise_yaw.clamp_(-self.cfg.noise_yaw_max, self.cfg.noise_yaw_max)

    def _kinematic_update(self):
        """Directly set robot pose — no physics forces, cannot crash."""
        dt = self.sim.cfg.dt  # physics step dt (0.01s at 100 Hz)
        N = self.num_envs

        # ── Advance random walk noise (smooth, correlated) ──
        self._step_ou_noise(dt)

        # Apply velocity noise from random walk
        noisy_vx = self._ll_vx_cmd + self._noise_vel[:, 0]
        noisy_vy = self._ll_vy_cmd + self._noise_vel[:, 1]
        noisy_vz = self._ll_vz_cmd + self._noise_vel[:, 2] * 0.5  # less vertical

        # Convert body-frame velocity to world-frame using tracked yaw
        cy = torch.cos(self._tracked_yaw)
        sy = torch.sin(self._tracked_yaw)
        vel_w_x = cy * noisy_vx - sy * noisy_vy
        vel_w_y = sy * noisy_vx + cy * noisy_vy
        vel_w_z = noisy_vz
        self._desired_vel_w[:, 0] = vel_w_x
        self._desired_vel_w[:, 1] = vel_w_y
        self._desired_vel_w[:, 2] = vel_w_z

        # Integrate position
        pos = self._robot.data.root_pos_w.clone()
        pos[:, 0] += vel_w_x * dt
        pos[:, 1] += vel_w_y * dt
        pos[:, 2] += vel_w_z * dt
        pos[:, 2].clamp_(min=0.05, max=self.cfg.max_flight_height)

        # Update tracked yaw (with random walk noise)
        self._tracked_yaw += (self._ll_yaw_rate_cmd + self._noise_yaw) * dt

        # Tilt proportional to acceleration (like a real quadrotor)
        g = 9.81
        max_tilt = self.cfg.pid_max_tilt

        # World-frame acceleration
        accel_w = (self._desired_vel_w - self._prev_vel_w) / dt
        self._prev_vel_w = self._desired_vel_w.clone()

        # Rotate world acceleration into body frame (using yaw)
        accel_body_x = cy * accel_w[:, 0] + sy * accel_w[:, 1]
        accel_body_y = -sy * accel_w[:, 0] + cy * accel_w[:, 1]

        # Body-frame velocity for drag tilt
        vel_body_x = cy * vel_w_x + sy * vel_w_y
        vel_body_y = -sy * vel_w_x + cy * vel_w_y

        # pitch/roll = acceleration tilt + drag tilt (steady-state) + noise
        # Drag: drone holds ~5° tilt at cruise to overcome drag
        # Accel: extra tilt during speed changes (braking tilts harder)
        # Signs: tilt TOWARD movement direction (nose down when moving forward)
        k_drag = self.cfg.drag_tilt_coeff
        pitch = (accel_body_x / g + vel_body_x * k_drag + self._noise_att[:, 0]).clamp(-max_tilt, max_tilt)
        roll = (-accel_body_y / g - vel_body_y * k_drag + self._noise_att[:, 1]).clamp(-max_tilt, max_tilt)
        quat = self._euler_to_quat(roll, pitch, self._tracked_yaw)

        # Write pose and velocity to sim
        root_pose = torch.cat([pos, quat], dim=-1)
        vel_state = torch.zeros(N, 6, device=self.device)
        vel_state[:, 0] = vel_w_x
        vel_state[:, 1] = vel_w_y
        vel_state[:, 2] = vel_w_z
        vel_state[:, 5] = self._ll_yaw_rate_cmd
        self._robot.write_root_pose_to_sim(root_pose)
        self._robot.write_root_velocity_to_sim(vel_state)

    # ── Physics Step (dual-rate control) ─────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        """Called once per env.step() at 50 Hz (decimation=2, matching multi_waypoint).

        High-level goal modifier updates every 5 env.steps (10 Hz).
        Low-level policy runs every env.step (50 Hz) — same as multi_waypoint training.
        PID computed here (same pattern as multi_waypoint).
        """
        # Update high-level goal modifier every 5 steps per environment (10 Hz).
        # FIX v2: per-env counter so a reset in one env does not disrupt others.
        self._hl_step_counter += 1
        update_mask = self._hl_step_counter >= 5
        if update_mask.any():
            self._hl_step_counter[update_mask] = 0
            a = actions.clone().clamp(-1.0, 1.0)
            lo, hi = self.cfg.speed_factor_range
            # Compute raw target values
            target_yaw = a[:, 0] * self.cfg.max_yaw_offset
            target_spd = lo + (a[:, 1] + 1.0) * 0.5 * (hi - lo)
            target_alt = a[:, 2] * self.cfg.max_altitude_offset
            # EMA smoothing: blend toward target to avoid sudden LL input jumps
            alpha = self.cfg.goal_modifier_smoothing
            self._hl_yaw_offset[update_mask] = ((1 - alpha) * self._hl_yaw_offset + alpha * target_yaw)[update_mask]
            self._hl_speed_factor[update_mask] = ((1 - alpha) * self._hl_speed_factor + alpha * target_spd)[update_mask]
            self._hl_alt_offset[update_mask] = ((1 - alpha) * self._hl_alt_offset + alpha * target_alt)[update_mask]

        # Run deterministic LL controller at 50 Hz
        self._run_low_level_policy()

    def _apply_action(self):
        """Kinematic update — directly set robot pose each physics step.
        No forces, no torques, no PID. Cannot crash."""
        self._kinematic_update()
        # Apply hover thrust to counteract gravity (keeps physics engine happy)
        self._thrust[:, 0, 2] = self._robot_weight
        self._moment[:] = 0
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )

    # ── Observations (high-level) ────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """High-level obs: N LiDAR rays + state (20) dims."""
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w

        goal_pos_b, _ = subtract_frame_transforms(root_pos, root_quat, self._goal_pos_w)

        # High-level LiDAR (450 rays)
        hl_lidar_hits = self._hl_lidar.data.ray_hits_w
        hl_lidar_pos = self._hl_lidar.data.pos_w.unsqueeze(1)
        hl_lidar_dists = torch.linalg.norm(hl_lidar_hits - hl_lidar_pos, dim=-1)
        # Augment with analytical obstacle intersections
        hl_lidar_dists = self._augment_hl_lidar_with_obstacles(hl_lidar_dists, root_pos, root_quat)
        hl_lidar_norm = (hl_lidar_dists / self.cfg.hl_lidar_max_distance).clamp(0.0, 1.0)

        # Store for debug vis and reward
        self._last_hl_lidar_distances = hl_lidar_dists
        if self._hl_lidar_horiz_indices is not None and len(self._hl_lidar_horiz_indices) > 0:
            horiz_dists = hl_lidar_dists[:, self._hl_lidar_horiz_indices]
            self._min_horiz_lidar_dist = horiz_dists.min(dim=1).values

        # Phase one-hot
        phase_one_hot = torch.nn.functional.one_hot(self._phase.long(), 5).float()

        # Nearest obstacle in body frame (3 dims: direction_xy(2) + normalized_dist(1))
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

        # Store for debug
        nearest_obs_xy = self._obstacle_pos_w[batch_idx, min_idx]
        self._nearest_obs_pos_w[:, 0] = nearest_obs_xy[:, 0]
        self._nearest_obs_pos_w[:, 1] = nearest_obs_xy[:, 1]
        self._nearest_obs_pos_w[:, 2] = root_pos[:, 2]
        self._nearest_obs_dist = nearest_dist

        # Layout: [lidar_1347, lin_vel_b, ang_vel_b, gravity_b, goal_pos_b, phase, nearest_obstacle]
        obs = torch.cat([
            hl_lidar_norm,                          # 1347 (3ch × 449)
            self._robot.data.root_lin_vel_b,        # 3
            self._robot.data.root_ang_vel_b,        # 3
            self._robot.data.projected_gravity_b,    # 3
            goal_pos_b,                              # 3
            phase_one_hot,                           # 5
            nearest_body_dir,                        # 2
            nearest_dist_normalized,                 # 1
        ], dim=-1)  # Total: 470
        return {"policy": obs}

    # ── Rewards ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self._robot.data.root_pos_w
        dt = self.step_dt  # 0.02s at 50 Hz (decimation=2)

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
        takeoff_ascent = torch.where(is_takeoff, alt_progress * self.cfg.takeoff_ascent_scale, torch.zeros_like(alt_progress))
        alt_fraction = (altitude / self.cfg.cruise_altitude).clamp(0.0, 1.0)
        takeoff_alt_reward = torch.where(is_takeoff, alt_fraction * self.cfg.takeoff_altitude_reward_scale * dt, torch.zeros_like(altitude))
        takeoff_drift = torch.where(is_takeoff, horizontal_speed * self.cfg.takeoff_drift_penalty * dt, torch.zeros_like(horizontal_speed))

        # ── Phase 1: STABILIZE ───────────────────────────────────────────────
        stabilize_position = torch.where(is_stabilize, torch.exp(-start_xy_dist) * self.cfg.stabilize_position_scale * dt, torch.zeros_like(start_xy_dist))
        stabilize_speed = torch.where(is_stabilize, torch.exp(-speed) * self.cfg.stabilize_low_speed_scale * dt, torch.zeros_like(speed))
        stabilize_altitude = torch.where(is_stabilize, alt_error * self.cfg.stabilize_altitude_scale * dt, torch.zeros_like(alt_error))
        stabilize_ang_vel = torch.where(is_stabilize, torch.exp(-ang_vel_magnitude) * self.cfg.stabilize_ang_vel_scale * dt, torch.zeros_like(ang_vel_magnitude))

        # ── Phase 2: NAVIGATE (high-level is active) ─────────────────────────
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

        # v5: Clearance bonus — reward for staying far from all obstacles
        is_clear = (min_obstacle_dist > self.cfg.obstacle_safe_distance).float()
        obstacle_clearance = torch.where(is_navigate, is_clear * self.cfg.obstacle_clearance_bonus_scale * dt, torch.zeros_like(is_clear))

        # LiDAR-based obstacle penalty
        lidar_dist = self._min_horiz_lidar_dist
        lidar_danger = (lidar_dist < self.cfg.lidar_danger_distance).float()
        lidar_closeness = (1.0 - lidar_dist / self.cfg.lidar_danger_distance).clamp(min=0.0)
        lidar_penalty = torch.where(is_navigate, lidar_danger * lidar_closeness * self.cfg.lidar_obstacle_penalty_scale * dt, torch.zeros_like(lidar_dist))

        # Goal-modifier quality rewards
        # Penalize large yaw deviations (prefer going straight when safe)
        yaw_deviation = torch.abs(self._hl_yaw_offset)
        subgoal_magnitude = torch.where(is_navigate, yaw_deviation * self.cfg.subgoal_magnitude_penalty * dt, torch.zeros_like(yaw_deviation))

        # Reward for keeping LiDAR clear (good avoidance direction chosen)
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

        # ── Phase 3: HOVER ───────────────────────────────────────────────────
        hover_position = torch.where(is_hover, torch.exp(-xy_dist) * self.cfg.hover_position_scale * dt, torch.zeros_like(xy_dist))
        hover_speed = torch.where(is_hover, torch.exp(-speed) * self.cfg.hover_low_speed_scale * dt, torch.zeros_like(speed))
        hover_altitude = torch.where(is_hover, alt_error * self.cfg.hover_altitude_scale * dt, torch.zeros_like(alt_error))

        # ── Phase 4: LAND ────────────────────────────────────────────────────
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

        # Crash penalty: detect if drone hit obstacle or is about to die
        drone_xy_r = root_pos[:, :2].unsqueeze(1)
        crash_obs_dists = torch.linalg.norm(drone_xy_r - self._obstacle_pos_w, dim=2)
        crash_surface_dists = crash_obs_dists - self._obstacle_radii.unsqueeze(0)
        min_crash_dist, min_crash_idx = crash_surface_dists.min(dim=1)
        crash_nearest_h = self._obstacle_heights[min_crash_idx]
        is_crashing = (min_crash_dist < self.cfg.obstacle_collision_radius) & (root_pos[:, 2] < crash_nearest_h)
        crash_penalty = is_crashing.float() * self.cfg.crash_penalty

        # ── All-phase rewards ────────────────────────────────────────────────
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

        # ── Total ────────────────────────────────────────────────────────────
        reward = (
            takeoff_ascent + takeoff_alt_reward + takeoff_drift
            + stabilize_position + stabilize_speed + stabilize_altitude + stabilize_ang_vel
            + nav_xy_progress + nav_velocity_align + nav_altitude + nav_speed_penalty
            + nav_lateral_penalty + nav_stability
            + obstacle_proximity + obstacle_clearance + lidar_penalty + subgoal_reachability + subgoal_magnitude
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

        # Store crash causes for per-episode logging
        self._crash_too_low = too_low
        self._crash_too_high = too_high
        self._crash_flipped = flipped
        self._crash_hit_obstacle = hit_obstacle

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

        # Per-cause crash counts
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

        # Reset goal modifier and step counter.
        # FIX v2: reset only the envs that are actually being reset.
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
        self._prev_vel_w[env_ids] = 0.0
        self._noise_vel[env_ids] = 0.0
        self._noise_att[env_ids] = 0.0
        self._noise_yaw[env_ids] = 0.0

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

            # Sub-goal marker (small yellow sphere)
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
        """Show current target (large cube) and all waypoints (small cubes).
        Matches multi_waypoint_sac exactly — static waypoints, one moving target."""
        # Current target — advances when an intermediate waypoint is reached
        self.goal_pos_visualizer.visualize(self._goal_pos_w)

        # All waypoints — static throughout the episode
        for wp_idx in range(self.cfg.max_num_waypoints):
            wp_positions = self._waypoints_w[:, wp_idx, :]
            valid = wp_idx < self._num_waypoints
            viz_pos = torch.where(
                valid.unsqueeze(1).expand(-1, 3),
                wp_positions,
                torch.full_like(wp_positions, -1000.0),
            )
            self.waypoint_visualizers[wp_idx].visualize(viz_pos)

        # Hide subgoal — internal HL modifier, not needed for viewing
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

        # Only draw every 5th ray to reduce overhead with 450 rays
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
# Gym Registration
# ═══════════════════════════════════════════════════════════════════════════════

gym.register(
    id="Isaac-HierarchicalObstacle-SAC-v5",
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

def train():
    env_cfg = HierarchicalObstacleNavEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    log_root = os.path.join(SCRIPT_DIR, "logs", "hierarchical_sac_v5", "crazyflie_hierarchical_obstacle_sac_v5")
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Logging to: {log_dir}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Total timesteps: {args_cli.total_timesteps}")
    print(f"[INFO] Low-level: deterministic proportional controller (no checkpoint)")
    print(f"[INFO] Waypoints: {env_cfg.min_num_waypoints}-{env_cfg.max_num_waypoints}")
    print(f"[INFO] Obstacles: {len(env_cfg.obstacle_positions)} trees")
    print(f"[INFO] High-level: 10 Hz, {env_cfg.observation_space - 20}-ray LiDAR ({env_cfg.hl_lidar_n_channels}ch), action=goal modifier")
    print(f"[INFO] Low-level: 50 Hz (deterministic), PID: 100 Hz")

    env = gym.make("Isaac-HierarchicalObstacle-SAC-v5", cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    if args_cli.checkpoint:
        print(f"[INFO] Resuming high-level from: {args_cli.checkpoint}")
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
                net_arch=[512, 256],  # v5: larger network for complex avoidance (was [256, 128])
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
    print(f"[INFO] High-level model saved to: {final_path}.zip")
    print(f"[INFO] Logs saved to: {log_dir}")
    env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Play
# ═══════════════════════════════════════════════════════════════════════════════

def play():
    env_cfg = HierarchicalObstacleNavEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 40.0
    env_cfg.debug_vis = True
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    env = gym.make("Isaac-HierarchicalObstacle-SAC-v5", cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    # Load HL agent if checkpoint provided, otherwise use zero actions (neutral modifiers)
    agent = None
    if args_cli.checkpoint is not None:
        agent = SAC.load(args_cli.checkpoint, env=env, device="cuda:0")
    obs = env.reset()

    cameras = setup_cameras(env.unwrapped)

    episode_count = 0
    total_reward = 0.0
    step_count = 0
    playback_dt = env_cfg.sim.dt * env_cfg.decimation * 3.0  # 3x slower

    print(f"\n{'='*60}")
    print(f"[INFO] v5: Improved avoidance (19 trees) + deterministic proportional LL controller")
    if agent:
        print(f"[INFO] High-level: trained SAC policy")
    else:
        print(f"[INFO] High-level: NONE (zero actions = neutral goal modifiers)")
    print(f"[INFO] Low-level: proportional controller (gains: xy={env_cfg.ll_gain_xy}, z={env_cfg.ll_gain_z}, yaw={env_cfg.ll_gain_yaw})")
    print(f"[INFO] PID max tilt: {env_cfg.pid_max_tilt:.2f} rad ({env_cfg.pid_max_tilt * 180 / 3.14159:.1f}°)")
    print(f"[INFO] Press Ctrl+C to stop.")
    print(f"{'='*60}\n")

    # Zero HL actions (3-dim: yaw_offset, speed_factor, alt_offset)
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
                # Noise state for env 0
                nv = unwrapped._noise_vel[0].cpu()
                na = unwrapped._noise_att[0].cpu()
                ny = unwrapped._noise_yaw[0].item()
                # Actual pitch/roll from robot quat (for verification)
                rq = unwrapped._robot.data.root_quat_w[0].cpu()
                # quat [w,x,y,z] -> approximate pitch/roll
                pitch_approx = math.asin(max(-1, min(1, 2*(rq[0]*rq[2] - rq[3]*rq[1]).item())))
                roll_approx = math.atan2(2*(rq[0]*rq[1] + rq[2]*rq[3]).item(),
                                         1 - 2*(rq[1]**2 + rq[2]**2).item())
                print(f"  Step {step_count} | {phase_names[phase]} | WP {wp_idx}/{num_wps} | "
                      f"Pos ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}) | "
                      f"P={math.degrees(pitch_approx):+.1f}° R={math.degrees(roll_approx):+.1f}° | "
                      f"nV=({nv[0]:.3f},{nv[1]:.3f},{nv[2]:.3f}) nA=({na[0]:.3f},{na[1]:.3f}) nY={ny:.3f} | "
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


# ═══════════════════════════════════════════════════════════════════════════════
# Eval
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate():
    assert args_cli.checkpoint is not None, "Must provide --checkpoint for eval mode."

    num_eval_envs = min(args_cli.num_envs, 16) if args_cli.num_envs == 256 else args_cli.num_envs

    env_cfg = HierarchicalObstacleNavEnvCfg()
    env_cfg.scene.num_envs = num_eval_envs
    env_cfg.episode_length_s = 40.0
    env_cfg.debug_vis = False
    env_cfg.min_num_waypoints = args_cli.min_waypoints
    env_cfg.max_num_waypoints = args_cli.max_waypoints

    env = gym.make("Isaac-HierarchicalObstacle-SAC-v5", cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)
    agent = SAC.load(args_cli.checkpoint, env=env, device="cuda:0")
    obs = env.reset()

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
    env_rewards = np.zeros(num_eval_envs)
    env_steps = np.zeros(num_eval_envs, dtype=int)
    step_dt = env_cfg.sim.dt * env_cfg.decimation

    print(f"\n{'='*60}")
    print(f"EVALUATION: Hierarchical SAC | {args_cli.num_episodes} episodes")
    print(f"HL Checkpoint: {os.path.basename(args_cli.checkpoint)}")
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
                    dist = extras.get("Metrics/final_distance_to_goal", -1)
                    died_count = extras.get("Episode_Termination/died", 0)
                    wp_rate = extras.get("Metrics/waypoint_completion_rate", 0)
                    final_distances.append(dist)
                    wp_completion_rates.append(wp_rate)
                    if died_count > 0:
                        crashes += 1
                        for pname in phase_crashes:
                            phase_crashes[pname] += extras.get(f"Episode_Termination/died_{pname}", 0)
                        for cname in cause_crashes:
                            cause_crashes[cname] += extras.get(f"Episode_Termination/cause_{cname}", 0)
                        ep_phase = next((p for p in phase_crashes if extras.get(f"Episode_Termination/died_{p}", 0) > 0), None)
                        ep_cause = next((c for c in cause_crashes if extras.get(f"Episode_Termination/cause_{c}", 0) > 0), None)
                        if ep_phase and ep_cause:
                            crash_details.append((ep_phase, ep_cause))
                    if dist >= 0 and dist < env_cfg.goal_threshold:
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
        print(f"RESULTS -- Hierarchical SAC ({n} episodes)", flush=True)
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
                print(f"    - {phase} → {cause}", flush=True)
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
