"""Debug: v2's exact environment with DUMMY HL actions + FIXED waypoints.

Uses v2's identical environment (same scene, LL, PID, timing, obstacles).
Instead of a trained HL, passes zero actions (= pass-through to frozen LL).
Monkey-patches _generate_waypoints so waypoints are ALWAYS fixed — even
when _reset_idx fires internally during env.step().

This isolates: can the frozen LL + PID fly a simple straight path in v2?

Usage:
    cd ~/projects/isaac/IsaacLab
    source ~/projects/isaac/env_isaaclab/bin/activate

    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/debug_frozen_ll.py
"""
import sys
import os
import argparse

# ── Parse our args BEFORE v2's parser runs ──────────────────────────────────
_debug_parser = argparse.ArgumentParser(add_help=False)
_debug_parser.add_argument("--num_envs", type=int, default=1)
_debug_args, _remaining = _debug_parser.parse_known_args()

# Set sys.argv so v2's argparse sees valid args
sys.argv = [
    sys.argv[0],
    "--mode", "play",
    "--num_envs", str(_debug_args.num_envs),
] + _remaining

# ── Import v2 (triggers its argparse + AppLauncher + gym.register) ──────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import hierarchical_obstacle_sac_v2 as v2  # noqa: E402

import gymnasium as gym  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402


# ── Fixed waypoints (local coords, before env_origin offset) ────────────────
# Drone starts at (-7, 0, 0.2).  Three waypoints in a straight line:
FIXED_WAYPOINTS = [
    (-6.0, 0.0, 1.5),   # WP0: 1 m ahead  (matches multi_waypoint training distribution)
    (-3.0, 0.0, 1.5),   # WP1: 4 m ahead
    ( 0.0, 0.0, 1.5),   # WP2: 7 m ahead
]
NUM_FIXED = len(FIXED_WAYPOINTS)


def _fixed_generate_waypoints(self, env_ids):
    """Drop-in replacement: always generates the same fixed waypoints."""
    num_reset = len(env_ids)
    device = self.device
    max_wps = self.cfg.max_num_waypoints

    # Build fixed waypoints tensor
    wp_local = torch.zeros(num_reset, max_wps, 3, device=device)
    for wi in range(max_wps):
        src = FIXED_WAYPOINTS[min(wi, NUM_FIXED - 1)]
        wp_local[:, wi, 0] = src[0]
        wp_local[:, wi, 1] = src[1]
        wp_local[:, wi, 2] = src[2]

    self._num_waypoints[env_ids] = NUM_FIXED

    # Offset to world coords
    env_origins_xy = self._terrain.env_origins[env_ids, :2]
    waypoints_w = wp_local.clone()
    waypoints_w[:, :, 0] += env_origins_xy[:, 0:1]
    waypoints_w[:, :, 1] += env_origins_xy[:, 1:2]
    self._waypoints_w[env_ids] = waypoints_w

    # Final goal = landing spot under last waypoint
    batch_idx = torch.arange(num_reset, device=device)
    self._final_goal_pos_w[env_ids] = waypoints_w[batch_idx, NUM_FIXED - 1].clone()
    self._final_goal_pos_w[env_ids, 2] = 0.2


def main():
    env_cfg = v2.HierarchicalObstacleNavEnvCfg()
    env_cfg.scene.num_envs = _debug_args.num_envs
    env_cfg.debug_vis = True
    env_cfg.min_num_waypoints = NUM_FIXED
    env_cfg.max_num_waypoints = max(NUM_FIXED, 5)

    env = gym.make("Isaac-HierarchicalObstacle-SAC-v0", cfg=env_cfg)

    # ── Monkey-patch: fixed waypoints on EVERY reset ────────────────────────
    raw_env = env.unwrapped
    raw_env._generate_waypoints = types.MethodType(_fixed_generate_waypoints, raw_env)

    env = Sb3VecEnvWrapper(env)
    obs = env.reset()   # first reset now uses fixed waypoints

    # Also prevent the episode_length_buf randomization from causing instant resets
    raw_env.episode_length_buf[:] = 0

    # Setup cameras
    cameras = None
    try:
        cameras = v2.setup_cameras(raw_env)
    except Exception:
        pass

    # Dummy HL actions: all zeros → pass real waypoint to LL unmodified
    dummy_actions = np.zeros((_debug_args.num_envs, 3), dtype=np.float32)

    step_count = 0
    episode_count = 0
    phase_names = ["TAKEOFF", "STABILIZE", "NAVIGATE", "HOVER", "LAND"]

    wp_str = "  ".join(f"WP{i}: ({w[0]}, {w[1]}, {w[2]})" for i, w in enumerate(FIXED_WAYPOINTS))
    print(f"\n{'='*70}")
    print(f"  DEBUG: v2 env, FIXED waypoints, DUMMY HL (pass-through)")
    print(f"  Start: (-7, 0, 0.2)")
    print(f"  {wp_str}")
    print(f"  HL actions: [0,0,0] → no yaw/speed/alt modification")
    print(f"  Obstacles: {len(env_cfg.obstacle_positions)}")
    print(f"  Ctrl+C to stop")
    print(f"{'='*70}\n")

    header = (
        f"{'Step':>5} | {'Phase':>10} | {'WP':>4} | "
        f"{'PosX':>6} {'PosY':>6} {'PosZ':>6} | "
        f"{'GoalX':>6} {'GoalY':>6} {'GoalZ':>5} | {'Dist':>5} | "
        f"{'vx':>6} {'vy':>6} {'vz':>6} {'yr':>6} | "
        f"{'Thrust':>7} {'RollT':>6} {'PitchT':>6}"
    )
    print(header)
    print("-" * len(header))

    playback_dt = env_cfg.sim.dt * env_cfg.decimation * 2.0

    try:
        while v2.simulation_app.is_running():
            step_start = time.time()
            obs, rewards, dones, infos = env.step(dummy_actions)
            step_count += 1

            u = env.unwrapped
            pos = u._robot.data.root_pos_w[0].cpu().numpy()

            if cameras is not None:
                try:
                    v2.update_cameras(cameras, pos)
                except Exception:
                    cameras = None

            if step_count % 10 == 0:
                goal = u._goal_pos_w[0].cpu().numpy()
                phase = u._phase[0].item()
                wp_idx = u._current_wp_idx[0].item()
                num_wps = u._num_waypoints[0].item()
                vx = u._ll_vx_cmd[0].item()
                vy = u._ll_vy_cmd[0].item()
                vz = u._ll_vz_cmd[0].item()
                yr = u._ll_yaw_rate_cmd[0].item()
                thrust = u._thrust[0, 0, 2].item()
                roll_t = u._moment[0, 0, 0].item()
                pitch_t = u._moment[0, 0, 1].item()
                dist = np.linalg.norm(goal[:2] - pos[:2])

                print(
                    f"{step_count:5d} | {phase_names[phase]:>10} | {wp_idx}/{num_wps:>1} | "
                    f"{pos[0]:6.2f} {pos[1]:6.2f} {pos[2]:6.2f} | "
                    f"{goal[0]:6.2f} {goal[1]:6.2f} {goal[2]:5.1f} | {dist:5.2f} | "
                    f"{vx:+6.2f} {vy:+6.2f} {vz:+6.2f} {yr:+6.2f} | "
                    f"{thrust:7.2f} {roll_t:+6.3f} {pitch_t:+6.3f}"
                )

                # Print LL observation breakdown every 50 steps
                if step_count % 50 == 0 and hasattr(u, '_last_ll_obs'):
                    o = u._last_ll_obs[0].cpu().numpy()
                    print(f"        LL_OBS: lin_vel_b=[{o[0]:+.2f},{o[1]:+.2f},{o[2]:+.2f}] "
                          f"ang_vel_b=[{o[3]:+.2f},{o[4]:+.2f},{o[5]:+.2f}] "
                          f"grav_b=[{o[6]:+.2f},{o[7]:+.2f},{o[8]:+.2f}]")
                    print(f"        LL_OBS: goal_b=[{o[9]:+.2f},{o[10]:+.2f},{o[11]:+.2f}] "
                          f"lidar=[{o[12]:.2f},{o[13]:.2f},{o[14]:.2f},{o[15]:.2f},{o[16]:.2f},"
                          f"{o[17]:.2f},{o[18]:.2f},{o[19]:.2f},{o[20]:.2f}] "
                          f"phase=[{o[21]:.0f},{o[22]:.0f},{o[23]:.0f},{o[24]:.0f},{o[25]:.0f}]")

            if dones.any():
                episode_count += 1
                final_phase = u._phase[0].item()
                final_alt = pos[2]
                reason = "DIED" if u.reset_terminated[0].item() else "TIMEOUT/GOAL"
                print(
                    f"\n  === Episode {episode_count}: {reason} "
                    f"phase={phase_names[final_phase]}, alt={final_alt:.2f} ===\n"
                )
                # episode_length_buf randomization already happened in _reset_idx;
                # reset it to avoid instant timeouts with 1 env
                u.episode_length_buf[:] = 0
                print(header)
                print("-" * len(header))

            elapsed = time.time() - step_start
            if (playback_dt - elapsed) > 0:
                time.sleep(playback_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

    env.close()


if __name__ == "__main__":
    main()
    v2.simulation_app.close()
