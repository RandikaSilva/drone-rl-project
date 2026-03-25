"""Test low-level controller stability in the obstacle environment.

Runs the frozen LL checkpoint in the obstacle env exactly as v2 does,
but with HL control disabled (neutral modifiers) and fixed clear-path waypoints.

Usage (from IsaacLab directory):
    cd ~/projects/isaac/IsaacLab
    source ~/projects/isaac/env_isaaclab/bin/activate

    # With GUI (watch the drone):
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/test_ll_stability.py

    # Headless:
    python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/test_ll_stability.py --headless
"""
import sys
import os

# ── Rewrite sys.argv so v2's module-level parser accepts it ──────────────
# v2 expects --mode, --num_envs, etc.  We extract --num_episodes for ourselves
# and forward everything else.  v2's __main__ block won't run on import.
_num_episodes = 10
_new_argv = [sys.argv[0]]
i = 1
while i < len(sys.argv):
    if sys.argv[i] == "--num_episodes" and i + 1 < len(sys.argv):
        _num_episodes = int(sys.argv[i + 1])
        i += 2
    else:
        _new_argv.append(sys.argv[i])
        i += 1

# Inject --mode train (required by v2's parser, but __main__ won't fire)
if "--mode" not in _new_argv:
    _new_argv.extend(["--mode", "train"])
# Default to 1 env if not specified
if "--num_envs" not in _new_argv:
    _new_argv.extend(["--num_envs", "1"])

sys.argv = _new_argv

# ── Import v2 — this launches AppLauncher, starts sim, registers gym env ─
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hierarchical_obstacle_sac_v2 as v2

# Everything below runs after Isaac Sim is up
import gymnasium as gym
import torch
import numpy as np
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

# ── Fixed waypoints along clear paths ────────────────────────────────────
# Interior obstacles at (±5, ±4) r=0.5.  Center corridor (y≈0) is clear.
FIXED_WAYPOINTS = [
    (-8.0, 0.0, 1.5),   # WP1: ~2m from spawn, centerline — clear of all obstacles
    (-4.0, 0.0, 1.5),   # WP2: 4m spacing, Y=0 — no direction reversal
    ( 0.0, 0.0, 1.5),   # WP3: centerline
    ( 4.0, 0.0, 1.5),   # WP4: centerline
]


def test_ll():
    env_cfg = v2.HierarchicalObstacleNavEnvCfg()
    env_cfg.scene.num_envs = v2.args_cli.num_envs
    env_cfg.episode_length_s = 40.0
    env_cfg.debug_vis = True
    env_cfg.min_num_waypoints = len(FIXED_WAYPOINTS)
    env_cfg.max_num_waypoints = len(FIXED_WAYPOINTS)

    env = gym.make("Isaac-HierarchicalObstacle-SAC-v0", cfg=env_cfg)
    unwrapped = env.unwrapped

    # ── Monkey-patch: fixed clear-path waypoints ─────────────────────────
    _orig_gen = unwrapped._generate_waypoints

    def _fixed_waypoints(env_ids):
        _orig_gen(env_ids)  # sets up all buffers normally
        num_reset = len(env_ids)
        device = unwrapped.device
        wp = torch.tensor(FIXED_WAYPOINTS, dtype=torch.float32, device=device)
        wp = wp.unsqueeze(0).expand(num_reset, -1, -1).clone()
        origins = unwrapped._terrain.env_origins[env_ids, :2]
        wp[:, :, 0] += origins[:, 0:1]
        wp[:, :, 1] += origins[:, 1:2]
        unwrapped._waypoints_w[env_ids, :len(FIXED_WAYPOINTS)] = wp
        unwrapped._num_waypoints[env_ids] = len(FIXED_WAYPOINTS)
        unwrapped._final_goal_pos_w[env_ids] = wp[:, -1].clone()
        unwrapped._final_goal_pos_w[env_ids, 2] = 0.2

    unwrapped._generate_waypoints = _fixed_waypoints

    # ── Monkey-patch: replicate native env control path EXACTLY ─────────
    # No EMA, no speed_factor, no nav clamp — raw LL actions → PID,
    # identical to multi_waypoint_sac.py _pre_physics_step.
    _step_n = [0]

    def _native_pre_physics(actions):
        """Replicate multi_waypoint_sac.py control path exactly."""
        # Lock HL modifiers to neutral (shouldn't matter — we bypass _run_low_level_policy)
        unwrapped._hl_yaw_offset[:] = 0.0
        unwrapped._hl_speed_factor[:] = 1.0
        unwrapped._hl_alt_offset[:] = 0.0

        # Build LL obs exactly like _build_low_level_obs but with no modifier
        is_nav = (unwrapped._phase == unwrapped.NAVIGATE)
        ll_obs = unwrapped._build_low_level_obs(use_modifier_mask=is_nav)

        # ── Step-1 diagnostic: check lidar shapes and full obs ──
        if _step_n[0] == 0:
            hl_shape = unwrapped._hl_lidar.data.ray_hits_w.shape
            ll_shape = unwrapped._ll_lidar.data.ray_hits_w.shape
            print(f"\n  HL lidar ray_hits_w shape: {hl_shape}", flush=True)
            print(f"  LL lidar ray_hits_w shape: {ll_shape}", flush=True)
            print(f"  ll_obs shape: {ll_obs.shape}", flush=True)
            print(f"  ll_obs[0] = {ll_obs[0].cpu().tolist()}", flush=True)
            # Break down the obs components
            o = ll_obs[0].cpu()
            print(f"    lin_vel_b:  {o[0:3].tolist()}", flush=True)
            print(f"    ang_vel_b:  {o[3:6].tolist()}", flush=True)
            print(f"    gravity_b:  {o[6:9].tolist()}", flush=True)
            print(f"    goal_pos_b: {o[9:12].tolist()}", flush=True)
            print(f"    lidar:      {o[12:21].tolist()}", flush=True)
            print(f"    phase:      {o[21:26].tolist()}", flush=True)
            if ll_obs.shape[-1] != 26:
                print(f"  *** WARNING: ll_obs has {ll_obs.shape[-1]} dims, expected 26! ***", flush=True)
            print(flush=True)

        # Query frozen LL actor
        with torch.no_grad():
            ll_actions = unwrapped._ll_actor(ll_obs, deterministic=True)
            ll_actions = ll_actions.clamp(-1.0, 1.0)

        # Convert to velocity commands — NO EMA, NO speed_factor
        vx_cmd = ll_actions[:, 0] * unwrapped.cfg.max_velocity_xy
        vy_cmd = ll_actions[:, 1] * unwrapped.cfg.max_velocity_xy
        vz_cmd = ll_actions[:, 2] * unwrapped.cfg.max_velocity_z
        yaw_rate_cmd = ll_actions[:, 3] * unwrapped.cfg.max_yaw_rate

        # Clamp lateral speed during NAVIGATE — matches v2's _run_low_level_policy.
        # Without this, LL commands up to ±2.0 m/s → PID creates near-max tilt → rolling.
        is_nav = (unwrapped._phase == unwrapped.NAVIGATE)
        nav_max = 1.0  # m/s — same as v2
        vx_cmd = torch.where(is_nav, vx_cmd.clamp(-nav_max, nav_max), vx_cmd)
        vy_cmd = torch.where(is_nav, vy_cmd.clamp(-nav_max, nav_max), vy_cmd)

        # Store in buffers for PID
        unwrapped._ll_vx_cmd = vx_cmd
        unwrapped._ll_vy_cmd = vy_cmd
        unwrapped._ll_vz_cmd = vz_cmd
        unwrapped._ll_yaw_rate_cmd = yaw_rate_cmd

        # PID — identical to native env
        unwrapped._run_pid()

        # Diagnostics (first 150 steps of first episode)
        _step_n[0] += 1
        if _step_n[0] <= 150 and _step_n[0] % 5 == 0:
            pos = unwrapped._robot.data.root_pos_w[0]
            grav = unwrapped._robot.data.projected_gravity_b[0]
            vel = unwrapped._robot.data.root_lin_vel_b[0]
            phase = unwrapped._phase[0].item()
            thrust_val = unwrapped._thrust[0, 0, 2].item()
            weight = unwrapped._robot_weight
            pnames = ["TK", "ST", "NV", "HV", "LD"]
            print(f"    s={_step_n[0]:4d} {pnames[phase]} | "
                  f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}) | "
                  f"vel=({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}) | "
                  f"grav_z={grav[2]:.3f} | "
                  f"ll=({ll_actions[0,0]:.2f},{ll_actions[0,1]:.2f},{ll_actions[0,2]:.2f},{ll_actions[0,3]:.2f}) | "
                  f"cmd=({vx_cmd[0]:.2f},{vy_cmd[0]:.2f},{vz_cmd[0]:.2f}) | "
                  f"T={thrust_val:.2f} W={weight:.2f}", flush=True)

    unwrapped._pre_physics_step = _native_pre_physics

    # ── Run ──────────────────────────────────────────────────────────────
    env = Sb3VecEnvWrapper(env)
    obs = env.reset()
    num_envs = v2.args_cli.num_envs
    dummy_action = np.zeros((num_envs, 3), dtype=np.float32)

    total_episodes = 0
    episode_rewards, episode_lengths = [], []
    crashes = 0
    goal_reached = 0
    wp_rates = []
    phase_crashes = {p: 0 for p in ["TAKEOFF", "STABILIZE", "NAVIGATE", "HOVER", "LAND"]}
    env_rewards = np.zeros(num_envs)
    env_steps = np.zeros(num_envs, dtype=int)
    step_dt = env_cfg.sim.dt * env_cfg.decimation

    print(f"\n{'='*60}", flush=True)
    print(f"LL STABILITY TEST — obstacle env, no HL, fixed waypoints", flush=True)
    print(f"Waypoints: {FIXED_WAYPOINTS}", flush=True)
    print(f"Episodes: {_num_episodes} | Envs: {num_envs}", flush=True)
    print(f"{'='*60}\n", flush=True)

    while total_episodes < _num_episodes and v2.simulation_app.is_running():
        obs, rewards, dones, infos = env.step(dummy_action)
        rewards_np = rewards.cpu().numpy().flatten() if isinstance(rewards, torch.Tensor) else np.array(rewards).flatten()
        dones_np = dones.cpu().numpy().flatten() if isinstance(dones, torch.Tensor) else np.array(dones).flatten()
        env_rewards += rewards_np
        env_steps += 1

        for i in range(num_envs):
            if dones_np[i] and total_episodes < _num_episodes:
                total_episodes += 1
                episode_rewards.append(env_rewards[i])
                episode_lengths.append(env_steps[i] * step_dt)

                extras = env.unwrapped.extras.get("log", {})
                dist = extras.get("Metrics/final_distance_to_goal", -1)
                died = extras.get("Episode_Termination/died", 0)
                wp_rate = extras.get("Metrics/waypoint_completion_rate", 0)
                final_phase = extras.get("Metrics/final_phase_mean", -1)
                wp_rates.append(wp_rate)

                if died > 0:
                    crashes += 1
                    for pn in phase_crashes:
                        phase_crashes[pn] += extras.get(f"Episode_Termination/died_{pn}", 0)
                if dist >= 0 and dist < env_cfg.goal_threshold:
                    goal_reached += 1

                pnames = ["TAKEOFF", "STABILIZE", "NAVIGATE", "HOVER", "LAND"]
                pstr = pnames[int(final_phase)] if 0 <= final_phase < 5 else f"?{final_phase:.1f}"
                dstr = "DIED" if died > 0 else "ok"
                print(f"  Ep {total_episodes:3d} | R={env_rewards[i]:8.1f} | "
                      f"len={env_steps[i]*step_dt:5.1f}s | dist={dist:5.2f} | "
                      f"wp={wp_rate*100:4.0f}% | {pstr} | {dstr}", flush=True)

                env_rewards[i] = 0.0
                env_steps[i] = 0

    # ── Summary ──────────────────────────────────────────────────────────
    n = len(episode_rewards)
    if n == 0:
        print("No episodes completed!", flush=True)
    else:
        print(f"\n{'='*60}", flush=True)
        print(f"RESULTS — LL Stability Test ({n} episodes)", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Success Rate:        {goal_reached/n*100:6.1f}%  ({goal_reached}/{n})", flush=True)
        print(f"  Crash Rate:          {crashes/n*100:6.1f}%  ({crashes}/{n})", flush=True)
        print(f"  Crashes by Phase:", flush=True)
        for pn, c in phase_crashes.items():
            print(f"    {pn:12s}: {c:4d}", flush=True)
        print(f"  Waypoint Completion: {np.mean(wp_rates)*100:6.1f}%", flush=True)
        print(f"  Avg Reward:          {np.mean(episode_rewards):8.2f}  (+/- {np.std(episode_rewards):.2f})", flush=True)
        print(f"  Avg Episode Length:   {np.mean(episode_lengths):5.2f} s", flush=True)
        print(f"{'='*60}\n", flush=True)

    env.close()
    v2.simulation_app.close()


if __name__ == "__main__":
    test_ll()
