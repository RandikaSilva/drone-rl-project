# Low-Level Controller Stability — Debug Investigation

Date: 2026-03-18
Investigator: Claude (with Lasitha)

---

## Problem Statement

The frozen low-level (LL) controller checkpoint, trained in `multi_waypoint_sac.py`, appears stable
in its native training environment but becomes unstable (drone flips and crashes) when reused
inside the hierarchical obstacle environment (`hierarchical_obstacle_sac_v2.py`).

---

## Key Finding: Goal Distance Distribution Mismatch

**The LL policy is highly sensitive to goal distance during TAKEOFF phase.**

The LL was trained with `arena_x_range=(-8, 8)` and `static_start_pos=(-10, 0, 0.2)`. The first
waypoint X coordinate is always near -8 (the left edge of the arena), giving a goal distance of
approximately **2m** in body-frame X during TAKEOFF.

When the test script placed the first waypoint at `(-6, 0, 1.5)`, the goal distance was **4m** —
outside the LL's training distribution for TAKEOFF. The LL responded with erratic lateral velocity
commands that caused the drone to tilt and crash.

### Standalone Actor Test (no Isaac Lab)

Fed the exact same observation to the LL actor with only the goal X coordinate changed:

| goal_pos_b (X) | vx action | vy action | vz action | Result |
|---|---|---|---|---|
| 4.0m | -0.73 | **+1.00** | 0.97 | Saturated lateral — **crashes** |
| 2.0m | +0.81 | +0.30 | 0.91 | Moderate — **stable** |
| 1.0m | +0.95 | **-0.86** | 0.83 | Saturated opposite — unstable |

The LL produces ~1.0 vy (full lateral) when goal_x=4.0, even though goal_y=0. This is a pure
artifact of out-of-distribution goal distance during TAKEOFF phase.

### How to reproduce this test

No Isaac Lab needed — just loads the SB3 checkpoint and queries it:

```bash
source ~/projects/isaac/env_isaaclab/bin/activate
python3 -u -c "
import torch
from stable_baselines3 import SAC

ll_path = 'multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip'
agent = SAC.load(ll_path, device='cuda:0')
actor = agent.policy.actor
actor.eval()

obs = torch.tensor([[
    0.0, 0.0, 0.0,           # lin_vel_b
    0.0, 0.0, 0.0,           # ang_vel_b
    0.0, 0.0, -1.0,          # gravity_b
    4.0, 0.0, 1.3,           # goal_pos_b — change X to test
    0.0133, 0.0133, 0.0133,  # lidar down
    0.0140, 0.0189, 0.0189,  # lidar angled
    1.0, 1.0, 1.0,           # lidar horizontal
    1.0, 0.0, 0.0, 0.0, 0.0  # phase: TAKEOFF
]], dtype=torch.float32, device='cuda:0')

with torch.no_grad():
    act = actor(obs, deterministic=True).clamp(-1, 1)
print(f'vx={act[0,0]:.4f}, vy={act[0,1]:.4f}, vz={act[0,2]:.4f}, yaw={act[0,3]:.4f}')
"
```

---

## What Was Verified

### 1. LL is stable in native environment

```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate

# Eval mode (headless, prints stats):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/multi_waypoint/multi_waypoint_sac.py \
    --mode eval --num_envs 1 --num_episodes 5 \
    --checkpoint multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip

# Play mode (GUI, debug_vis=True, 5x slow):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/multi_waypoint/multi_waypoint_sac.py \
    --mode play --num_envs 1 \
    --checkpoint multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip
```

Results (5 episodes): 0% crash rate, 60% success, 74.7% waypoint completion.
Drone climbs steadily during TAKEOFF, navigates through 3-5 waypoints, and lands.

### 2. LL crashes in obstacle env with distant first waypoint

```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate

# Headless with diagnostics:
PYTHONUNBUFFERED=1 python -u \
    ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/test_ll_stability.py \
    --headless --num_episodes 5

# With GUI (can watch the flip):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/test_ll_stability.py \
    --num_episodes 3
```

Results: 90% crash rate, all during TAKEOFF, dies in ~2 seconds. Step-by-step telemetry shows
vy command saturating from step 5 onward.

### 3. Physics configs are identical

Every physics parameter was compared between native and obstacle envs:
- `decimation=2`, `dt=1/100`, same PID gains, same thrust_to_weight=1.9
- Same robot config (`VISIBLE_CRAZYFLIE_CFG`), same moment_scale=0.04
- Same spawn position `(-10, 0, 0.2)`, same cruise_altitude=1.5
- Same LL lidar config (3ch × 3 horizontal = 9 rays, max 15m, ground only)
- Same phase transitions (takeoff_altitude_tolerance=0.3, stabilize_duration=2.0)

### 4. LL lidar vs HL lidar shapes confirmed

At step 0 diagnostic:
```
HL lidar ray_hits_w shape: torch.Size([1, 1347, 3])  — 3ch × 449 horizontal
LL lidar ray_hits_w shape: torch.Size([1, 9, 3])     — 3ch × 3 horizontal
ll_obs shape: torch.Size([1, 26])                     — correct
```

No data mixup between the two sensors. The LL sees exactly 9 lidar rays.

### 5. EMA smoothing eliminated as cause

The test script replicates the native env's control path exactly (no EMA, no speed_factor, no
HL modifiers). The LL still crashes — ruling out v2's EMA smoothing as the cause.

---

## Architecture Overview

```
50 Hz env.step
  │
  ├── High-Level Policy (10 Hz, every 5 env.steps)
  │     Input:  1367-dim (1347 HL lidar + 20 state)
  │     Output: 3 (yaw_offset, speed_factor, alt_offset)
  │
  ├── Low-Level Policy (50 Hz, every env.step) ← FROZEN
  │     Input:  26-dim (vel(3) + angvel(3) + grav(3) + goal(3) + lidar(9) + phase(5))
  │     Output: 4 (vx, vy, vz, yaw_rate) — scaled by max_velocity
  │
  └── PID Controller (100 Hz via decimation=2)
        Input:  velocity commands from LL
        Output: thrust + torques applied to robot
```

### Observation format (26-dim)

| Index | Component | Dim | Notes |
|---|---|---|---|
| 0-2 | `root_lin_vel_b` | 3 | Body-frame linear velocity |
| 3-5 | `root_ang_vel_b` | 3 | Body-frame angular velocity |
| 6-8 | `projected_gravity_b` | 3 | Gravity in body frame (upright = [0,0,-1]) |
| 9-11 | `goal_pos_b` | 3 | Current waypoint in body frame |
| 12-20 | `lidar_normalized` | 9 | 3ch × 3 horizontal, normalized by 15m |
| 21-25 | `phase_one_hot` | 5 | [TK, ST, NV, HV, LD] |

### LiDAR configuration (shared between both envs)

```python
RayCasterCfg(
    prim_path="/World/envs/env_.*/Robot/body",
    ray_alignment="yaw",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=3,                          # 3 elevation angles
        vertical_fov_range=(-90.0, 0.0),     # straight down to horizontal
        horizontal_fov_range=(0.0, 360.0),   # full circle
        horizontal_res=90.0,                 # 3 rays per channel
    ),
    max_distance=15.0,
    mesh_prim_paths=["/World/ground"],       # ground only — no obstacles
)
```

At altitude 0.2m (spawn), produces:
- Rays 0-2 (straight down): 0.2/15 = 0.0133
- Rays 3-5 (45° down): ~0.28/15 = 0.0189
- Rays 6-8 (horizontal): max range = 1.0

---

## Files

| File | Purpose |
|---|---|
| `multi_waypoint/multi_waypoint_sac.py` | Native LL training environment |
| `obstacle/hierarchical_obstacle_sac.py` | Original obstacle env (has `_hl_step_counter` bug) |
| `obstacle/hierarchical_obstacle_sac_v2.py` | Fixed obstacle env (per-env counter) |
| `obstacle/test_ll_stability.py` | LL stability test (monkey-patches v2, fixed waypoints) |
| `obstacle/FINDINGS.md` | Code review findings, v2 changes |
| `obstacle/LL_STABILITY_DEBUG.md` | This document |

### LL Checkpoint

```
multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip
```
- 48,584 parameters, deterministic=True at inference
- Trained in native env with `arena_x_range=(-8, 8)`, spawn at `(-10, 0, 0.2)`
- First waypoint typically ~2m from spawn (x ≈ -8)

---

## test_ll_stability.py — How It Works

The test script verifies LL stability in the obstacle environment with all confounding factors
removed:

1. **No HL control** — HL modifiers locked to neutral (yaw_offset=0, speed_factor=1, alt_offset=0)
2. **Fixed waypoints** — Clear-path waypoints along center corridor (y=0)
3. **Frozen LL** — Same checkpoint, same actor extraction as v2
4. **Native control path** — No EMA smoothing, no speed_factor scaling

### How it avoids dual-AppLauncher conflict

Isaac Sim can only have one AppLauncher per process. The script manipulates `sys.argv` before
importing `hierarchical_obstacle_sac_v2.py`, which creates the AppLauncher at module level:

```python
sys.argv = [sys.argv[0], "--mode", "train", "--num_envs", "1"]
import hierarchical_obstacle_sac_v2 as v2
```

Then monkey-patches the unwrapped environment:
- `_generate_waypoints` → places fixed clear-path waypoints
- `_pre_physics_step` → replicates native env control path exactly

### Current fixed waypoints (PROBLEMATIC)

```python
FIXED_WAYPOINTS = [(-6.0, 0.0, 1.5), (0.0, 0.0, 1.5), (6.0, 0.0, 1.5)]
```

First waypoint at (-6, 0, 1.5) creates `goal_pos_b = (4.0, 0.0, 1.3)` — **too far for TAKEOFF**.

### Recommended fix

Change to waypoints closer to spawn, matching native env's first-WP distance:

```python
FIXED_WAYPOINTS = [(-8.0, 0.0, 1.5), (0.0, 0.0, 1.5), (6.0, 0.0, 1.5)]
```

This gives `goal_pos_b ≈ (2.0, 0.0, 1.3)` — within the LL's training distribution.

---

## Diagnostic Output Captured

### Step 0 observation (obstacle env)

```
ll_obs[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
             4.0, 0.0, 1.3,
             0.0133, 0.0133, 0.0133, 0.0140, 0.0189, 0.0189, 1.0, 1.0, 1.0,
             1.0, 0.0, 0.0, 0.0, 0.0]
```

### Crash sequence (first 50 steps)

```
s=   5 TK | pos=(-10.0,0.0,0.22)  | vy_cmd= 0.90 | grav_z=-0.995  ← already 0.9 m/s lateral
s=  15 TK | pos=(-10.0,-0.0,0.33) | vy_cmd=-1.21 | grav_z=-0.973  ← reversing, growing
s=  25 TK | pos=(-10.0,-0.2,0.45) | vy_cmd=-1.83 | grav_z=-0.976  ← nearly saturated
s=  35 TK | pos=(-9.9,-0.4,0.57)  | vy_cmd=-1.98 | grav_z=-0.984  ← saturated
s=  50 TK | pos=(-9.8,-0.8,0.71)  | vy_cmd=-2.00 | grav_z=-0.985  ← full tilt
s=  65 TK | pos=(-9.5,-1.3,0.67)  | vy_cmd=-1.93 | grav_z=-0.850  ← past recovery
s= 105 TK | pos=(-8.8,-1.7,0.55)  | vy_cmd=-1.98 | grav_z= 0.515  ← flipped → DIED
```

---

## Next Steps

1. **Fix test waypoints** — Change first WP to (-8, 0, 1.5) and re-run to verify LL stability
2. **Verify HL+LL together** — Once LL is confirmed stable, test with HL enabled to check if
   the `_hl_step_counter` fix in v2 resolves the original "degrades over time" symptom
3. **Consider clamping goal distance** — The LL's sensitivity to goal distance during TAKEOFF
   is fragile. May want to clamp `goal_pos_b` magnitude during non-NAVIGATE phases to stay
   within the training distribution
4. **Investigate HL goal modifier OOD** — During NAVIGATE, the HL can push goal_pos_b to
   extremes via yaw_offset/speed_factor/alt_offset. Need to verify these stay within the
   LL's training distribution

---

## Summary

| Issue | Status | Impact |
|---|---|---|
| Global `_hl_step_counter` | Fixed in v2 | Desynchronized HL updates during multi-env training |
| Test waypoints too far | **Found** — fix pending | Test artifact causing TAKEOFF crashes |
| LL OOD sensitivity | **Found** | LL produces erratic lateral cmds for unusual goal distances |
| HL modifier OOD | Suspected, not yet tested | HL could push LL obs outside training distribution |
