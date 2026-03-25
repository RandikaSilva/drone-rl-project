# Incremental Debug Log: LL Controller Stability

Date: 2026-03-19 — 2026-03-20

---

## Overview

### Problem

A frozen low-level (LL) drone controller flies perfectly in its native training
environment but crashes when reused inside the hierarchical obstacle avoidance
environment. This document tracks the incremental process of finding every culprit.

### Approach

Start with the working LL in its native env and incrementally add features from
the obstacle env one at a time until it breaks. Each test creates a new file —
working files are never modified.

### Architecture

The system uses a **hierarchical SAC architecture** with three control layers:

```
┌─────────────────────────────────────────────────────┐
│  High-Level (HL) Policy — 10 Hz (every 5 env steps) │
│  Input:  1347-ray lidar + drone state (20 dims)     │
│  Output: goal modifiers (yaw_offset, speed_factor,  │
│          alt_offset) that steer LL around obstacles  │
└────────────────────┬────────────────────────────────┘
                     │ modified goal vector
┌────────────────────▼────────────────────────────────┐
│  Low-Level (LL) Policy — 50 Hz (frozen checkpoint)   │
│  Input:  26-dim obs (vel, angvel, gravity, goal,     │
│          9-ray lidar, phase one-hot)                 │
│  Output: velocity commands (vx, vy, vz, yaw_rate)   │
└────────────────────┬────────────────────────────────┘
                     │ velocity commands
┌────────────────────▼────────────────────────────────┐
│  PID Controller — 100 Hz (physics rate)              │
│  Converts velocity commands → thrust + torques       │
│  Cascade: vel error → desired tilt → attitude PD     │
└─────────────────────────────────────────────────────┘
```

### Key Files

**Native LL training environment** (`multi_waypoint/` directory):
- `multi_waypoint_sac.py` — Original training env. Single-layer: RL agent directly
  outputs velocity commands through the PID. 5-phase state machine (TAKEOFF →
  STABILIZE → NAVIGATE → HOVER → LAND). 26-dim observation, 4-dim action (vx, vy,
  vz, yaw_rate). Trains with SB3 SAC. Has train/play/eval modes.

**Obstacle avoidance environment** (`obstacle/` directory):
- `hierarchical_obstacle_sac_v2.py` — Two-layer hierarchical env. HL policy (trainable)
  outputs goal modifiers every 5 steps. LL policy (frozen checkpoint) outputs velocity
  commands every step. 14 tree obstacles. HL uses 1347-ray lidar for obstacle sensing.
  LL uses 9-ray downward lidar for ground/height sensing.

**LL Checkpoint:**
```
multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip
```
48,584 parameters, frozen, deterministic inference.

### LL Observation Vector (26 dims)

Both environments construct the same 26-dim observation for the LL:

| Component | Dims | Source | Notes |
|-----------|------|--------|-------|
| Linear velocity (body frame) | 3 | `root_lin_vel_b` | Unbounded |
| Angular velocity (body frame) | 3 | `root_ang_vel_b` | Unbounded |
| Gravity projection (body frame) | 3 | `projected_gravity_b` | Used for attitude sensing |
| Goal position (body frame) | 3 | `subtract_frame_transforms` | Relative to current waypoint |
| LiDAR distances (normalized) | 9 | 9 rays, distance/15m | Downward-facing, ground only |
| Phase one-hot | 5 | TAKEOFF/STABILIZE/NAVIGATE/HOVER/LAND | 5-class encoding |

### Critical Discovery: Action Space Mismatch

The action space is `Box(-100, 100)`, NOT `Box(-1, 1)`.

During training, SB3's `predict()` calls `unscale_action()` which maps the actor's
output [-1,1] → [-100,100]. Then the env's `_pre_physics_step` clamps to [-1,1],
**saturating everything to ±1.0**. The LL was effectively trained with binary ±1.0
actions. The PID does all fine control.

The obstacle env calls `actor()` directly, getting nuanced values like [0.808, 0.296]
instead of the saturated ±1.0. This mismatch breaks the LL.

```
Training pipeline: actor → [-1,1] → unscale → [-100,100] → clamp [-1,1] → ±1.0
Obstacle env:      actor → [-1,1] → clamp [-1,1] → nuanced values ≠ training
```

---

## How to Run Any Test

```bash
# Step 1: Navigate to IsaacLab and activate environment
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate

# Step 2: IMPORTANT — Kill ALL sim windows before running a new one
pkill -9 -f "python|isaac|omni|kit" 2>/dev/null; sleep 2

# Step 3: Run the test file
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/multi_waypoint/<FILE> \
    --mode play --num_envs 1 \
    --checkpoint ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip
```

Replace `<FILE>` with any test file listed below.

---

## Test Results

### Baseline: LL in Native Environment

**File:** `multi_waypoint_sac.py`
**Result: STABLE — flies perfectly**

The LL checkpoint flies stably through 3-5 waypoints with 0% crash rate. Takes off,
stabilizes, navigates zigzag waypoints, hovers, and lands. ~1000+ steps per episode.

Config: observation_space=26, decimation=2 (50 Hz), env_spacing=50, 9-ray lidar
(3ch × 3 horizontal, vertical_fov -90° to 0°, max 15m, ground only), spawn at
(-10, 0, 0.2), cruise altitude 1.5m, no obstacles, single raycaster.

---

### Test 1: Add Obstacle Geometry (Trees)

**File:** `multi_waypoint_sac_debug_test1_2.py` (includes Test 2)
**Result: STABLE — no effect**

Added 14 tree obstacles (trunk + canopy cylinders with collision) matching the obstacle
env layout. Kinematic rigid bodies with collision enabled.

**Conclusion:** Obstacle geometry does NOT affect LL stability.

---

### Test 2: Add Second Raycaster (HL LiDAR, 1347 Rays)

**File:** `multi_waypoint_sac_debug_test1_2.py` (includes Test 1)
**Result: STABLE — no effect**

Added a second RayCaster on the same body prim matching the HL lidar config (3 channels ×
449 horizontal = 1347 rays, vertical_fov -45° to 0°, max 20m). Registered with scene but
NOT used for LL observation.

**Conclusion:** Two raycasters on the same prim do NOT interfere.

---

### Test 3: EMA Smoothing on Velocity Commands

**File:** `multi_waypoint_sac_debug_test1_2_3_4.py` (includes Test 4)
**Result: CRASHES — confirmed culprit #1**

Added EMA smoothing (alpha=0.6) on lateral velocity commands (vx, vy) only — not vz or
yaw_rate. Episodes became very short (~200-300 steps vs 1000+ baseline).

**Root cause:** The LL outputs vx, vy, vz, yaw_rate as a coordinated set. Smoothing only
the lateral components desynchronizes this coordination, causing rolling crashes. The
drone can't maintain attitude when two channels are lagged and two are immediate.

**Status in v2:** Already identified and removed (see v2 lines 864-869 comment).

---

### Test 4: env_spacing Change (50 → 70)

**File:** `multi_waypoint_sac_last_working.py`
**Result: STABLE — no effect**

Changed env_spacing from 50.0 to 70.0 (matching obstacle env). This file also includes
Tests 1+2 (obstacles + HL lidar) and is the **first "last known working" baseline**.

**Conclusion:** env_spacing does NOT affect LL stability.

---

### Test 5: Direct Actor Extraction (Without Fix)

**File:** `multi_waypoint_sac_debug_test5.py`
**Result: CRASHES — confirmed culprit #2**

Changed the play loop from `agent.predict(obs, deterministic=True)` to extracting the
actor network and calling it directly: `actor(obs_tensor, deterministic=True).clamp(-1, 1)`.
This is how the obstacle env calls the LL. Episodes significantly shorter (~200-750 steps).

**Root cause:** See "Critical Discovery: Action Space Mismatch" above.

```
agent.predict():  [80.8, 29.6, 90.7, -16.6]  → after clamp: [1.0, 1.0, 1.0, -1.0]
actor() direct:   [0.808, 0.296, 0.907, -0.166]  → NOT what training saw
```

**Status in v2:** NOT yet fixed. Current v2 code at line 860-862:
```python
ll_actions = self._ll_actor(ll_obs, deterministic=True)
ll_actions = ll_actions.clamp(-1.0, 1.0)  # WRONG — missing ×100
```

**Fix:** `ll_actions = (ll_actions * 100.0).clamp(-1.0, 1.0)`

---

### Test 5-fix: Actor Extraction WITH ×100 Scaling

**File:** `multi_waypoint_sac_test5_fix.py`
**Result: STABLE — fix confirmed**

Applied `(actions * 100.0).clamp(-1.0, 1.0)` to replicate the training pipeline when
using the extracted actor. Episodes comparable to baseline (~739 steps, WP 3/5).

This file is built on top of last_working (Tests 1+2+4) + the actor ×100 fix. It became
the new working baseline for subsequent tests.

---

### Test 7: Goal XY Magnitude Clamping (3.0m)

**File:** `multi_waypoint_sac_debug_test7_goal_clamp.py`
**Result: MILD INSTABILITY — minor contributor**

Added XY magnitude clamping to 3.0m in `_get_observations`, matching v2's
`_build_low_level_obs` (lines 810-817). Preserves direction, only clamps magnitude.

```python
MAX_LL_GOAL_XY = 3.0
goal_xy = goal_pos_b[:, :2]
xy_dist = goal_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
xy_scale = (MAX_LL_GOAL_XY / xy_dist).clamp(max=1.0)
goal_pos_b = torch.cat([goal_xy * xy_scale, goal_pos_b[:, 2:3]], dim=1)
```

**Root cause:** The LL was trained with unbounded goal distances (routinely 6-8m). Clamping
to 3m changes the magnitude signal the LL uses for speed regulation. Not a hard crash but
noticeable degradation.

---

### Test 8: Goal Modifier Pipeline (Neutral Values)

**File:** `multi_waypoint_sac_debug_test8_goal_modifiers.py`
**Result: MILD INSTABILITY — same as Test 7**

Added the full goal modifier pipeline from v2's `_build_low_level_obs` (lines 786-831)
but with neutral values: yaw_offset=0, speed_factor=1.0, alt_offset=0.0. With neutral
values, only the XY clamp has any real effect. Same mild instability as Test 7.

The pipeline includes:
1. Yaw rotation of goal XY (NAVIGATE phase only)
2. Altitude offset (NAVIGATE phase only)
3. Conditional selection (modified goal in NAVIGATE, real goal in other phases)
4. XY magnitude clamping to 3.0m (ALL phases)
5. Speed factor scaling on XY (NAVIGATE phase only)

---

### Test 9: Random HL Goal Modifiers

**File:** `multi_waypoint_sac_debug_test9_random_hl.py`
**Result: UNSTABLE — culprit #3**

Simulated an untrained HL policy by randomizing goal modifiers every 5 steps during
NAVIGATE phase (matching v2's 10 Hz HL update rate):

- `yaw_offset`: uniform random in [-0.5, +0.5] radians (~±28°)
- `speed_factor`: uniform random in [0.6, 1.0]
- `alt_offset`: uniform random in [-0.8, +0.8] meters

These ranges match v2's configured limits (`max_yaw_offset=0.5`, `speed_factor_range=(0.6, 1.0)`,
`max_altitude_offset=0.8`).

**Root cause:** Random goal perturbations every 5 steps during NAVIGATE create a rapidly
changing, incoherent goal signal. The LL tries to track each new goal direction/distance
but the constant changes prevent stable flight. This is what happens at the start of HL
training when the HL outputs effectively random actions.

**Implication:** The HL cannot be trained from scratch with these modifier ranges — the LL
will crash before the HL can learn anything useful. Possible mitigations:
- Tighten modifier ranges during early HL training (curriculum)
- Start HL with a warm-start policy that outputs near-neutral values
- Isolate which modifier (yaw vs speed vs alt) is most destabilizing

### Test 10: Stabilized Random HL Goal Modifiers (Fix for Test 9)

**File:** `multi_waypoint_sac_debug_test10_stabilized_hl.py`
**Result: STABLE — fix confirmed (avg reward 1232.92 vs Test 9's 73.11)**

Applied three fixes to address the instability caused by random HL goal modifiers:

1. **Tightened modifier ranges** (curriculum-safe bounds for untrained HL):
   - Yaw offset: ±0.5 → **±0.15 rad** (~±8.6° vs ±28°)
   - Speed factor: 0.6-1.0 → **0.85-1.0** (gentle slowdown only)
   - Alt offset: ±0.8 → **±0.2 m** (minimal altitude perturbation)

2. **EMA smoothing on goal modifiers** (alpha=0.3):
   - Gradual transitions instead of sudden jumps every 5 steps
   - Applied to modifiers, NOT to velocity commands (that was culprit #1)
   - Prevents goal whiplash that caused LL coordination breakdown

3. **Increased XY clamp** from 3.0m to 8.0m:
   - Matches LL training distribution (routinely 6-8m distances)
   - Restores speed regulation signal the LL relies on

```python
# Tightened ranges
target_yaw = (torch.rand(...) * 2 - 1) * 0.15      # was 0.5
target_speed = torch.rand(...) * 0.15 + 0.85         # was 0.4 + 0.6
target_alt = (torch.rand(...) * 2 - 1) * 0.2         # was 0.8

# EMA on modifiers (NOT velocity commands)
alpha = 0.3
self._hl_yaw_offset[mask] = (1 - alpha) * self._hl_yaw_offset[mask] + alpha * target_yaw

# XY clamp
MAX_LL_GOAL_XY = 8.0  # was 3.0
```

**Comparison:**
| | Test 9 (random, full ranges) | Test 10 (stabilized) |
|---|---|---|
| Episode 1 | 236 steps, crashed | 739 steps, WP 3/5 |
| Episode 2 | 103 steps, crashed | 813 steps, all 4/4 WPs, landed |
| Avg reward | 73.11 | **1232.92** |
| Avg steps/ep | ~192 | ~1552 |

**Implication for HL training:** The HL can safely be trained from scratch using these
tightened ranges. As training progresses and the HL learns coherent obstacle avoidance
behavior, the ranges can be gradually widened (curriculum) toward the original v2 values.

---

## Summary of Results

| Test | Change | Result | Culprit? |
|------|--------|--------|----------|
| Baseline | Native env, no changes | Stable | — |
| 1 | Add obstacle geometry (14 trees) | Stable | No |
| 2 | Add HL lidar (1347 rays, 2nd raycaster) | Stable | No |
| 3 | **EMA smoothing on lateral velocity** | **Crashes** | **Yes — #1** |
| 4 | env_spacing 50 → 70 | Stable | No |
| 5 | **Direct actor extraction (no ×100)** | **Crashes** | **Yes — #2** |
| 5-fix | Actor extraction WITH ×100 scale | Stable | Fix confirmed |
| 7 | Goal XY clamp to 3.0m | Mild instability | Minor contributor |
| 8 | Goal modifier pipeline (neutral) | Mild instability | Same as 7 |
| 9 | **Random HL goal modifiers** | **Unstable** | **Yes — #3** |
| 10 | **Stabilized HL (tightened ranges + EMA + XY clamp 8m)** | **Stable** | **Fix for #3** |

---

## All Code Files

All test files are in `~/Desktop/Lasitha/drone_rl_project/drone-rl-project/multi_waypoint/`.

| File | Based On | What It Adds | Result |
|------|----------|--------------|--------|
| `multi_waypoint_sac.py` | — | Original native LL training env | Stable (baseline) |
| `multi_waypoint_sac_debug_test1_2.py` | Baseline | + 14 tree obstacles + HL lidar (1347 rays) | Stable |
| `multi_waypoint_sac_debug_test1_2_3_4.py` | Test 1+2 | + EMA smoothing + env_spacing=70 | Crashes (EMA) |
| `multi_waypoint_sac_last_working.py` | Test 1+2 | + env_spacing=70 (no EMA) | Stable |
| `multi_waypoint_sac_debug_test5.py` | last_working | + actor extraction (no ×100 fix) | Crashes |
| `multi_waypoint_sac_test5_fix.py` | last_working | + actor extraction WITH ×100 fix | Stable |
| `multi_waypoint_sac_test5_fix_backup.py` | test5_fix | Backup copy | Stable |
| `multi_waypoint_sac_debug_test7_goal_clamp.py` | test5_fix | + goal XY clamp 3.0m | Mild instability |
| `multi_waypoint_sac_debug_test8_goal_modifiers.py` | test5_fix | + full goal modifier pipeline (neutral) | Mild instability |
| `multi_waypoint_sac_debug_test9_random_hl.py` | test8 | + random HL goal modifiers every 5 steps | Unstable |
| **`multi_waypoint_sac_debug_test10_stabilized_hl.py`** | **test9** | **+ tightened ranges + EMA on modifiers + XY clamp 8m** | **Stable (fix for #3)** |
| `multi_waypoint_sac_last_working_v2.py` | test5_fix | Backup of current last working config | Stable |

### What each test file contains vs baseline

All test files are copies of `multi_waypoint_sac.py` with incremental modifications.
They are self-contained (env + config + train/play/eval) and can be run independently.

**`multi_waypoint_sac_debug_test1_2.py`:**
- Config: Added `obstacle_positions` list (14 trees) and `hl_lidar` RayCasterCfg
- `_setup_scene`: Added tree spawning loop + `self._hl_lidar = RayCaster(self.cfg.hl_lidar)`
- Everything else identical to baseline

**`multi_waypoint_sac_last_working.py`:**
- Same as test1_2 + `env_spacing=70.0` in scene config

**`multi_waypoint_sac_test5_fix.py` (current working baseline):**
- Same as last_working + modified `play()` function:
  - Extracts actor: `actor = agent.policy.actor`
  - Calls with ×100 fix: `actions = (actor(obs, deterministic=True) * 100.0).clamp(-1, 1)`

**`multi_waypoint_sac_debug_test7_goal_clamp.py`:**
- Same as test5_fix + XY magnitude clamping in `_get_observations`:
  - Clamps goal XY to max 3.0m, preserving direction

**`multi_waypoint_sac_debug_test8_goal_modifiers.py`:**
- Same as test5_fix + full goal modifier pipeline in `_get_observations`:
  - Yaw rotation, alt offset, XY clamp, speed factor (all neutral values)
  - Goal modifier buffers in `__init__` and reset in `_reset_idx`

**`multi_waypoint_sac_debug_test9_random_hl.py`:**
- Same as test8 + random modifier updates in `_pre_physics_step`:
  - Step counter, updates every 5 steps during NAVIGATE
  - Random yaw ±0.5, speed 0.6-1.0, alt ±0.8

**`multi_waypoint_sac_debug_test10_stabilized_hl.py` (fix for Test 9):**
- Same as test9 with three fixes:
  - Tightened modifier ranges: yaw ±0.15, speed 0.85-1.0, alt ±0.2
  - EMA smoothing (alpha=0.3) on goal modifiers for gradual transitions
  - XY magnitude clamp increased from 3.0m to 8.0m

---

## Confirmed Culprits & Fix Status

### Culprit #1: EMA Smoothing on Lateral Velocity (Test 3)

- **Severity:** Hard crash
- **Mechanism:** Asymmetric EMA on vx/vy but not vz/yaw_rate desynchronizes the LL's coordinated 4-channel output
- **Status in v2:** Already removed (v2 lines 864-869 comment)
- **Action needed:** None

### Culprit #2: Actor Extraction Without ×100 Scaling (Test 5)

- **Severity:** Hard crash
- **Mechanism:** `actor()` returns [-1,1], but training saw ±1.0 (saturated) due to action space Box(-100,100) + unscale + clamp
- **Status in v2:** NOT fixed. Line 860-862 uses `.clamp(-1, 1)` without ×100
- **Fix:** Change line 862 from `ll_actions = ll_actions.clamp(-1.0, 1.0)` to `ll_actions = (ll_actions * 100.0).clamp(-1.0, 1.0)`
- **Action needed:** Apply one-line fix to v2

### Culprit #3: Random HL Goal Modifiers (Test 9) — FIXED in Test 10

- **Severity:** Unstable (not instant crash, but degrades quickly)
- **Mechanism:** Untrained HL outputs random goal perturbations that create incoherent goal signal for LL
- **Status in v2:** Present by design — the HL needs to modify goals to steer around obstacles
- **Fix (Test 10):** Three combined changes restore stability:
  1. Tighten modifier ranges: yaw ±0.5→±0.15, speed 0.6-1.0→0.85-1.0, alt ±0.8→±0.2
  2. EMA smoothing (alpha=0.3) on goal modifiers (gradual transitions)
  3. XY clamp 3.0→8.0m (match LL training distribution)
- **Result:** Avg reward 73.11 → **1232.92**, avg steps 192 → **1552**
- **Action needed:** Apply these safe ranges to v2 as starting curriculum. Widen gradually as HL learns.

### Minor Contributor: Goal XY Magnitude Clamping (Test 7)

- **Severity:** Mild instability
- **Mechanism:** LL trained with unbounded goal distances (6-8m), clamping to 3.0m changes speed regulation signal
- **Status in v2:** Present in `_build_low_level_obs` lines 810-817
- **Fix options:** Increase or remove the 3.0m cap
- **Action needed:** Consider removing or increasing cap

---

## v3: Deterministic Kinematic Controller

**File:** `obstacle/hierarchical_obstacle_sac_v3.py`

New approach: replace the frozen LL checkpoint entirely with a deterministic proportional
controller that uses kinematic pose setting (no physics forces). The robot is directly
positioned each frame — it literally cannot crash. This gives the HL a perfectly stable,
predictable base to learn obstacle avoidance against.

### What v3 changes vs v2

1. **No LL checkpoint** — removed checkpoint loading, `--ll_checkpoint` arg, LL lidar
2. **Deterministic proportional controller** (`_run_low_level_policy`):
   - Computes velocity proportional to goal direction/distance
   - Phase-specific behavior: TAKEOFF (ascend), STABILIZE/HOVER (hold), NAVIGATE (fly toward goal), LAND (descend)
   - HL goal modifiers (yaw, speed, alt) still applied during NAVIGATE
3. **Kinematic pose setting** (`_kinematic_update` + `_apply_action`):
   - Position: integrated from velocity each physics step (100 Hz)
   - Orientation: set directly — mostly level with small cosmetic tilt (~7° max)
   - Yaw: tracked separately, updated by yaw_rate command
   - No PID, no thrust/torques computation
   - Hover thrust applied only to counteract gravity (keeps physics engine happy)
4. **Config changes:**
   - Conservative velocities: max_xy=1.0 m/s, max_z=0.8 m/s
   - Controller gains: ll_gain_xy=0.8, ll_gain_z=0.8, ll_gain_yaw=0.5
   - PID config still present but unused (kept for reference)

### How to run v3

```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate

# IMPORTANT: Kill all sim windows first
pkill -9 -f "python|isaac|omni|kit" 2>/dev/null; sleep 2

# Play mode (no checkpoint needed — HL uses zero actions = neutral modifiers):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v3.py \
    --mode play --num_envs 1

# Play mode with trained HL checkpoint:
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v3.py \
    --mode play --num_envs 1 --checkpoint /path/to/hl_checkpoint.zip

# Train HL:
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v3.py \
    --mode train --num_envs 256 --total_timesteps 16000000 --headless
```

---

## How to Run Everything

### Prerequisites

```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate

# ALWAYS kill existing sims before launching new ones
pkill -9 -f "python|isaac|omni|kit" 2>/dev/null; sleep 2
```

### LL Debug Tests (multi_waypoint/ directory)

All use the same checkpoint and command pattern:

```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/multi_waypoint/<FILE> \
    --mode play --num_envs 1 \
    --checkpoint ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip
```

| FILE | What it tests | Expected result |
|------|---------------|-----------------|
| `multi_waypoint_sac.py` | Baseline LL | Stable |
| `multi_waypoint_sac_debug_test1_2.py` | + obstacles + HL lidar | Stable |
| `multi_waypoint_sac_last_working.py` | + env_spacing=70 | Stable |
| `multi_waypoint_sac_debug_test5.py` | + actor extraction (broken) | Crashes |
| `multi_waypoint_sac_test5_fix.py` | + actor ×100 fix | Stable |
| `multi_waypoint_sac_debug_test7_goal_clamp.py` | + XY clamp 3m | Mild instability |
| `multi_waypoint_sac_debug_test8_goal_modifiers.py` | + goal modifiers (neutral) | Mild instability |
| `multi_waypoint_sac_debug_test9_random_hl.py` | + random HL modifiers | Unstable |
| `multi_waypoint_sac_debug_test10_stabilized_hl.py` | + tightened ranges + EMA | Stable |
| `multi_waypoint_sac_last_working_v2.py` | Last known good (backup) | Stable |

### Obstacle Env (obstacle/ directory)

```bash
# v2 — frozen LL checkpoint (original, has unfixed bugs):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v2.py \
    --mode play --num_envs 1 \
    --checkpoint /path/to/hl_checkpoint.zip

# v3 — deterministic kinematic controller (no LL checkpoint needed):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v3.py \
    --mode play --num_envs 1
```

---

## Next Steps

1. **Verify v3 kinematic controller** — confirm drone flies stably with zero HL actions
2. **Train HL on v3** — the deterministic LL is perfectly stable, HL can learn safely
3. v2 fixes documented but superseded by v3 approach
