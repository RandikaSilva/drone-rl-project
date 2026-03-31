# Multi-Waypoint SAC — Version History & Changes

Date: 2026-03-28
Environment: Multi-waypoint zigzag navigation, 10x Crazyflie, hierarchical PID control
Architecture: Single-level SAC (early) -> 2-level Hierarchical SAC (obstacle avoidance)

---

## Version Timeline

| Version | File | Architecture | Key Focus |
|---------|------|-------------|-----------|
| Baseline | `multi_waypoint_sac.py` | Flat SAC (4D velocity) | Original multi-waypoint navigation |
| Debug Tests 1-4 | `multi_waypoint_sac_debug_test1_2_3_4.py` | Flat SAC + EMA | EMA smoothing, env spacing increase |
| Debug Test 5 | `multi_waypoint_sac_debug_test5.py` / `test5_fix.py` | Flat SAC | Goal clamping experiments |
| Debug Test 7 | `multi_waypoint_sac_debug_test7_goal_clamp.py` | Flat SAC | Goal clamp refinement |
| Debug Test 8 | `multi_waypoint_sac_debug_test8_goal_modifiers.py` | Flat SAC | Goal modifier system (yaw, speed, alt) |
| Debug Test 9 | `multi_waypoint_sac_debug_test9_random_hl.py` | Proto-hierarchical | Random high-level goal generation |
| Debug Test 10 | `multi_waypoint_sac_debug_test10_stabilized_hl.py` | Hierarchical (HL+LL) | Stabilized hierarchical control (foundation) |
| OA v1 | `multi_waypoint_sac_obstacle_avoidance.py` | Hierarchical (HL+LL) | First obstacle avoidance (14 trees) |
| OA v2 | `multi_waypoint_sac_obstacle_avoidance_v2.py` | Hierarchical (HL+LL) | Wider steering + stronger penalties (18 trees) |
| OA v3 | `multi_waypoint_sac_obstacle_avoidance_v3.py` | Hierarchical (HL+LL) | Anti-oscillation + corridor detection |

---

## Key Configuration Comparison

### Core Architecture

| Parameter | Baseline | Tests 1-4 | Test 10 / OA v1 | OA v2 | OA v3 |
|-----------|----------|-----------|-----------------|-------|-------|
| Architecture | Flat SAC | Flat SAC | **2-level Hierarchical** | Hierarchical | Hierarchical |
| HL action space | N/A | N/A | **3** (yaw_offset, speed_factor, alt_offset) | 3 | 3 |
| LL action space | 4 (vx, vy, vz, yaw) | 4 | 4 (frozen) | 4 (frozen) | 4 (frozen) |
| HL obs space | N/A | N/A | **1367** (LiDAR 1347 + state 20) | 1367 | 1367 |
| LL obs space | 26 | 26 | 26 | 26 | 26 |
| HL frequency | N/A | N/A | **10 Hz** (every 5 steps) | **16.7 Hz** (every 3) | 16.7 Hz |
| LL frequency | 50 Hz | 50 Hz | 50 Hz | 50 Hz | 50 Hz |
| Features extractor | MLP | MLP | **LidarCNNExtractor** (1D CNN) | LidarCNNExtractor | LidarCNNExtractor |
| env_spacing | 50.0 m | **70.0 m** | 70.0 m | 70.0 m | 70.0 m |

### Goal Modifier Parameters (Hierarchical versions only)

| Parameter | Test 10 / OA v1 | OA v2 | OA v3 |
|-----------|-----------------|-------|-------|
| max_yaw_offset | 0.15 rad (8.6deg) | **0.5 rad (28deg)** | **0.35 rad (20deg)** |
| speed_factor_range | (0.85, 1.0) | **(0.3, 1.0)** | (0.3, 1.0) |
| max_altitude_offset | 0.2 m | **0.5 m** | 0.5 m |
| goal_modifier_smoothing (EMA) | 0.3 | **0.5** | 0.5 (base), **0.2 in danger** |
| max_yaw_rate_per_update | (none) | (none) | **0.1 rad** |

### Obstacle Environment

| Parameter | Test 10 / OA v1 | OA v2 | OA v3 |
|-----------|-----------------|-------|-------|
| Number of trees | 14 | **18** (+4 interior) | 18 |
| obstacle_collision_radius | 0.5 m | 0.5 m | 0.5 m |
| obstacle_safe_distance | 1.5 m | **2.5 m** | 2.5 m |
| lidar_danger_distance | 3.5 m | **5.0 m** | 5.0 m |
| emergency_brake_distance | (none) | **2.0 m** | **3.5 m** |
| emergency_brake_min_factor | (none) | (implicit 0.2) | **0.15** |
| corridor_danger_distance | (none) | (none) | **4.0 m** |
| corridor_escape_vz | (none) | (none) | **0.3 m/s** |

### Reward Weights (Obstacle-Specific)

| Parameter | Test 10 / OA v1 | OA v2 | OA v3 |
|-----------|-----------------|-------|-------|
| obstacle_proximity_penalty | -25.0 | **-60.0** | -60.0 |
| lidar_obstacle_penalty | -25.0 | **-60.0** | -60.0 |
| subgoal_reachability_scale | 5.0 | **8.0** | 8.0 |
| subgoal_magnitude_penalty | -0.3 | **-0.1** | -0.1 |
| crash_penalty | -500 | **-1500** | -1500 |
| xy_progress_scale (NAV) | 5.0 | 5.0 | 5.0 |
| velocity_align_scale (NAV) | 6.0 | 6.0 | 6.0 |
| intermediate_waypoint_bonus | 200.0 | 200.0 | 200.0 |
| goal_reached_bonus | 750.0 | 750.0 | 750.0 |

### Training Configuration

| Parameter | Baseline | OA v1 | OA v2 | OA v3 |
|-----------|----------|-------|-------|-------|
| total_timesteps | 16M | 16M | **20M** | 20M |
| learning_rate | 3e-4 | 3e-4 | 3e-4 | 3e-4 |
| batch_size | 256 | 256 | 256 | 256 |
| gamma | 0.98 | 0.98 | 0.98 | 0.98 |
| tau | 0.005 | 0.005 | 0.005 | 0.005 |
| net_arch (HL) | [256, 128, 64] | [256, 128] | [256, 128] | [256, 128] |
| buffer_size | 1M | 1M | 1M | 1M |
| ent_coef | auto | auto | auto | auto |

### EMA Smoothing Evolution

| Version | EMA Alpha | Notes |
|---------|-----------|-------|
| Baseline | (none) | No smoothing |
| Tests 1-4 | **0.6** | First introduction of EMA on lateral velocity |
| Test 10 / OA v1 | **0.3** | Tightened for stability |
| OA v2 | **0.5** | Faster reaction |
| OA v3 | **0.5** (base), **0.2** (danger zone) | Adaptive — slower in tight spaces |

---

## Changes: Baseline -> Tests 1-4

First debug cycle addressing circling behavior and oscillation.

### Summary Table

| Parameter | Baseline | Tests 1-4 |
|-----------|----------|-----------|
| env_spacing | 50.0 m | **70.0 m** |
| EMA smoothing | (none) | **alpha = 0.6** |
| HL LiDAR | (none) | **Added** (3ch, 449 rays/ch, 20m range) |
| Obstacle positions | (none) | **14 trees defined** (not yet used in training) |
| Everything else | (baseline) | (identical) |

### Detailed Explanations

#### 1. Increased Environment Spacing (50m -> 70m)
- **Root cause**: 50m spacing caused inter-environment interference with 512 parallel envs.
- **Fix**: 70m gives enough clearance for zigzag waypoint generation without overlap.

#### 2. EMA Smoothing on Velocity Commands (NEW, alpha=0.6)
- **Root cause**: Raw RL policy outputs caused jittery velocity commands, leading to unstable flight.
- **Fix**: Exponential moving average with alpha=0.6 smooths lateral velocity, reducing oscillation while preserving responsiveness.

#### 3. High-Level LiDAR Infrastructure (Added, not yet active)
- Prepared HL LiDAR config (3 channels, -45 to 0 deg vertical, 0.8 deg horizontal resolution).
- Not yet used in observations — infrastructure for upcoming hierarchical architecture.

---

## Changes: Tests 1-4 -> Test 10 (Hierarchical Foundation)

The transition from flat single-level SAC to hierarchical 2-level control. Tests 5-9 were iterative experiments that led to the stabilized test 10 architecture.

### Summary Table

| Parameter | Tests 1-4 | Test 10 |
|-----------|-----------|---------|
| Architecture | Flat SAC (4D velocity) | **2-level Hierarchical (HL trained + LL frozen)** |
| HL action | N/A | **3D goal modifier** (yaw, speed, alt) |
| HL obs | N/A | **1367D** (LiDAR 1347 + state 20) |
| HL frequency | N/A | **10 Hz** |
| Features extractor | MLP | **LidarCNNExtractor** (1D CNN) |
| net_arch | [256, 128, 64] | **[256, 128]** |
| EMA alpha | 0.6 | **0.3** |
| max_yaw_offset | N/A | **0.15 rad** |
| speed_factor_range | N/A | **(0.85, 1.0)** |
| max_altitude_offset | N/A | **0.2 m** |

### Detailed Explanations

#### 1. 2-Level Hierarchical Architecture (NEW)
- **Root cause**: Flat SAC couldn't handle obstacle avoidance + multi-waypoint navigation simultaneously. Too many competing objectives for a single policy.
- **Fix**: Split into High-Level (obstacle-aware goal modifier) and Low-Level (frozen waypoint navigator).
- **HL agent**: Observes LiDAR (1347 rays) + drone state (20D). Outputs goal modifiers (yaw offset, speed factor, altitude offset) at 10 Hz.
- **LL agent**: Pre-trained multi-waypoint policy (frozen). Takes modified goals and outputs velocity commands at 50 Hz.
- **PID**: Converts velocity commands to thrust + torques at 100 Hz (unchanged).

#### 2. LidarCNNExtractor (NEW)
- 1D CNN processes 1347-ray LiDAR input (3 vertical channels x 449 horizontal rays).
- More parameter-efficient than MLP for spatial LiDAR data.

#### 3. Tightened Goal Modifiers
- Conservative initial ranges: yaw +-0.15 rad, speed 0.85-1.0, alt +-0.2m.
- Prevents HL agent from destabilizing the LL policy with aggressive modifications.

#### 4. Reduced Network Size ([256,128,64] -> [256,128])
- Smaller network sufficient for 3D goal modifier output (vs 4D velocity commands).

### Intermediate Test History (Tests 5-9)

| Test | Focus | Key Change | Outcome |
|------|-------|------------|---------|
| Test 5 | Goal clamping | Constrain goal positions to reachable range | Reduced divergence |
| Test 7 | Goal clamp refinement | Tighter clamp distance (8m) | More stable targets |
| Test 8 | Goal modifiers | Yaw/speed/alt modifier system | First hierarchical prototype |
| Test 9 | Random HL | Randomized sub-goal generation | Exploration improvement |
| Test 10 | Stabilization | Tightened ranges, EMA=0.3, goal clamp=8m | **Stable foundation** |

---

## Changes: Test 10 -> OA v1 (First Obstacle Avoidance)

Productionized hierarchical architecture with obstacle avoidance focus.

### Summary Table

| Parameter | Test 10 | OA v1 |
|-----------|---------|-------|
| Obstacle integration | Config only | **Active in training** |
| obstacle_safe_distance | 1.5 m | 1.5 m |
| obstacle_proximity_penalty | -25.0 | -25.0 |
| crash_penalty | -500 | -500 |

### Detailed Explanation

- Test 10's stabilized hierarchical architecture was cleaned up and used as the first proper obstacle avoidance version.
- 14 tree obstacles active during training with collision detection.
- **Problem discovered**: Conservative goal modifiers (yaw +-0.15 rad, speed 0.85-1.0) couldn't steer the drone away from obstacles fast enough. The HL agent lacked the authority to make meaningful avoidance maneuvers.

---

## Changes: OA v1 -> OA v2

v2 addresses tree collisions by giving the HL agent much more steering authority and stronger obstacle penalties.

### Summary Table

| Parameter | OA v1 | OA v2 |
|-----------|-------|-------|
| max_yaw_offset | 0.15 rad (8.6deg) | **0.5 rad (28deg)** |
| speed_factor_range | (0.85, 1.0) | **(0.3, 1.0)** |
| max_altitude_offset | 0.2 m | **0.5 m** |
| HL update frequency | 10 Hz (every 5 steps) | **16.7 Hz (every 3 steps)** |
| goal_modifier_smoothing | 0.3 | **0.5** |
| crash_penalty | -500 | **-1500** |
| obstacle_proximity_penalty | -25 | **-60** |
| lidar_obstacle_penalty | -25 | **-60** |
| lidar_danger_distance | 3.5 m | **5.0 m** |
| obstacle_safe_distance | 1.5 m | **2.5 m** |
| Emergency brake | (none) | **2.0 m trigger** |
| Number of trees | 14 | **18** (+4 interior) |
| subgoal_reachability_scale | 5.0 | **8.0** |
| subgoal_magnitude_penalty | -0.3 | **-0.1** |
| total_timesteps | 16M | **20M** |

### Detailed Explanations

#### 1. 3.3x Wider Yaw Range (0.15 -> 0.5 rad)
- **Root cause**: +-8.6 deg yaw offset was too small to steer around trees. The HL agent could see obstacles via LiDAR but couldn't turn sharply enough to avoid them.
- **Fix**: +-28 deg gives the HL agent enough authority for meaningful avoidance turns.

#### 2. Much Wider Speed Range ((0.85,1.0) -> (0.3,1.0))
- **Root cause**: Minimum 85% speed meant the drone couldn't slow down near obstacles.
- **Fix**: Speed factor down to 0.3 allows the drone to crawl through tight spaces.

#### 3. Wider Altitude Offset (0.2 -> 0.5m)
- Allows the drone to adjust height more aggressively to avoid obstacles at certain heights.

#### 4. Faster HL Decisions (10Hz -> 16.7Hz)
- **Root cause**: At 10 Hz, the HL agent updated every 100ms — too slow for fast-approaching obstacles.
- **Fix**: 16.7 Hz (every 60ms) gives 67% faster reaction time.

#### 5. Faster EMA Response (0.3 -> 0.5)
- Higher alpha = less smoothing = faster reaction to new HL commands.
- Trade-off: slightly more jitter, but critical for obstacle avoidance.

#### 6. 3x Stronger Crash Penalty (-500 -> -1500)
- Much stronger discouragement for collisions during training.

#### 7. 2.4x Stronger Proximity Penalties (-25 -> -60)
- Both obstacle_proximity and lidar_obstacle penalties increased to dominate near obstacles.

#### 8. Earlier Detection Zones (3.5m -> 5.0m danger, 1.5m -> 2.5m safe)
- Danger zone 43% wider: penalties start earlier.
- Safe distance 67% wider: avoidance margin increased.

#### 9. Emergency Speed Clamp (NEW, 2.0m trigger)
- Hardcoded speed reduction when LiDAR detects obstacles within 2.0m.
- Safety net independent of learned policy.

#### 10. Denser Obstacle Field (14 -> 18 trees)
- 4 new interior trees create tighter corridors and more complex navigation challenges.

#### 11. Reduced Subgoal Magnitude Penalty (-0.3 -> -0.1)
- Allows larger goal modifiers without excessive penalty, complementing the wider steering ranges.

---

## Changes: OA v2 -> OA v3

v3 fixes roll crashes in narrow corridors caused by yaw oscillation. When obstacles flank both sides, v2's aggressive yaw corrections caused rapid left-right oscillation leading to roll instability and crashes.

### Summary Table

| Parameter | OA v2 | OA v3 |
|-----------|-------|-------|
| max_yaw_offset | 0.5 rad (28deg) | **0.35 rad (20deg)** |
| max_yaw_rate_per_update | (none) | **0.1 rad per HL update** |
| Corridor detection | (none) | **Suppress yaw when both sides < 4.0m** |
| Corridor escape altitude | (none) | **+0.3 m/s upward boost** |
| emergency_brake_distance | 2.0 m | **3.5 m** |
| emergency_brake_min_factor | ~0.2 | **0.15** |
| goal_modifier_smoothing (danger) | 0.5 (fixed) | **0.2 (adaptive in danger zone)** |
| LiDAR noise | (none) | **HL: 0.02 std, LL: 0.01 std** |
| obstacle_min_tilt | (none) | **0.25 rad (14deg)** |
| Roll penalty (NAVIGATE) | 1x | **2x** |
| Everything else | (baseline) | (identical to v2) |

### Detailed Explanations

#### 1. Yaw Rate Limiting (NEW — max 0.1 rad per HL update)
- **Root cause**: In narrow corridors, the HL agent alternated between large positive and negative yaw offsets each update (oscillation). This whiplash destabilized the drone's attitude.
- **Fix**: Clamp yaw change to +-0.1 rad per HL update. The agent can still steer +-20 deg total, but must do so gradually over multiple updates.
- **Why 0.1 rad**: Fast enough for avoidance turns, slow enough to prevent oscillation-induced roll.

#### 2. Corridor Detection & Yaw Suppression (NEW)
- **Root cause**: When obstacles are on both sides (corridor), any yaw offset pushes the drone toward one wall. The agent panics and over-corrects toward the other wall.
- **Fix**: When both left AND right LiDAR detect obstacles within `corridor_danger_distance` (4.0m), suppress yaw offset entirely — the drone flies straight through.
- **Why this works**: In a corridor, the safest path is straight ahead. Trying to steer only makes things worse.

#### 3. Corridor Escape Altitude Boost (NEW — +0.3 m/s)
- **Root cause**: If the drone gets stuck in a corridor (both sides blocked, can't turn), it needs an escape route.
- **Fix**: Automatic +0.3 m/s upward velocity when corridor is detected. The drone rises above obstacle height to escape.

#### 4. Reduced Yaw Range (0.5 -> 0.35 rad)
- Balances steering authority vs stability. 20 deg is still enough for avoidance but less likely to trigger oscillation.

#### 5. Wider Emergency Brake Zone (2.0 -> 3.5m)
- Braking starts 75% earlier, giving more time to slow down before collision.

#### 6. Adaptive EMA Smoothing (0.5 base -> 0.2 in danger)
- **Root cause**: Fixed 0.5 alpha was too reactive in tight spaces, amplifying oscillation.
- **Fix**: When in danger zone, alpha drops to 0.2 — much heavier smoothing dampens rapid yaw changes.

#### 7. LiDAR Noise Injection (NEW)
- HL: std=0.02, LL: std=0.01
- Adds robustness to sensor noise. Prevents overfitting to perfect LiDAR readings.

#### 8. Dynamic Tilt Cap (obstacle_min_tilt = 0.25 rad)
- Tighter tilt limit (14 deg vs 28 deg baseline) when near obstacles.
- Prevents the PID from commanding aggressive tilts that could flip the drone in tight spaces.

#### 9. Doubled Roll Penalty in NAVIGATE Phase
- Extra discouragement for roll oscillation during the critical navigation phase.

### Problem Evolution Summary

```
OA v1: HL can SEE obstacles (LiDAR) but can't STEER away (too narrow yaw range)
   |
   v
OA v2: HL can steer (wider yaw) but OSCILLATES in corridors (too aggressive)
   |
   v
OA v3: Anti-oscillation (yaw rate limit + corridor detection + adaptive smoothing)
```

---

## PID Controller (Unchanged Across All Versions)

| Parameter | Value |
|-----------|-------|
| max_velocity_xy | 2.0 m/s |
| max_velocity_z | 1.0 m/s |
| max_yaw_rate | 1.5 rad/s |
| pid_vel_kp | 0.25 |
| pid_att_kp | 6.0 |
| pid_att_kd | 1.0 |
| pid_vz_kp | 0.5 |
| pid_yaw_kp | 0.4 |
| pid_max_tilt | 0.5 rad (28 deg) |
| thrust_to_weight | 1.8 |

---

## Checkpoints

| Version | Checkpoint Location | Training Steps | Notes |
|---------|--------------------|----------------|-------|
| Baseline | `multi_waypoint/logs/hierarchical_sac/...` | 16M | Original flat SAC |
| OA v1 | `multi_waypoint/logs/hierarchical_sac/...` | 16M | First hierarchical |
| OA v2 | `multi_waypoint/logs/hierarchical_sac/...` | 20M | Wider steering |
| OA v3 | `multi_waypoint/logs/hierarchical_sac/...` | 20M | Anti-oscillation |

---

## How to Reproduce

```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate

# Train OA v3:
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/multi_waypoint/multi_waypoint_sac_obstacle_avoidance_v3.py \
    --mode train --num_envs 512 --headless

# Play OA v3 (visual):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/multi_waypoint/multi_waypoint_sac_obstacle_avoidance_v3.py \
    --mode play \
    --checkpoint <path_to_checkpoint>

# Evaluate OA v3:
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/multi_waypoint/multi_waypoint_sac_obstacle_avoidance_v3.py \
    --mode eval \
    --checkpoint <path_to_checkpoint> \
    --num_episodes 100 --num_envs 16 --headless
```

---

## Key Lessons Learned

1. **Hierarchical control is essential** for combining waypoint navigation with obstacle avoidance. A flat policy couldn't handle both objectives.
2. **Goal modifier authority matters**: Too narrow (v1: +-0.15 rad) = can't avoid. Too wide (v2: +-0.5 rad) = oscillation. v3 found the balance at +-0.35 rad with rate limiting.
3. **Corridor detection is critical**: Generic obstacle avoidance fails in narrow passages. Detecting "flanked by both sides" and switching to "fly straight + rise" is more effective than trying to steer.
4. **Adaptive smoothing**: One EMA alpha doesn't fit all situations. Higher alpha (0.5) for open spaces, lower (0.2) for danger zones.
5. **Conservative modifier ranges + stabilization first**: Test 10's lesson — tighten everything first, then gradually widen only what's needed.
6. **Emergency brakes are safety nets, not strategies**: Hardcoded speed clamps catch what the RL agent misses, but the agent should learn proper avoidance.
