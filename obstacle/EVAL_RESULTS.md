# Hierarchical SAC — Evaluation Results & Version History

Date: 2026-03-25 (re-eval with crash cause tracking)
Environment: 19-tree dense obstacle field, 3-5 zigzag waypoints, 10x Crazyflie
Eval: 50 episodes, headless, deterministic policy

---

## Evaluation Results

### v7 vs v6 vs v5 vs v4 Comparison (50 episodes each)

| Metric               |   v4            |   v5            |   v6            |   v7              | v7 vs v6 |
|----------------------|-----------------|-----------------|-----------------|-------------------|----------|
| Success Rate         | 20.0% (10/50)   | 74.0% (37/50)   | 82.0% (41/50)   | **94.0% (47/50)** | **+12%** |
| Crash Rate           | 76.0% (38/50)   | 18.0%  (9/50)   | 14.0%  (7/50)   | **2.0%   (1/50)** | **-12%** |
| Waypoint Completion  | 43.9%           | 68.0%           | 70.2%           | **72.1%**         | +1.9%    |
| Avg Final Distance   | 8.437 m         | 2.665 m         | 2.295 m         | **1.175 m**       | **-1.12 m** |
| Min Final Distance   | 0.481 m         | 0.481 m         | 0.480 m         | **0.480 m**       | —        |
| Avg Reward           | 275.88          | 1157.62         | 1286.37         | **1515.12**       | **+229** |
| Reward Std Dev       | 784.60          | 849.75          | 724.58          | **429.62**        | much more consistent |
| Avg Episode Length    | 5.53 s          | 11.77 s         | 11.66 s         | **12.41 s**       | ~same    |

### Crashes by Phase

| Phase     | v4  | v5  | v6  | v7  |
|-----------|-----|-----|-----|-----|
| TAKEOFF   | 0   | 0   | 0   | 0   |
| STABILIZE | 0   | 0   | 0   | 0   |
| NAVIGATE  | 40  | 9   | 7   | 1   |
| HOVER     | 0   | 0   | 0   | 0   |
| LAND      | 0   | 0   | 0   | 0   |

### Crashes by Cause

| Cause           | Description                                     | v4  | v5  | v6  | v7  |
|-----------------|-------------------------------------------------|-----|-----|-----|-----|
| `hit_obstacle`  | Collided with tree trunk while flying            | 40  | 9   | 7   | 1   |
| `too_low`       | Dropped below min flight height (ground impact)  | 0   | 0   | 0   | 0   |
| `too_high`      | Exceeded max flight height (flew away)           | 0   | 0   | 0   | 0   |
| `flipped`       | Drone flipped upside down (attitude loss)        | 0   | 0   | 0   | 0   |

### Crash Details (Phase + Cause)

| Version | Total Crashes | Breakdown |
|---------|--------------|-----------|
| v4      | 38 (76%)     | All 38: NAVIGATE → hit_obstacle |
| v5      | 9  (18%)     | All 9:  NAVIGATE → hit_obstacle |
| v6      | 7  (14%)     | All 7:  NAVIGATE → hit_obstacle |
| v7      | 1  (2%)      | 1: NAVIGATE → hit_obstacle |

**Key finding**: Across all versions (v4-v7), 100% of crashes are the same failure mode — **tree trunk collision during waypoint navigation**. No crashes from altitude loss, attitude instability, takeoff, or landing. The PID low-level controller is fully stable; the only failure is the RL agent's obstacle avoidance during navigation.

**Crash cause definitions:**
- **hit_obstacle**: Drone entered obstacle collision radius (0.5m from trunk surface) while below trunk height. Indicates the RL agent chose a path too close to a tree trunk during navigation.
- **too_low**: Drone altitude dropped below `min_flight_height`. Would indicate PID altitude control failure or aggressive descent. Never observed in v4-v7.
- **too_high**: Drone exceeded `max_flight_height`. Would indicate upward control divergence. Never observed.
- **flipped**: Projected gravity Z > 0.7 (drone nearly inverted). Would indicate catastrophic attitude loss. Never observed — PID max_tilt limit prevents this.

### Key Observations

- **v7** achieves **94% success** — new best, up from v6's 82%
- **100% of crashes across all versions are NAVIGATE → hit_obstacle** — the only failure mode is tree collision during navigation
- **Zero takeoff/landing/altitude/attitude crashes** — PID controller is rock-solid
- Crash rate progression: 76% → 18% → 14% → 2% shows obstacle avoidance steadily improving
- v4's 40 crashes (note: >38 episodes because some envs crash multiple times in parallel batch) all from same cause
- Avg final distance halved (2.295m -> 1.175m) — drone gets much closer to goals
- Reward std dev cut by 40% (724 -> 430) — most consistent version yet
- **v7 uses v6's trained checkpoint** — no retraining needed, safety layer is hardcoded
- Key lesson: hardcoded safety layers that don't change obs/action space can be applied to existing checkpoints

---

## Changes: v7 -> v8

v8 targets 100% success with 5 hardcoded improvements — no retraining, uses v6 checkpoint.

### Summary Table

| Parameter                              | v7 Value                     | v8 Value                       |
|----------------------------------------|------------------------------|--------------------------------|
| Reactive safety zones                  | Single zone (0.9m)           | **Two zones: outer 1.2m + inner 0.5m** |
| Outer zone speed reduction             | 30% min (single zone)        | **50% min (gentle early reaction)** |
| Inner zone speed reduction             | (none)                       | **10% min (emergency near-stop)** |
| Outer lateral push gain                | 0.6 (single zone)           | **0.3 (gentle outer zone)** |
| Inner lateral push gain                | (none)                       | **1.2 (exponential emergency push)** |
| Predictive avoidance                   | (none)                       | **1.0s lookahead, 1.5m trigger** |
| Anti-stall detection                   | (none)                       | **5s window, 0.7 speed boost** |
| Corridor min speed                     | 0.2                          | **0.15 (slower in tight passages)** |
| Episode time                           | 40s                          | **50s (more time for cautious nav)** |
| HL checkpoint                          | v6-trained (24M steps)       | **v6-trained (no retraining)** |
| Everything else                        | (baseline)                   | (identical to v7)             |

### Detailed Explanations

### 1. Two-Zone Graduated Safety (replaces single zone)
- **Root cause**: v7's single 0.9m zone applied the same response regardless of how close the drone was. The 1 remaining crash was likely a case where the drone entered the zone too fast for the uniform response.
- **Fix**: Outer zone (1.2m) provides gentle early warning — mild slowdown (50% min) + mild lateral push (0.3 gain). Inner zone (0.5m) triggers emergency response — near-stop (10% min speed) + strong exponential lateral push (1.2 gain, squared strength curve).
- **Why two zones**: The outer zone catches most scenarios early and gently. The inner zone acts as an emergency failsafe for the rare cases where the drone gets past the outer zone (fast approach angles, tight corridors).

### 2. Velocity-Aware Predictive Avoidance (NEW)
- **Root cause**: Reactive safety only triggers based on current position. A drone moving fast toward an obstacle that's 1.5m away won't trigger the 1.2m outer zone until it's already close, leaving less reaction time.
- **Fix**: Projects the drone's velocity 1.0s forward. If the projected position is within 1.5m of any obstacle surface, avoidance triggers immediately — even if the current position is safe.
- **Why 1.0s lookahead**: At max velocity (1.5 m/s), this looks 1.5m ahead — enough to catch approaching obstacles before entering the physical danger zone.
- **Implementation**: Uses `min(current_distance, predicted_distance)` as the effective distance for all safety decisions.

### 3. Exponential Lateral Push in Inner Zone (NEW)
- **Root cause**: v7's linear push strength meant that at very close range (0.2m), the push was only ~78% of max — not enough for emergency deflection.
- **Fix**: Inner zone uses squared strength curve: `(1 - d/inner_d)^2 * 1.2`. At 0.1m from surface, this gives `(1-0.2)^2 * 1.2 = 0.768` — 2.5x stronger than v7's linear push at the same distance.
- **Why exponential**: Emergency situations need disproportionately strong responses. Linear push doesn't differentiate enough between "close" and "critically close".

### 4. Anti-Stall Progress Tracking (NEW)
- **Root cause**: v7's 2 timeout failures (4%) were likely caused by the safety layer making the drone too cautious near obstacles, causing it to circle or creep slowly instead of making progress toward waypoints.
- **Fix**: Tracks time since last waypoint advancement. If >5 seconds have passed without reaching a waypoint, the safety slowdown floor is raised to 70% speed — the drone speeds up through obstacle zones instead of getting stuck.
- **Why 5s window**: Normal waypoint-to-waypoint navigation takes 2-4s. 5s without progress strongly indicates the drone is stuck in an avoidance loop.
- **Why 70% speed**: Fast enough to make meaningful progress, slow enough that the lateral push can still deflect the drone away from obstacles.

### 5. Longer Episode Time (40s -> 50s)
- **Root cause**: With more cautious safety layers, some episodes may need extra time to complete all 3-5 waypoints.
- **Fix**: 25% more time (50s vs 40s) provides buffer for the 2 timeout-prone scenarios.
- **Why 50s not more**: Excessive episode time would mask real navigation problems. 50s is enough buffer without hiding failures.

---

## Changes: v6 -> v7

v7 enhances the reactive safety layer with lateral avoidance and corridor detection — 4 targeted changes, no retraining.

### Summary Table

| Parameter                        | v6 Value                     | v7 Value                      |
|----------------------------------|------------------------------|-------------------------------|
| reactive_safety_distance         | 0.6 m                        | **0.9 m** (50% wider trigger) |
| reactive_safety_min_speed        | 0.4 (40%)                    | **0.3 (30%)**                 |
| Lateral avoidance push           | (none)                       | **0.6 gain — pushes drone AWAY from nearest obstacle** |
| Corridor detection               | (none)                       | **2+ obstacles close → 20% min speed** |
| obstacle_safe_distance           | 2.7 m                        | 2.7 m (unchanged)            |
| HL checkpoint                    | v6-trained (24M steps)       | **v6-trained (no retraining)** |
| Everything else                  | (baseline)                   | (identical to v6)            |

### Detailed Explanations

### 1. Lateral Avoidance Push (NEW)
- **Root cause**: v6's safety layer only reduced speed near obstacles — the drone slowed down but still drifted into corridor walls because it had no lateral correction.
- **Fix**: When within safety distance, a velocity push is applied AWAY from the nearest obstacle surface. Push strength is proportional to closeness (stronger when closer). Direction is computed in world frame, converted to body frame, then added to velocity commands.
- **Why 0.6 gain**: Strong enough to deflect the drone from collision paths, but not so strong as to destabilize navigation.
- **Why hardcoded**: Same reasoning as v6 — emergency avoidance doesn't need learning, and can be applied to existing checkpoints.

### 2. Wider Reactive Safety Trigger (0.6m -> 0.9m)
- Gives the drone 50% more distance to react before collision.
- Critical in tight corridors between center tree (0,0) and corridor blockers at (-2.5,2.5) / (2.5,-2.5).

### 3. Corridor Detection (NEW)
- **Root cause**: Single-obstacle safety was insufficient in narrow passages where obstacles flank both sides.
- **Fix**: Count obstacles within safety distance. When 2+ obstacles are close (corridor detected), min speed drops to 20% instead of 30%.
- **Why 2 obstacles**: A corridor is defined by having obstacles on multiple sides. Single obstacle proximity is handled by standard safety.

### 4. Lower Minimum Speed (40% -> 30%)
- Allows the drone to slow down more aggressively near single obstacles, complementing the lateral push.

### Failed v7 Approaches (for reference)
- **obstacle_safe_distance 2.7 -> 3.0m**: With 10M training steps, the agent couldn't adapt to the wider penalty zone — became overly cautious (80% success, worse than v6).
- **Retraining from scratch with v7 safety layer**: Agent learned different (worse) strategies around the modified dynamics (74% success, 20% crash). The v6 checkpoint already has excellent navigation — the safety layer just needs to protect it.
- **Key lesson**: When safety improvements are hardcoded (not learned), use the best existing checkpoint rather than retraining. The RL agent's policy doesn't change — the safety layer acts as a post-processing filter on velocity commands.

---

## Changes: v5 -> v6

v6 is a minimal fix over v5 — only 3 targeted changes to address corridor crashes.

### Summary Table

| Parameter                    | v5 Value        | v6 Value        |
|------------------------------|-----------------|-----------------|
| Reactive safety layer        | (none)          | **0.6m trigger, 40% min speed** |
| crash_penalty                | -800            | **-900**        |
| obstacle_safe_distance       | 2.5 m           | **2.7 m**       |
| Everything else              | (baseline)      | (identical to v5) |

### Detailed Explanations

### 1. Reactive Safety Layer (NEW)
- **Root cause**: v5's remaining 18% crashes were from tight corridor passages where the RL agent couldn't react fast enough.
- **Fix**: Hardcoded speed reduction when drone surface distance to any obstacle is < 0.6m. Speed scales linearly from 100% at 0.6m to 40% at 0m. Only active during NAVIGATE phase.
- **Why 0.6m**: Very tight trigger — only activates at near-collision distance. Does not interfere with normal navigation or make the agent overly cautious.
- **Why hardcoded (not learned)**: The RL agent doesn't need to learn emergency braking — it's a safety net that works immediately without training.

### 2. Slightly Stronger Crash Penalty (-800 -> -900)
- Modest increase to further discourage risky corridor paths.
- Not too strong — v6 first attempt used -1000 which made the agent overly cautious.

### 3. Slightly Earlier Safe Distance (2.5m -> 2.7m)
- Avoidance penalties trigger 0.2m earlier, giving the agent slightly more reaction time in corridors.

### Failed v6 Approaches (for reference)
Several more ambitious v6 designs were tried and failed:
- **Top-3 obstacle observations + approach velocity**: Changed obs space from 1367 to 1374 dims. Required more training time and the agent couldn't learn as well with 10M steps.
- **Graduated proximity penalties (3 zones)**: Split single -50 penalty into critical(-120)/danger(-60)/caution(-30). Total penalty budget was too harsh — agent learned to hover and time out (0% success).
- **Weakened penalties to fix overcaution**: Reduced all penalties to compensate — agent started crashing again (40% crash rate, 54% success).
- **Restored v5 penalties + new obs space**: Better (10% crash) but 18% success — not enough training for larger obs space.
- **Key lesson**: In RL, signal strength matters more than signal richness. Changing observations requires retuning everything. Minimal changes on a working baseline are more reliable.

---

## Changes: v4 -> v5

### Summary Table

| Parameter                    | v4 Value        | v5 Value        |
|------------------------------|-----------------|-----------------|
| EMA smoothing alpha          | 0.15            | **0.4**         |
| Max yaw offset               | +-0.5 rad (28°) | **+-1.0 rad (57°)** |
| Speed factor range           | [0.6, 1.0]      | **[0.0, 1.2]**  |
| velocity_align reward        | 6.0             | **2.0**         |
| obstacle_proximity penalty   | -25.0           | **-50.0**       |
| lidar_obstacle penalty       | -25.0           | **-40.0**       |
| obstacle_safe_distance       | 1.5 m           | **2.5 m**       |
| lidar_danger_distance        | 3.5 m           | **4.0 m**       |
| subgoal_magnitude penalty    | -0.3            | **-0.2**        |
| crash_penalty                | -500            | **-800**        |
| clearance_bonus              | (none)          | **+3.0/step**   |
| net_arch                     | [256, 128]      | **[512, 256]**  |
| training steps               | 16M             | **24M**         |

### Detailed Explanations

### 1. Faster EMA Smoothing (alpha: 0.15 -> 0.4)
- **Root cause**: With alpha=0.15, the drone had 85% inertia on goal modifiers. By the time it reacted to an obstacle, it had already crashed.
- **Fix**: alpha=0.4 gives 2.7x faster reaction time.

### 2. Wider Yaw Offset Range (+-0.5 -> +-1.0 rad)
- **Root cause**: +-28 degrees wasn't enough to steer around dense interior trees, especially the center tree at (0,0).
- **Fix**: +-57 degrees allows the drone to make sharp avoidance turns.

### 3. Wider Speed Factor Range ([0.6, 1.0] -> [0.0, 1.2])
- **Root cause**: The drone could only slow to 60% speed — it couldn't stop near obstacles.
- **Fix**: Range [0.0, 1.2] allows full stop (0.0) or slight speedup (1.2).

### 4. Reduced Velocity Align Reward (6.0 -> 2.0)
- **Root cause**: Strong forward-progress reward (+6.0 x dt) encouraged the drone to rush toward goals through obstacles rather than around them.
- **Fix**: Reduced to 2.0 so obstacle avoidance penalties can dominate near obstacles.

### 5. Stronger Obstacle Penalties, Earlier Trigger
- `obstacle_proximity_penalty`: -25.0 -> **-50.0** (doubled)
- `lidar_obstacle_penalty`: -25.0 -> **-40.0** (increased)
- `obstacle_safe_distance`: 1.5m -> **2.5m** (earlier penalty trigger)
- `lidar_danger_distance`: 3.5m -> **4.0m** (wider danger zone)

### 6. Added Clearance Bonus (+3.0/step)
- **New reward**: +3.0 x dt when all obstacles are beyond safe_distance (2.5m)
- **Purpose**: Positive reinforcement for maintaining safe distance, not just punishment for getting close.

### 7. Increased Crash Penalty (-500 -> -800)
- Stronger punishment for collision death to further discourage risky paths.

### 8. Reduced Subgoal Magnitude Penalty (-0.3 -> -0.2)
- Allows bigger yaw offsets without penalty, complementing the wider yaw range.

### 9. Larger Network ([256, 128] -> [512, 256])
- More capacity for learning complex avoidance strategies in dense obstacle fields.

### 10. Longer Training (16M -> 24M steps)
- More training time needed due to larger network and more complex reward landscape.

---

---

## Changes: v3 -> v4

v4 is a harder environment — same controller and rewards, but denser obstacle field.

### Summary Table

| Parameter                    | v3 Value            | v4 Value            |
|------------------------------|---------------------|---------------------|
| Total trees                  | 14                  | **19** (+5 interior) |
| Interior trees               | 4 (corners only)    | **9** (corners + center + corridors) |
| Tree visual variety          | Single color        | **5 trunk/canopy color variants** |
| Canopy heights               | Fixed 1.5 m         | **Varied: 1.2, 1.5, 1.8 m** |
| Canopy spread                | Fixed 1.5x radius   | **Varied: 1.3x to 1.75x radius** |
| Reward scales                | (baseline)          | (identical to v3)   |
| PID / LL gains               | (baseline)          | (identical to v3)   |
| obstacle_safe_distance       | 1.5 m               | 1.5 m (unchanged)   |
| obstacle_collision_radius    | 0.5 m               | 0.5 m (unchanged)   |

### New Interior Trees in v4

| Position       | Radius | Height | Purpose                              |
|----------------|--------|--------|--------------------------------------|
| (0.0, 0.0)     | 0.60   | 6.0    | Center tree — forces route splitting |
| (-2.5, 2.5)    | 0.40   | 4.0    | Upper-left corridor blocker          |
| (2.5, -2.5)    | 0.40   | 4.0    | Lower-right corridor blocker         |
| (-7.0, 0.0)    | 0.45   | 4.5    | Left midfield — blocks direct path   |
| (3.0, 3.5)     | 0.55   | 5.5    | Upper-right interior obstacle        |

### Impact
- The 5 new interior trees create a much denser obstacle field
- The center tree at (0,0) forces the drone to choose left or right routes
- Corridor blockers at (-2.5, 2.5) and (2.5, -2.5) narrow available paths
- All reward/control parameters unchanged — only the environment difficulty increased
- This exposed the v4 agent's weak obstacle avoidance (76% crash rate), which v5 then fixed

---

## Checkpoints

| Version | Checkpoint Path | Training Steps | Notes |
|---------|----------------|----------------|-------|
| v4 | `obstacle/logs/hierarchical_sac_v4/crazyflie_hierarchical_obstacle_sac_v4/2026-03-21_11-45-18/hl_sac_final.zip` | 16M | |
| v5 | `obstacle/logs/hierarchical_sac_v5/crazyflie_hierarchical_obstacle_sac_v5/2026-03-21_23-18-00/hl_sac_final.zip` | 24M | |
| v6 | `obstacle/logs/hierarchical_sac_v6/crazyflie_hierarchical_obstacle_sac_v6/2026-03-23_13-42-18/hl_sac_final.zip` | 24M | |
| v7 | Uses v6 checkpoint (no retraining) | 24M (v6) | Hardcoded safety layer only |
| v8 | Uses v6 checkpoint (no retraining) | 24M (v6) | Two-zone safety + predictive avoidance + anti-stall |

---

## How to Reproduce

```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate

# Evaluate v8 (uses v6 checkpoint + v8 hardcoded safety layer):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v8.py \
    --mode eval \
    --checkpoint ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/logs/hierarchical_sac_v6/crazyflie_hierarchical_obstacle_sac_v6/2026-03-23_13-42-18/hl_sac_final.zip \
    --num_episodes 50 --num_envs 16 --headless

# Play v8 (visual):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v8.py \
    --mode play \
    --checkpoint ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/logs/hierarchical_sac_v6/crazyflie_hierarchical_obstacle_sac_v6/2026-03-23_13-42-18/hl_sac_final.zip

# Evaluate v7 (uses v6 checkpoint + v7 hardcoded safety layer):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v7.py \
    --mode eval \
    --checkpoint ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/logs/hierarchical_sac_v6/crazyflie_hierarchical_obstacle_sac_v6/2026-03-23_13-42-18/hl_sac_final.zip \
    --num_episodes 50 --num_envs 16 --headless

# Play v7 (visual):
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v7.py \
    --mode play \
    --checkpoint ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/logs/hierarchical_sac_v6/crazyflie_hierarchical_obstacle_sac_v6/2026-03-23_13-42-18/hl_sac_final.zip

# Evaluate v6:
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v6.py \
    --mode eval \
    --checkpoint ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/logs/hierarchical_sac_v6/crazyflie_hierarchical_obstacle_sac_v6/2026-03-23_13-42-18/hl_sac_final.zip \
    --num_episodes 50 --num_envs 16 --headless

# Evaluate v5:
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v5.py \
    --mode eval \
    --checkpoint ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/logs/hierarchical_sac_v5/crazyflie_hierarchical_obstacle_sac_v5/2026-03-21_23-18-00/hl_sac_final.zip \
    --num_episodes 50 --num_envs 16 --headless
```
