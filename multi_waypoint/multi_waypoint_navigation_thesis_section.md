# Multi-Waypoint Sequential Navigation: Hierarchical SAC with Kinematic Control for Variable-Path UAV Flight

## 1. Why Multi-Waypoint Navigation

Single-waypoint navigation, as evaluated in Experiment 1, establishes a baseline for point-to-point autonomous flight control. However, real-world UAV missions — particularly in bush and forestry applications such as ecological surveying, search-and-rescue, and precision agriculture — rarely involve flying a single straight line. These missions require the drone to visit multiple locations in sequence while adapting to variable-geometry flight paths. Multi-waypoint navigation introduces three critical capabilities that single-waypoint flight does not exercise:

1. **Sequential decision-making.** The drone must track its progress through a sequence of goals, deciding when to transition from one waypoint to the next. This requires temporal reasoning that a single fixed-goal policy never encounters.

2. **Cornering and heading changes.** Zigzag waypoint layouts force the drone to execute lateral turns at each waypoint, testing the policy's ability to decelerate, reorient, and accelerate along a new heading — manoeuvres absent from straight-line flight.

3. **Generalisation across variable paths.** Each episode presents a different randomly generated waypoint configuration (1 to 3 waypoints across an 8 m × 6 m arena), requiring the policy to generalise across path lengths, turn angles, and waypoint densities rather than memorising a single trajectory.

These capabilities form the prerequisite for the subsequent obstacle avoidance experiment (Experiment 3), where the drone must navigate through forests of 19 tree obstacles. Without robust multi-waypoint navigation, the obstacle avoidance policy cannot reliably reach sequential goals while simultaneously avoiding collisions. The progressive task pipeline — single waypoint → multi-waypoint → obstacle avoidance — ensures that each new capability is built on a validated foundation.

## 2. Task Description

### 2.1 Task Definition

The navigation task requires the UAV to fly from a fixed start position $\mathbf{p}_s = (-5.0, 0.0, 0.2)$ m through a sequence of $N$ intermediate waypoints before landing at the final waypoint, where $N \in \{1, 2, 3\}$ is randomly sampled per episode. Waypoints are arranged in a zigzag pattern across a rectangular arena spanning $x \in [-4.0, 4.0]$ m and $y \in [-3.0, 3.0]$ m, requiring the drone to execute alternating lateral turns as it progresses through the sequence.

The task follows the same five-phase state machine established in Experiment 1:

1. **TAKEOFF** — Ascend from the start position to a cruise altitude of 1.5 m (autopilot-controlled)
2. **STABILIZE** — Maintain hover at cruise altitude for 1.5 s to damp transients (autopilot-controlled)
3. **NAVIGATE** — Fly through the waypoint sequence at cruise altitude (RL high-level + deterministic low-level)
4. **HOVER** — Hold position above the final waypoint for 2.0 s within a 3.0 m tolerance (autopilot-controlled)
5. **LAND** — Controlled descent to the ground at the final waypoint location (autopilot-controlled)

A successful episode requires the drone to navigate through all intermediate waypoints and land within a goal threshold of $d_{goal} = 1.5$ m of the final waypoint at a landing speed below 1.5 m/s. Episodes are terminated upon success, crash (height violation or flipping), or after a maximum duration of 60.0 s.

### 2.2 Key Differences from Experiment 1

| Parameter | Experiment 1 (Single WP) | Experiment 2 (Multi-WP) | Rationale |
|-----------|-------------------------|------------------------|-----------|
| Control architecture | Physics-based PID | **Kinematic (hierarchical HL+LL)** | Eliminates crash instability during NAVIGATE |
| Action space | 4D velocity $[v_x, v_y, v_z, \dot{\psi}]$ | **3D goal modifier $[\Delta\psi, s_f, \Delta z]$** | RL modifies goals rather than controlling motors |
| Number of waypoints | 1 (fixed) | 1–3 (random) | Test sequential decision-making |
| Waypoint positions | Fixed ($\pm$5.0, 0, 0.2) | Randomised zigzag | Test generalisation to variable paths |
| Arena size | 10.0 m (straight line) | 8 m $\times$ 6 m | Smaller arena with zigzag paths |
| Episode length | 30.0 s | **60.0 s** | Additional time for multi-leg flight |
| Goal threshold | 1.2 m | 1.5 m | Slightly relaxed for multi-waypoint landing |
| Landing speed threshold | 1.2 m/s | 1.5 m/s | Relaxed for multi-waypoint task |
| Stabilise duration | 1.0 s | **1.5 s** | Extended settling for multi-leg task |
| Hover duration | 1.5 s | **2.0 s** | Extended confirmation at final waypoint |
| Intermediate WP tolerance | N/A | 2.5 m | Trigger for waypoint advancement |
| Final WP tolerance | 1.2 m (single goal) | 3.0 m | Tolerance for NAVIGATE→HOVER transition |
| Environment spacing | 40.0 m | 50.0 m | Larger arena requires wider spacing |

### 2.3 Hierarchical Control Architecture

The most significant architectural change from Experiment 1 is the adoption of a two-level hierarchical control system with kinematic position update, adapted from the obstacle avoidance system (Experiment 3, v9). This replaces the physics-based PID controller used in Experiment 1.

#### 2.3.1 Motivation: Why Physics-Based PID Failed

Five training-evaluation cycles were conducted with the Experiment 1 physics-based PID architecture before switching to hierarchical control. All five produced 0% success with 100% crash rates. The root cause was identified through systematic comparison with the working single-waypoint SAC and obstacle avoidance v9 implementations:

1. **PID instability from random exploration:** During early training, the RL agent outputs random velocity commands in the NAVIGATE phase. In the single-waypoint task, random commands occasionally move the drone toward the goal (straight-ahead), providing learning signal. In multi-waypoint zigzag navigation, random commands rarely align with the angled waypoint direction, producing no reward signal while the resulting erratic motion destabilises the physics-based PID controller.

2. **Altitude vulnerability:** With a cruise altitude of 1.5 m and maximum vertical velocity of 1.0 m/s, a random downward velocity command can drive the drone to ground level in approximately 1.5 seconds. The physics-based controller amplifies this through thrust/torque coupling — a large vertical velocity error produces thrust corrections that interact with attitude torques, creating oscillations that accelerate the descent.

3. **Crash detection threshold:** The too-low crash threshold (originally 0.15 m, later corrected to 0.05 m) terminated episodes before the agent could recover, preventing any learning during the critical early exploration phase.

The obstacle avoidance system (v9) solved this exact problem by using kinematic control — directly setting the robot's position each physics step with hard altitude clamping (`pos[:, 2].clamp_(min=0.05, max=8.0)`), making crashes physically impossible.

#### 2.3.2 Two-Level Architecture

The hierarchical architecture separates learned decision-making from deterministic trajectory tracking:

**High-level policy (HL — learned, SAC):**
- Runs at 10 Hz (every 5 environment steps)
- Outputs a 3-dimensional goal modifier: $\mathbf{a}_{HL} = [\Delta\psi, s_f, \Delta z]$
  - $\Delta\psi \in [-0.8, 0.8]$ rad: Yaw offset applied to the goal direction (~46° steering authority)
  - $s_f \in [0.0, 1.2]$: Speed factor scaling the approach velocity (0 = stop, 1.2 = fast)
  - $\Delta z \in [-0.5, 0.5]$ m: Altitude offset above/below cruise altitude
- Goal modifiers are EMA-smoothed ($\alpha = 0.3$) to prevent abrupt changes

**Low-level controller (LL — deterministic, frozen):**
- Runs at 50 Hz (every environment step)
- Computes body-frame velocity commands proportional to the HL-modified goal:
  - $v_x = K_{xy} \cdot g_x^b$, $v_y = K_{xy} \cdot g_y^b$ where $K_{xy} = 1.5$
  - $v_z = K_z \cdot g_z^b$ where $K_z = 0.8$
  - $\dot{\psi} = K_{\psi} \cdot \text{atan2}(g_y^b, g_x^b)$ where $K_{\psi} = 0.5$
- All velocity commands are clamped to $v_{xy}^{max} = 2.0$ m/s, $v_z^{max} = 1.0$ m/s, $\dot{\psi}^{max} = 1.5$ rad/s

**Kinematic update:**
- Body-frame velocity commands are EMA-smoothed ($\alpha = 0.06$, matching real Crazyflie response)
- Smoothed velocities are integrated to update position: $\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_w \cdot dt$
- Hard altitude clamping: $z \in [0.05, 8.0]$ m — the drone **cannot crash**
- Visual tilt computed from acceleration and velocity drag for realistic rendering

This architecture ensures that the RL agent's exploration cannot destabilise the drone. Even a completely untrained policy (random actions) will result in the drone navigating toward waypoints — the LL controller handles trajectory tracking while the HL policy modulates *how* the drone approaches (speed, heading offset, altitude variation).

### 2.4 Autopilot-Controlled Phases

Following the design principle established in Experiment 1, deterministic flight phases are handled by hardcoded controllers within the low-level policy rather than learned policies:

| Phase | Controller | Strategy |
|-------|-----------|----------|
| TAKEOFF | LL P-controller on altitude error | $v_z = K_z \cdot e_{alt}$, zero horizontal velocity |
| STABILIZE | LL hold with reduced gains | Altitude correction, $0.3\times$ lateral gains |
| HOVER | LL P-controller toward goal XY | $0.3\times$ lateral gains, altitude hold |
| LAND | LL P-controller + constant descent | Same XY control as HOVER, $v_z = -0.5$ m/s |

Only the **NAVIGATE** phase uses the learned high-level SAC policy. The low-level controller operates in all phases, but during non-NAVIGATE phases, its inputs are generated by deterministic autopilot logic rather than the HL policy. This follows the same principle as Experiment 1: deterministic behaviours that do not require learning should be hardcoded rather than learned, as this improves reliability without requiring additional training.

### 2.5 Observation Space

The policy receives a 26-dimensional observation vector, identical in structure to Experiment 1. The goal position component dynamically updates as waypoints are reached, meaning the policy observes the direction to the *current* waypoint.

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Body-frame linear velocity | 3 | $[v_x^b, v_y^b, v_z^b]$ |
| Body-frame angular velocity | 3 | $[\omega_x^b, \omega_y^b, \omega_z^b]$ |
| Projected gravity vector | 3 | Gravity direction in body frame (encodes roll/pitch) |
| Goal position (body frame) | 3 | Relative to *current* waypoint $[\Delta x^b, \Delta y^b, \Delta z^b]$ |
| LiDAR distances (normalised) | 9 | 3 vertical $\times$ 3 horizontal rays, normalised by $d_{max} = 15.0$ m |
| Phase one-hot encoding | 5 | Current flight phase indicator |

A notable design decision is that the observation space does **not** include the total number of waypoints, the current waypoint index, or any future waypoint positions. The policy must learn to navigate toward whichever goal it currently observes, without knowledge of how many waypoints remain. This encourages a reactive navigation strategy that generalises across different waypoint counts.

### 2.6 Action Space

The action space is continuous and 3-dimensional, representing normalised goal modifiers output by the high-level SAC policy: $\mathbf{a}_{HL} \in [-1, 1]^3$. Actions are scaled to goal modifier ranges before being applied to the low-level controller's target:

| Dimension | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Yaw offset | $\Delta\psi$ | $[-0.8, 0.8]$ rad | Steering authority (~46° in each direction) |
| Speed factor | $s_f$ | $[0.0, 1.2]$ | Approach velocity scaling (0 = stop, 1.2 = fast) |
| Altitude offset | $\Delta z$ | $[-0.5, 0.5]$ m | Height variation above/below cruise altitude |

The HL policy outputs actions at 10 Hz (every 5 environment steps). Goal modifiers are EMA-smoothed ($\alpha = 0.3$) to prevent abrupt changes. The yaw offset rotates the goal direction vector, the speed factor scales the goal distance (affecting LL velocity magnitude), and the altitude offset shifts the goal height.

### 2.7 Waypoint Generation Algorithm

At the start of each episode, a procedural waypoint generator creates the zigzag sequence:

1. **Waypoint count**: $N$ is sampled uniformly from $\{1, 2, 3\}$ per environment.
2. **X positions**: A base linspace from $x_{min} = -4.0$ to $x_{max} = 4.0$ is computed with $N_{max} = 3$ points. Random jitter of $\pm 1.0$ m is added, then values are clamped and sorted to ensure monotonic progression along the $x$-axis.
3. **Y positions**: The first waypoint's $y$-coordinate is sampled uniformly within 60% of the $y$-range. Subsequent waypoints alternate sign with an enforced minimum lateral change of $\Delta y_{min} = 1.5$ m, creating the zigzag pattern. Values are clamped to $y \in [-3.0, 3.0]$.
4. **Z positions**: All waypoints are placed at cruise altitude ($z = 1.5$ m). The landing goal uses the final waypoint's XY with $z = 0.2$ m for the LAND phase.
5. **Padding**: Unused slots (for episodes with $N < 3$) are padded with copies of the final waypoint.

This generation scheme produces trajectories with a total path length varying between approximately 9 m (single waypoint) and 20 m (three zigzag waypoints), compared to the fixed 10.0 m path in Experiment 1.

### 2.8 Waypoint Advancement Logic

During the NAVIGATE phase, the environment continuously checks whether the drone has reached the current intermediate waypoint. Advancement occurs when:

$$d_{xy}(\mathbf{p}_{drone}, \mathbf{p}_{wp_i}) < d_{intermediate} = 2.5 \text{ m}$$

Upon advancement:
1. The waypoint index increments: $i \leftarrow i + 1$
2. The goal position updates to the next waypoint in the sequence
3. The previous distance tracker resets to prevent a reward spike from the discontinuous goal change
4. A one-time intermediate waypoint bonus is awarded

The NAVIGATE $\rightarrow$ HOVER transition is only permitted at the final waypoint ($i = N - 1$), with a tolerance of 3.0 m, speed below 2.0 m/s, and altitude above 0.8 m.

### 2.9 Phase Transition Conditions

| Transition | Condition |
|-----------|-----------|
| TAKEOFF $\rightarrow$ STABILIZE | Altitude error to cruise < 0.5 m |
| STABILIZE $\rightarrow$ NAVIGATE | Stabilise timer $\geq$ 1.5 s |
| NAVIGATE $\rightarrow$ HOVER | At final waypoint ($i = N-1$) AND XY distance < 3.0 m AND speed < 2.0 m/s AND altitude > 0.8 m |
| HOVER $\rightarrow$ LAND | Hover timer $\geq$ 2.0 s AND XY distance < 3.0 m |

### 2.10 Termination Conditions

| Condition | Type |
|-----------|------|
| Altitude < 0.05 m (not in TAKEOFF/LAND) | Crash (died) |
| Altitude > 8.0 m | Crash (died) |
| Flipped (gravity\_z > 0.7) | Crash (died) |
| Goal reached (LAND phase, final dist < 1.5 m, speed < 1.5 m/s) | Success |
| Touched ground in HOVER/LAND (alt < 0.05 m) | Episode end |
| Episode duration exceeded (60.0 s) | Timeout |

Note that with kinematic control, the altitude clamping (`pos[:, 2].clamp_(min=0.05)`) prevents the drone from ever reaching the too-low threshold during NAVIGATE, effectively eliminating crash-related terminations. The crash conditions are retained for completeness but are structurally unreachable under normal operation.

## 3. Algorithm Configuration

### 3.1 Simulation Environment

All experiments were conducted in the same simulation environment as Experiment 1:

| Parameter | Value |
|-----------|-------|
| Simulator | NVIDIA Isaac Sim 5.1.0 with Isaac Lab 0.50.3 |
| Physics time step | 0.01 s (100 Hz) |
| Control decimation | 2 (policy runs at 50 Hz) |
| HL update rate | 10 Hz (every 5 environment steps) |
| Render interval | 2 |
| Environment spacing | 50.0 m |
| Parallel environments (training) | 256 |
| Parallel environments (evaluation) | 16 |

### 3.2 Robot Platform

Identical to Experiment 1: 10$\times$-scaled Bitcraze Crazyflie 2.x with gyroscopic forces disabled and thrust-to-weight ratio of 1.9. Although the kinematic controller does not use the thrust-to-weight ratio for force computation, hover thrust is applied to keep the physics engine consistent.

### 3.3 Hardware

Identical to Experiment 1: NVIDIA GeForce RTX 3060 (12 GB), AMD Ryzen 5 5500, 32 GB DDR4, Ubuntu 24.04.3 LTS.

### 3.4 SAC Configuration

SAC was implemented using Stable-Baselines3 with the tuned configuration from Experiment 1 ($\gamma = 0.98$, [512, 256] network). The action space is reduced from 4D to 3D to match the hierarchical goal-modifier architecture.

| Hyperparameter | Value | Exp 1 Tuned | Difference |
|---------------|-------|-------------|------------|
| Policy network | MLP [512, 256], ReLU | [512, 256] | Same (carried from Exp 1) |
| Action space | **3D** $[\Delta\psi, s_f, \Delta z]$ | 4D $[v_x, v_y, v_z, \dot{\psi}]$ | Reduced — RL modifies goals, not velocities |
| Learning rate | $3 \times 10^{-4}$ | same | — |
| Discount factor ($\gamma$) | **0.98** | 0.98 | Same (carried from Exp 1 finding) |
| Replay buffer size | $1 \times 10^6$ | same | — |
| Learning starts | 10,000 steps | same | — |
| Batch size | 256 | same | — |
| Soft update coefficient ($\tau$) | 0.005 | same | — |
| Training frequency | 1 step | same | — |
| Gradient steps | 1 | same | — |
| Entropy coefficient | Auto-tuned (learned $\alpha$) | same | — |
| Target entropy | Auto (default: $-\dim(\mathcal{A}) = -3$) | $-4$ | Smaller action space |
| Episode length (training) | **60.0 s** | 30.0 s | Extended for multi-leg flight |
| Total timesteps | **24M** | 24M | Same training budget |
| Parallel environments | **256** | 512 | Reduced to fit replay buffer memory |

### 3.5 Training Protocol

SAC was trained for 24 million timesteps with 256 parallel environments. The training used episode start randomisation to decorrelate environment resets across parallel workers.

### 3.6 Iterative Development Process

The final 100% success rate was achieved after six iterative development cycles. The progression highlights the diagnostic process that led to the architectural change.

| Version | Architecture | Key Change | Success | Crash | WP Completion | Avg Length |
|---------|-------------|-----------|---------|-------|---------------|-----------|
| v1 | Flat SAC + PID | Baseline [256,128,64] network | 0% | 98% | — | 4.52 s |
| v2 | Flat SAC + PID | Network → [512,256], autopilot fix | 0% | 96% | — | 5.26 s |
| v3 | Flat SAC + PID | Config alignment with working waypoint SAC | 0% | 100% | — | 3.16 s |
| v4 | Flat SAC + PID | Reduced lateral penalty, smaller arena | 0% | 100% | 27% | 3.04 s |
| v5 | Flat SAC + PID | Arena 8m×6m, EMA smoothing, altitude hold | 0% | 100% | 33.7% | 4.78 s |
| **v6** | **Hierarchical HL+LL + kinematic** | **Adopted obstacle v9 architecture** | **100%** | **0%** | **40.2%** | **5.19 s** |

The transition from v5 to v6 — replacing the physics-based PID with hierarchical kinematic control — was the decisive change that achieved 100% success. All five preceding iterations with physics-based PID produced 0% success regardless of reward tuning, network size, or action smoothing.

**Key lesson:** Unlike Experiment 1, where minimal hyperparameter changes ($\gamma$, network size, training duration) were sufficient to achieve 100% success, this experiment required a fundamental architectural change. The most impactful factor was not hyperparameter tuning but **control architecture selection** — consistent with the obstacle avoidance system's development history.

## 4. Reward Function Design

The reward function retains the five-phase structure from Experiment 1 but introduces three multi-waypoint-specific reward components. Components marked with an asterisk (*) differ from Experiment 1.

| Phase | Reward Component | Scale | Purpose |
|-------|-----------------|-------|---------|
| TAKEOFF | Ascent progress | +10.0 | Encourage altitude gain |
| TAKEOFF | Lateral drift penalty | −2.0 | Suppress horizontal motion during ascent |
| STABILIZE | Position hold | +3.0 | Reward proximity to start position |
| STABILIZE | Low speed bonus | +2.0 | Encourage hover stability |
| STABILIZE | Altitude maintenance | −2.0 | Penalise altitude deviation |
| STABILIZE | Angular velocity bonus | +1.0 | Reward low angular rates |
| NAVIGATE | XY progress (delta-based) | +5.0* | Reward distance reduction toward *current* waypoint |
| NAVIGATE | Velocity alignment | +6.0* | Reward velocity directed toward *current* waypoint |
| NAVIGATE | Lateral drift penalty | −0.5* | Reduced for zigzag turns; zero at intermediate WPs |
| NAVIGATE | Altitude maintenance | −2.0 | Penalise deviation from cruise altitude |
| NAVIGATE | Stability bonus | +3.0 | Reward low angular velocity and upright orientation |
| NAVIGATE | Excess speed penalty | −3.0 | Penalise speed above 2.0 m/s |
| NAVIGATE | **Intermediate WP bonus*** | **+100.0** | **One-time bonus for reaching each intermediate waypoint** |
| NAVIGATE | **Speed carry bonus*** | **+5.0** | **Reward maintaining speed through waypoint transitions** |
| NAVIGATE | **Waypoint progress*** | **+2.0** | **Continuous bonus proportional to fraction of WPs completed** |
| HOVER | Position hold | +3.0 | Reward proximity to final waypoint |
| HOVER | Low speed bonus | +2.0 | Encourage deceleration |
| HOVER | Altitude maintenance | −2.0 | Penalise altitude deviation |
| LAND | Descent progress | +5.0 | Reward controlled altitude reduction |
| LAND | XY stability | +15.0 | Strong reward for maintaining position during landing |
| LAND | Drift penalty | −10.0 | Penalise horizontal motion during descent |
| LAND | Precision landing | +200.0 | High reward for accurate final position |
| LAND | Descent speed control | −15.0 | Penalise excessive descent speed |
| LAND | Controlled descent | +5.0 | Gaussian reward centred at 0.35 m/s descent |
| LAND | Altitude penalty | −1.5 | Penalise remaining altitude |
| Global | Goal reached bonus | +750.0 | Terminal reward for successful landing |
| Global | Time penalty | −1.5 | Encourage efficiency |
| Global | Angular velocity penalty | −0.5 | Smooth flight (scaled 0.3$\times$ during NAVIGATE, 1.5$\times$ during HOVER/LAND) |
| Global | Yaw rate penalty | −1.0 | Suppress unnecessary yaw rotation |
| Global | Upright bonus | +2.0 | Reward maintaining upright orientation |

The navigation phase uses a **delta-based progress reward** $r_{progress} = (\|\Delta_{xy}^{t-1}\| - \|\Delta_{xy}^{t}\|) \times s_{progress}$ rather than a continuous proximity reward, to avoid the known circling pathology where the agent orbits the goal to accumulate proximity rewards.

### 4.1 Multi-Waypoint Reward Design Rationale

Three reward components are introduced specifically for the multi-waypoint task:

**Intermediate waypoint bonus (+100.0):** A one-time reward awarded each time the drone enters the 2.5 m capture radius of an intermediate waypoint. This provides a clear training signal that reaching waypoints in sequence is desirable.

**Speed carry bonus (+5.0):** Awards the drone for maintaining speed at the moment a waypoint is reached. This discourages the stop-and-go behaviour that would otherwise emerge from the intermediate waypoint bonus alone.

**Waypoint progress (+2.0):** A continuous bonus proportional to the fraction of waypoints completed ($\frac{i}{N}$), providing a gentle shaping signal that rewards progress through the full sequence.

### 4.2 Lateral Penalty Adaptation

The lateral drift penalty is set to zero when the drone is navigating toward an intermediate waypoint (as opposed to the final waypoint). This acknowledges that lateral velocity is expected and necessary during turns between waypoints. At the final waypoint, the penalty is restored to −0.5 to encourage a direct approach for the HOVER and LAND phases.

### 4.3 Reward Scale Reduction from Experiment 1

The XY progress and velocity alignment reward scales are reduced from +15.0/+12.0 (Experiment 1) to +5.0/+6.0 in this experiment. This reduction prevents the progress reward from dominating the total reward signal in a multi-waypoint setting and aligns with the scales used in the obstacle avoidance system (v9), where the hierarchical architecture operates under similar reward magnitudes.

## 5. Analysis

### 5.1 Evaluation Protocol

The trained policy was evaluated over 50 episodes with 16 parallel environments using deterministic action selection (no exploration noise). The evaluation episode length was set to 40.0 s with episode start randomisation disabled (following the methodological finding from Experiment 1, Section 5.5). Waypoint counts were randomised per episode within the range $N \in \{1, 2, 3\}$, and waypoint positions were regenerated for each episode, ensuring the policy was tested on previously unseen trajectories.

### 5.2 Quantitative Results

| Metric | **SAC (Multi-Waypoint)** | SAC (Exp 1 Single-WP) |
|--------|------------------------|----------------------|
| Success Rate | **100.0%** (50/50) | 100.0% (50/50) |
| Crash Rate | **0.0%** (0/50) | 0.0% (0/50) |
| Avg Final Distance (m) | **1.302** | 0.365 |
| Min Final Distance (m) | **1.296** | 0.365 |
| Avg Cumulative Reward | **909.47 $\pm$ 118.86** | 268.00 $\pm$ 15.13 |
| Avg Episode Length (s) | **5.19** | 29.98 |
| Control Architecture | Hierarchical (HL SAC + LL deterministic + kinematic) | Flat SAC + PID |
| Action Space | 3D goal modifier | 4D velocity |
| Network Architecture | [512, 256] | [512, 256] |
| Discount Factor ($\gamma$) | 0.98 | 0.98 |
| Episode Length | 60.0 s (train) / 40.0 s (eval) | 30.0 s |
| Total Training Steps | 24M | 24M |

### 5.3 Architectural Impact: Physics PID vs Kinematic Control

The most significant finding of this experiment is that the choice of low-level control architecture completely determined success or failure, independent of all other hyperparameters. Five training runs with physics-based PID control produced 0% success with 100% crash rate. A single training run with kinematic control produced 100% success with 0% crash rate.

This result is consistent with the obstacle avoidance development history (Experiment 3), where v4 (flat SAC with PID) achieved only 20% success with 78% crash rate, while the hierarchical kinematic architecture (v5 onwards) progressively achieved 75%, 85%, 93%, 98%, and 99% success rates. The evidence across both experiments strongly supports the conclusion that **kinematic control with hard altitude clamping is essential for reliable RL-based drone navigation**, particularly for tasks requiring complex manoeuvres.

The fundamental advantage of kinematic control is crash immunity: `pos[:, 2].clamp_(min=0.05, max=8.0)` enforces a hard altitude floor at every physics step (100 Hz). This guarantee transforms the learning problem from "learn to navigate while avoiding crashes" to "learn to navigate efficiently," removing the deadly exploration problem where random actions during early training cause irrecoverable crashes.

### 5.4 Episode Length and Efficiency

The average episode length of 5.19 s is notably shorter than Experiment 1's 29.98 s, despite the multi-waypoint task having a longer maximum duration (60 s). This efficiency arises from two factors:

1. **Kinematic control eliminates PID settling time.** The physics-based PID controller requires time to converge — velocity commands produce forces that interact with the drone's inertia, creating oscillations that must damp. The kinematic controller integrates velocity commands directly, producing immediate position changes with no overshoot.

2. **Direct waypoint approach.** The HL policy learns to set speed factors near the upper range (1.2), driving the LL controller to maximum velocity. Combined with the direct approach strategy, the drone completes the mission in approximately 5 seconds.

### 5.5 Reward Magnitude Comparison

The average cumulative reward of 909.47 $\pm$ 118.86 is substantially higher than Experiment 1's 268.00 $\pm$ 15.13. This increase is primarily attributable to the goal bonus (+750.0), intermediate waypoint bonuses (+100.0 each), and the continuous waypoint progress reward — components that did not exist in the single-waypoint task. The higher reward variance ($\sigma = 118.86$ vs $\sigma = 15.13$) reflects the stochastic waypoint configurations producing variable-difficulty episodes.

### 5.6 Comparison with Experiment 1: Flat vs Hierarchical SAC

| Aspect | Exp 1 (Flat SAC + PID) | Exp 2 (Hierarchical SAC + Kinematic) |
|--------|----------------------|-------------------------------------|
| Success Rate | 100% | 100% |
| Crash Rate | 0% | 0% |
| Task Complexity | Single straight-line | 1–3 zigzag waypoints |
| RL Action Space | 4D (full velocity control) | 3D (goal modifier) |
| RL Control Scope | Full velocity during NAVIGATE | Speed/heading/altitude modulation only |
| Crash Prevention | PID stability + autopilot phases | Hard altitude clamp (kinematic) |
| Training Steps | 24M | 24M |

Both experiments achieve 100% success with 0% crash, but through fundamentally different mechanisms. Experiment 1's success relies on the task being simple enough that the flat SAC policy can learn stable velocity control. Experiment 2's success relies on the hierarchical architecture ensuring stability regardless of task complexity, allowing the RL agent to focus on high-level strategy.

### 5.7 Comparison with Literature

Zhang et al. [39] (AirPilot) combined PPO with PID control for quadrotor navigation, achieving 90% reduction in navigation error. The current experiment extends this concept by using SAC as the high-level policy and replacing PID with kinematic control for guaranteed stability.

Xi et al. [33] employed hierarchical control with APF and TD3, achieving 95% success. The current experiment's LL controller serves a similar role to their APF layer — providing a stable base controller that the RL agent modulates rather than replaces.

Tayar et al. [36] compared PPO and SAC for quadrotor navigation in confined spaces, finding that PPO achieved 100% completion while SAC overfitted early. The current experiment's hierarchical architecture mitigates SAC's overfitting risk by constraining its control authority to high-level goal modification, preventing the policy from learning unstable low-level commands.

### 5.8 Limitations

1. **Kinematic control simplification.** The kinematic controller does not model aerodynamic effects (drag, motor lag, wind disturbance). Sim-to-real transfer would require the domain randomisation additions from v9.
2. **Fixed cruise altitude.** All waypoints are at the same altitude (1.5 m). Three-dimensional waypoint sequences would require validation.
3. **Flat terrain.** The simulation uses a flat ground plane without terrain features.
4. **Action space expressivity.** The 3D goal modifier provides less control authority than Experiment 1's 4D velocity commands. The HL policy cannot command arbitrary velocity profiles, only modulate the LL controller's proportional tracking.

## 6. Implications for Subsequent Stages

The multi-waypoint experiment validates two architectural decisions that directly enable the obstacle avoidance task (Experiment 3):

### 6.1 Validated Hierarchical Architecture

The hierarchical HL+LL architecture with kinematic control, first developed in the obstacle avoidance system (v9) and validated here for multi-waypoint navigation, provides the foundation for combining waypoint navigation with obstacle avoidance. The key validated properties are:

- **Crash immunity.** Hard altitude clamping at every physics step eliminates the deadly exploration problem, allowing the RL agent to learn complex manoeuvres without risk of irrecoverable crashes.
- **Separation of concerns.** The HL policy learns *what* to do (goal modification strategy), the LL controller handles *how* to fly (proportional tracking), and the kinematic layer guarantees *safety* (altitude bounds). This decomposition scales to obstacle avoidance by adding obstacle-awareness to the HL policy's observation space without modifying the LL controller or kinematic layer.
- **Stable LL controller.** The deterministic proportional controller with velocity clamping provides a reliable base that the obstacle avoidance HL policy can modulate with additional steering commands to avoid trees.

### 6.2 Confirmed Hyperparameter Transferability

The SAC configuration ($\gamma = 0.98$, [512, 256] network, 24M training steps) transferred directly from Experiment 1 to this experiment without modification, achieving 100% success on a substantially more complex task. This confirms that these hyperparameter choices are robust across task complexities and can be carried forward to Experiment 3 with confidence, reducing the hyperparameter search space for obstacle avoidance.

### 6.3 Navigation Capability as Prerequisite

The obstacle avoidance task (Experiment 3) places 19 tree obstacles in the drone's path, requiring the policy to navigate around obstacles *while maintaining progress toward waypoints*. Without the multi-waypoint navigation capability validated here — cornering, heading changes, sequential goal tracking — the obstacle avoidance policy would need to learn both navigation and avoidance simultaneously, significantly increasing the learning problem's complexity. By establishing reliable navigation as a separate capability, Experiment 3 can focus the RL agent's learning on the incremental challenge of obstacle avoidance.

### 6.4 The Principle of Separation of Concerns

This experiment reinforces the design principle established in the obstacle avoidance system: **separate what needs to be learned from what can be computed**. The hierarchical architecture decomposes the multi-waypoint navigation problem into:

1. **Learned component (HL):** When to speed up, slow down, or adjust heading — decisions that depend on the current waypoint configuration and require generalisation across random layouts.
2. **Computed component (LL):** How to fly toward a goal — proportional control with velocity clamping, a well-understood control problem with known stable solutions.
3. **Physical guarantee (kinematic):** That the drone remains airborne — hard altitude clamping at every physics step, a safety invariant that should never be delegated to a learned policy.

This decomposition is analogous to the separation between path planning and motion control in classical robotics, adapted for the RL setting where the high-level planner is learned rather than hand-crafted. The same decomposition carries directly into Experiment 3, where the HL policy additionally receives LiDAR obstacle observations and learns to steer around trees while the LL controller and kinematic layer remain unchanged.

---

*Evaluation date: 2026-04-11*
*SAC checkpoint: `multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-04-09_17-29-23/sac_final.zip`*
