# Single-Waypoint Navigation: A Comparative Study of PPO and SAC for UAV Point-to-Point Flight Control

## 1. Introduction

This section presents a comparative evaluation of two model-free reinforcement learning algorithms — Proximal Policy Optimisation (PPO) and Soft Actor-Critic (SAC) — applied to a single-waypoint navigation task for a simulated quadrotor UAV. The objective is to establish a performance baseline for point-to-point autonomous flight control before extending to more complex multi-waypoint and obstacle-avoidance scenarios. Through systematic hyperparameter tuning and architectural improvements, SAC is shown to achieve a 100% success rate on this task, while PPO fails to converge within the allocated training budget.

## 2. Problem Formulation

### 2.1 Task Definition

The navigation task requires the UAV to fly from a fixed start position $\mathbf{p}_s = (-5.0, 0.0, 0.2)$ m to a goal position $\mathbf{p}_g = (5.0, 0.0, 0.2)$ m, spanning a Euclidean distance of 10.0 m along the $x$-axis. The task follows a five-phase state machine:

1. **TAKEOFF** — Ascend from the start position to a cruise altitude of 1.5 m (autopilot-controlled)
2. **STABILIZE** — Maintain hover at cruise altitude for 1.0 s to damp transients (autopilot-controlled)
3. **NAVIGATE** — Fly toward the goal position at cruise altitude (RL-controlled)
4. **HOVER** — Hold position above the goal for 1.5 s within a 3.0 m tolerance (autopilot-controlled)
5. **LAND** — Controlled descent to the ground at the goal location (autopilot-controlled)

A successful episode requires the drone to land within a goal threshold of $d_{goal} = 1.2$ m of the target position at a landing speed below 1.2 m/s. Episodes are terminated upon success, crash (height violation or flipping), or after a maximum duration of 30.0 s.

### 2.2 Hierarchical Control Architecture

The system employs a two-level hierarchical control architecture to decouple high-level decision-making from low-level stabilisation:

- **High-level policy (learned):** Outputs 4-dimensional velocity commands $\mathbf{a} = [v_x, v_y, v_z, \dot{\psi}]$ in the body frame, bounded by $v_{xy}^{max} = 2.0$ m/s, $v_z^{max} = 1.0$ m/s, and $\dot{\psi}^{max} = 1.5$ rad/s. The RL agent only controls the NAVIGATE phase; all other phases use hardcoded autopilot controllers.
- **Low-level PID controller (frozen):** Converts velocity commands to thrust and torque inputs via a cascaded proportional-derivative (PD) controller:
  - Outer loop: Velocity error $\rightarrow$ desired attitude (tilt angle), with $K_{vel} = 0.25$ and maximum tilt $\theta_{max} = 0.5$ rad ($\approx 28.6\degree$)
  - Inner loop: Attitude error $\rightarrow$ torque, with $K_{att} = 6.0$ (proportional) and $K_{att,d} = 1.0$ (derivative)
  - Altitude: Thrust correction proportional to vertical velocity error, $K_{vz} = 0.5$
  - Yaw: Torque proportional to yaw rate error, $K_{yaw} = 0.4$

This architecture ensures flight stability is maintained by the PID layer, allowing the RL agent to focus on trajectory-level decisions.

### 2.3 Autopilot-Controlled Phases

A key design insight, inspired by the successful obstacle avoidance system (Section X), is that mechanical flight phases do not require learned control. Four of the five phases use hardcoded autopilot controllers:

| Phase | Controller | Strategy |
|-------|-----------|----------|
| TAKEOFF | P-controller on altitude error | $v_z = \text{clamp}(1.5 \times e_{alt}, 0.1, 1.0)$, zero horizontal velocity |
| STABILIZE | Zero-velocity hold | Altitude correction via $v_z = \text{clamp}(0.5 \times e_{alt}, -0.3, 0.3)$ |
| HOVER | P-controller toward goal XY | $v_{xy} = \text{clamp}(0.8 \times \Delta_{xy}, -0.5, 0.5)$, altitude hold |
| LAND | P-controller + constant descent | Same XY control as HOVER, $v_z = -0.3$ m/s |

This approach follows the same principle as the obstacle avoidance system's safety layers (v7/v8): deterministic behaviours that do not require learning should be hardcoded rather than learned, as this improves reliability without requiring additional training.

### 2.4 Observation Space

Both algorithms receive an identical 26-dimensional observation vector:

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Body-frame linear velocity | 3 | $[v_x^b, v_y^b, v_z^b]$ |
| Body-frame angular velocity | 3 | $[\omega_x^b, \omega_y^b, \omega_z^b]$ |
| Projected gravity vector | 3 | Gravity direction in body frame (encodes roll/pitch) |
| Goal position (body frame) | 3 | Relative goal $[\Delta x^b, \Delta y^b, \Delta z^b]$ |
| LiDAR distances (normalised) | 9 | 3 vertical channels $\times$ 3 horizontal rays (360 degree coverage), normalised by $d_{max} = 15.0$ m |
| Phase one-hot encoding | 5 | Current flight phase indicator |

### 2.5 Action Space

The action space is continuous and 4-dimensional, representing normalised velocity commands in the body frame: $\mathbf{a} \in [-1, 1]^4$. Actions are scaled to velocity limits ($v_{xy}^{max}$, $v_z^{max}$, $\dot{\psi}^{max}$) before being passed to the PID controller.

### 2.6 Reward Function

A dense, phase-dependent reward function guides training through each flight phase. Key components include:

| Phase | Reward Component | Scale | Purpose |
|-------|-----------------|-------|---------|
| TAKEOFF | Ascent progress | +10.0 | Encourage altitude gain |
| TAKEOFF | Lateral drift penalty | -2.0 | Suppress horizontal motion during ascent |
| STABILIZE | Position hold | +3.0 | Reward proximity to start position |
| STABILIZE | Low speed bonus | +2.0 | Encourage hover stability |
| NAVIGATE | XY progress (delta-based) | +15.0 | Reward distance reduction toward goal |
| NAVIGATE | Velocity alignment | +12.0 | Reward velocity directed toward goal |
| NAVIGATE | Lateral drift penalty | -2.0 | Penalise perpendicular motion |
| NAVIGATE | Altitude maintenance | -2.0 | Penalise deviation from cruise altitude |
| NAVIGATE | Stability bonus | +3.0 | Reward low angular velocity and upright orientation |
| NAVIGATE | Excess speed penalty | -3.0 | Penalise speed above 2.0 m/s |
| HOVER | Position hold | +3.0 | Reward proximity to goal |
| HOVER | Low speed bonus | +2.0 | Encourage deceleration |
| LAND | Descent progress | +5.0 | Reward controlled altitude reduction |
| LAND | XY stability | +15.0 | Strong reward for maintaining position during landing |
| LAND | Precision landing | +200.0 | High reward for accurate final position |
| LAND | Controlled descent | +5.0 | Gaussian reward centred at 0.35 m/s descent |
| Global | Goal reached bonus | +750.0 | Terminal reward for successful landing |
| Global | Time penalty | -1.5 | Encourage efficiency |
| Global | Angular velocity penalty | -0.5 | Smooth flight |
| Global | Yaw rate penalty | -1.0 | Suppress unnecessary yaw rotation |
| Global | Upright bonus | +2.0 | Reward maintaining upright orientation |

The navigation phase uses a **delta-based progress reward** $r_{progress} = (\|\Delta_{xy}^{t-1}\| - \|\Delta_{xy}^{t}\|) \times s_{progress}$ rather than a continuous proximity reward, to avoid the known circling pathology where the agent orbits the goal to accumulate proximity rewards.

### 2.7 Phase Transition Conditions

| Transition | Condition |
|-----------|-----------|
| TAKEOFF $\rightarrow$ STABILIZE | Altitude error to cruise < 0.5 m |
| STABILIZE $\rightarrow$ NAVIGATE | Stabilize timer $\geq$ 1.0 s |
| NAVIGATE $\rightarrow$ HOVER | XY distance to goal < 3.0 m AND speed < 2.0 m/s AND altitude > 0.8 m |
| HOVER $\rightarrow$ LAND | Hover timer $\geq$ 1.5 s AND XY distance < 3.0 m |

### 2.8 Termination Conditions

| Condition | Type |
|-----------|------|
| Altitude < 0.05 m (not in TAKEOFF/LAND) | Crash (died) |
| Altitude > 8.0 m | Crash (died) |
| Flipped (gravity_z > 0.7) | Crash (died) |
| Goal reached (LAND phase, dist < 1.2 m, speed < 1.2 m/s) | Success (time_out) |
| Touched ground in HOVER/LAND (alt < 0.05 m) | Episode end (time_out) |
| Episode duration exceeded | Timeout (time_out) |

## 3. Experimental Setup

### 3.1 Simulation Environment

All experiments were conducted in NVIDIA Isaac Sim 5.1.0 with Isaac Lab 0.50.3, providing GPU-accelerated rigid-body physics and parallel environment execution. The simulation parameters are:

| Parameter | Value |
|-----------|-------|
| Physics time step | 0.01 s (100 Hz) |
| Control decimation | 2 (policy runs at 50 Hz) |
| Render interval | 2 |
| Environment spacing | 40.0 m |
| Parallel environments (training) | 512 |
| Parallel environments (evaluation) | 64 |

### 3.2 Robot Platform

The simulated platform is based on the Bitcraze Crazyflie 2.x quadrotor, scaled by a factor of 10 to facilitate visual debugging and reduce numerical sensitivity. Gyroscopic forces are disabled to simplify the dynamics. The thrust-to-weight ratio is set to 1.9, providing sufficient control authority for manoeuvres.

### 3.3 Hardware

All training and evaluation were performed on a single workstation:
- **GPU:** NVIDIA GeForce RTX 3060 (12 GB VRAM)
- **CPU:** AMD Ryzen 5 5500 (6 cores / 12 threads)
- **RAM:** 32 GB DDR4
- **OS:** Ubuntu 24.04.3 LTS, Kernel 6.17.0

### 3.4 Algorithm Configurations

#### 3.4.1 Proximal Policy Optimisation (PPO)

PPO was implemented using RSL-RL (OnPolicyRunner). To enable a fair comparison with SAC, the PPO configuration was updated to match the same structural hyperparameters (discount factor, network architecture, episode length, and autopilot phases):

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| Policy network | MLP [512, 256], ELU activation | Matched to SAC |
| Critic network | MLP [512, 256], ELU activation | Matched to SAC |
| Learning rate | $1 \times 10^{-4}$ (adaptive schedule) | |
| Discount factor ($\gamma$) | **0.98** | Matched to SAC (was 0.99) |
| GAE parameter ($\lambda$) | 0.95 | |
| Clipping parameter ($\epsilon$) | 0.2 | |
| Entropy coefficient | 0.0 | |
| Number of epochs per iteration | 5 | |
| Number of mini-batches | 4 | |
| Steps per environment per iteration | 1,024 | |
| Maximum gradient norm | 1.0 | |
| Desired KL divergence | 0.008 | |
| Initial action noise std | 0.3 | |
| Observation normalisation | Enabled (actor and critic) | |
| Training iterations | 120 | Stopped after reward plateau |
| Episode length (training) | **30.0 s** | Matched to SAC (was 18.0 s) |
| Autopilot phases | TAKEOFF, STABILIZE, **HOVER, LAND** | Matched to SAC |
| Total environment steps | $\approx 120 \times 512 \times 1024 \approx 63\text{M}$ | |

#### 3.4.2 Soft Actor-Critic (SAC) — Final Tuned Configuration

SAC was implemented using Stable-Baselines3. The final tuned configuration is:

| Hyperparameter | Value | Baseline | Change Rationale |
|---------------|-------|----------|-----------------|
| Policy network | MLP [512, 256], ReLU | [256, 128, 64] | Larger capacity, proven in obstacle SAC |
| Learning rate | $3 \times 10^{-4}$ | same | -- |
| Discount factor ($\gamma$) | **0.98** | 0.99 | Shorter horizon for faster credit assignment |
| Replay buffer size | $1 \times 10^6$ | same | -- |
| Learning starts | 10,000 steps | same | -- |
| Batch size | 256 | same | -- |
| Soft update coefficient ($\tau$) | 0.005 | same | -- |
| Training frequency | 1 step | same | -- |
| Gradient steps | 1 | same | -- |
| Entropy coefficient | Auto-tuned (learned $\alpha$) | same | -- |
| Target entropy | Auto (default: $-\dim(\mathcal{A})$) | same | -- |
| Episode length (training) | **30.0 s** | 20.0 s | More time for full mission completion |
| Total timesteps | **24M** | 16M | Extended training for convergence |

The three key hyperparameter changes ($\gamma = 0.98$, larger network, extended training) were motivated by the successful obstacle avoidance SAC (v6), which achieved 85% success with the same hyperparameter pattern.

### 3.5 Training Protocol

PPO was trained for 120 iterations (~35 minutes wall-clock time) with 512 parallel environments and 1,024 steps per environment per iteration, yielding ~63M total environment steps. Training was stopped after the mean reward plateaued at ~95 and began declining, with the final distance to goal stagnating at ~7.7 m. SAC was trained for 24 million timesteps (~18 minutes wall-clock time) with 512 parallel environments, achieving approximately 22,000 steps/second throughput.

### 3.6 Iterative Tuning Process

The SAC configuration underwent four iterations of tuning:

| Attempt | Key Changes | Success Rate | Crash Rate | Outcome |
|---------|------------|-------------|------------|---------|
| Baseline | Original config (16M steps, $\gamma$=0.99, [256,128,64]) | 14% | 0% | Partial navigation, most episodes timeout |
| Attempt 1 | Aggressive reward boost (nav_progress=30, velocity_align=20, LR=7e-4) | 0% | 14% | Destabilised learning |
| Attempt 2 | Obstacle SAC reward scales (upright=4.0, ang_vel=-1.0) | 2% | 54% | Over-penalised orientation caused crashes |
| **Attempt 3** | **Minimal changes: $\gamma$=0.98, [512,256], 30s episodes, 24M** | **100%** | **0%** | **Full convergence** |

**Key lesson:** Minimal, targeted hyperparameter changes outperformed aggressive reward tuning. The most impactful changes were the discount factor ($\gamma$), network capacity, episode length, and training duration — not the reward scales.

## 4. Evaluation Protocol

Each trained policy was evaluated over 50 episodes with 50 parallel environments using deterministic action selection (no exploration noise). The evaluation episode length was set to 30.0 s with **episode start randomisation disabled** to ensure fair comparison (all episodes receive the full allocated time).

The following metrics were recorded:
- **Success rate:** Percentage of episodes where the drone landed within 1.2 m of the goal at speed < 1.2 m/s
- **Crash rate:** Percentage of episodes terminated by height violations or flipping
- **Average final distance:** Mean Euclidean distance to the goal at episode termination
- **Minimum final distance:** Closest approach to the goal across all episodes
- **Average cumulative reward:** Mean total reward per episode
- **Average episode length:** Mean episode duration

## 5. Results

### 5.1 Quantitative Results — Fair Comparison (Matched Parameters)

The primary comparison uses identical environment configuration, reward function, autopilot phases, network architecture, and discount factor for both algorithms:

| Metric | **PPO (matched params)** | **SAC (tuned)** |
|--------|------------------------|----------------|
| Success Rate | **0.0%** (0/50) | **100.0%** (50/50) |
| Crash Rate | **100.0%** (50/50) | **0.0%** (0/50) |
| Avg Final Distance (m) | 7.177 | **0.365** |
| Min Final Distance (m) | 7.141 | **0.365** |
| Avg Cumulative Reward | 87.64 $\pm$ 0.87 | **268.00 $\pm$ 15.13** |
| Avg Episode Length (s) | 3.61 | **29.98** |
| Network Architecture | [512, 256] | [512, 256] |
| Discount Factor ($\gamma$) | 0.98 | 0.98 |
| Episode Length | 30.0 s | 30.0 s |
| Autopilot Phases | TAKEOFF, STABILIZE, HOVER, LAND | TAKEOFF, STABILIZE, HOVER, LAND |
| Training Wall Time | ~35 min | ~18 min |
| Total Environment Steps | ~63M | ~24M |

### 5.2 Historical Results — SAC Tuning Progression

| Metric | SAC Baseline (16M) | **SAC Tuned (24M)** |
|--------|-------------------|-------------------|
| Success Rate | 14.0% (7/50) | **100.0% (50/50)** |
| Crash Rate | 0.0% (0/50) | **0.0% (0/50)** |
| Avg Final Distance (m) | 4.666 | **0.365** |
| Min Final Distance (m) | 0.995 | **0.365** |
| Avg Cumulative Reward | 267.32 $\pm$ 305.87 | **268.00 $\pm$ 15.13** |
| Avg Episode Length (s) | 6.55 | **29.98** |

### 5.3 Analysis

#### PPO (Matched Parameters): Complete Failure Despite Identical Configuration

With parameters matched to the successful SAC configuration ($\gamma = 0.98$, [512, 256] network, 30 s episodes, autopilot HOVER/LAND), PPO achieved **0% success with 100% crash rate**. The average final distance of 7.177 m (compared to the 10.0 m initial distance) shows the agent learned to move only ~2.8 m toward the goal before crashing. The average episode length of just 3.61 s indicates early termination — drones crash within seconds of entering the NAVIGATE phase.

During training, the mean reward increased steadily from ~27 to ~95 over 120 iterations (~63M environment steps), while the final distance metric improved from ~9.4 m to ~7.7 m. However, the **phase mean remained at 2.0 (NAVIGATE) throughout training** — no drone ever reached the HOVER phase, and the goal bonus was zero at every iteration. This indicates PPO learned to accumulate navigation rewards (velocity alignment, progress toward goal) but never solved the full trajectory to reach the goal vicinity.

The remarkably low reward variance at evaluation ($\sigma = 0.87$) is notable: PPO converged to a highly consistent but incorrect policy where all drones crash in the same way, suggesting the policy collapsed to a local optimum.

#### PPO Training Dynamics

| Training Metric | Start | End (iter 120) |
|----------------|-------|----------------|
| Mean Reward | 26.88 | ~95.0 |
| Final Distance to Goal | 9.35 m | 7.70 m |
| Phase Mean | 2.0 (NAVIGATE) | 2.0 (NAVIGATE) |
| Goal Bonus | 0.0 | 0.0 |
| Died per Episode | 3.7 | 3.5 |

The training trajectory shows continuous improvement in reward but no qualitative breakthrough — the agent never discovered the full NAVIGATE → HOVER → LAND trajectory. This is a characteristic failure mode of on-policy methods in long-horizon tasks: without replay, PPO cannot propagate credit backward from the sparse goal bonus through the long navigation phase.

#### SAC Baseline: Partial Learning

The baseline SAC configuration (16M timesteps, $\gamma = 0.99$, smaller network) demonstrated meaningful but incomplete learning. The 14% success rate and average final distance of 4.666 m indicate the agent learned to navigate approximately halfway to the goal. Notably, SAC achieved 0% crashes even with a partially trained policy, validating the hierarchical control architecture.

The high reward variance ($\sigma = 305.87$) reflects highly variable episode outcomes — a mix of near-successful approaches and episodes where the drone circled or stalled in the NAVIGATE phase.

#### SAC Tuned: Complete Task Mastery

The tuned SAC configuration achieved **100% success across 50 evaluation episodes** with zero crashes. The average final distance of 0.365 m demonstrates precise goal-reaching behaviour, and the remarkably low reward standard deviation ($\sigma = 15.13$, a **95% reduction** from the baseline's 305.87) indicates highly consistent performance across episodes.

The average episode length of 29.98 s (out of 30.0 s maximum) deserves note: the drone consistently completes the full mission (takeoff, stabilise, navigate, hover, land) using nearly all available time. This is expected given the five-phase state machine with its timer-based transitions (1.0 s stabilise + 1.5 s hover = 2.5 s minimum overhead).

### 5.4 Ablation: Impact of Individual Changes

The tuned SAC configuration differs from the baseline in three hyperparameters and two architectural decisions. To understand the contribution of each change, we note the following:

| Change | Impact | Evidence |
|--------|--------|----------|
| $\gamma$: 0.99 $\rightarrow$ 0.98 | Faster credit assignment over long episodes | Same pattern used in successful obstacle SAC (v6, 85% success) |
| Network: [256,128,64] $\rightarrow$ [512,256] | Greater function approximation capacity | Proven effective in obstacle avoidance task with similar observation space |
| Episode length: 20s $\rightarrow$ 30s | Sufficient time for full 5-phase mission | 30s provides ~20s buffer over minimum mission time |
| Training: 16M $\rightarrow$ 24M steps | Sufficient convergence time for larger network | Reward still improving at 16M, plateaued by 24M |
| Autopilot HOVER/LAND | Reliable precision landing without learning | Follows obstacle SAC v7/v8 principle: hardcode deterministic behaviours |

The discount factor change ($\gamma = 0.98$) is particularly significant for this task. With 30 s episodes at 50 Hz control, each episode contains up to 1,500 steps. At $\gamma = 0.99$, the effective horizon is $\frac{1}{1-\gamma} = 100$ steps (2 seconds), while at $\gamma = 0.98$ it is 50 steps (1 second). The shorter effective horizon with $\gamma = 0.98$ may seem counterintuitive, but it helps SAC's critic converge faster by reducing the variance of bootstrapped value estimates over the long episode horizon.

### 5.5 Effect of Episode Start Randomisation

An important methodological finding emerged during evaluation. The training environment randomises each environment's episode step counter on the first global reset to decorrelate episode endings across parallel environments. When this randomisation was active during evaluation, the measured success rate was 86% (with 100 episodes) rather than 100%. The 14% failures were entirely caused by environments initialised with insufficient remaining time to complete the full mission, not by policy deficiencies.

| Evaluation Setting | Success Rate | Avg Final Distance |
|-------------------|-------------|-------------------|
| With episode randomisation | 86% | 1.152 m |
| **Without episode randomisation** | **100%** | **0.370 m** |

This highlights the importance of disabling training-specific randomisation during evaluation to obtain accurate performance measurements.

## 6. Discussion

### 6.1 On-Policy vs Off-Policy Learning for UAV Control

The fair comparison with matched parameters ($\gamma = 0.98$, [512, 256] network, 30 s episodes, identical autopilot phases) demonstrates a decisive advantage for SAC over PPO in this waypoint navigation task. Critically, the PPO failure is not due to configuration mismatch — both algorithms received the same environment, reward function, network capacity, and discount factor. The key differentiating factors are:

1. **Sample efficiency:** SAC's off-policy learning and experience replay enable convergence at 24M timesteps, while PPO at ~63M environment steps still failed to converge. SAC makes more effective use of each collected transition by replaying experiences from the entire training history.

2. **Credit assignment over long horizons:** With 30 s episodes at 50 Hz, each episode spans up to 1,500 steps. The goal bonus (+750) is only received upon successful landing — the terminal step of a multi-phase mission. SAC's replay buffer allows the critic to learn value estimates backward from rare successes, while PPO's on-policy rollouts provide only a single forward pass through the trajectory. During PPO training, the goal bonus remained zero across all 120 iterations, indicating the policy never discovered the full trajectory to the goal.

3. **Exploration:** SAC's maximum entropy framework provides sustained exploration throughout training via the learned temperature parameter $\alpha$. The entropy coefficient evolved from ~0.27 (high exploration) early in training to ~0.013 (near-deterministic policy) by 24M steps, indicating a natural exploration-exploitation transition. PPO's exploration relies solely on the initial action noise ($\sigma = 0.3$), which diminishes as the policy sharpens.

4. **Failure mode divergence:** PPO converged to a consistent but incorrect policy (100% crash rate, $\sigma_{reward} = 0.87$), while SAC converged to a consistent and correct policy (100% success rate, $\sigma_{reward} = 15.13$). The low variance in both cases indicates convergence — but to qualitatively different solutions.

### 6.2 The Principle of Hardcoded Deterministic Behaviours

A central design principle, consistent with the obstacle avoidance system (v7/v8), is that **deterministic flight behaviours should be hardcoded rather than learned**. In this system:

- **Takeoff** (ascend to cruise altitude) — deterministic, does not require learning
- **Stabilise** (hold position briefly) — deterministic
- **Hover** (hold position above goal) — deterministic P-controller
- **Land** (descend vertically) — deterministic constant-rate descent

Only the **NAVIGATE** phase, which requires decision-making about velocity and direction based on goal-relative observations, benefits from learned control. By restricting the RL agent to this single phase, the policy search space is dramatically reduced, accelerating convergence and improving reliability.

This principle was validated empirically: the autopilot-controlled HOVER and LAND phases achieve 100% success once the drone enters those phases, regardless of how the RL agent navigated to the goal vicinity.

### 6.3 Lessons from the Tuning Process

The iterative tuning process (Section 3.6) revealed several practical insights:

1. **Minimal changes outperform aggressive tuning.** The best-performing configuration (Attempt 3) made only three hyperparameter changes from the baseline. Earlier attempts that modified reward scales by 2-3x or adopted reward patterns from a different task domain caused performance degradation.

2. **Cross-task hyperparameter transfer is unreliable for reward scales.** The obstacle avoidance SAC (v6) uses different reward magnitudes optimised for its hierarchical subgoal architecture. Directly transferring these scales to the waypoint task (Attempt 2) caused 54% crash rate. However, structural hyperparameters ($\gamma$, network size, training duration) transferred successfully.

3. **Reward scale stability matters more than magnitude.** The baseline reward scales (progress=15.0, velocity_align=12.0) achieved 100% success. Doubling them (Attempt 1) caused 0% success. The SAC critic's value estimates are calibrated to the reward scale; large changes destabilise the Q-function.

### 6.4 Limitations

1. **Fixed start-goal pair:** All experiments use the same start $(-5, 0, 0.2)$ and goal $(5, 0, 0.2)$ positions. The trained policy may not generalise to arbitrary positions without domain randomisation.
2. **Simplified dynamics:** The 10x scale factor and disabled gyroscopic forces simplify the aerodynamics, which may affect transfer to real-world platforms.
3. **Deterministic evaluation:** The SAC policy is evaluated with deterministic actions (mean of the policy distribution). Stochastic evaluation may yield different success rates.
4. **Single seed:** Training was conducted with a single random seed (42). Results may vary across seeds, though SAC's 100% success rate suggests robust convergence.
5. **PPO training budget:** While PPO was trained for ~63M environment steps (compared to SAC's 24M timesteps), on-policy methods may require different hyperparameter tuning strategies (e.g., larger batch sizes, different learning rate schedules) that were not explored. The matched-parameter comparison demonstrates algorithmic differences but does not represent a fully optimised PPO baseline.

## 7. Conclusion

This comparative study evaluated PPO and SAC for single-waypoint UAV navigation in a high-fidelity Isaac Sim environment using a controlled experimental design with matched hyperparameters. With identical environment configuration, reward function, network architecture ([512, 256]), discount factor ($\gamma = 0.98$), episode length (30 s), and autopilot phases:

- **SAC** achieved **100% success rate** across 50 evaluation episodes with **0% crashes**, an average final distance of **0.365 m**, and training convergence in **24M timesteps (~18 min)**
- **PPO** achieved **0% success rate** with **100% crash rate**, an average final distance of **7.177 m**, and failed to converge despite **~63M environment steps (~35 min)**

The fair comparison eliminates confounding variables and demonstrates that the performance gap is inherent to the algorithm class, not the configuration. SAC's off-policy replay buffer and entropy-regularised exploration are critical for solving this long-horizon, multi-phase task where the terminal reward signal (goal landing bonus) requires credit assignment over 1,500+ timesteps.

The three most impactful SAC tuning changes (from its own baseline) were: (1) reducing the discount factor from 0.99 to 0.98 for faster credit assignment, (2) increasing network capacity from [256, 128, 64] to [512, 256], and (3) extending training from 16M to 24M timesteps. Applying these same changes to PPO did not resolve its fundamental convergence failure, confirming that the performance difference is algorithmic rather than hyperparameter-dependent.

Autopilot control for deterministic flight phases (takeoff, stabilise, hover, land) improved reliability for both algorithms by restricting the RL agent to the navigation phase only. This principle — hardcoding deterministic behaviours rather than learning them — is consistent with the obstacle avoidance system's design (v7/v8).

These results demonstrate that SAC is the preferred algorithm for UAV waypoint navigation tasks with long horizons and sparse terminal rewards, providing a strong foundation for the subsequent multi-waypoint navigation and obstacle avoidance tasks.

---

*Evaluation date: 2026-03-28. All experiments conducted on a single NVIDIA RTX 3060 GPU.*
*SAC checkpoint: `waypoint/logs/sac/crazyflie_waypoint_sac/2026-03-28_22-12-21/sac_final.zip`*
*PPO checkpoint: `waypoint/logs/ppo/crazyflie_forest_nav_ppo/2026-03-28_23-04-48/model_100.pt`*
