# Single-Waypoint Navigation: A Comparative Study of PPO and SAC for UAV Point-to-Point Flight Control

## 1. Introduction

This section presents a comparative evaluation of two model-free reinforcement learning algorithms — Proximal Policy Optimisation (PPO) and Soft Actor-Critic (SAC) — applied to a single-waypoint navigation task for a simulated quadrotor UAV. The objective is to establish a performance baseline for point-to-point autonomous flight control before extending to more complex multi-waypoint and obstacle-avoidance scenarios.

## 2. Problem Formulation

### 2.1 Task Definition

The navigation task requires the UAV to fly from a fixed start position $\mathbf{p}_s = (-5.0, 0.0, 0.2)$ m to a goal position $\mathbf{p}_g = (5.0, 0.0, 0.2)$ m, spanning a Euclidean distance of 10.0 m along the $x$-axis. The task follows a five-phase state machine:

1. **TAKEOFF** — Ascend from the start position to a cruise altitude of 1.5 m
2. **STABILIZE** — Maintain hover at cruise altitude for 2.0 s to damp transients
3. **NAVIGATE** — Fly toward the goal position at cruise altitude
4. **HOVER** — Hold position above the goal for 2.0 s within a 0.5 m tolerance
5. **LAND** — Controlled descent to the ground at the goal location

A successful episode requires the drone to land within a goal threshold of $d_{goal} = 0.5$ m of the target position at a landing speed below 1.0 m/s. Episodes are terminated upon success, crash (height violation), or after a maximum duration of 18.0 s (15.0 s during evaluation).

### 2.2 Hierarchical Control Architecture

The system employs a two-level hierarchical control architecture to decouple high-level decision-making from low-level stabilisation:

- **High-level policy (learned):** Outputs 4-dimensional velocity commands $\mathbf{a} = [v_x, v_y, v_z, \dot{\psi}]$ in the body frame, bounded by $v_{xy}^{max} = 2.0$ m/s, $v_z^{max} = 1.0$ m/s, and $\dot{\psi}^{max} = 1.5$ rad/s.
- **Low-level PID controller (frozen):** Converts velocity commands to thrust and torque inputs via a cascaded proportional-derivative (PD) controller:
  - Outer loop: Velocity error $\rightarrow$ desired attitude (tilt angle), with $K_{vel} = 0.25$ and maximum tilt $\theta_{max} = 0.5$ rad ($\approx 28.6°$)
  - Inner loop: Attitude error $\rightarrow$ torque, with $K_{att} = 6.0$ (proportional) and $K_{att,d} = 1.0$ (derivative)
  - Altitude: Thrust correction proportional to vertical velocity error, $K_{vz} = 0.5$
  - Yaw: Torque proportional to yaw rate error, $K_{yaw} = 0.4$

This architecture ensures flight stability is maintained by the PID layer, allowing the RL agent to focus on trajectory-level decisions.

### 2.3 Observation Space

Both algorithms receive an identical 26-dimensional observation vector:

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Body-frame linear velocity | 3 | $[v_x^b, v_y^b, v_z^b]$ |
| Body-frame angular velocity | 3 | $[\omega_x^b, \omega_y^b, \omega_z^b]$ |
| Projected gravity vector | 3 | Gravity direction in body frame (encodes roll/pitch) |
| Goal position (body frame) | 3 | Relative goal $[\Delta x^b, \Delta y^b, \Delta z^b]$ |
| LiDAR distances (normalised) | 9 | 3 vertical channels $\times$ 4 horizontal rays (360° coverage), normalised by $d_{max} = 15.0$ m |
| Phase one-hot encoding | 5 | Current flight phase indicator |

### 2.4 Action Space

The action space is continuous and 4-dimensional, representing velocity commands in the body frame: $\mathbf{a} \in \mathbb{R}^4 = [v_x, v_y, v_z, \dot{\psi}]$. Actions are clipped to the velocity limits defined above before being passed to the PID controller.

### 2.5 Reward Function

A dense, phase-dependent reward function guides training through each flight phase. Key components include:

| Phase | Reward Component | Scale | Purpose |
|-------|-----------------|-------|---------|
| TAKEOFF | Ascent progress | +10.0 | Encourage altitude gain |
| TAKEOFF | Lateral drift penalty | -2.0 | Suppress horizontal motion during ascent |
| STABILIZE | Position hold | +3.0 | Reward proximity to start position |
| STABILIZE | Low speed bonus | +2.0 | Encourage hover stability |
| NAVIGATE | XY progress (delta-based) | +3.0 | Reward distance reduction toward goal |
| NAVIGATE | Velocity alignment | +8.0 | Reward velocity directed toward goal |
| NAVIGATE | Lateral drift penalty | -2.0 | Penalise perpendicular motion |
| NAVIGATE | Altitude maintenance | -2.0 | Penalise deviation from cruise altitude |
| NAVIGATE | Stability bonus | +3.0 | Reward low angular velocity and upright orientation |
| HOVER | Position hold | +3.0 | Reward proximity to goal |
| LAND | Descent progress | +5.0 | Reward controlled altitude reduction |
| LAND | XY stability | +15.0 | Strong reward for maintaining position during landing |
| LAND | Precision landing | +200.0 | High reward for accurate final position |
| Global | Goal reached bonus | +750.0 | Terminal reward for successful landing |
| Global | Time penalty | -1.5 | Encourage efficiency |
| Global | Angular velocity penalty | -0.5 | Smooth flight |

The navigation phase uses a delta-based progress reward $r_{progress} = (\|\Delta_{xy}^{t-1}\| - \|\Delta_{xy}^{t}\|) \times s_{progress}$ rather than a continuous proximity reward, to avoid the known circling pathology where the agent orbits the goal to accumulate proximity rewards.

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
| Parallel environments (evaluation) | 1 |

### 3.2 Robot Platform

The simulated platform is based on the Bitcraze Crazyflie 2.x quadrotor, scaled by a factor of 10 to facilitate visual debugging and reduce numerical sensitivity. Gyroscopic forces are disabled to simplify the dynamics. The thrust-to-weight ratio is set to 1.9, providing sufficient control authority for aggressive manoeuvres.

### 3.3 Hardware

All training and evaluation were performed on a single workstation:
- **GPU:** NVIDIA GeForce RTX 3060 (12 GB VRAM)
- **CPU:** AMD Ryzen 5 5500 (6 cores / 12 threads)
- **RAM:** 32 GB DDR4
- **OS:** Ubuntu 24.04.3 LTS, Kernel 6.17.0

### 3.4 Algorithm Configurations

#### 3.4.1 Proximal Policy Optimisation (PPO)

PPO was implemented using RSL-RL (OnPolicyRunner). The configuration is as follows:

| Hyperparameter | Value |
|---------------|-------|
| Policy network | MLP [256, 128, 64], ELU activation |
| Critic network | MLP [256, 128, 64], ELU activation |
| Learning rate | $1 \times 10^{-4}$ (adaptive schedule) |
| Discount factor ($\gamma$) | 0.98 |
| GAE parameter ($\lambda$) | 0.95 |
| Clipping parameter ($\epsilon$) | 0.2 |
| Entropy coefficient | 0.01 |
| Number of epochs per iteration | 8 |
| Number of mini-batches | 4 |
| Steps per environment per iteration | 64 |
| Maximum gradient norm | 1.0 |
| Desired KL divergence | 0.008 |
| Initial action noise std | 0.5 |
| Observation normalisation | Enabled (actor and critic) |
| Training iterations | 5,000 |
| Total environment steps | $\approx 5000 \times 512 \times 64 = 163.8\text{M}$ |

#### 3.4.2 Soft Actor-Critic (SAC)

SAC was implemented using Stable-Baselines3. The configuration is as follows:

| Hyperparameter | Value |
|---------------|-------|
| Policy network | MLP [256, 128, 64], ReLU activation |
| Learning rate | $3 \times 10^{-4}$ |
| Discount factor ($\gamma$) | 0.98 |
| Replay buffer size | $1 \times 10^6$ |
| Learning starts | 10,000 steps |
| Batch size | 256 |
| Soft update coefficient ($\tau$) | 0.005 |
| Training frequency | 1 step |
| Gradient steps | 1 |
| Entropy coefficient | Auto-tuned (learned $\alpha$) |
| Target entropy | Auto (default: $-\dim(\mathcal{A})$) |
| Total timesteps | $\approx 10\text{M}$ |

### 3.5 Training Protocol

Both algorithms were trained for approximately 2 hours of wall-clock time on the RTX 3060. Training was performed sequentially (one algorithm at a time) to avoid GPU memory contention. PPO completed 5,000 iterations with 512 parallel environments; SAC completed approximately 10 million timesteps with 512 parallel environments.

## 4. Evaluation Protocol

Each trained policy was evaluated over 100 episodes in a single-environment configuration with deterministic action selection (no exploration noise). The evaluation episode length was capped at 15.0 s. The following metrics were recorded:

- **Success rate:** Percentage of episodes where the drone landed within 0.5 m of the goal
- **Crash rate:** Percentage of episodes terminated by height or boundary violations
- **Average final distance:** Mean Euclidean distance to the goal at episode termination
- **Minimum final distance:** Closest approach to the goal across all episodes
- **Average cumulative reward:** Mean total reward per episode
- **Average episode length:** Mean episode duration

## 5. Results

### 5.1 Quantitative Results

| Metric | PPO | SAC |
|--------|-----|-----|
| Success Rate | 0.0% (0/100) | 0.0% (0/100) |
| Crash Rate | 0.0% (0/100) | 0.0% (0/100) |
| Avg Final Distance (m) | 10.008 $\pm$ 0.003 | 4.850 $\pm$ 2.21 |
| Min Final Distance (m) | 10.003 | 1.992 |
| Avg Cumulative Reward | 2.08 $\pm$ 0.20 | 78.41 $\pm$ 44.27 |
| Avg Episode Length (s) | 7.99 | 8.29 |
| Training Wall Time (hrs) | ~2 | ~2 |

### 5.2 Analysis

**Neither algorithm achieved the goal-landing objective** within the allocated training budget. However, the two algorithms exhibited qualitatively different failure modes:

#### PPO: Non-convergence

The PPO agent produced an average final distance of $10.008 \pm 0.003$ m — virtually identical to the initial start-to-goal distance of 10.0 m. The extremely low variance ($\sigma = 0.003$) and near-zero cumulative reward ($2.08 \pm 0.20$) indicate that the policy failed to learn any meaningful navigation behaviour. The drone effectively remained stationary at or near the start position throughout all episodes, only accumulating the small phase-dependent rewards from the TAKEOFF and STABILIZE phases.

This non-convergence may be attributed to several factors:
1. **Sparse terminal reward dominance:** The goal-reached bonus ($+750$) is a sparse signal that PPO's on-policy updates may struggle to propagate backward through the 18-second episodes without sufficient trajectory diversity.
2. **Adaptive learning rate schedule:** PPO's adaptive KL-based learning rate may have reduced the step size prematurely if early policy updates caused large KL divergence, effectively stalling optimisation.
3. **Insufficient exploration:** The 5,000 iterations, while substantial in wall-clock time, may not have provided sufficient policy gradient signal for convergence given the phase-dependent reward structure.

#### SAC: Partial Learning

SAC demonstrated meaningful learning, achieving an average final distance of $4.850$ m — a 51.5% reduction from the initial 10.0 m. The minimum final distance of 1.992 m shows that the agent successfully navigated to within 2 m of the goal in its best episode. The higher reward variance ($\sigma = 44.27$) reflects diverse episode outcomes, ranging from near-successful approaches to episodes where the drone deviated from the goal.

SAC's superior sample efficiency in this setting can be attributed to:
1. **Experience replay:** The $10^6$-step replay buffer enables off-policy learning from past transitions, improving data efficiency compared to PPO's on-policy approach.
2. **Entropy regularisation:** The automatically tuned entropy coefficient encourages sustained exploration throughout training, helping the agent discover reward-producing trajectories in the NAVIGATE phase.
3. **Continuous updates:** SAC performs one gradient step per environment step, accumulating more parameter updates over the same number of interactions.

Despite approaching the goal, SAC failed to achieve successful landings (0% success rate). The failure likely occurs in the HOVER and LAND phases, where the policy must transition from navigation to precision positioning — a qualitatively different control objective that requires extended training to master.

#### Flight Stability

Both algorithms achieved 0% crash rates, validating the hierarchical control architecture. The frozen PID controller successfully maintained attitude stability regardless of the high-level policy's output quality, preventing catastrophic flight failures even from untrained or poorly trained policies.

## 6. Discussion

### 6.1 On-Policy vs Off-Policy Learning for UAV Control

The results highlight a notable sample efficiency advantage for SAC over PPO in this waypoint navigation task. While PPO benefits from stable policy updates through its clipped surrogate objective, the on-policy constraint discards all collected data after each update cycle, making it less efficient in environments with high-dimensional reward structures and long episode horizons.

SAC's off-policy nature, combined with its maximum entropy framework, enables more effective exploration and data reuse — particularly advantageous for multi-phase tasks where the agent must discover temporally extended behaviours spanning takeoff, navigation, and landing.

### 6.2 Limitations

Several limitations of this baseline experiment should be acknowledged:

1. **Training budget:** The 2-hour training window is modest for both algorithms. Prior work suggests that PPO may require significantly longer training (or more parallel environments) to converge on complex continuous control tasks, while SAC typically benefits from 20M+ timesteps.
2. **Single start-goal pair:** The fixed start and goal positions limit the generalisability of the trained policies. Future experiments should evaluate with randomised positions.
3. **Simplified dynamics:** The 10x scale factor and disabled gyroscopic forces simplify the aerodynamics, which may affect transfer to real-world platforms.
4. **Evaluation seed:** A single evaluation seed was used; results may exhibit variance across different random seeds.

### 6.3 Recommendations for Future Training

Based on these results, the following adjustments are recommended:

- **Extended training:** Increase SAC training to 20-30M timesteps and PPO to 10,000+ iterations to allow sufficient convergence.
- **Reward shaping refinement:** Consider increasing the navigation progress reward scale or adding intermediate waypoint bonuses to provide denser gradient signals for PPO.
- **Curriculum learning:** Gradually increase the start-to-goal distance during training to help the agent learn navigation incrementally.
- **Hyperparameter tuning:** PPO's learning rate, entropy coefficient, and GAE parameters may benefit from systematic optimisation for this specific task.

## 7. Conclusion

This comparative study evaluated PPO and SAC for single-waypoint UAV navigation in a high-fidelity Isaac Sim environment. Within a 2-hour training budget, SAC demonstrated partial learning capability (51.5% distance reduction, min distance 1.99 m) while PPO failed to converge (0% distance reduction). Neither algorithm achieved successful goal landings, indicating that extended training or algorithmic modifications are required. The hierarchical PID control architecture proved effective at maintaining flight stability regardless of high-level policy quality, with 0% crash rates across all 200 evaluation episodes. These baseline results inform the design of subsequent experiments in multi-waypoint navigation and obstacle avoidance.

---

*Evaluation date: 2026-03-27. All experiments conducted on a single NVIDIA RTX 3060 GPU.*
