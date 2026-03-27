# Waypoint Navigation: PPO vs SAC — Training & Evaluation

## Task
Single-waypoint navigation: `(-5, 0, 0.2)` → `(5, 0, 0.2)` (10m straight line)
5-phase state machine: TAKEOFF → STABILIZE → NAVIGATE → HOVER → LAND

---

## 1. Train Both Simultaneously (~2 Hours)

Open **two terminals** and run both at the same time.

### Prerequisites (both terminals)
```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate
```

### Terminal 1 — PPO (GPU 0)
```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/waypoint/waypoint_ppo.py \
    --mode train --num_envs 512 --max_iterations 1000 --headless
```
- **~1000 iters** with 512 envs × 64 steps/iter = ~32K steps/iter
- Estimated wall time: **~2 hours** on RTX 3060
- Checkpoints saved every 100 iterations

### Terminal 2 — SAC (GPU 0)
> **WARNING**: Running both on the same RTX 3060 12GB will likely OOM.
> **Option A** — Run sequentially (one after the other, ~2hrs each = ~4hrs total)
> **Option B** — Reduce envs to 256 each to fit both on GPU:

```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/waypoint/waypoint_sac.py \
    --mode train --num_envs 512 --total_timesteps 2000000 --headless
```
- **2M timesteps** with 512 envs
- Estimated wall time: **~2 hours** on RTX 3060
- Checkpoints saved every ~50K timesteps

### Option B — Both at Once (256 envs each to avoid OOM)
**Terminal 1:**
```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/waypoint/waypoint_ppo.py \
    --mode train --num_envs 256 --max_iterations 1500 --headless
```

**Terminal 2:**
```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/waypoint/waypoint_sac.py \
    --mode train --num_envs 256 --total_timesteps 2000000 --headless
```

### One-Liner (background both, sequential on same GPU)
```bash
cd ~/projects/isaac/IsaacLab && source ~/projects/isaac/env_isaaclab/bin/activate

# Run PPO first, then SAC (total ~4hrs but safe)
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/waypoint/waypoint_ppo.py \
    --mode train --num_envs 512 --max_iterations 1000 --headless && \
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/waypoint/waypoint_sac.py \
    --mode train --num_envs 512 --total_timesteps 2000000 --headless
```

---

## 2. Evaluate Both (100 Episodes Each)

After training, find the latest checkpoints:
```bash
# PPO checkpoint (latest .pt file)
PPO_CKPT=$(ls -t ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/waypoint/logs/ppo/crazyflie_forest_nav_ppo/*/model_*.pt 2>/dev/null | head -1)
echo "PPO checkpoint: $PPO_CKPT"

# SAC checkpoint (final .zip or latest checkpoint)
SAC_CKPT=$(ls -t ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/waypoint/logs/sac/crazyflie_waypoint_sac/*/sac_final.zip 2>/dev/null | head -1)
echo "SAC checkpoint: $SAC_CKPT"
```

### Evaluate PPO (100 episodes)
```bash
cd ~/projects/isaac/IsaacLab && source ~/projects/isaac/env_isaaclab/bin/activate

python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/waypoint/waypoint_ppo.py \
    --mode eval --checkpoint "$PPO_CKPT" --num_episodes 100 --num_envs 64 --headless
```

### Evaluate SAC (100 episodes)
```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/waypoint/waypoint_sac.py \
    --mode eval --checkpoint "$SAC_CKPT" --num_episodes 100 --num_envs 64 --headless
```

---

## 3. Expected Eval Output Format

Both scripts print the same metrics table:

```
============================================================
RESULTS - PPO/SAC (100 episodes)
============================================================
  Success Rate:       XX.X%  (XX/100 landed at goal)
  Crash Rate:          X.X%  (X/100 crashed)
  Avg Final Distance:  X.XXX m
  Min Final Distance:  X.XXX m
  Avg Reward:         XXXX.XX  (+/- XXX.XX)
  Avg Episode Length:  XX.XX s
============================================================
```

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **Success Rate** | % of episodes where drone landed within 0.5m of goal |
| **Crash Rate** | % of episodes terminated by crash (height/bounds violation) |
| **Avg Final Distance** | Mean distance to goal at episode end |
| **Min Final Distance** | Closest any episode got to the goal |
| **Avg Reward** | Mean cumulative reward per episode |
| **Avg Episode Length** | Mean episode duration in seconds (max 15s in eval) |

---

## 4. Comparison Table — Eval Results (2026-03-27, 100 Episodes Each)

| Metric | PPO (5000 iters) | SAC (10M steps) |
|--------|-------------------|-----------------|
| Success Rate (%) | 0.0% (0/100) | 0.0% (0/100) |
| Crash Rate (%) | 0.0% (0/100) | 0.0% (0/100) |
| Avg Final Distance (m) | 10.008 | 4.850 |
| Min Final Distance (m) | 10.003 | 1.992 |
| Avg Reward | 2.08 (+/- 0.20) | 78.41 (+/- 44.27) |
| Avg Episode Length (s) | 7.99 | 8.29 |
| Training Time (hrs) | ~2 | ~2 |
| Training Iterations/Steps | 5000 iters | ~10M timesteps |

### Checkpoints Used
- **PPO**: `logs/ppo/crazyflie_forest_nav_ppo/2026-03-26_22-37-16/model_4999.pt`
- **SAC**: `logs/sac/crazyflie_waypoint_sac/2026-03-26_22-23-53/sac_final.zip`

### Analysis
- **Neither algorithm achieved goal landing** — 0% success for both.
- **SAC navigated closer** — avg 4.85m vs PPO's 10.0m (PPO barely moved from start).
- **PPO appears stuck** — min distance 10.003m means the drone never moved toward the goal (start-to-goal is 10m). Likely the policy did not converge; may need reward tuning or longer training.
- **SAC showed learning** — min distance 1.99m shows it can approach the goal but fails to land. The higher reward variance (44.27) indicates diverse episode outcomes.
- **No crashes** — both policies are stable (PID controller prevents flipping).

---

## 5. Training Hyperparameters

| Parameter | PPO | SAC |
|-----------|-----|-----|
| Framework | RSL-RL (OnPolicyRunner) | Stable-Baselines3 |
| Network | [256, 128, 64] ELU | [256, 128, 64] ReLU |
| Learning Rate | 1e-4 (adaptive) | 3e-4 |
| Gamma | 0.98 | 0.98 |
| Batch Size | 4 mini-batches | 256 |
| Steps/Iter | 64 per env | 1 per env |
| Epochs/Iter | 8 | 1 gradient step |
| Entropy | 0.01 (fixed coef) | auto (learned alpha) |
| Buffer | On-policy (no buffer) | 1M replay buffer |

---

## 6. Notes

- Both use identical environment, reward, and PID controller
- 10x scaled Crazyflie, gyroscopic forces OFF
- Action space: 4D velocity commands [vx, vy, vz, yaw_rate]
- PID controller converts velocity commands to thrust + torques (frozen)
- RTX 3060 12GB: 512 envs max per process; reduce to 256 if running both
