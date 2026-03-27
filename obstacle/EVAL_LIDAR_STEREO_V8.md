# Evaluation Guide — Hierarchical Obstacle LiDAR+Stereo SAC v8

## Overview

**File:** `hierarchical_obstacle_lidar_stereo_sac_v8.py`

This is the **LiDAR-only v8** (`hierarchical_obstacle_sac_v8.py`) with one addition: a **ZED 2i stereo depth camera**.
Everything else — rewards, safety layers, PID controller, obstacle config — is identical to v8.

### What Changed (Stereo vs LiDAR-only v8)

| Parameter              | LiDAR-only v8                  | LiDAR+Stereo v8                          |
|------------------------|--------------------------------|-------------------------------------------|
| **Sensors**            | LiDAR 1347 rays (360°)        | LiDAR 1347 rays + ZED 2i depth (48x64)   |
| **Observation space**  | 1367                           | 4439 (+3072 depth pixels)                 |
| **Feature extractor**  | 1D CNN (LiDAR), dim=128       | Dual-branch CNN (1D+2D), dim=192         |
| **Checkpoint**         | Compatible with v7             | INCOMPATIBLE — must train from scratch    |
| Rewards                | Same                           | Same                                      |
| Safety layers          | Same                           | Same                                      |
| PID controller         | Same                           | Same                                      |
| Episode length         | 180s                           | 180s                                      |
| Goal threshold         | 0.5m                           | 0.5m                                      |

### v8 LiDAR-only Baseline (to beat)

| Metric                  | v8 Result         |
|-------------------------|--------------------|
| **Success Rate**        | 98%               |
| **Crash Rate**          | 2%                |
| **Waypoint Completion** | —                 |

## Prerequisites

```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate
```

## Step 1: Train (must train from scratch)

```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_lidar_stereo_sac_v8.py \
    --mode train --num_envs 256 --total_timesteps 24000000 --headless
```

Checkpoints saved to:
`obstacle/logs/hierarchical_lidar_stereo_sac_v8/crazyflie_hierarchical_obstacle_lidar_stereo_sac_v8/<timestamp>/`

**NOTE:** LiDAR-only v8 checkpoints are INCOMPATIBLE (obs space 1367 vs 4439). Must train from scratch.

## Step 2: Play (visual inspection)

```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_lidar_stereo_sac_v8.py \
    --mode play --num_envs 1 \
    --checkpoint <path_to>/hl_sac_final.zip
```

Watch for:
- Does the drone navigate tight corridors better than LiDAR-only?
- Any new failure modes from the added depth input?
- Smooth transitions between waypoints

## Step 3: Evaluate (quantitative — stereo camera impact)

### Default evaluation (50 episodes, 3-5 waypoints)

```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_lidar_stereo_sac_v8.py \
    --mode eval --num_envs 16 --num_episodes 50 --headless \
    --checkpoint <path_to>/hl_sac_final.zip
```

### Extended evaluation (100 episodes)

```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_lidar_stereo_sac_v8.py \
    --mode eval --num_envs 16 --num_episodes 100 --headless \
    --checkpoint <path_to>/hl_sac_final.zip
```

### Custom waypoint counts

```bash
# Harder: more waypoints
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_lidar_stereo_sac_v8.py \
    --mode eval --num_envs 16 --num_episodes 50 --headless \
    --min_waypoints 5 --max_waypoints 7 \
    --checkpoint <path_to>/hl_sac_final.zip

# Easier: fewer waypoints
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_lidar_stereo_sac_v8.py \
    --mode eval --num_envs 16 --num_episodes 50 --headless \
    --min_waypoints 2 --max_waypoints 3 \
    --checkpoint <path_to>/hl_sac_final.zip
```

## Metrics Reported

| Metric                  | Description                                              | v8 LiDAR-only Baseline |
|-------------------------|----------------------------------------------------------|------------------------|
| **Success Rate**        | Episodes where final distance < 0.5m                    | 98%                    |
| **Crash Rate**          | Episodes ending in collision                             | 2%                     |
| **Crashes by Phase**    | Breakdown: TAKEOFF / STABILIZE / NAVIGATE / HOVER / LAND | —                     |
| **Crashes by Cause**    | Breakdown: hit_obstacle / too_low / too_high / flipped   | —                     |
| **Waypoint Completion** | Average % of waypoints reached per episode               | —                      |
| **Avg Final Distance**  | Mean distance to goal at episode end (m)                 | —                      |
| **Min Final Distance**  | Closest approach to goal across episodes (m)             | —                      |
| **Avg Reward**          | Mean cumulative episode reward (+/- std)                 | —                      |
| **Avg Episode Length**  | Mean episode duration (seconds)                          | —                      |

## What to Look For (Stereo Camera Impact)

The only question: **does adding the ZED 2i stereo depth camera improve on 98% success?**

1. **Crash reduction in NAVIGATE phase** — v8's remaining 2% crashes are in tight corridors where sparse LiDAR (360° but low density) misses obstacles directly ahead. The ZED 2i provides dense 110° forward depth (3072 pixels) to fill this gap.
2. **Success rate >= 98%** — must match or exceed the LiDAR-only baseline. If lower, the added observation complexity may be hurting more than helping.
3. **Episode length** — shorter episodes = more efficient paths from better spatial awareness. Longer episodes = the agent may be confused by extra input.
4. **No regression in other phases** — TAKEOFF/STABILIZE/HOVER/LAND should remain stable since stereo only helps during NAVIGATE.

## CLI Arguments Reference

| Argument             | Default | Description                          |
|----------------------|---------|--------------------------------------|
| `--mode`             | train   | `train`, `play`, or `eval`           |
| `--num_envs`         | 256     | Number of parallel environments      |
| `--total_timesteps`  | 10M     | Training steps (use 24M for stereo)  |
| `--seed`             | 42      | Random seed                          |
| `--checkpoint`       | None    | Path to `hl_sac_final.zip`           |
| `--num_episodes`     | 50      | Episodes for evaluation              |
| `--min_waypoints`    | 3       | Min waypoints per episode            |
| `--max_waypoints`    | 5       | Max waypoints per episode            |
| `--headless`         | False   | Run without GUI (faster eval)        |
