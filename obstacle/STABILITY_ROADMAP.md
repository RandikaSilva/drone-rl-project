# Hierarchical Obstacle SAC — Stability Roadmap

Date: 2026-03-19
Based on: `LL_STABILITY_DEBUG.md`

---

## Phase 1: Confirm LL is Stable in Isolation

**Goal**: Eliminate test artifacts before touching any architecture.

### Steps

1. Fix first waypoint in `test_ll_stability.py`:
   ```python
   # Before (PROBLEMATIC — 4m goal distance, OOD)
   FIXED_WAYPOINTS = [(-6.0, 0.0, 1.5), (0.0, 0.0, 1.5), (6.0, 0.0, 1.5)]

   # After (matches native env ~2m first WP distance)
   FIXED_WAYPOINTS = [(-8.0, 0.0, 1.5), (0.0, 0.0, 1.5), (6.0, 0.0, 1.5)]
   ```

2. Re-run headless test (5 episodes):
   ```bash
   cd ~/projects/isaac/IsaacLab
   source ~/projects/isaac/env_isaaclab/bin/activate
   PYTHONUNBUFFERED=1 python -u \
       ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/test_ll_stability.py \
       --headless --num_episodes 5
   ```

3. Run with GUI to visually confirm clean TAKEOFF → NAVIGATE → LAND:
   ```bash
   python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/test_ll_stability.py \
       --num_episodes 3
   ```

**Pass criterion**: 0% crash rate, drone reaches all 3 waypoints.
**Blocker**: Do NOT proceed to Phase 2+ if LL still crashes.

---

## Phase 2: Add OOD Guard — Goal Distance Clamping

**Goal**: Make LL robust to unusual goal distances during non-NAVIGATE phases.

### Problem

The LL was trained with first waypoint ~2m away. Any goal_pos_b magnitude beyond ~3m during
TAKEOFF causes erratic lateral commands (vy saturates to ±1.0) and crashes.

### Fix

In `hierarchical_obstacle_sac_v2.py`, clamp `goal_pos_b` magnitude inside `_pre_physics_step`
**before** feeding to the LL, gated on phase:

```python
# Clamp goal distance during TAKEOFF and STABILIZE phases
TAKEOFF_IDX  = 0  # phase_one_hot index
STABILIZE_IDX = 1

MAX_LL_GOAL_DIST = 2.5  # meters — stay within LL training distribution

def _clamp_ll_goal(goal_pos_b, phase_one_hot):
    is_early_phase = (phase_one_hot[:, TAKEOFF_IDX] + phase_one_hot[:, STABILIZE_IDX]) > 0
    dist = goal_pos_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = (MAX_LL_GOAL_DIST / dist).clamp(max=1.0)
    clamped = goal_pos_b * torch.where(is_early_phase.unsqueeze(-1), scale, torch.ones_like(scale))
    return clamped
```

**Why now**: Without this guard, any waypoint placement change silently causes crashes.
The system is fragile to goal distance — this makes it robust.

**Note**: This is a defensive clamp only. It does not change NAVIGATE behavior.

---

## Phase 3: Verify HL + LL Together (v2)

**Goal**: Confirm the `_hl_step_counter` fix in v2 resolves the "degrades over time" symptom.

### Background

The original `hierarchical_obstacle_sac.py` used a single global `_hl_step_counter` shared
across all envs. This caused HL updates to desynchronize during multi-env training.
`v2` uses a per-env tensor counter.

### Steps

1. Run `hierarchical_obstacle_sac_v2.py` in play mode with the fixed LL checkpoint:
   ```bash
   cd ~/projects/isaac/IsaacLab
   source ~/projects/isaac/env_isaaclab/bin/activate
   python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v2.py \
       --mode play --num_envs 1 \
       --ll_checkpoint multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip
   ```

2. Monitor HL step frequency — should fire every 5 env steps (10 Hz), no drift.

3. Watch for late-episode performance degradation (the original symptom).

**Pass criterion**: Consistent HL update cadence across full episode length, no crash rate increase over time.
**Blocker**: Must confirm HL+LL stability before starting HL training.

---

## Phase 4: Audit HL Modifier OOD

**Goal**: Ensure the HL cannot push LL observations outside its training distribution during NAVIGATE.

### HL Output → LL Obs Impact

| HL Output | Effect on LL `goal_pos_b` | OOD Risk |
|---|---|---|
| `yaw_offset` | Rotates goal in XY plane | Large yaw offsets create lateral goal components |
| `speed_factor` | Scales velocity commands post-LL | Does NOT affect LL obs directly |
| `alt_offset` | Shifts goal Z component | Can push Z outside LL's ±1.5m training range |

### Steps

1. Add diagnostic logging of `goal_pos_b` magnitude during NAVIGATE with HL active:
   ```python
   if self.phase[env_id] == NAVIGATE:
       goal_mag = ll_goal_pos_b[env_id].norm().item()
       print(f"NAVIGATE goal_mag={goal_mag:.2f}m")
   ```

2. Collect distribution of `goal_pos_b` over a full eval run.

3. If any dimension consistently goes OOD, clamp HL modifier output ranges:
   ```python
   # Example — tighten yaw_offset range if causing lateral OOD
   hl_action[:, 0] = hl_action[:, 0].clamp(-0.5, 0.5)  # yaw_offset
   hl_action[:, 2] = hl_action[:, 2].clamp(-0.5, 0.5)  # alt_offset
   ```

**Note**: This is an audit/diagnostic phase — not all modifiers will need clamping.
Results feed directly into HL training hyperparameters (Phase 5).

---

## Phase 5: HL Training

**Goal**: Train the HL policy on top of the confirmed-stable LL.

### Curriculum

| Stage | Arena | Obstacles | Episodes | Success Target |
|---|---|---|---|---|
| 1 | Small (8m) | None | 500 | LL crash rate < 5% |
| 2 | Medium (16m) | 2–4 static | 1000 | 50% waypoint completion |
| 3 | Full (20m) | 6–10 static | 2000 | 70% waypoint completion |
| 4 | Full | Mixed static+dynamic | 3000 | Full eval suite |

### Key Monitoring Metrics

- **LL crash rate per episode** — must stay near 0% (Phase 2 guard helps)
- **HL update cadence** — verify per-env counter is correct
- **Waypoint completion rate** — primary success metric
- **`goal_pos_b` magnitude distribution** — watch for OOD creep as HL explores

### Training Command (template)

```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v2.py \
    --mode train --num_envs 512 \
    --ll_checkpoint multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip
```

---

## Summary Table

| Phase | Action | Depends On | Status |
|---|---|---|---|
| 1 | Fix test waypoints, re-run LL standalone | — | Pending |
| 2 | Add goal-distance clamping in v2 | Phase 1 pass | Pending |
| 3 | Verify HL+LL in v2 (counter fix) | Phase 1 pass | Pending |
| 4 | Audit HL modifier OOD ranges | Phase 3 pass | Pending |
| 5 | Train HL policy (curriculum) | Phase 3 + 4 | Pending |

---

## Related Files

| File | Purpose |
|---|---|
| `multi_waypoint/multi_waypoint_sac.py` | Native LL training environment |
| `obstacle/hierarchical_obstacle_sac_v2.py` | Fixed obstacle env (per-env HL counter) |
| `obstacle/test_ll_stability.py` | LL standalone stability test |
| `obstacle/LL_STABILITY_DEBUG.md` | Root cause investigation |
| `obstacle/FINDINGS.md` | v1 → v2 code review findings |

### LL Checkpoint

```
multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip
```
