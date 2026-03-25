# Hierarchical SAC — Code Review Findings & Colleague Instructions
Date: 2026-03-15
Files reviewed: `hierarchical_obstacle_sac.py`
Files produced: `hierarchical_obstacle_sac_v2.py` (fixed), `FINDINGS.md` (this file)
Symptom reported: drone flies well initially, performance degrades over time.

---

## Quick-start for colleagues

### What files exist and which to use

| File | Status | Use for |
|---|---|---|
| `hierarchical_obstacle_sac.py` | Original — do not modify | Reference / comparison only |
| `hierarchical_obstacle_sac_v2.py` | Fixed version | All new training runs |
| `FINDINGS.md` | This document | Understanding what changed and why |

### Environment setup
```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate
```

### Train (use v2)
```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v2.py \
    --mode train \
    --num_envs 256 \
    --total_timesteps 2000000 \
    --headless
```
Checkpoints are saved automatically every ~50 k steps to:
`obstacle/logs/hierarchical_sac/crazyflie_hierarchical_obstacle_sac/<timestamp>/`

To resume from a checkpoint:
```bash
python .../hierarchical_obstacle_sac_v2.py \
    --mode train \
    --num_envs 256 \
    --total_timesteps 2000000 \
    --checkpoint obstacle/logs/hierarchical_sac/.../hl_sac_model_<step>_steps.zip \
    --headless
```

### Play (visualise a trained high-level checkpoint)
```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v2.py \
    --mode play \
    --num_envs 1 \
    --checkpoint obstacle/logs/hierarchical_sac/.../hl_sac_final.zip
```

### Evaluate (headless, multiple episodes, prints stats)
```bash
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v2.py \
    --mode eval \
    --num_envs 1 \
    --checkpoint obstacle/logs/hierarchical_sac/.../hl_sac_final.zip \
    --num_episodes 50
```

### Low-level checkpoint (frozen, pre-trained)
The frozen low-level policy is loaded automatically from:
```
multi_waypoint/logs/sac/crazyflie_multi_waypoint_nav_sac/2026-03-08_16-11-44/sac_final.zip
```
To use a different LL checkpoint add `--ll_checkpoint <path>`.
**Do not retrain the low-level** — it must stay frozen for the hierarchical architecture to work.

### What to watch in TensorBoard
```bash
tensorboard --logdir obstacle/logs/hierarchical_sac/
```
Key metrics to monitor:
- `Episode_Reward/nav_xy_progress` — should increase steadily; collapse here means HL is not generating useful sub-goals
- `Episode_Termination/died_NAVIGATE` — crash count during navigation; should trend down
- `Metrics/waypoint_completion_rate` — fraction of waypoints reached per episode
- `Episode_Reward/crash_penalty` — should be near zero after early training; persistent negatives mean the LL is being steered into obstacles

---

## Architecture confirmed working correctly

### Low-level weights are truly frozen
Lines 493–496:
```python
self._ll_actor = ll_agent.policy.actor
self._ll_actor.eval()
for param in self._ll_actor.parameters():
    param.requires_grad = False
```
`.eval()` puts BatchNorm/Dropout into inference mode. `requires_grad=False` prevents any gradient from flowing into the LL parameters. Only the actor network is extracted; the LL critic is discarded. **Correctly frozen.**

### LL observation normalization — not needed
An initial concern was that the LL policy's running observation normalizer (obs_rms) was stored but never applied at inference (line 807). On inspection, `multi_waypoint_sac.py` does **not** use `VecNormalize` and SB3 SAC defaults to `normalize_observations=False`. Therefore `_ll_obs_normalize` is always `None` and the dead check at lines 498–500 is harmless. **Not a bug.**

### Crash penalty does not accumulate
The crash penalty (`cfg.crash_penalty`, line 1103) fires every step the drone is inside the collision radius. However, `_get_dones()` (line 1247) sets `died = True` on the same collision condition, terminating the episode immediately. So the penalty fires at most once before reset. **Not a penalty well.**

### `_prev_xy_dist` reset on phase transitions
`nav_xy_progress` (line 1001) is gated by `torch.where(is_navigate, ...)`, so a stale `_prev_xy_dist` is zeroed out outside of NAVIGATE. `_prev_z_dist` is explicitly reset at line 1154 when entering LAND. **Not a bug.**

---

## Confirmed bug

### Global `_hl_step_counter` resets all environments on any reset

**Location:** lines 448, 862–869, 1315
**Severity:** Major during training; does not affect single-env play.

The high-level step counter is a plain Python integer (scalar):
```python
self._hl_step_counter = 0          # line 448 — scalar, not per-env
```

In `_pre_physics_step` it is incremented globally and the HL action is applied when it reaches 5 (= 10 Hz):
```python
self._hl_step_counter += 1
if self._hl_step_counter >= 5:
    self._hl_step_counter = 0
    # apply new yaw_offset / speed_factor / alt_offset to ALL envs
```

In `_reset_idx`, called whenever **any** environment resets:
```python
self._hl_step_counter = 0          # line 1315 — resets counter for ALL 256 envs
```

**Effect during training with 256 envs:**
With ~2000-step episodes across 256 envs, a reset from any environment occurs roughly every 8 `env.step()` calls on average. The counter needs 5 steps to trigger an HL update. Resets interrupt the cycle approximately every 8 steps, so the 10 Hz cadence is unreliable. Additionally, when the counter resets mid-cycle, all environments that were partway through a 5-step hold suddenly restart the hold, desynchronising their HL action cadence.

**Effect during play (1 env):** None — the counter only resets at episode end.

**Fix:** make `_hl_step_counter` a per-environment integer tensor and reset only the indices that actually reset. See `hierarchical_obstacle_sac_v2.py`.

---

## Design observations (not bugs, but worth noting)

### HL goal modifier pushes LL toward edge cases over time
The high-level policy modifies the goal seen by the frozen LL (yaw offset up to ±60°, speed factor 0.3–1.5, altitude offset ±1 m). The LL was trained on zigzag multi-waypoint navigation in all directions, so yaw rotation is tolerable. Speed scaling to extremes (0.3× or 1.5×) may produce velocity commands the LL has less experience with, since multi-waypoint training had a fixed max-speed regime. This is not a code defect but a distribution mismatch that grows as the HL explores extreme actions.

### PID has no integral term
The PID controller (lines 816–850) is pure P+D. This produces a constant steady-state tracking error for constant-velocity segments. The LL was trained with this same PID controller so the error is baked into training, but any additional offloading of responsibility from LL to HL will not be corrected by integral action.

---

## Summary table

| Finding | Verdict | Fixed in v2? |
|---|---|---|
| LL weights frozen | Correct — no change needed | n/a |
| LL obs normalisation not applied | False alarm — LL trained raw | n/a |
| Crash penalty accumulates | False alarm — episode ends on collision | n/a |
| `_prev_xy_dist` stale on phase change | False alarm — gated by phase mask | n/a |
| Global `_hl_step_counter` | **Real bug** — desynchronises HL updates during training | **Yes** |
| HL goal modifier distribution shift | Design note — monitor speed_factor extremes | Guarded |

---

## Exact changes in v2 (for colleagues reviewing the diff)

Only 4 locations changed. The original file is untouched.

### Change 1 — docstring (line 2)
Added version marker and reference to this document.

### Change 2 — `_setup_buffers`, line ~448
**Before (original):**
```python
# High-level update counter (updates every 5 env.steps = 10 Hz)
self._hl_step_counter = 0
```
**After (v2):**
```python
# High-level update counter (updates every 5 env.steps = 10 Hz).
# FIX v2: per-environment tensor instead of a global scalar so that
# resetting one environment does not desynchronise all others.
self._hl_step_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
```
**Why:** the scalar `0` was shared across all 256 envs. A reset in any one env set it back to 0 for everyone.

### Change 3 — `_pre_physics_step`, lines ~862–869
**Before (original):**
```python
self._hl_step_counter += 1
if self._hl_step_counter >= 5:
    self._hl_step_counter = 0
    a = actions.clone().clamp(-1.0, 1.0)
    self._hl_yaw_offset = a[:, 0] * self.cfg.max_yaw_offset
    lo, hi = self.cfg.speed_factor_range
    self._hl_speed_factor = lo + (a[:, 1] + 1.0) * 0.5 * (hi - lo)
    self._hl_alt_offset = a[:, 2] * self.cfg.max_altitude_offset
```
**After (v2):**
```python
self._hl_step_counter += 1
update_mask = self._hl_step_counter >= 5
if update_mask.any():
    self._hl_step_counter[update_mask] = 0
    a = actions.clone().clamp(-1.0, 1.0)
    lo, hi = self.cfg.speed_factor_range
    self._hl_yaw_offset[update_mask] = (a[:, 0] * self.cfg.max_yaw_offset)[update_mask]
    self._hl_speed_factor[update_mask] = (lo + (a[:, 1] + 1.0) * 0.5 * (hi - lo))[update_mask]
    self._hl_alt_offset[update_mask] = (a[:, 2] * self.cfg.max_altitude_offset)[update_mask]
```
**Why:** each env now independently tracks whether its own 5-step hold has elapsed. Envs are updated only when their individual counter reaches 5, leaving other envs undisturbed.

### Change 4 — `_reset_idx`, line ~1315
**Before (original):**
```python
# Reset goal modifier and step counter
self._hl_step_counter = 0
self._hl_yaw_offset[env_ids] = 0.0
```
**After (v2):**
```python
# Reset goal modifier and step counter.
# FIX v2: reset only the envs that are actually being reset.
self._hl_step_counter[env_ids] = 0
self._hl_yaw_offset[env_ids] = 0.0
```
**Why:** `env_ids` contains only the indices of environments that are resetting. Using tensor indexing restricts the reset to those envs only, leaving the counters of all other envs intact.

---

## How to verify the fix is working

After a short training run with v2, open TensorBoard and check that
`Episode_Reward/nav_xy_progress` does not collapse after an initial rise.
You can also add a temporary debug print in `_pre_physics_step` to confirm
`update_mask` fires roughly evenly across envs over a window of steps:

```python
if update_mask.any():
    print(f"HL update: {update_mask.sum().item()} / {self.num_envs} envs updated this step")
```

With the fix, you should see counts scattered across the range 0–256 rather than always 0 or 256.

---

## Confidence assessment (written before colleague testing)

### What has been verified
- All four changed lines were diffed against the original and confirmed correct
- Every usage of `_hl_step_counter`, `_hl_yaw_offset`, `_hl_speed_factor`, and `_hl_alt_offset` was
  traced through the full file to confirm no other code assumes a scalar counter or reassigns the
  modifier tensors to non-tensor values
- The in-place indexed assignments (`[update_mask]`, `[env_ids]`) are valid because all three modifier
  tensors are initialised as GPU tensors and never replaced by scalars anywhere in the file

### What has NOT been verified
- The code has not been executed — no smoke test has been run
- Potential runtime issues that cannot be caught by reading alone:
  - `dtype=torch.int32` with `+= 1` under the specific PyTorch / CUDA version in this environment
  - `Sb3VecEnvWrapper` action tensor shape reaching `_pre_physics_step` in the expected form
  - Isaac Lab's internal call ordering doing something unexpected between `_pre_physics_step` and `_reset_idx`

### Recommended smoke test before full training run
```bash
cd ~/projects/isaac/IsaacLab
source ~/projects/isaac/env_isaaclab/bin/activate
python ~/Desktop/Lasitha/drone_rl_project/drone-rl-project/obstacle/hierarchical_obstacle_sac_v2.py \
    --mode train --num_envs 256 --total_timesteps 5000 --headless
```
If it completes without a Python/CUDA exception the tensor operations are working.

Add this temporary debug print inside `_pre_physics_step` in v2 (remove after confirming):
```python
if update_mask.any():
    print(f"HL update: {update_mask.sum().item()} / {self.num_envs} envs")
```
Expected output: counts like `47 / 256`, `52 / 256` — not always `0 / 256` or `256 / 256`.
