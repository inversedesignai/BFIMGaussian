# Autotune playbook — D-OPTIMAL run (first-of-kind, UNCALIBRATED)

You are a fresh Claude session fired by cron to tune the D-optimal geometry
run `train_dopt_interactive.jl`. You have no memory of prior fires in this
chat, but your project memory (`MEMORY.md` and referenced entries) has been
loaded and carries cross-run wisdom — read it.

**Your cwd is `/home/zlin/BFIMGaussian/`.** Project context is in `CLAUDE.md`.
Per-run chronicle is `autotune_dopt_log.md`. Prior tuning decisions are there.

## What this run is (differences from the BFIM autotune run)

The completed autotune you may find in memory (2026-04-16) was for
**`PhEnd2End_interactive.jl`** — minimizing EKF relative-squared-error.
**This run is different**:

- Script: `train_dopt_interactive.jl` (NOT `PhEnd2End_interactive.jl`).
- Objective: `-logdet(J_N)` — more-negative means higher information.
  The loss is **NEGATIVE** and **decreases** during progress. Do not read
  the loss on a log scale; use linear.
- Checkpoints: `checkpoints_dopt/dopt_step_*.jls` (not `checkpoints/`).
- Control file: `checkpoints_dopt/control.toml`.
- Control keys: `beta, beta_max, freeze_beta, clear_schedule, lr_geom,
  lr_s, save_every, stop`. Note two lrs (geom + sensor), not one.
- Log: `nohup_dopt.out`.
- β starts at **1.0 (greyscale)**, not 16, and the passive schedule is
  **deliberately long** (`[200×8]`). Autotune is expected to *drive* β
  transitions when the geometry is ready, not just watch them happen.
- The sensor params `s_list` co-optimize with the geometry — the inspector
  writes `sensor_trajectory.png`. Watch that `φ` values stabilize before
  concluding a phase is done.

**Calibration priors from the BFIM run do not transfer.** Spike sizes,
recovery windows, binary ceiling — all unknown for d-optimality. Treat
this run as exploratory; build new calibration as phases complete (see
Step 5 for memory-writing guidance).

## Mission

Steer the d-optimal run toward a **binary, coherent, structured** geometry
that achieves good `-logdet`. The v1 run (archived at
`checkpoints_dopt_v1_hotβ_archive/`) started at β=16 and produced **pixel-
scale Nyquist speckle** (peak feature wavelength ~1.4 px, 85.9% binary with
*no coherent features*) — a known TO failure mode. Avoiding that is the
main reason this run starts ungreyscale.

**Watch the peak_feature_wavelength_px metric.** If at any β it collapses
to <2 px, the geometry is re-falling into the speckle trap; diagnose before
advancing β further. A healthy structured geometry should have λ_feat
roughly in the range of λ_physical / dx_pixel — for this problem that's
roughly 10–30 px.

## Hard safety rails (never violate)

1. `lr_geom ∈ [1e-5, 5e-3]`. `lr_s ∈ [1e-4, 5e-2]`. `beta ≤ 256`.
2. If the Julia process is dead → log and exit. No restart.
3. If `checkpoints_dopt/control.toml` exists (pending, not consumed) → do
   not overwrite. Log "deferred" and exit.
4. 20-min cooldown after any prior action. Check `autotune_dopt_log.md`
   for the timestamp of the last non-hold verdict.
5. Inaction is the default.
6. At most one action per fire.

## Step 1 — Load state

Run the inspector (fast — no re-simulation, just re-filter + parse log):

```bash
julia --project=. autotune_dopt_inspect.jl
```

It prints key=value stats to stdout and writes five PNGs to
`autotune_dopt_snapshot/`:

- `geometry.png` — raw / filtered / projected ε side-by-side. The header
  includes `λ_feat≈<N>px` — your speckle detector.
- `gray_zone.png` — where undecided pixels live
- `loss_trajectory.png` — linear y, -logdet over time (lower=better),
  β doublings marked
- `grad_hist.png` — geometry gradient magnitude distribution
- `sensor_trajectory.png` — φ₁,φ₂ for each of 3 sensor steps over time

Use the Read tool on each PNG. You are multimodal — actually look.

Also:
- `tail -50 autotune_dopt_log.md` — prior fires' reasoning.
- Check memory: `feedback_autotune_principles.md` still applies. The
  BFIM-specific calibration (phases/calibration/failure_modes) is for a
  different run — use as loose prior, not ground truth.

## Step 2 — Diagnose (answer these in the log, before deciding)

**Q1. What phase are we in?**

Phases for d-optimality (first-principles, since no prior run):

- **Warmup (β=1):** continuous exploration. Loss (−logdet) should descend
  rapidly in the first ~50-100 iters as the design moves off the random
  init. Geometry should show *visible structure emerging* in the filtered
  ε (middle panel) — not just noise.
- **Continuous refine (β=1–4):** structure crystallizes. Projected and
  filtered panels start to look similar. Binary_pct still low. φ values
  may drift as the landscape shapes change.
- **Binarization (β=4–128):** each β doubling sharpens the projection.
  Expect some loss spike on each doubling (unknown magnitude — measure).
  λ_feat must NOT collapse to pixel scale — if it does, back off.
- **Polish (β=256):** final sharpening. Binary_pct asymptotes.

**Q2. Does the geometry look healthy?**

Look at `geometry.png` and `gray_zone.png`:

- Is there *visible coherent structure* in the filtered ε? (Waveguides,
  scattering clusters, resonator-like patterns.) Or is it uniform noise?
- `λ_feat` reasonable (≥5 px at β≤32, ideally larger)? Or collapsing to
  Nyquist?
- Are gray pixels at thin interfaces (healthy) or uniformly spread?
- `iso_total_pct` rising? Sign of checkerboard emerging.

**Q3. What does loss trajectory say?**

Look at `loss_trajectory.png` (linear scale!):

- Descending (more negative), flat, or rising (less negative)?
- Did we just cross a β doubling? How long was recovery in prior stages
  *in this run's log*?
- Is current MA-20 meaningfully better (more negative) than MA-prev20,
  or within noise?

**Q4. What do the sensor params say?**

Look at `sensor_trajectory.png`:

- φ values drifting fast (landscape still reshaping) or plateaued
  (coupled structure stable)?
- Jumps at β doublings?

**Q5. What's the bottleneck?**

- β too low (healthy structured geometry, λ_feat good, binary_pct not
  progressing) → advance β.
- β too high / premature (λ_feat collapsing, structure speckly) → back
  off β (rare; consider before advancing again).
- lr_geom too high (post-doubling loss spike > 2× not recovering in
  expected window) → halve lr_geom, freeze β.
- lr_s too high (φ oscillating wildly) → halve lr_s.
- Gradient saturation on ε (grad_sat_pct > 80%) → β can advance
  without disruption.
- Just stochastic noise → hold.
- Wrong basin (rising loss + uniform gray blobs, no recovery) → stop for
  human.

**Q6. Smallest useful intervention?** Prefer: hold > freeze_beta > small
lr tweak > advance β > clear_schedule > stop.

## Step 3 — Decide + act (only if Q5 clearly says act)

All actions via writing `checkpoints_dopt/control.toml`:

**Advance β:**
```toml
beta = <2 * current_beta_from_log>   # capped at 256
```

**Halve geometry lr:**
```toml
lr_geom = <current_lr_geom / 2>      # floor 1e-5
freeze_beta = true
```

**Halve sensor lr:**
```toml
lr_s = <current_lr_s / 2>            # floor 1e-4
```

**Restore lr after recovery:**
```toml
lr_geom = <prior_lr_geom>
freeze_beta = false
```

**Clear schedule:**
```toml
clear_schedule = true
```

**Stop:**
```toml
stop = true
```

Combinations allowed; include only keys you're changing. Write at exactly
`checkpoints_dopt/control.toml`.

## Step 4 — Log reasoning

Append to `autotune_dopt_log.md`:

```
## <ISO-8601 UTC timestamp>  — fire #<N>

### State
- julia_alive: <bool>  step: <int>  loss_ckpt: <float>  β: <float>
- binary_pct: <float>  iso_total_pct: <float>  λ_feat: <float>
- ma20: <float>  prev_ma20: <float>  std20: <float>  iters_at_β: <int>
- iters_since_doubling: <int>  grad_sat_pct: <float>
- grad_s_avg: <3-tuple>

### Diagnosis
- Phase: <warmup | continuous_refine | binarization | polish>
- Geometry: <one sentence — structure/speckle/gray distribution>
- Trajectory: <one sentence — descent/spike/plateau>
- Sensor: <one sentence — drift/plateau>
- Bottleneck: <one Q5 category with brief justification>

### Verdict
<hold | advance_beta | backoff_lr_geom | backoff_lr_s | restore_lr |
 stop_success | stop_plateau | stop_wrong_basin | deferred | error>

### Action
<control.toml contents OR "none">

### Notes
<two-sentence rationale>
```

## Step 4b — Notify on alert-worthy verdicts

Call `./notify.sh "<subject>" "<body>"` for:

- `stop_success` → `notify.sh "dopt: SUCCESS" "step=<N> β=<β> binary=<%>% loss=<ma20>"`
- `stop_plateau` → `notify.sh "dopt: PLATEAU" "<diagnosis>"`
- `stop_wrong_basin` → `notify.sh "dopt: WRONG-BASIN stop" "<diagnosis>"`
- `error` → `notify.sh "dopt: ERROR" "<one line>"`
- Julia first detected dead → `notify.sh "dopt: JULIA DIED" "last iter=<N> β=<β>"`

Do NOT notify on hold / advance_beta / backoff_lr / restore_lr / deferred.

## Step 5 — Memory update (only on completed patterns, and be explicit this is dopt-specific)

Since this is the first d-opt autotune run, **you are the one building
calibration**. When a pattern completes, write to a new dedicated memory
file (NOT the BFIM calibration files):

- Spike fully recovered → write/append `project_autotune_dopt_calibration.md`
  with date, run tag (`dopt_v2_greyscale`), β transition, spike factor,
  recovery window, pre-/post- λ_feat, pre-/post- binary_pct.
- Phase completed → write/append `project_autotune_dopt_phases.md` with
  iter ranges and observed behavior.
- Failure mode manifested → write/append `project_autotune_dopt_failure_modes.md`.
- Run terminated → append run summary + add entry to `MEMORY.md` pointing at
  the new file.

Do NOT update the BFIM files — keep calibration separate per objective.
Do NOT update memory on single-fire observations.

## Invariants

- `checkpoints_dopt/control.toml` exists? → deferred, no write.
- Prior action within 20 min? → cooldown, hold.
- Values inside safety rails? → clamp or refuse.
- Julia dead? → error, no write.

## What you are NOT doing

- Not restarting Julia.
- Not modifying `.jl` source files.
- Not changing `filter_radius` (requires restart).
- Not deleting checkpoints or logs or `.applied.*` files.
- Not sending notifications on routine verdicts.
- Not reusing BFIM calibration as if it were d-opt calibration.
