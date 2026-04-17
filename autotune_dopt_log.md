# D-Optimal Autotune Log

Per-fire chronicle for the `train_dopt_interactive.jl` autotune cron agent.
Each cron fire appends one block here with its state, diagnosis, and verdict.

Run: **dopt_v2_greyscale** (β₀=1.0, passive schedule [200×8]).
Launched: 2026-04-17 (UTC+0 TBD) after v1 speckle failure.

---

## 2026-04-17T18:37Z — fire #1

### State
- julia_alive: true  step: 20  loss_ckpt: -92.696  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.046  λ_feat: 1.41 px
- ma20: -92.001  prev_ma20: NaN  std20: 1.43  iters_at_β: 27
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 3.0e-3  grad_s_avg: (0.258, 0.062, 0.169)

### Diagnosis
- Phase: warmup (β=1, step 20, ~beginning of run)
- Geometry: uniform random init, gray everywhere (96.7% center-gray); filtered panel still noisy because structure hasn't crystallized.
- Trajectory: descending steadily from -86.7 → -94.1 over 27 iters; MA20 = -92.0, MA40 = -90.96 (still in early descent).
- Sensor: φ values drifting smoothly (~0.05–0.25 rad over 27 iters), landscape still reshaping — expected at warmup.
- Bottleneck: none — descent is healthy and we are far too early for any β / lr change.

### Verdict
hold

### Action
none

### Notes
First fire of dopt_v2_greyscale autotune. λ_feat=1.41 px is alarming on its face but is just the random init's high-frequency content showing through the filter at step 20 — structure hasn't emerged yet to lengthen the dominant wavelength. Will revisit at ~step 80–100 when warmup descent should start to flatten and structure should be visible in the filtered panel. No action.

---

## 2026-04-17T18:52Z — fire #2

### State
- julia_alive: true  step: 40  loss_ckpt: -96.159  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.047  λ_feat: 1.41 px
- ma20: -94.805  prev_ma20: -90.368  std20: 0.98  iters_at_β: 41
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 2.28e-3  grad_s_avg: (0.174, 0.0247, 0.0959)

### Diagnosis
- Phase: warmup (β=1, step 40, descent still ongoing)
- Geometry: still uniform random, filtered panel remains noisy; gray_center 94.8% (down 1.9 pts from fire #1). No coherent structure yet — expected.
- Trajectory: smooth monotonic descent, MA20 −92.0 → −94.8 over 20 iters (−2.8). std20 tightened from 1.43 → 0.98. No spikes.
- Sensor: all six φ values drifting smoothly toward lower values (−0.15 rad over 20 iters typical); landscape still reshaping — no plateau.
- Bottleneck: none. Warmup progressing as expected; λ_feat=1.41 still reflects random-init high-freq content, not a speckle trap (β=1, no projection sharpening yet).

### Verdict
hold

### Action
none

### Notes
Descent rate ~0.14 units/iter (linear scale), consistent with warmup. Will revisit near step 80–100 to check whether structure has started emerging in filtered ε before the passive schedule's first β=2 transition at step 200. No action.

---

## 2026-04-17T19:07Z — fire #3

### State
- julia_alive: true  step: 50  loss_ckpt: -97.943  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.04  λ_feat: 1.41 px
- ma20: -97.341  prev_ma20: -93.919  std20: 1.106  iters_at_β: 56
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 2.43e-3  grad_s_avg: (0.0624, 0.0783, 0.0429)

### Diagnosis
- Phase: warmup (β=1, step 50, descent ongoing).
- Geometry: filtered ε still uniform high-freq noise; gray_center 93.5% (down 1.3 pts from fire #2). No coherent structure yet — expected at this stage with β=1.
- Trajectory: smooth monotonic descent, MA20 −94.8 → −97.34 (∆=−2.5 over 10 iters). std20 tightened 0.98 → 1.11 (essentially flat). Loss_min touched −99.24. No spikes.
- Sensor: φ magnitudes shrinking (grad_s_avg 0.04–0.08, down from 0.025–0.17 at fire #2). Some inflections (s2.φ1, s3.φ2 turning) — landscape still reshaping but slowing.
- Bottleneck: none. Warmup proceeding as expected; ~150 iters until passive β=2 transition at step 200.

### Verdict
hold

### Action
none

### Notes
λ_feat=1.41 still reflects unprojected random-init high-freq content; not actionable at β=1. Descent rate slowing slightly (∆=−2.5/10 vs −2.8/20 in fire #2) as expected as the warmup landscape flattens. Will revisit near step 80–100 to check for emerging structure in filtered ε. No action.

## 2026-04-17T19:22Z — fire #4

### State
- julia_alive: true  step: 70  loss_ckpt: -102.048  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.031  λ_feat: 1.41 px
- ma20: -100.361  prev_ma20: -96.452  std20: 1.203  iters_at_β: 71
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 2.45e-3  grad_s_avg: (0.0398, 0.0952, 0.0427)

### Diagnosis
- Phase: warmup (β=1, step 70, descent still ongoing).
- Geometry: filtered ε still noise-dominated, no coherent structure yet; gray_center 89.93% (down 3.6 pts from fire #3 — weakly crystallizing). iso_total_pct 0.031 (flat, no checkerboard pressure).
- Trajectory: smooth monotonic descent, MA20 −97.34 → −100.36 (∆=−3.0 over 20 iters; rate slightly accelerated vs fire #3's −2.5/10). No spikes. Loss_min −102.2.
- Sensor: φ values beginning to plateau — grad_s_avg (0.04, 0.095, 0.043) down from (0.06, 0.08, 0.04), several tracks showing inflection points. Landscape starting to stabilize.
- Bottleneck: none — warmup descent healthy; structure hasn't visibly emerged yet but gray_center dropping slowly. ~130 iters until passive β=2 transition at step 200.

### Verdict
hold

### Action
none

### Notes
Descent continues at ~0.15 units/iter linear. λ_feat=1.41 still not actionable at β=1 (unprojected high-freq random content dominates spectrum). Will revisit near step 100–120 to check whether filtered panel shows any emerging structure before the passive β=2 point at step 200. No action.

## 2026-04-17T19:52Z — fire #6

### State
- julia_alive: true  step: 100  loss_ckpt: -110.305  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.024  λ_feat: 1.41 px
- ma20: -107.481  prev_ma20: -102.213  std20: 1.761  iters_at_β: 100
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 3.09e-3  grad_s_avg: (0.047, 0.042, 0.072)

### Diagnosis
- Phase: warmup (β=1, step 100, descent still ongoing and *accelerating*).
- Geometry: filtered ε still noise-dominated — no coherent waveguide-like features; gray_center 82.1% (down 5.6 pts from fire #5). iso_total_pct 0.024 (no checkerboard). raw_saturated_pct 5.17% — a small fraction of pixels now hitting bounds, first weak sign of crystallization pressure.
- Trajectory: smooth monotonic descent, MA20 −103.59 → −107.48 (∆=−3.89 over ~20 iters); last quarter (iter 75→100) shows faster ~0.28/iter rate vs ~0.15/iter earlier. std20 widened 1.46 → 1.76. No spikes. Loss_best −110.3.
- Sensor: φ values clearly plateaued in last 30 iters — all 6 tracks converged near (1.88–1.93, 2.44–2.47). grad_s_avg (0.047, 0.042, 0.072) slightly elevated vs fire #5's (0.034, 0.076, 0.055) but consistent with noise at low magnitudes.
- Bottleneck: none — warmup descent healthy and accelerating; 100 iters until passive β=2 transition at step 200. Cutting warmup short would forfeit landscape exploration.

### Verdict
hold

### Action
none

### Notes
Accelerating descent + plateaued sensor + monotonic gray_center drop are all consistent with the landscape continuing to shape at β=1. λ_feat=1.41 still reflects unprojected high-freq noise (β≈1 is near-linear) — not actionable. First appearance of raw_saturated_pct ≈ 5% hints structure is beginning to form at pixel scale; will watch next fire whether filtered panel starts to show coherent features before passive β=2 at step 200. No action.

## 2026-04-17T19:37Z — fire #5

### State
- julia_alive: true  step: 80  loss_ckpt: -104.364  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.026  λ_feat: 1.41 px
- ma20: -103.592  prev_ma20: -99.332  std20: 1.458  iters_at_β: 86
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 2.79e-3  grad_s_avg: (0.034, 0.076, 0.055)

### Diagnosis
- Phase: warmup (β=1, step 80, descent still ongoing).
- Geometry: filtered ε still noise-dominated; gray_center 87.7% (down 2.2 pts from fire #4). iso_total_pct 0.026 (slightly decreased — no checkerboard tendency). No coherent structure visible yet in filtered panel.
- Trajectory: monotonic descent, MA20 −100.36 → −103.59 (∆=−3.23 over ~16 iters since last fire). std20 1.20 → 1.46 (slight widening). Loss_min recent −106.10. No spikes.
- Sensor: φ values nearly plateaued — grad_s_avg (0.034, 0.076, 0.055) similar to fire #4. Sensor trajectory plot shows clear flattening over last 20 iters around (φ1≈2.0, φ2≈2.45).
- Bottleneck: none — warmup descent healthy and continuing; ~120 iters until passive β=2 transition at step 200.

### Verdict
hold

### Action
none

### Notes
Descent rate ~0.20 units/iter (linear), slightly faster than fire #4. λ_feat=1.41 still reflects unprojected high-freq init content at β=1 — not actionable. Sensor stabilization is the first weak signal that landscape is settling, but gray fraction still dominant; structure has not crystallized. Will revisit near step 100–120. No action.

## 2026-04-17T20:07Z — fire #7

### State
- julia_alive: true  step: 110  loss_ckpt: -113.349  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.014  λ_feat: 1.41 px
- ma20: -111.987  prev_ma20: -106.012  std20: 1.809  iters_at_β: 115
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 3.09e-3  grad_s_avg: (0.116, 0.091, 0.090)

### Diagnosis
- Phase: warmup (β=1, step 110, descent still ongoing and accelerating).
- Geometry: filtered ε still noise-dominated — no coherent waveguide-like features visible; gray_center 79.3% (down 2.9 pts from fire #6's 82.1%, continuing monotonic decrease). iso_total_pct 0.014 (dropped from 0.024 — actively moving away from checkerboard). Projected at 0% binary as expected at β=1.
- Trajectory: smooth monotonic descent, MA20 −107.48 → −111.99 (∆=−4.51 over ~10 iters). 20-v-prev20 ∆=−5.98 (accelerated vs fire #6's ∆=−5.27). No spikes. Loss_best_ever −114.9 (set at recent iter).
- Sensor: all 6 φ tracks plateaued in (1.88–2.55) since ~iter 90; slight uptick at tail but within noise band. grad_s_avg (0.12, 0.09, 0.09) marginally elevated but consistent with descent phase.
- Bottleneck: none — warmup descent healthy and still accelerating; 85–90 iters until passive β=2 transition at step 200.

### Verdict
hold

### Action
none

### Notes
Descent rate has nearly doubled since fire #4 (∆=−0.30/iter now vs ∆=−0.15/iter at step 70). iso_total_pct trending down is a good sign — no checkerboard pressure emerging. λ_feat=1.41 still reflects high-freq content on unprojected greyscale (β=1 is near-linear); not actionable and not expected to show coherent structure until β ≥ 4. Will revisit near step 150–180 for early signs of filtered-panel structure before passive β=2. No action.

## 2026-04-17T20:22Z — fire #8

### State
- julia_alive: true  step: 120  loss_ckpt: -116.578  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.014  λ_feat: 1.41 px
- ma20: -116.418  prev_ma20: -110.168  std20: 1.925  iters_at_β: 129
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 3.18e-3  grad_s_avg: (0.124, 0.106, 0.074)

### Diagnosis
- Phase: warmup (β=1, step 120, descent still ongoing).
- Geometry: filtered ε still noise-dominated — no coherent waveguide-like features; gray_center 76.5% (continuing monotonic drop: fire #5 87.7 → #6 82.1 → #7 79.3 → now 76.5). iso_total_pct 0.014 (no checkerboard). raw_saturated_pct 5.76% (up from 5.17% at fire #6) — crystallization pressure slowly building.
- Trajectory: smooth monotonic descent, MA20 -111.99 → -116.42 (∆=-4.43 over ~10 iters). 20-vs-prev20 ∆=-6.25 (comparable to fire #7's -5.98). No spikes. Loss_best_ever -119.5.
- Sensor: plateau broken — upward drift in last ~20 iters (s1.φ1 from 1.88 to 1.92, s3.φ2 from 2.44 to 2.65). grad_s_avg (0.12, 0.11, 0.074) elevated vs prior fires. Landscape still reshaping on the sensor side as geometry begins to commit.
- Bottleneck: none — descent healthy, gray dropping steadily, ~70 iters until passive β=2 transition at step 200.

### Verdict
hold

### Action
none

### Notes
Cooldown: 15 min since fire #7 (< 20 min window). Independent of cooldown, descent continues robustly (∆MA20 ≈ -6 per 20 iters, on par with recent fires). λ_feat=1.41 expected at β=1 (unprojected high-freq content). Sensor re-drift is an interesting first weak signal that geometry commitment is starting to perturb the sensor basin — will watch for re-plateau before β=2. No action.

## 2026-04-17T20:37Z — fire #9

### State
- julia_alive: true  step: 140  loss_ckpt: -122.023  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.012  λ_feat: 1.41 px
- ma20: -120.754  prev_ma20: -114.808  std20: 1.386  iters_at_β: 144
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 9.78e-3  grad_s_avg: (0.055, 0.054, 0.047)

### Diagnosis
- Phase: warmup (β=1, step 140, descent still ongoing and smooth).
- Geometry: filtered ε still noise-dominated, no coherent waveguide-like features; gray_center 72.08% (monotonic drop continues: #5 87.7 → #6 82.1 → #7 79.3 → #8 76.5 → now 72.1, ∆=4.4 pts in 20 iters). iso_total_pct 0.012 (dropped further from 0.014 — actively avoiding checkerboard). raw_saturated_pct 6.0% (rising from 5.76 at fire #8 — crystallization pressure still building).
- Trajectory: smooth monotonic descent, MA20 −116.42 → −120.75 (∆=−4.33 over ~20 iters). 20-vs-prev20 ∆=−5.95 (comparable to fire #8's −6.25, very slight deceleration). No spikes. Loss_best_ever −122.6.
- Sensor: φ values drifting upward in last ~40 iters — s1.φ2 and s3.φ2 climbing from 2.45 to ~2.67; s2.φ1 rising from 2.0 to 2.45. grad_s_avg moderate (~0.05 all tracks). Landscape still reshaping on sensor side as geometry commits — second re-drift phase after earlier plateau.
- Bottleneck: none — warmup descent healthy, sensor still evolving, structure not yet visible in filtered panel; 60 iters until passive β=2 at step 200.

### Verdict
hold

### Action
none

### Notes
Cooldown: all prior fires were holds, so no active cooldown restriction. Descent rate sustained (~0.30/iter linear). λ_feat=1.41 expected and not actionable at β=1 (projection near-linear — nothing about speckle detection yet). Sensor re-drift is consistent with the landscape continuing to shape as raw_saturated_pct climbs; this is why cutting warmup short would be premature. Will revisit near step 170–190 before the passive β=2 transition. No action.

## 2026-04-17T20:52Z — fire #10

### State
- julia_alive: true  step: 150  loss_ckpt: -123.397  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.014  λ_feat: 1.41 px
- ma20: -123.219  prev_ma20: -119.478  std20: 0.659  iters_at_β: 159
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 0.01499  grad_s_avg: (0.044, 0.049, 0.050)

### Diagnosis
- Phase: warmup (β=1, step 150, descent slowing near a likely early-plateau bottom of the continuous objective).
- Geometry: filtered ε still noise-dominated, no coherent waveguide features; gray_center 70.79% (monotonic drop continues: #7 79.3 → #8 76.5 → #9 72.1 → now 70.8, decelerating). iso_total_pct 0.014 stable. raw_saturated_pct 5.91 (slight dip from 6.0 at fire #9 — crystallization pressure pausing).
- Trajectory: smooth monotonic descent, MA20 −120.75 → −123.22 (∆=−2.46 over ~20 iters). 20-vs-prev20 ∆=−3.74 (continued deceleration: #7 −5.98 → #8 −6.25 → #9 −5.95 → now −3.74). std20 dropped to 0.66 (tighter variance, consistent with flattening). Loss_best_ever −124.2.
- Sensor: re-plateaued — after the upward drift at fires #8–#9 settled with s1.φ2/s3.φ2 at ~2.65 and φ1 tracks at ~2.00 (all 6 tracks nearly flat over last ~10 iters). grad_s_avg (0.04, 0.05, 0.05) lowest recorded this run — clearest "sensor done reshaping" signal yet.
- Bottleneck: none — descent slowing is expected near continuous minimum; λ_feat=1.41 at β=1 is projection artifact not speckle trap; 50 iters until passive β=2 transition at step 200.

### Verdict
hold

### Action
none

### Notes
Cooldown: 15 min since fire #9 (< 20-min window) — independent of cooldown, no action warranted. Descent decelerating and sensor plateauing together is the expected signature of approaching the continuous-β minimum; this is what we want to see before β=2. Gray_center drop decelerating from 4.4 pts/20iters (fire #9) to 1.3 pts/20iters now confirms geometry is nearing its continuous fixed point. Will revisit near step 180–200 before the passive β=2 transition to check whether an early β advance is warranted or the schedule can fire naturally.

## 2026-04-17T21:07Z — fire #11

### State
- julia_alive: true  step: 170  loss_ckpt: -124.770  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.008  λ_feat: 1.41 px
- ma20: -124.458  prev_ma20: -122.596  std20: 0.350  iters_at_β: 174
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 0.01836  grad_s_avg: (0.038, 0.036, 0.035)

### Diagnosis
- Phase: warmup (β=1, step 170, descent near continuous-β fixed point but still nonzero).
- Geometry: filtered ε still noise-dominated — no coherent waveguide/resonator features yet; gray_center 69.1% (continuing slow drop: #7 79.3 → #8 76.5 → #9 72.1 → #10 70.8 → now 69.1, ∆=1.7 pts/20 iters — further decelerated). iso_total_pct dropped to 0.008 (run-min — actively avoiding checkerboard). raw_saturated_pct 5.84 (stable, no further crystallization pressure).
- Trajectory: smooth monotonic descent, MA20 −123.22 → −124.46 (∆=−1.24 over ~20 iters). 20-vs-prev20 ∆=−1.86 (halving each fire: #8 −6.25 → #9 −5.95 → #10 −3.74 → now −1.86). std20 tightest yet at 0.35 (was 0.66 at fire #10). Loss_best_ever −125.0.
- Sensor: fully plateaued — all 6 φ tracks nearly flat since ~iter 140; grad_s_avg (0.038, 0.036, 0.035) lowest of run, three-way symmetric. Coupled geometry-sensor landscape has settled at β=1.
- Bottleneck: none — approaching continuous minimum, still residual signal; 30 iters from passive β=2 at step 200.

### Verdict
hold

### Action
none

### Notes
No cooldown (all prior fires were holds). Descent halving per-fire + std tightening + sensor fully converged = classic signature of β=1 fixed point; but ~∆−1.24 MA improvement in last 20 iters is still non-zero, so passive β=2 at step 200 will fire with slightly deeper continuous minimum than now. Geometry still shows no coherent features in filtered panel — β advance wouldn't help structure emerge (projection doesn't create features, gradients do), and β=2 projection is near-identity anyway. Will evaluate passive β=1→2 transition response at next fire (step 190–200 window).

## 2026-04-17T21:22Z — fire #12

### State
- julia_alive: true  step: 180  loss_ckpt: -125.313  β: 1.0
- binary_pct: 0.0  iso_total_pct: 0.011  λ_feat: 1.41 px
- ma20: -125.232  prev_ma20: -124.131  std20: 0.276  iters_at_β: 189
- iters_since_doubling: -1 (no doubling yet)  grad_sat_pct: 0.0
- grad_avg: 0.01943  grad_s_avg: (0.031, 0.031, 0.033)

### Diagnosis
- Phase: warmup (β=1, step 180, ~10 iters from passive β=2 at step 200).
- Geometry: filtered ε still noise-dominated — no coherent waveguide/resonator features; gray_center 68.46% (slow drop continues: #9 72.1 → #10 70.8 → #11 69.1 → now 68.5, ∆=0.7 pts/20 iters — asymptoting). iso_total_pct 0.011 (still run-low; no checkerboard). raw_saturated_pct 5.85 (flat).
- Trajectory: smooth monotonic descent with clear asymptotic curvature in plot (tail flattens); MA20 −124.46 → −125.23 (∆=−0.77 over ~20 iters). 20-vs-prev20 ∆=−1.10 (continued halving: #10 −3.74 → #11 −1.86 → now −1.10). std20 at run-min 0.276 (was 0.35 at fire #11). Loss_best_ever −125.7.
- Sensor: fully plateaued — all 6 φ tracks essentially flat since ~iter 150; clear three-fold symmetry (three φ₁ at 1.99, three φ₂ at 2.63). grad_s_avg (0.031, 0.031, 0.033) — run-min, three-way symmetric.
- Bottleneck: none — classic β=1 fixed-point approach (descent halving + std tightening + sensor symmetric plateau). No speckle threat at β=1 (projection near-identity so λ_feat=1.41 is projection artifact, not design flaw).

### Verdict
hold

### Action
none

### Notes
No cooldown constraint (all prior verdicts holds). The β=1 continuous minimum is essentially reached — residual descent (∆−0.77 per 20 iters, std₂₀ 0.28) is near-noise. Passive β=1→2 transition at step 200 will fire in ~10 min; deferring all action until we can observe the response — that's the first real calibration datapoint for this objective and the first real test of whether structure emerges under projection. Acting now (manual advance or lr tweak) would destroy that natural-response observation.

## 2026-04-17T21:37Z — fire #13

### State
- julia_alive: true  step: 200  loss_ckpt: -106.839  β: 2.0
- binary_pct: 0.0  iso_total_pct: 0.011  λ_feat: 1.41 px
- ma20: -121.813  prev_ma20: -124.993  std20: 7.111  iters_at_β: 5
- iters_since_doubling: 4  grad_sat_pct: 0.0
- grad_avg: 0.02477  grad_s_avg: (0.130, 0.129, 0.129)

### Diagnosis
- Phase: continuous_refine (β=2, step 200, just 4 iters past passive β=1→2 transition).
- Geometry: filtered ε shows small blob texture (~5-10 px scale, consistent with R=5 filter) — no coherent waveguide features yet. Projected panel gray_center dropped sharply from 68.5% (fire #12) to 57.0% — β=2 projection is actively committing pixels, but commit pattern matches filtered blobs, not speckle. iso_total_pct 0.011 stable (no checkerboard). raw_saturated_pct 5.87 flat. λ_feat=1.41 is projection-edge artifact (tanh sharpens the filtered-blob boundaries); the filtered panel does NOT show pixel-scale Nyquist noise.
- Trajectory: classic post-doubling spike. Loss_best_ever −126.0 (iter 199), checkpoint at iter 200 −106.84 (spike factor ~0.85 retained, Δloss +19.2 linear). MA20 −124.99 → −121.81 (+3.18, first MA rise of run). std20 exploded 0.276 → 7.111. Only 4 iters of recovery data — insufficient.
- Sensor: plateau briefly broken at the β step — all 6 φ tracks bent downward at iter 200 (visible in sensor_trajectory.png). grad_s_avg (0.13, 0.13, 0.13) 4× pre-doubling (was 0.031 at fire #12) — sensor landscape reshaping, expected. Three-way φ symmetry preserved.
- Bottleneck: none actionable — post-doubling spike is expected; recovery window unknown for this objective (first data point).

### Verdict
hold

### Action
none

### Notes
Cooldown: 15 min since fire #12 (< 20 min) — hold mandatory. Independent of cooldown, acting 4 iters into an unknown-magnitude recovery would destroy the first d-opt β-transition calibration datapoint. The filtered panel (middle) shows coherent ~5-10px blob structure — NOT speckle — so the playbook's λ_feat<2 alarm is a false positive at β=2 (projection-edge artifact on otherwise-smooth filtered field). Will revisit at step 215-230 to assess spike recovery rate and when MA20 begins descending again. Calibration target: note iters-to-recover-to-best-ever and pre/post λ_feat when recovery completes.

## 2026-04-17T21:52Z — fire #14

### State
- julia_alive: true  step: 210  loss_ckpt: -116.955  β: 2.0
- binary_pct: 0.0  iso_total_pct: 0.008  λ_feat: 1.41 px
- ma20: -111.92  prev_ma20: -125.656  std20: 3.649  iters_at_β: 20
- iters_since_doubling: 19  grad_sat_pct: 0.0
- grad_avg: 0.0545  grad_s_avg: (0.486, 0.486, 0.496)

### Diagnosis
- Phase: continuous_refine (β=2, step 210, 19 iters past first β transition of run).
- Geometry: filtered ε shows small blob texture (~5-10 px scale, R=5 filter); no coherent waveguide features yet. Projected gray_center 57.12% (flat since fire #13's 57.0%). iso_total_pct 0.008 (run-min, no checkerboard). raw_saturated_pct dropped 5.87 → 1.99 since fire #13 — β=2 projection pressure is reshaping extreme ε values toward a gentler distribution. λ_feat=1.41 is projection-edge artifact (sharpened blob boundaries), not speckle — filtered panel has clear ~5-10 px coherence.
- Trajectory: oscillating recovery from post-doubling spike. Losses visibly bounce -107 → -120 → -109 → -120 → settling near -112. MA20 -125.66 → -111.92 (Δ+13.74 regression — but expected, window is now all post-doubling). std20 3.65 (was 7.11 at fire #13 — variance tightening as oscillation damps). Loss_best_ever still -126.0 (pre-doubling iter 199). Not yet recovered.
- Sensor: three-fold φ₁ symmetry BROKE at the β=1→2 transition — visible in sensor_trajectory.png at iter 200: φ₁ tracks diverged from symmetric 2.00 to ~1.65 / ~1.82. φ₂ tracks remain nearly symmetric at ~2.68. grad_s_avg (0.49, 0.49, 0.50) is 4× fire #13 — sensor landscape still actively reshaping. New d-opt observation: projection can break prior sensor symmetries.
- Bottleneck: none actionable — recovery window still open (19 iters post-doubling, oscillation not damped); acting now would corrupt the first β-transition calibration datapoint. Grad_avg elevated (3× pre-doubling) confirms active correction.

### Verdict
hold

### Action
none

### Notes
No cooldown constraint (all prior verdicts holds). Standard d-opt-unknown recovery window: I chose to hold for the full first spike to build the calibration datapoint (pre-ma, post-ma, iters-to-recover, λ_feat trend, sensor-symmetry-broken flag). Current recovery trajectory looks healthy: std20 halved since fire #13, grad tail shortening (histogram). Next fire target: step 225-235, by which point either (a) MA20 is within ~2 of -125 pre-doubling → full recovery, or (b) still oscillating — diagnose lr_geom. No notification (not alert-worthy).

## 2026-04-17T22:07Z — fire #15

### State
- julia_alive: true  step: 230  loss_ckpt: -117.190  β: 2.0
- binary_pct: 0.0  iso_total_pct: 0.008  λ_feat: 1.41 px
- ma20: -115.872  prev_ma20: -115.906  std20: 4.191  iters_at_β: 35
- iters_since_doubling: 34  grad_sat_pct: 0.0
- grad_avg: 0.03877  grad_s_avg: (0.483, 0.510, 0.556)

### Diagnosis
- Phase: continuous_refine (β=2, step 230, 34 iters past the first β=1→2 transition).
- Geometry: filtered ε still shows coherent ~5-10 px blob texture (R=5 imposed scale), no coherent waveguide/resonator features yet. Projected panel gray_center 57.48% (stable since fire #13 post-doubling commit; no further commitment in last 20 iters). iso_total_pct 0.008 (run-min, no checkerboard). raw_saturated_pct 2.52 (further dropped from 1.99 at fire #14 — β=2 projection continues to reshape extreme ε values). λ_feat=1.41 is projection-edge artifact (tanh sharpens filtered-blob boundaries); filtered panel lacks pixel-scale Nyquist noise, so the playbook's λ_feat<2 alarm is a false positive at β=2 with R=5.
- Trajectory: post-doubling recovery still in progress. Loss trajectory plot shows MA20 climbed to ~-111 peak near iter 220 then is descending again toward ~-116 at iter 230. MA20 -111.92 → -115.87 (Δ=-3.95 since fire #14), but ma20 vs prev_ma20 within this fire is near-flat (-115.87 vs -115.91) — meaning the bulk of the -3.95 drop happened in the prev20 window (iters ~210-220 recovery tail), and current 20 has flattened. std20 4.19 (slight uptick from 3.65 at fire #14 — still oscillating). Loss_best_ever still -126.0 (pre-doubling iter 199); gap ~10 remaining.
- Sensor: three-fold φ₁ symmetry-break progressing — φ₁ tracks now at 1.52 / 1.65 / 1.80 (was 1.65 / 1.82 at fire #14), still actively diverging. φ₂ tracks remain nearly symmetric at ~2.68. grad_s_avg (0.48, 0.51, 0.56) barely moved from fire #14's (0.49, 0.49, 0.50) — sensor landscape reshaping is *not* done. Three-fold symmetry may not be preserved under β=2 projection; waiting for sensor to find its new fixed point.
- Bottleneck: none actionable. Recovery is partially complete (Δ=-3.95 MA20 over 20 iters), but (a) sensor not settled (grad_s still ~15× pre-doubling) and (b) MA20 in last 20 iters is flat (-115.87 vs -115.91) — either genuine plateau at a β=2 local min, or pause before next descent. Acting mid-pause would corrupt calibration.

### Verdict
hold

### Action
none

### Notes
No cooldown (all prior verdicts holds). First d-opt β-transition recovery is still building — 34 iters post-doubling, sensor still actively reshaping, MA20 just flattened in last 20 iters (first plateau-or-pause signal). Spike factor so far: pre-doubling best -126.0, post-doubling MA20 peak -111.92, now -115.87 → partial recovery (~40% of the -15-unit gap closed so far). Decision to wait longer: (1) need more data to distinguish plateau from pause, (2) sensor grad still elevated, (3) premature β advance on incomplete recovery risks double-spike. Next fire target step 245-255 — if MA20 still -116 ± 1 and sensor grad_s still > 0.3, diagnose as β=2 plateau and consider advance; if MA20 continuing to descend, continue hold.

## 2026-04-17T22:22Z — fire #16

### State
- julia_alive: true  step: 240  loss_ckpt: -122.651  β: 2.0
- binary_pct: 0.0  iso_total_pct: 0.011  λ_feat: 1.41 px
- ma20: -122.138  prev_ma20: -114.13  std20: 2.105  iters_at_β: 50
- iters_since_doubling: 49  grad_sat_pct: 0.0
- grad_avg: 0.04073  grad_s_avg: (0.452, 0.456, 0.494)

### Diagnosis
- Phase: continuous_refine (β=2, step 240, 49 iters past first β=1→2 transition).
- Geometry: filtered ε shows coherent ~5-10 px blob texture (R=5 imposed scale) — no waveguide/resonator features yet. Projected gray_center 57.69% (flat vs fire #15's 57.48 — commitment plateau at β=2). iso_total_pct 0.011 (no checkerboard). raw_saturated_pct 1.98 (down from fire #15's 2.52). λ_feat=1.41 remains projection-edge artifact, not Nyquist speckle; gray_zone.png shows uniform under-commitment (whole field hovering at gray_center≈0.7), not high-frequency pattern.
- Trajectory: post-doubling recovery in its strongest descent. MA20 -122.14 vs prev_ma20 -114.13 = Δ-8.0 (biggest recovery drop since fire #13). std20 2.11 halved from fire #15's 4.19 and quartered from fire #13's 7.11 — oscillation damping rapidly. loss_min_recent = loss_best_ever = -126.0 → current window has already matched pre-doubling best. Plot's light-cyan trace dives toward -125 at iter 250. Full MA20 recovery to ≤-125 imminent (within ~10-20 iters at current rate).
- Sensor: φ₁ tracks rebounding from dip — bottomed ~1.5 at iter ~230, now rising through 1.8/1.9/2.0 at step 250. φ₂ tracks at ~2.6-2.7 (slight separation between s₁,s₂ and s₃ for φ₂ emerging at iter 250). grad_s_avg (0.45, 0.46, 0.49) slightly down from fire #15's (0.48, 0.51, 0.56), still ~10× pre-doubling. Sensor landscape still actively reshaping — not stable.
- Bottleneck: none actionable. Holding through the strongest descent phase of the first calibration datapoint.

### Verdict
hold

### Action
none

### Notes
No cooldown constraint (all prior verdicts holds since 17:14Z applied). This fire captures the meat of the first d-opt β-transition recovery: ma20 descent of -8 over 20 iters, std halved, loss_min_recent back to best_ever. Not yet recorded as "recovered" because (a) MA20 -122.14 still above pre-doubling MA20 ~-125, (b) sensor grad_s still ~10× baseline — reshaping ongoing. Next fire target step 255-270: if ma20 ≤ -125 and grad_s ≤ 0.15, declare recovery complete and write first entry to project_autotune_dopt_calibration.md (β=1→2 transition datapoint). If ma20 plateaus at -122 ± 1 with grad_s still elevated, diagnose as β=2 local min and consider advance.
