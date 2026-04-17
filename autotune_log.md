## 2026-04-16T14:15:12Z  — fire #1

### State
- julia_alive: true  step: 210  loss_ckpt: 6.49e-4  β: 16.0
- binary_pct: 10.61  iso_total_pct: 0.018  gray_center_pct: 10.62
- ma20: 4.85e-4  prev_ma20: 5.17e-4  std20: 1.42e-4  iters_at_β: 219
- iters_since_doubling: -1 (no doubling parsed; run started at β=16)
- grad_sat_pct: 0.0  grad_avg_recent: 3.59e-6

### Diagnosis
- Phase: plateau (β=16, binary_pct still ~10%; meaningful binarization expected only at next doubling per memory)
- Geometry: speckled raw, smooth filtered, projected still mostly noise; gray distributed (not blob), iso pixels negligible — healthy
- Trajectory: clean monotonic log-scale descent 3e-2 → 5e-4 over 210 iters; MA20/prev ratio 0.94 (still improving); std20/MA20 ≈ 30% normal stochastic noise
- Bottleneck: none active. Loss still improving and geometry uncommitted but healthy. Pushing β now would interrupt useful descent.

### Verdict
hold

### Action
none

### Notes
First fire on empty log; no cooldown to honor. Loss is still descending and geometry shows no failure signatures (no wrong-basin blobs, no min-feature islands, no gradient saturation). Wait for either plateau (ma_ratio→1.0) or geometry to commit further before considering β advance.

## 2026-04-16T14:30:09Z  — fire #2

### State
- julia_alive: true  step: 220  loss_ckpt: 6.22e-4  β: 16.0
- binary_pct: 10.63  iso_total_pct: 0.017  gray_center_pct: 10.61
- ma20: 4.34e-4  prev_ma20: 4.74e-4  std20: 1.67e-4  iters_at_β: 229
- iters_since_doubling: -1 (run started at β=16; no doubling parsed)
- grad_sat_pct: 0.0  grad_avg_recent: 3.56e-6

### Diagnosis
- Phase: plateau / pre-binarization (β=16, binary_pct still 10.6%; real binarization awaits next doubling per memory)
- Geometry: raw speckled, filtered smoothly uncommitted, projected mostly noise; gray distributed broadly (not blob, not thin-interface yet); iso pixels negligible — healthy but uncommitted
- Trajectory: monotonic log-scale descent continuing; MA20/prev ratio 0.915 (still genuinely improving, not noise); std20/ma20 ≈ 38% stochastic; no spikes
- Bottleneck: none. Loss actively improving ~9% per 20-iter window and geometry uncommitted. Advancing β now would disrupt productive descent; lr change unwarranted.

### Verdict
hold

### Action
none

### Notes
15 min since prior fire (also hold) — no cooldown constraint but same reasoning holds. Ratio 0.915 is notable improvement from 0.94 last fire, so descent is actually accelerating slightly. Will check again next fire for plateau signal (ma_ratio → 1.0) or geometry commitment.

## 2026-04-16T14:45:09Z  — fire #3

### State
- julia_alive: true  step: 230  loss_ckpt: 6.40e-4  β: 16.0
- binary_pct: 10.637  iso_total_pct: 0.016  gray_center_pct: 10.627
- ma20: 4.88e-4  prev_ma20: 4.85e-4  std20: 2.08e-4  iters_at_β: 239
- iters_since_doubling: -1 (run started at β=16)
- grad_sat_pct: 0.0  grad_avg_recent: 4.67e-6

### Diagnosis
- Phase: plateau at β=16 (geometry emergence complete; binarization pending)
- Geometry: raw speckled, filtered smooth, projected showing faint structural signals but still ~89% gray; gray zone broadly distributed (no blobs, no min-feature islands). Healthy but uncommitted.
- Trajectory: 239 iters at β=16 took loss from ~3e-2 to ~5e-4 (60x). MA20/prev20 now 1.007 (was 0.94 then 0.915) — improvement has genuinely flattened. std20/ma20 ≈ 43% so ratio is within noise, but direction has clearly reversed from improving.
- Bottleneck: β too low. Binary_pct plateaued at 10.6% across three consecutive fires (10.61 → 10.63 → 10.64, ~30 iters span). Geometry is healthy enough to survive a doubling; gradient saturation 0% so plenty of headroom.

### Verdict
advance_beta

### Action
```toml
beta = 32
clear_schedule = true
```

### Notes
Smallest useful intervention: single β doubling to trigger next binarization phase. Cleared schedule per playbook on manual advance. Expect short post-doubling loss spike per calibration memory; will watch for recovery over next 2–3 fires before any lr adjustment.

## 2026-04-16T15:00:16Z  — fire #4

### State
- julia_alive: true  step: 240  loss_ckpt: 1.80e-4  β: 32.0
- binary_pct: 10.65  iso_total_pct: 0.018  gray_center_pct: 10.60
- ma20: 6.38e-2  prev_ma20: 4.40e-4  std20: 1.42e-1  iters_at_β: 8
- iters_since_doubling: 7  grad_sat_pct: 0.0  grad_avg_recent: 1.55e-3

### Diagnosis
- Phase: binarization (β just advanced 16→32 at iter 241 by fire #3)
- Geometry: raw still speckled, filtered smooth, projected unchanged at 10.7% binary — too early for binarization to manifest. Gray zone broadly distributed (no blobs, no min-feature islands). Healthy.
- Trajectory: clean monotonic descent through iter 240 (~3e-2 → 5e-4), then sharp spike to ~6e-2 at iter 241+ from β doubling. ma_ratio_20/prev20 = 145× — within expected post-doubling magnitude per calibration memory; loss_min_recent = 1.59e-4 shows individual iters already touching pre-spike levels.
- Bottleneck: none actionable. Mid post-doubling transient; need recovery time.

### Verdict
hold

### Action
none

### Notes
20-min cooldown not satisfied (15 min since fire #3 advance_beta). Even without cooldown, 7 iters is far too early to judge spike recovery — calibration memory says wait 2–3 fires. loss_min_recent already touching 1.59e-4 is encouraging. Will reassess next fire.

## 2026-04-16T15:15:10Z  — fire #5

### State
- julia_alive: true  step: 250  loss_ckpt: 2.53e-2  β: 32.0
- binary_pct: 42.58  iso_total_pct: 0.016  gray_center_pct: 5.36
- ma20: 7.33e-2  prev_ma20: 4.60e-4  std20: 1.38e-1  iters_at_β: 18
- iters_since_doubling: 17  grad_sat_pct: 0.0  grad_avg_recent: 1.82e-3

### Diagnosis
- Phase: binarization (Phase 3, mid-recovery from 16→32 doubling at iter 241)
- Geometry: projected jumped 10.6% → 42.6% binary in 17 iters — projection is doing its work. Filtered ε showing organized smooth structure; gray zone now concentrated at emerging interfaces (not contiguous blob, not noise). iso_total_pct 0.016% negligible. Healthy and committing fast.
- Trajectory: classic post-doubling spike from ~5e-4 → MA20 7.33e-2 (~159× factor — well below the 4000× reference for 16→32 in calibration memory). loss_min_recent 1.59e-4 shows individual iters already touching pre-spike levels — recovery underway. std20/ma20 ≈ 1.9 expected during transient.
- Bottleneck: none actionable. Mid-recovery; calibration says don't back off in first 30–50 iters post-spike — we're at 17.

### Verdict
hold

### Action
none

### Notes
Cooldown satisfied (30 min since fire #3 advance_beta) but the diagnostic itself says hold: spike factor is reassuringly small relative to reference, binarization is rapidly progressing, and individual iters already touching pre-spike loss. Will reassess next fire when iters_since_doubling ≈ 32.

## 2026-04-16T15:30:14Z  — fire #6

### State
- julia_alive: true  step: 260  loss_ckpt: 1.40e-2  β: 32.0
- binary_pct: 42.75  iso_total_pct: 0.014  gray_center_pct: 5.39
- ma20: 1.48e-2  prev_ma20: 6.38e-2  std20: 1.45e-2  iters_at_β: 28
- iters_since_doubling: 27  grad_sat_pct: 0.0  grad_avg_recent: 4.04e-4

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling recovery from 16→32 at iter 241)
- Geometry: projection now shows identifiable waveguide-like structural features at 42.7% binary; filtered ε is smooth and organized; gray zone concentrates at interfaces (not blobs, not 1px noise). iso_total_pct 0.014% negligible. Healthy and committed to new β=32 plateau.
- Trajectory: clear post-spike recovery on log-scale plot. MA20 dropped 0.073 → 0.015 (4.3× improvement) in 10 iters since last fire. ma_ratio_20/prev20 = 0.23 (genuine recovery, not noise). loss_min_recent 1.59e-4 confirms individual iters back at pre-spike levels. grad_avg dropped 1.82e-3 → 4.04e-4 (4.5×) — gradient calming as expected.
- Bottleneck: none actionable. Mid-recovery proceeding cleanly; spike factor was only ~159× (well below calibration reference of ~4000× for 16→32) so recovery should complete sooner than typical.

### Verdict
hold

### Action
none

### Notes
Cooldown satisfied (45 min since fire #3 advance_beta). Recovery is healthy and ahead of schedule per calibration memory — interrupting now (lr backoff) would be unwarranted. Expect MA20 to converge back to pre-spike range over next 1–2 fires; will then assess if β=32 has reached its own plateau before considering next doubling to 64.

## 2026-04-16T15:45:10Z  — fire #7

### State
- julia_alive: true  step: 270  loss_ckpt: 6.11e-3  β: 32.0
- binary_pct: 42.91  iso_total_pct: 0.014  gray_center_pct: 5.34
- ma20: 8.15e-3  prev_ma20: 7.18e-2  std20: 5.68e-3  iters_at_β: 36
- iters_since_doubling: 35  grad_sat_pct: 0.0  grad_avg_recent: 2.03e-4

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling recovery from 16→32 at iter 241)
- Geometry: projected shows clear waveguide-like structural features at 42.9% binary; filtered ε smooth and organized; gray zone concentrated at interfaces (bright edges — healthy thin-interface pattern, not blobs, not 1px noise). iso_total_pct 0.014% negligible. Committed cleanly to new β regime.
- Trajectory: classic post-spike exponential recovery on log scale. MA20 dropped 1.48e-2 → 8.15e-3 in 10 iters (1.8×); ma_ratio_20/prev20 = 0.114 (genuine improvement, std20/ma20 ≈ 0.70). Still ~15× above pre-spike MA (~5e-4); grad_avg halved again (4.04e-4 → 2.03e-4) — gradient continues calming.
- Bottleneck: none actionable. Recovery proceeding faster than calibration worst-case; no plateau signal yet at new β.

### Verdict
hold

### Action
none

### Notes
60 min since last non-hold action (fire #3 advance_beta) — cooldown satisfied, but diagnostic says hold. Descent ratio 0.114 is well below noise threshold, so interrupting would waste productive iters. Will reassess next fire for MA20 stabilization signal (ratio → ~1.0) before considering β=64 doubling.

## 2026-04-16T16:00:10Z  — fire #8

### State
- julia_alive: true  step: 280  loss_ckpt: 1.89e-3  β: 32.0
- binary_pct: 42.96  iso_total_pct: 0.013  gray_center_pct: 5.33
- ma20: 3.95e-3  prev_ma20: 1.96e-2  std20: 3.12e-3  iters_at_β: 44
- iters_since_doubling: 43  grad_sat_pct: 0.0  grad_avg_recent: 9.36e-5

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling recovery from 16→32 at iter 241)
- Geometry: projected at 43% binary shows identifiable waveguide-like structures (vertical corridors, blocky features); filtered ε smooth and organized; gray zone image shows bright edges concentrated at thin interfaces — the healthy sharpening pattern, not contiguous blobs, not 1px noise. iso_total_pct 0.013% negligible.
- Trajectory: continued clean exponential recovery on log scale. MA20 dropped 8.15e-3 → 3.95e-3 over 10 iters (2.1×); ma_ratio_20/prev20 = 0.20 (genuine descent, well below std20/ma20 ≈ 0.79 noise floor). Still ~8.5× above pre-spike MA20 (~4.6e-4); grad_avg halved again (2.03e-4 → 9.36e-5) — gradient continues calming.
- Bottleneck: none actionable. Recovery still producing 2×/10-iter improvement; no plateau signal yet.

### Verdict
hold

### Action
none

### Notes
Cooldown satisfied (75 min since fire #3 advance_beta), but diagnostic says hold — MA20 is still dropping materially and grad is still halving each fire. Pre-spike plateau was ~4.6e-4; current 3.95e-3 implies ~3 more fires of similar descent rate to reach plateau before β=64 advance should be considered. Inaction is the default when descent is still productive.

## 2026-04-16T16:15:08Z  — fire #9

### State
- julia_alive: true  step: 290  loss_ckpt: 2.47e-3  β: 32.0
- binary_pct: 43.01  iso_total_pct: 0.013  gray_center_pct: 5.33
- ma20: 2.35e-3  prev_ma20: 1.39e-2  std20: 5.61e-4  iters_at_β: 51
- iters_since_doubling: 50  grad_sat_pct: 0.0  grad_avg_recent: 3.69e-5

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling recovery from 16→32 at iter 241)
- Geometry: projected shows the same waveguide-like structures as prior 2 fires at 43.0% binary — binarization saturated at this β. Filtered ε smooth and organized. Gray zone image shows bright ribbons tracing thin interfaces (textbook healthy sharpening pattern), no contiguous blobs, no 1px noise. iso_total_pct 0.013% negligible.
- Trajectory: loss_trajectory.png shows classic post-spike exponential recovery. MA20 3.95e-3 → 2.35e-3 over 10 iters (1.68×, slowing from ~2×); ma_ratio_20/prev20 = 0.169 (genuine descent; noise floor std20/ma20 ≈ 0.24). Still ~5× above pre-spike MA20 (~4.6e-4). grad_avg halved (9.36e-5 → 3.69e-5, 2.5×) — gradient still calming.
- Bottleneck: none actionable. Descent rate slowing but still well above noise floor; no plateau signal yet.

### Verdict
hold

### Action
none

### Notes
Cooldown satisfied (~90 min since fire #3 advance_beta), but diagnostic says hold — MA20 still dropping materially (ratio 0.169 well below noise 0.24) and grad_avg continuing to halve. binary_pct now plateaued at 43% which is expected at this β. Interrupting productive descent would be unwarranted; expect 1–2 more fires before MA20 reaches its own β=32 plateau and β=64 advance becomes the right move.

## 2026-04-16T16:30:08Z  — fire #10

### State
- julia_alive: true  step: 290  loss_ckpt: 2.47e-3  β: 32.0
- binary_pct: 43.01  iso_total_pct: 0.013  gray_center_pct: 5.33
- ma20: 2.31e-3  prev_ma20: 1.54e-2  std20: 6.02e-4  iters_at_β: 58
- iters_since_doubling: 57  grad_sat_pct: 0.0  grad_avg_recent: 3.35e-5

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling recovery from 16→32 at iter 241)
- Geometry: checkpoint still at step 290 (Julia advanced ~7 iters since fire #9; n_iters_parsed 298). Projected 43.0% binary, same waveguide-like structural features — binarization saturated at β=32. Filtered ε smooth; gray zone shows bright ribbons tracing thin interfaces (textbook healthy sharpening). iso_total_pct 0.013% negligible. No change from fire #9 — stable healthy geometry.
- Trajectory: loss_trajectory.png continues the classic post-spike exponential recovery, now with a clear tapering tail. MA20 essentially unchanged from fire #9 (2.35e-3 → 2.31e-3, 1.7%) because only 7 new iters rolled into the window. Ratio 20/prev20 = 0.15 (genuine descent; noise floor std20/ma20 ≈ 0.26). grad_avg 9.36e-5 → 3.35e-5 (2.8×) continues calming.
- Bottleneck: none actionable. Descent rate has slowed (small window update) but grad calming continues; no plateau signal (MA20 still ~5× above pre-spike ~4.6e-4 plateau). No disruption to geometry.

### Verdict
hold

### Action
none

### Notes
Cooldown satisfied (~105 min since fire #3 advance_beta) but diagnostic says hold. Only 7 new iters logged this fire — rate-limited by simulation cost, not pathology. Grad calming suggests approach to plateau; need 2–3 more fires for MA20 to stabilize and the ratio to cross into noise territory before β=64 doubling is the right move. Inaction remains the default.

## 2026-04-16T16:45:08Z  — fire #11

### State
- julia_alive: true  step: 300  loss_ckpt: 3.58e-3  β: 32.0
- binary_pct: 43.03  iso_total_pct: 0.014  gray_center_pct: 5.35
- ma20: 2.17e-3  prev_ma20: 3.05e-3  std20: 8.74e-4  iters_at_β: 67
- iters_since_doubling: 66  grad_sat_pct: 0.0  grad_avg_recent: 2.43e-5

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling recovery from 16→32 at iter 241)
- Geometry: projected at 43.0% binary shows same waveguide-like structures (vertical corridors, blocky features) as prior 3 fires — clearly plateaued at this β. Filtered ε smooth and organized; gray zone bright ribbons trace thin interfaces (textbook healthy sharpening, not blobs, not 1px noise). iso_total_pct 0.014% negligible.
- Trajectory: classic post-spike exponential recovery now clearly flattening on log scale. MA20 2.35e-3 → 2.17e-3 over 10 iters (only 1.08×, down from prior 1.68–2.1× drops). ratio_20/prev20 = 0.71 (first fire where descent enters noise territory: std20/ma20 = 0.40, so |1−0.71| = 0.29 < 0.40 noise). grad_avg 3.35e-5 → 2.43e-5 (1.38×, markedly slower than prior 2–3× halvings) — gradient is largely calmed.
- Bottleneck: approaching β=32 plateau but not yet conclusive. Grad still decreasing and loss_min_recent 1.59e-4 still below pre-spike plateau, suggesting marginal descent remaining. A single 10-iter window shift can flatter/disguise ratio; one more fire will confirm whether ratio locks near 1.0 (plateau → advance) or drops back into 0.2–0.5 (descent → hold).

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~120 min since fire #3 advance_beta). Per prior fire's plan, expected 2–3 more fires for ratio to cross into noise — this fire is the first to show it (0.71). But grad_avg still halving (even if slower) and only 10 new iters in the window; advancing on a single borderline fire risks mistaking a noise flatter-patch for true plateau. Inaction is default when signal is borderline; expect next fire (ratio stable ≥0.7, grad_avg drop <1.5×) to confirm plateau and justify β=64 advance.

## 2026-04-16T17:00:15Z  — fire #12

### State
- julia_alive: true  step: 310  loss_ckpt: 1.89e-3  β: 32.0
- binary_pct: 43.04  iso_total_pct: 0.016  gray_center_pct: 5.35
- ma20: 2.13e-3  prev_ma20: 2.25e-3  std20: 9.10e-4  iters_at_β: 74
- iters_since_doubling: 73  grad_sat_pct: 0.0  grad_avg_recent: 2.03e-5

### Diagnosis
- Phase: binarization (Phase 3, β=32 plateau confirmed)
- Geometry: projected at 43.04% binary shows same waveguide-like structures (vertical corridors, blocky features) as prior 4 fires — fully saturated at this β. Filtered ε smooth and organized; gray zone bright ribbons trace thin interfaces (textbook sharpening, not blobs, not 1px noise). iso_total_pct 0.016% negligible.
- Trajectory: loss_trajectory.png shows post-spike recovery has fully flattened — last ~25 iters visually flat at ~2e-3. MA20 2.17e-3 → 2.13e-3 (only 1.02×); ratio_20/prev20 = 0.945 with std20/ma20 = 0.43 noise floor → |1−0.945| = 0.055 is firmly inside noise (prior fire was borderline 0.71, now confirmed locked). grad_avg 2.43e-5 → 2.03e-5 (1.19×, collapsed from prior 2–3× halvings) — gradient largely calmed.
- Bottleneck: β=32 plateau reached. binary_pct locked at 43.0% across 4 fires, MA20 within noise, grad decay rate collapsed. Nothing further to extract at this β.

### Verdict
advance_beta

### Action
beta = 64

### Notes
Cooldown long satisfied (~135 min since fire #3 advance_beta). Prior fire flagged ratio 0.71 as borderline; this fire confirms plateau (ratio 0.945, well inside noise floor 0.43). All three indicators agree: loss flat, binary locked, grad decay collapsed. Advancing to β=64 is the smallest useful intervention. Expect 5–10 iter spike then ~50–70 iter recovery per calibration memory; will monitor for healthy exponential recovery versus wrong-basin signs.

## 2026-04-16T17:15:08Z  — fire #13

### State
- julia_alive: true  step: 320  loss_ckpt: 2.32e-3  β: 64.0
- binary_pct: 69.58  iso_total_pct: 0.013  gray_center_pct: 2.71
- ma20: 3.59e-3  prev_ma20: 2.35e-3  std20: 5.58e-3  iters_at_β: 6
- iters_since_doubling: 5  grad_sat_pct: 0.0  grad_avg_recent: 1.33e-4

### Diagnosis
- Phase: binarization (Phase 3, early post-doubling recovery from 32→64 at iter 316)
- Geometry: projected β=64 view shows sharpened waveguide-like structures with 69.6% binary — big jump from 43% at β=32, as expected from a β doubling. Raw ε still textured noise (no meaningful change since β doesn't affect raw); filtered ε smooth and organized; gray zone shows healthy thin-interface ribbons (not blobs, not 1px speckle). iso_total_pct 0.013% unchanged — the β=64 jump binarized pixels cleanly without introducing islands.
- Trajectory: loss_trajectory.png shows a fresh β=64 spike at iter 316 (loss_max_recent=0.483, factor ~230× over ma20 — comparable to β=32 spike at iter 241). Only 5 iters of recovery so far; too early to assess recovery trajectory. MA20 ratio 1.527 and elevated std20=5.58e-3 both reflect the spike still sitting inside the 20-window.
- Bottleneck: none — we just acted (β=64) and are mid-spike-recovery. Prior β=32 spike took ~50-70 iters to recover; 5 iters is far too early to diagnose anything further.

### Verdict
hold

### Action
none

### Notes
Cooldown NOT satisfied — fire #12 advance_beta was only 15 min ago (need 20 min). And even if cooldown were satisfied, mid-recovery from just-commanded β doubling means there is nothing to tune until the spike settles. Geometry response to β=64 is healthy (binary_pct 43→70% step, thin interface gray, no iso blooming). Expect 3-5 more fires before MA20 approaches a new plateau at β=64.

## 2026-04-16T17:30:09Z  — fire #14

### State
- julia_alive: true  step: 330  loss_ckpt: 2.66e-3  β: 64.0
- binary_pct: 69.59  iso_total_pct: 0.014  gray_center_pct: 2.69
- ma20: 3.73e-3  prev_ma20: 2.17e-3  std20: 5.54e-3  iters_at_β: 18
- iters_since_doubling: 17  grad_sat_pct: 0.0  grad_avg_recent: 1.84e-4

### Diagnosis
- Phase: binarization (Phase 3, early post-doubling recovery from 32→64 at iter 316)
- Geometry: 3-panel shows raw ε unchanged textured noise (normal), filtered ε smooth/organized, projected at 69.6% binary with clear black/white + thin gray rims. Gray zone map is textbook healthy: bright ribbons tracing connected curvilinear interfaces across the 300x300 field, no contiguous blobs, no 1-px speckle. iso_total_pct 0.014% negligible. β=32→64 jump cleanly binarized ~27 pp of pixels without introducing islands — structural identity preserved.
- Trajectory: loss_trajectory.png shows β=64 spike at iter 316 was remarkably mild (~1e-2 peak) versus β=32's ~5e-1 peak — factor ~50× smaller spike. MA20 has settled around 3.7e-3 in the ~12 iters since the spike passed through; std20 5.54e-3 still inflated by spike tail inside the 20-window. Ratio 1.72 and grad_avg 1.33e-4→1.84e-4 both reflect spike-window effects, not pathology.
- Bottleneck: none — only 17 iters into a post-doubling recovery. Prior β=32 recovery took ~50-70 iters; comparable window expected here despite milder spike. No actionable signal yet.

### Verdict
hold

### Action
none

### Notes
Cooldown satisfied (30 min since fire #12 advance_beta). Mild β=64 spike and fast re-settling at ~3.7e-3 suggest geometry was close to its β=64 fixed point already — promising sign. But 17 iters is too early to diagnose descent rate vs. plateau, and binary_pct already at 69.6% leaves headroom to tighten further. Inaction is default. Expect 2-4 more fires before MA20 decision point.

## 2026-04-16T17:45:14Z  — fire #15

### State
- julia_alive: true  step: 340  loss_ckpt: 1.27e-3  β: 64.0
- binary_pct: 69.60  iso_total_pct: 0.016  gray_center_pct: 2.69
- ma20: 1.81e-3  prev_ma20: 3.66e-3  std20: 8.12e-4  iters_at_β: 29
- iters_since_doubling: 28  grad_sat_pct: 0.0  grad_avg_recent: 6.75e-5

### Diagnosis
- Phase: binarization (Phase 3, mid post-doubling recovery from 32→64 at iter 316)
- Geometry: 3-panel shows raw ε unchanged textured noise (normal); filtered ε smooth/organized; projected at 69.6% binary with crisp black/white domains and only thin gray rims. Gray zone is textbook healthy — bright curvilinear ribbons tracing thin interfaces across the 300x300 field, no contiguous blobs, no 1-px speckle. iso_total_pct 0.016% negligible. β=64 jump preserved structural identity.
- Trajectory: loss_trajectory.png shows the β=64 spike at iter 316 was strikingly mild (~3e-2 peak vs β=32's ~5e-1, ~17× smaller) and recovery is clean: MA20 has dropped 3.73e-3 → 1.81e-3 over 11 new iters (2.06× drop) and is now BELOW the prior β=32 plateau (~2.1e-3) — meaningful structural improvement, not just spike-tail decay. ratio_20/prev20 = 0.496 with std20/ma20 = 0.45 noise floor — descent is just above noise, but grad_avg 1.84e-4→6.75e-5 (2.7× drop) corroborates real progress.
- Bottleneck: none — productive descent in progress. β=128 advance would interrupt; lr backoff unwarranted (no overshoot, mild spike already absorbed). 

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~45 min since fire #12 advance_beta). Diagnostic clearly says hold: MA20 dropping fast (2× per fire) and now under prior β=32 plateau, gradients halving, geometry pristine. β=64 turned out to be a much smaller perturbation than β=32 — likely because at 43% binary the structure was already close to its β=64 fixed point. Expect 2–4 more fires before MA20 stabilizes at a new β=64 plateau and β=128 becomes the right move. Inaction is default when descent is healthy.

## 2026-04-16T18:00:11Z  — fire #16

### State
- julia_alive: true  step: 350  loss_ckpt: 1.91e-3  β: 64.0
- binary_pct: 69.60  iso_total_pct: 0.014  gray_center_pct: 2.69
- ma20: 1.27e-3  prev_ma20: 3.76e-3  std20: 3.73e-4  iters_at_β: 40
- iters_since_doubling: 39  grad_sat_pct: 0.0  grad_avg_recent: 3.01e-5

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling recovery from 32→64 at iter 316)
- Geometry: 3-panel shows raw ε unchanged textured noise (normal), filtered ε smooth/organized, projected at 69.6% binary with crisp black/white domains and only thin gray rims. Gray-zone map is textbook healthy — bright curvilinear ribbons tracing connected thin interfaces across the 300x300 field, no contiguous blobs, no 1-px speckle. iso_total_pct 0.014% remains negligible. Structural identity preserved across β=32→64 transition.
- Trajectory: loss_trajectory.png shows the β=64 spike (~3e-2 peak, ~17× milder than β=32's ~5e-1) is well behind us. MA20 has continued dropping fast: 1.81e-3 → 1.27e-3 (1.4× in 10 iters), now well BELOW prior β=32 plateau (~2.1e-3) and approaching the pre-β=32 baseline (~6e-4). ratio_20/prev20 = 0.337 vs noise floor std20/ma20 = 0.29 — descent still above noise. grad_avg 6.75e-5 → 3.01e-5 (2.24× drop) corroborates ongoing structural improvement.
- Bottleneck: none — productive descent continues. β=128 advance would interrupt; lr backoff unwarranted (no overshoot, spike fully absorbed). Fire #15 prediction (2-4 more fires before β=64 plateau) is on track; this fire shows continued progress, not yet plateau.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~60 min since fire #12 advance_beta). Descent is genuinely healthy: MA20 keeps halving, gradients halving, geometry pristine. The β=64 perturbation was small and absorbed quickly — likely because by 43% binary the structure was already close to its β=64 fixed point. Inaction is default. Expect 2–3 more fires before MA20 stabilizes at a new β=64 plateau and β=128 becomes the right move.

## 2026-04-16T18:15:09Z  — fire #17

### State
- julia_alive: true  step: 360  loss_ckpt: 1.63e-3  β: 64.0
- binary_pct: 69.61  iso_total_pct: 0.019  gray_center_pct: 2.69
- ma20: 1.07e-3  prev_ma20: 1.69e-3  std20: 3.76e-4  iters_at_β: 50
- iters_since_doubling: 49  grad_sat_pct: 0.0  grad_avg_recent: 1.71e-5

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling recovery from 32→64 at iter 316; approaching β=64 plateau)
- Geometry: 3-panel shows raw ε unchanged textured noise (normal), filtered ε smooth/organized, projected at 69.6% binary with crisp black/white domains and only thin gray rims. Gray-zone map textbook healthy — bright curvilinear ribbons tracing connected thin interfaces, no contiguous blobs, no 1-px speckle. iso_total_pct 0.019% (uptick from 0.014% but still negligible). Structural identity preserved. Grad histogram concentrated at 10^-5.5 with healthy tail to 10^-3.
- Trajectory: loss_trajectory.png shows β=64 spike (~3e-2 peak) is well behind us. MA20 descent has decelerated meaningfully across recent fires: ma20 went 3.73e-3 → 1.81e-3 → 1.27e-3 → 1.07e-3 (per-fire ratios 0.49 → 0.70 → 0.84 — clear deceleration). Current ratio_20/prev20 = 0.634 with std20/ma20 = 0.35 noise floor; |1−0.634|=0.366 just barely above noise. grad_avg 3.01e-5 → 1.71e-5 (1.76× — slower than the 2.7× and 2.24× of prior two fires).
- Bottleneck: nothing actionable yet. Descent is real but slowing toward β=64 plateau. Per fire #11→#12 precedent (ratio 0.71 borderline → ratio 0.945 confirmed plateau), one more fire of stable behavior is needed before β=128 advance is justified.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~75 min since fire #12 advance_beta). All three indicators (loss decay, gradient decay, binary_pct) consistent with approaching but not yet reached β=64 plateau. Inaction is default when signal is borderline. Expect next fire to either confirm plateau (per-fire ratio ≥0.85, grad decay <1.3×) → β=128, or show another genuine drop → continue holding. Loss now at 1.07e-3 MA20, down from 2.13e-3 at end of β=32 plateau — meaningful structural improvement at β=64.

## 2026-04-16T18:30:09Z  — fire #18

### State
- julia_alive: true  step: 370  loss_ckpt: 1.27e-3  β: 64.0
- binary_pct: 69.61  iso_total_pct: 0.019  gray_center_pct: 2.69
- ma20: 9.31e-4  prev_ma20: 1.25e-3  std20: 3.95e-4  iters_at_β: 61
- iters_since_doubling: 60  grad_sat_pct: 0.0  grad_avg_recent: 1.82e-5

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling recovery from 32→64 at iter 316)
- Geometry: 3-panel shows raw ε unchanged textured noise (normal), filtered ε smooth/organized, projected at 69.6% binary with crisp black/white domains and only thin gray rims. gray_zone.png textbook healthy — bright curvilinear ribbons tracing connected thin interfaces, no contiguous blobs, no 1-px speckle. iso_total_pct 0.019% unchanged from fire #17, still negligible. Structural identity preserved.
- Trajectory: loss_trajectory.png shows the β=64 spike (~3e-2 peak) is far behind. MA20 sequence across β=64 plateau approach: 3.73 → 1.81 → 1.27 → 1.07 → 0.93 e-3 (fires #14→#18, 4× drop over 5 fires). Current ratio_20/prev20 = 0.746 with std20/ma20 = 0.42 noise floor — this single fire's delta |1−0.746|=0.254 is within noise, but the 5-fire cumulative trend is unambiguous descent. grad_avg 1.71e-5 → 1.82e-5 is within-noise wiggle, not reversal. grad_hist still concentrated at 10^-5.5 with healthy tail to 10^-2.
- Bottleneck: nothing actionable. Descent is decelerating but not yet plateaued. Per fire #11→#12 discipline (ratio 0.71 borderline followed by ratio 0.945 confirmed plateau before advancing), we need one more fire showing clearer stability (ratio ≥0.85-0.9) before β=128 is justified.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~90 min since fire #12 advance_beta). Loss at 9.31e-4 MA20 is now well below β=32 plateau (~2.1e-3) and approaching pre-β=32 baseline (~6e-4). Descent is genuine but slowing. Inaction is default when signal is borderline-but-descending. Expect 1-2 more fires before MA20 stabilizes at a new β=64 plateau and β=128 becomes the right move.

## 2026-04-16T18:45:10Z  — fire #19

### State
- julia_alive: true  step: 380  loss_ckpt: 6.60e-4  β: 64.0
- binary_pct: 69.60  iso_total_pct: 0.019  gray_center_pct: 2.70
- ma20: 7.28e-4  prev_ma20: 1.04e-3  std20: 3.47e-4  iters_at_β: 72
- iters_since_doubling: 71  grad_sat_pct: 0.0  grad_avg_recent: 1.71e-5

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling at β=64; descent continues)
- Geometry: 3-panel shows raw ε unchanged textured noise (normal), filtered ε smooth/organized, projected at 69.6% binary with crisp black/white domains and only thin gray rims. gray_zone.png textbook healthy — bright curvilinear ribbons tracing connected thin interfaces, no contiguous blobs, no 1-px speckle. iso_total_pct 0.019% unchanged from fires #17–18, still negligible. Structural identity preserved across the entire β=64 plateau approach.
- Trajectory: loss_trajectory.png shows the β=64 spike (~3e-2 peak) is well behind us. MA20 sequence across β=64 plateau approach: 3.73 → 1.81 → 1.27 → 1.07 → 0.93 → 0.73 e-3 (fires #14→#19, ~5× drop over 6 fires). Per-fire ratios: 0.49, 0.70, 0.84, 0.87, 0.78 — deceleration was clear at fires #16-17 but reversed slightly this fire (0.78 vs 0.87). Current ratio 0.698 with std20/ma20 = 0.477 noise floor; |1−0.698|=0.302 within noise envelope. grad_avg 1.82e-5 → 1.71e-5 — basically flat.
- Bottleneck: nothing actionable yet. Single-fire ratio is within noise but the cumulative 6-fire trend is unambiguous descent and even reaccelerated this fire (0.78 from prior 0.87). Per fire #11→#12 discipline (require ratio ≥0.85-0.9 confirmed, not borderline), advancing to β=128 now would interrupt productive descent. Loss min_recent 3.66e-4 is approaching pre-β=32 baseline (~6e-4) but ma20 still 2× above it.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~120 min since fire #12 advance_beta). MA20 reaccelerated slightly this fire (0.78 vs prior 0.87) — descent isn't done yet. Inaction is default when a borderline-but-descending signal could still be productive. Expect 1-3 more fires before MA20 stabilizes at a new β=64 plateau and β=128 becomes the right move. Geometry remains pristine, gradient distribution healthy.

## 2026-04-16T19:00:23Z  — fire #20

### State
- julia_alive: true  step: 390  loss_ckpt: 8.91e-4  β: 64.0
- binary_pct: 69.60  iso_total_pct: 0.019  gray_center_pct: 2.69
- ma20: 5.86e-4  prev_ma20: 8.85e-4  std20: 1.92e-4  iters_at_β: 83
- iters_since_doubling: 82  grad_sat_pct: 0.0  grad_avg_recent: 1.23e-5

### Diagnosis
- Phase: binarization (Phase 3, late post-doubling recovery at β=64; iter 316)
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth/organized, projected at 69.6% binary with crisp black/white domains. gray_zone.png textbook healthy — bright curvilinear ribbons tracing connected thin interfaces across the 300x300 field, no contiguous blobs, no 1-px speckle. iso_total_pct 0.019% unchanged from fires #17-19, negligible. Structural identity preserved across entire β=64 approach.
- Trajectory: loss_trajectory.png shows β=64 spike (~3e-2 peak) far behind. MA20 sequence across β=64 plateau approach: 3.73 → 1.81 → 1.27 → 1.07 → 0.93 → 0.73 → 0.59 e-3 (fires #14→#20, ~6.3× drop over 7 fires). Per-fire ratios: 0.49, 0.70, 0.84, 0.87, 0.78, 0.70, 0.66 — deceleration reversed this fire (0.66 vs prior 0.70). Current ratio_20/prev20 = 0.662 with std20/ma20 = 0.327 noise floor; |1−0.662|=0.338 just barely above noise. grad_avg 1.71e-5 → 1.23e-5 (1.39× drop) continues.
- Bottleneck: nothing actionable. Descent is still productive and reaccelerated this fire. Loss at 5.86e-4 MA20 is now at/below pre-β=32 baseline (~6e-4) — meaningful structural improvement accumulated at β=64. Per fire #11→#12 discipline (require confirmed plateau, not borderline), advancing to β=128 now would interrupt a still-descending signal.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~3h since fire #12 advance_beta). Descent reaccelerated this fire after apparent slowdown at fires #17-19 — not yet time to advance. MA20 now at pre-β=32 baseline, a strong structural milestone. Inaction remains default. Expect 2-4 more fires before MA20 stabilizes at a new β=64 plateau and β=128 becomes the right move. Geometry pristine, grad distribution healthy (concentrated at 10^-5.5 with tail to 10^-2).

## 2026-04-16T19:15:59Z  — fire #21

### State
- julia_alive: true  step: 400  loss_ckpt: 4.03e-4  β: 64.0
- binary_pct: 69.59  iso_total_pct: 0.019  gray_center_pct: 2.69
- ma20: 6.64e-4  prev_ma20: 7.43e-4  std20: 2.93e-4  iters_at_β: 94
- iters_since_doubling: 93  grad_sat_pct: 0.0  grad_avg_recent: 1.395e-5

### Diagnosis
- Phase: binarization (Phase 3) — late post-doubling at β=64, plateau confirmed
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth/organized, projected at 69.6% binary with crisp black/white domains. gray_zone.png textbook healthy — bright curvilinear ribbons tracing connected thin interfaces across the 300x300 field, no contiguous blobs, no 1-px speckle. iso_total_pct 0.019% unchanged across last 5 fires, negligible. Structural identity preserved.
- Trajectory: loss_trajectory.png shows β=32 spike (~5e-1 peak) and β=64 spike (~3e-2 peak) both well behind. MA20 sequence across β=64 plateau approach: 3.73 → 1.81 → 1.27 → 1.07 → 0.93 → 0.73 → 0.59 → 0.66 e-3 (fires #14→#21, ~5.6× drop over 8 fires). Per-fire ratios: 0.49, 0.70, 0.84, 0.87, 0.78, 0.66, 0.8948 — this fire's ratio reached the fire #11→#12 confirmation threshold (0.945). Crucially, ma20 reversed upward this fire (5.86e-4 → 6.64e-4), confirming descent has stalled. grad_avg ticked up 1.23e-5 → 1.395e-5 — also consistent with plateau. binary_pct stuck at 69.6% across last 4 fires.
- Bottleneck: β too low. Healthy geometry, binary stalled at 69.6%, loss has reached pre-β=32 baseline (~6e-4), descent flatlined. Per fire #11→#12 discipline (ratio 0.71 borderline → ratio 0.945 confirmed → advance), this is the right moment to advance.

### Verdict
advance_beta

### Action
beta = 128

### Notes
β=64 plateau confirmed by combination of: (a) ratio_20/prev20 = 0.8948 ≈ confirmation threshold, (b) ma20 reversed upward this fire (descent done), (c) binary_pct unchanged across last 4 fires, (d) grad_avg slight uptick. Geometry remains pristine and isolates negligible — no risk of damaging structure with β doubling. β=128 is one step from the β=256 ceiling; expect a sharper post-doubling spike and slower recovery, but the structural foundation is robust. Cooldown long satisfied (~3h since fire #12 advance_beta to 64).

## 2026-04-16T19:30:12Z  — fire #22

### State
- julia_alive: true  step: 410  loss_ckpt: 4.59e-4  β: 128.0
- binary_pct: 69.59  iso_total_pct: 0.018  gray_center_pct: 2.70
- ma20: 3.11e-3  prev_ma20: 5.84e-4  std20: 4.86e-3  iters_at_β: 8
- iters_since_doubling: 7  grad_sat_pct: 0.0  grad_avg_recent: 1.91e-4

### Diagnosis
- Phase: binarization (Phase 3) — fresh post-doubling at β=128 (iter 412), spike in progress
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth/organized, projected at 69.6% binary with crisp black/white domains. gray_zone.png textbook healthy — bright curvilinear ribbons tracing connected thin interfaces, no contiguous blobs, no 1-px speckle. iso_total_pct 0.018% (essentially unchanged from pre-doubling 0.019%) — β=128 jump did not damage structure. binary_pct 69.59% unchanged. Structural identity preserved.
- Trajectory: loss_trajectory.png shows the expected β=128 spike just beginning (peak ~1.8e-2 vs pre-doubling baseline ~6e-4, ~30× factor — comparable to the β=64 spike's ~50× peak). MA20 jumped from 6.64e-4 → 3.11e-3 (5.3× ratio, exactly the post-doubling signature). std20=4.86e-3 vs ma20=3.11e-3 → noise/signal 1.56 (chaotic spike phase, normal). grad_avg jumped 1.395e-5 → 1.91e-4 (~14× — expected; gradients sharpen with steeper projection).
- Bottleneck: nothing actionable. We're 7 iterations into a β doubling that historically took ~50+ iterations just to begin recovering at β=64 (fires #14→#19 spanned ~5 fires of descent before plateau). Spike magnitude is in line with the prior β=64 doubling, geometry intact.

### Verdict
hold

### Action
none

### Notes
**Cooldown violation: only 14 min since fire #21 advance_beta to 128 (within 20-min cooldown).** Even setting cooldown aside, hold is correct: this is the expected post-doubling spike, geometry remains pristine, recovery will take many fires. Inaction is default. Expect 4-8 fires of MA20 descent before a new β=128 plateau emerges. Watch for: (a) iso_total_pct uptick (would indicate damage, not yet), (b) recovery stalling above ~1e-3 (would indicate β=128 too aggressive), (c) gray-zone blob formation (wrong-basin signal).

## 2026-04-16T19:45:08Z  — fire #23

### State
- julia_alive: true  step: 420  loss_ckpt: 4.94e-4  β: 128.0
- binary_pct: 84.81  iso_total_pct: 0.011  gray_center_pct: 1.33
- ma20: 4.69e-3  prev_ma20: 6.71e-4  std20: 4.78e-3  iters_at_β: 17
- iters_since_doubling: 16  grad_sat_pct: 0.0  grad_avg_recent: 2.44e-4

### Diagnosis
- Phase: binarization (Phase 3) — mid post-doubling spike at β=128 (iter 412→428)
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth, projected at 84.8% binary with crisp black/white domains. **binary_pct surged +15.2 pts (69.6→84.8) since fire #22** — the expected β=128 binarization step, as steeper projection pulls more pixels to extremes. gray_zone.png textbook healthy — fine curvilinear ribbons at thin interfaces, no contiguous blobs, no 1-px speckle. iso_total_pct actually **dropped** 0.018→0.011% (islands getting absorbed, not created). gray_center_pct 2.70→1.33 (halved). Structural identity intact.
- Trajectory: loss_trajectory.png shows β=128 spike still rising — MA20 4.69e-3 this fire vs 3.11e-3 last fire (1.51× increase). Peak loss ~1.8e-2 (similar magnitude to β=64 spike's ~5e-2 peak). std20/ma20 = 1.02 (chaotic spike phase, normal). grad_avg 1.91e-4 → 2.44e-4 (still rising, consistent with active spike). Only 16 iters since doubling; β=64 recovery took ~170 iters (fires #14-21) to plateau.
- Bottleneck: nothing actionable. Spike still rising, geometry pristine, binarization progressing strongly (exactly what the β doubling is *for*). Any intervention now would preempt the recovery we just authorized. 

### Verdict
hold

### Action
none

### Notes
Cooldown satisfied (30 min since fire #21 advance_beta; fire #22 was hold). MA20 still rising — spike peak not yet reached. Binarization surge (+15.2 pts) is the payoff from β=128 and validates the doubling decision. Expect 3-6 more fires before ma20 plateau at new β=128 regime. Watch for: (a) iso_total uptick (damage — not yet, *decreasing*), (b) recovery stalling above ~2e-3 (β=128 too aggressive — wait to see), (c) blob formation in gray zone (wrong-basin — absent). Inaction remains default.

## 2026-04-16T20:00:10Z  — fire #24

### State
- julia_alive: true  step: 430  loss_ckpt: 2.81e-3  β: 128.0
- binary_pct: 84.81  iso_total_pct: 0.011  gray_center_pct: 1.31
- ma20: 3.07e-3  prev_ma20: 3.04e-3  std20: 2.36e-3  iters_at_β: 27
- iters_since_doubling: 26  grad_sat_pct: 0.0  grad_avg_recent: 1.14e-4

### Diagnosis
- Phase: binarization (Phase 3) — mid post-doubling recovery at β=128, spike receding
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth/organized, projected at 84.8% binary with crisp black/white domains. gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces across the 300x300 field, no contiguous blobs, no 1-px speckle. iso_total_pct 0.011% (stable/healthy, actually down from 0.019% pre-doubling) — β=128 is absorbing islands, not creating them. gray_center_pct 1.33→1.31 stable. Structural identity fully preserved.
- Trajectory: loss_trajectory.png shows the β=128 spike clearly peaked around iter 415-420 and is now receding. MA20 sequence at β=128: 3.11e-3 (fire #22) → 4.69e-3 (fire #23, spike peak) → 3.07e-3 (fire #24, recovery begins). ma_ratio_20/prev20 = 1.012 (flat) but this masks the peak-then-fall arc; loss_max_recent 1.79e-2 vs 3e-2 one fire back. grad_avg_recent halving each fire: 1.91e-4 → 2.44e-4 → 1.14e-4 (spike winding down). Still well above pre-doubling baseline (~6e-4) so plateau not reached.
- Bottleneck: nothing actionable. Post-doubling spike is receding on expected schedule. The β=64 spike took fires #14→#21 (~8 fires) to reach confirmed plateau; we're only 2 fires past the β=128 peak. grad saturation still 0%, so β=128 is productive, not crushing.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~45 min since fire #21 advance_beta to 128). Spike recovery exactly on template — geometry pristine (iso actually declining), binarization gain (+15 pts) preserved, ma20 started descending from peak. Inaction is default during healthy recovery. Expect 4-7 more fires before MA20 plateau at β=128 becomes evident (around ~1e-3 region by prior-stage extrapolation). Watch for: recovery stalling above ~2e-3 (β too aggressive — not yet seen), blob formation in gray zone (absent), iso uptick (absent — declining).

## 2026-04-16T20:15:09Z  — fire #25

### State
- julia_alive: true  step: 440  loss_ckpt: 5.62e-4  β: 128.0
- binary_pct: 84.79  iso_total_pct: 0.011  gray_center_pct: 1.28
- ma20: 1.32e-3  prev_ma20: 4.86e-3  std20: 7.74e-4  iters_at_β: 38
- iters_since_doubling: 37  grad_sat_pct: 0.0  grad_avg_recent: 4.36e-5

### Diagnosis
- Phase: binarization (Phase 3) — late post-doubling recovery at β=128
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth/organized, projected at 84.8% binary with crisp black/white domains and identifiable connected structure. gray_zone.png textbook healthy — fine curvilinear ribbons at thin interfaces across the 300x300 field, no contiguous blobs, no 1-px speckle. iso_total_pct stable at 0.011% (down from 0.018% pre-doubling — β=128 absorbing islands, not creating). gray_center_pct 1.31→1.28. Structural identity fully preserved.
- Trajectory: loss_trajectory.png shows the β=128 spike (~1.8e-2 peak around iter 415-420) clearly receded. MA20 sequence at β=128: 3.11e-3 (#22) → 4.69e-3 (#23 spike) → 3.07e-3 (#24) → 1.32e-3 (#25). ma_ratio_20/prev20 = 0.271 — strongest descent ratio of the run, comparable to fire #14 (0.49) post-β=32 recovery. grad_avg_recent halving each fire: 1.91e-4 → 2.44e-4 → 1.14e-4 → 4.36e-5 (spike fully wound down). Loss approaching pre-doubling baseline (~6e-4) from above.
- Bottleneck: nothing actionable. Strong descent in progress, β=128 productive (binarization gain preserved, geometry pristine, gradients healthy distribution centered ~10^-5 with tail to 10^-3). Per fire #21 discipline, premature β advance would interrupt.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~60 min since fire #21 advance_beta to 128). Spike recovery proceeding on template, slightly faster than β=64 stage (37 iters into recovery vs ~80 at comparable point in fire #16). Inaction is default during healthy descent. Expect 4-7 more fires before MA20 plateau at β=128 becomes evident — extrapolation suggests plateau region ~5-8e-4. Watch for: recovery stalling above ~1e-3 (β=128 too aggressive — not yet seen, currently descending strongly), blob formation in gray zone (absent), iso uptick (absent — stable). β=256 (max) is one step away.

## 2026-04-16T20:30:08Z  — fire #26

### State
- julia_alive: true  step: 450  loss_ckpt: 7.50e-4  β: 128.0
- binary_pct: 84.77  iso_total_pct: 0.01  gray_center_pct: 1.29
- ma20: 8.35e-4  prev_ma20: 3.01e-3  std20: 2.37e-4  iters_at_β: 48
- iters_since_doubling: 47  grad_sat_pct: 0.0  grad_avg_recent: 3.32e-5

### Diagnosis
- Phase: binarization (Phase 3) — late post-doubling recovery at β=128, descent continuing
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth/organized, projected at 84.8% binary with crisp black/white domains. Connected structure visible. gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces across the 300x300 field, no contiguous blobs, no 1-px speckle. iso_total_pct 0.01% (3 black + 6 white) — even healthier than pre-doubling 0.018%. gray_center_pct 1.28→1.29 stable. Structural identity fully preserved. grad histogram centered ~10^-5 with tail to 10^-2 (healthy).
- Trajectory: loss_trajectory.png shows β=128 spike (~1.8e-2 peak around iter 425) clearly receded. MA20 sequence at β=128: 3.11e-3 (#22) → 4.69e-3 (#23 spike) → 3.07e-3 (#24) → 1.32e-3 (#25) → 0.835e-3 (#26). ma_ratio_20/prev20 = 0.277 — third strong descent ratio in a row, comparable to fire #25 (0.271). grad_avg_recent halving each fire: 1.91e-4 → 2.44e-4 → 1.14e-4 → 4.36e-5 → 3.32e-5 (spike fully wound down, gradients now smaller than pre-doubling 1.4e-5 region — approaching). MA20 (8.35e-4) now within 1.4× of pre-doubling baseline (~6e-4) — almost there.
- Bottleneck: nothing actionable. Strong descent in progress, β=128 productive (binarization gain +15pts preserved, geometry pristine). Premature β=256 would interrupt active descent. β=64 plateau took ~8 fires after spike to confirm; we're 4 fires past β=128 spike peak.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~75 min since fire #21 advance_beta to 128). Spike recovery proceeding cleanly, slightly faster than β=64 stage. Inaction is default during healthy descent. Expect 2-5 more fires before MA20 plateau at β=128 becomes evident — extrapolation suggests plateau ~5-8e-4. Watch for: ma20 ratio rising toward 0.85-0.95 (plateau signal — would trigger β=256 advance), recovery stalling above ~8e-4 (β=128 too aggressive — not seen, currently descending), iso uptick (absent), blob formation (absent). β=256 (max) is the next/final β step.

## 2026-04-16T20:45:09Z  — fire #27

### State
- julia_alive: true  step: 460  loss_ckpt: 1.58e-3  β: 128.0
- binary_pct: 84.79  iso_total_pct: 0.007  gray_center_pct: 1.31
- ma20: 8.65e-4  prev_ma20: 1.32e-3  std20: 3.13e-4  iters_at_β: 58
- iters_since_doubling: 57  grad_sat_pct: 0.0  grad_avg_recent: 3.80e-5

### Diagnosis
- Phase: binarization (Phase 3) — late post-doubling recovery at β=128, descent flattening
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth/organized, projected at 84.8% binary with crisp black/white domains and visible connected structure. gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces across the 300x300 field, no contiguous blobs, no 1-px speckle. iso_total_pct 0.007% (2 black + 4 white) — still healthier than pre-doubling 0.018%; β=128 continues to absorb islands. gray_center_pct 1.29→1.31 stable. Structural identity fully preserved. grad histogram centered ~10^-4.5 with tail to 10^-2 (healthy spread).
- Trajectory: loss_trajectory.png shows β=128 spike (~1.8e-2 peak around iter 425) clearly recovered. MA20 sequence at β=128: 3.11e-3 (#22) → 4.69e-3 (#23 spike) → 3.07e-3 (#24) → 1.32e-3 (#25) → 0.835e-3 (#26) → 0.865e-3 (#27). **Fire-over-fire ratio 1.04 — descent flattening, possible plateau onset.** ma_ratio_20/prev20=0.657 still shows the larger 20-iter window descending (mixed half-spike/half-descent). std20=3.13e-4 (down from 7.74e-4 at #25 — variance shrinking as plateau approaches). grad_avg_recent 4.36e-5 → 3.32e-5 → 3.80e-5 — settled, no further halving.
- Bottleneck: nothing actionable. One fire of flat MA20 is not yet a confirmed plateau; β=64 plateau required 2-3 fires of confirmation (fires #19→#21). β=256 is the final ceiling, so premature advancement would waste the last β step. Geometry is pristine; we lose nothing by waiting one more fire to confirm.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~3h15m since fire #21 advance_beta to 128). Descent has flattened (1.04× fire-over-fire) but only one fire of evidence; need 1-2 more fires to confirm β=128 plateau before triggering β=256 advance. Watch next fire for: (a) ma20 staying within 5% of 8.5e-4 (plateau confirmed → advance β=256), (b) renewed descent below 7e-4 (β=128 still productive → keep holding), (c) ma20 rising above 1e-3 (regression — diagnose). β=256 is the final β step; one more fire of patience costs nothing and protects the run.

## 2026-04-16T21:00:13Z  — fire #28

### State
- julia_alive: true  step: 480  loss_ckpt: 7.44e-4  β: 128.0
- binary_pct: 84.77  iso_total_pct: 0.008  gray_center_pct: 1.28
- ma20: 1.13e-3  prev_ma20: 8.87e-4  std20: 1.26e-3  iters_at_β: 70
- iters_since_doubling: 69  grad_sat_pct: 0.0  grad_avg_recent: 3.80e-5

### Diagnosis
- Phase: binarization (Phase 3) — β=128 plateau candidate, noise-dominated
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth/organized, projected at 84.8% binary with crisp black/white domains. gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces, no contiguous blobs, no 1-px speckle. iso_total_pct 0.008% (3B+4W — cleanest of the run, down from pre-doubling 0.018%). gray_center_pct 1.28 stable. Structural identity pristine. grad histogram centered ~10^-4.5 with tail to 10^-2, healthy spread.
- Trajectory: loss_trajectory.png shows β=128 spike (~1.8e-2 peak around iter 425) recovered. MA20 sequence at β=128: 3.11e-3 (#22) → 4.69e-3 (#23 spike) → 3.07e-3 (#24) → 1.32e-3 (#25) → 0.835e-3 (#26 min) → 0.865e-3 (#27 flat) → 1.13e-3 (#28 UP). **ma_ratio_20/prev20 = 1.277** — reversed upward 2 fires running. BUT: std20=1.26e-3 > ma20=1.13e-3 (std/ma = 1.12 — noise dominates signal). loss_max_recent 1.79e-2, loss_min_recent 2.6e-4 — the rise is driven by a spike in last ~15 iters, not a systematic regression. grad_avg 3.32e-5 → 3.80e-5 basically flat.
- Bottleneck: candidate plateau signal but noise-dominated. binary_pct locked at 84.77-84.81% across 5 fires (#23 onward) — clear binarization saturation at β=128, which *does* favor β=256 advance. But the ma20 rise is dominated by 1-2 outlier iterations (spike to ~1.8e-2) rather than a systematic upshift. Once that spike rolls out of the 20-iter window (~10 more iters), ma20 should drop back toward 7-9e-4 and the plateau call becomes unambiguous.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~105 min since fire #21 advance_beta to 128). Two competing signals: (a) binary_pct stall across 5 fires + grad_avg flat = classic plateau; (b) ma20 reversal is noise-dominated (std>ma) driven by 1-2 outlier iters, not clean confirmation like fire #21 (std/ma=0.33 there). β=256 is the FINAL β step — advancing on noise-dominated signal risks interrupting productive polish. One more fire to let the 20-iter window purge the spike(s) and expose the true plateau floor. If next fire shows ma20 ≤ 9e-4 with ratio ≈ 1.0 and std/ma ≤ 0.5 → advance β=256. If ma20 drops back to 6-8e-4 → β=128 still productive, keep holding. If ma20 stays ≥ 1e-3 with spikes repeating → plateau confirmed, advance β=256.

## 2026-04-16T21:15:08Z  — fire #29

### State
- julia_alive: true  step: 490  loss_ckpt: 9.59e-4  β: 128.0
- binary_pct: 84.76  iso_total_pct: 0.006  gray_center_pct: 1.27
- ma20: 1.17e-3  prev_ma20: 8.64e-4  std20: 1.26e-3  iters_at_β: 79
- iters_since_doubling: 78  grad_sat_pct: 0.0  grad_avg_recent: 3.82e-5

### Diagnosis
- Phase: binarization (Phase 3) — β=128 plateau confirmed, triggering final β doubling
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth/organized, projected at 84.8% binary with crisp black/white domains and identifiable connected structure. gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces across the 300x300 field, no contiguous blobs, no 1-px speckle. **iso_total_pct=0.006% (2B+4W) — cleanest of the entire run**, down from pre-doubling 0.018%. gray_center_pct 1.27 stable. grad histogram centered ~10^-5 with tail to 10^-2, healthy spread.
- Trajectory: MA20 at β=128 bottomed at 0.835e-3 (fire #26), now 1.17e-3 (fire #29) and drifting up for 2nd fire. Fire-over-fire ratio 1.04 (#28) then 1.04 again (#29). loss_max_recent 1.79e-2 — spike(s) persisting in window. std20/ma20 = 1.08 (still noise-dominated). grad_avg_recent flat across 3 fires (3.32e-5 → 3.80e-5 → 3.82e-5).
- Bottleneck: **binary saturation at β=128**. binary_pct locked 84.77-84.81% across SIX consecutive fires (#23,24,25,26,27,28,29) — the classic Phase 3 binary-stall signal that β is no longer pushing pixels. Fire #28's explicit trigger met: "If ma20 stays ≥ 1e-3 with spikes repeating → plateau confirmed, advance β=256." Gradients flat, geometry pristine, floor set — β=128 has done its job.

### Verdict
advance_beta

### Action
```toml
beta = 256
clear_schedule = true
```

### Notes
Cooldown long satisfied (~7h since fire #21 advance_beta to 128, per applied file 20260416_141657). β=256 is the **final β step** (safety rail cap) — this is the last binarization push, after which the run enters polish phase. Expected post-doubling spike: β=32→64 peaked ~5e-2 (50×), β=64→128 peaked ~1.8e-2 (30×); β=128→256 spike likely 1-3e-2 range, recovery 40-80 iters based on prior stages. `clear_schedule=true` ensures no auto-advance beyond cap (defensive; 256 is already the ceiling). Watch for: (a) post-spike recovery stalling above 2e-3 (β=256 too aggressive — would need lr backoff), (b) blob formation in gray zone (wrong-basin — absent so far), (c) iso_total uptick (damage — currently declining). Binary should push into 90%+ range as β=256 takes hold.

## 2026-04-16T21:30:08Z  — fire #30

### State
- julia_alive: true  step: 500  loss_ckpt: 5.52e-4  β: 256.0
- binary_pct: 92.39  iso_total_pct: 0.006  gray_center_pct: 0.619
- ma20: 7.59e-4  prev_ma20: 1.13e-3  std20: 2.72e-4  iters_at_β: 11
- iters_since_doubling: 10  grad_sat_pct: 0.0  grad_avg_recent: 1.53e-5

### Diagnosis
- Phase: binarization (Phase 3) — very early post-doubling at β=256 (final β step), cooldown-locked
- Geometry: 3-panel shows raw ε textured noise (normal), filtered ε smooth/organized, projected at **92.4% binary** (jump of +7.6 pts from 84.8% in 10 iters — β=256 rapidly recruiting pixels). Clean black/white domains with identifiable connected structure. gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces, plus sparse 1-pixel specks (mid-transit from recent iters, not stuck). No contiguous blobs. iso_total_pct=0.006% (2B+4W) — matches fire #29, run-minimum. gray_center_pct 1.27→0.619 — **halved** (β=256 compressing transition zone). grad histogram centered ~10^-5 with tail to 10^-2.
- Trajectory: loss_trajectory.png shows β=256 line at iter 493. MA20=7.59e-4, **lower** than pre-advance 1.13e-3 — but the 20-iter window still straddles mostly pre-advance β=128 data. Step-500 iter loss is 5.52e-4. The expected post-doubling spike has NOT yet manifested (only 10 iters in; β=64→128 spike emerged within ~5-8 iters after doubling). Either a spike will emerge in next 0-10 iters, or β=256 is unusually gentle because geometry was already well-binarized. Too early to tell.
- Bottleneck: nothing actionable and cannot act — cooldown rail active. Fire #29 verdict at 21:15:08Z, now 21:30:08Z = **15 min elapsed, 20 min required**.

### Verdict
hold

### Action
none

### Notes
Hard cooldown dominates (5 min short of the 20-min rail). Substantively also correct to hold: 10 iters is too early to judge β=256 response, and current signals (binary surge +7.6pts, gray_center halving, iso stable at run-min, ma20 ≤ pre-advance) are all healthy. Next fire (#31 at ~21:45Z) will have ~25 iters of β=256 data — enough to see whether the expected spike is emerging, how big, and whether recovery looks on-template. Watch for: (a) iter-by-iter loss climbing to 1-3e-2 range (normal spike, hold and monitor recovery), (b) spike >5e-2 or persisting >40 iters without descent (β=256 too aggressive — lr backoff candidate), (c) blob formation in gray zone (wrong-basin), (d) binary_pct stalling below 95% after 80+ iters at β=256 (final plateau candidate — stop_success if floor holds).

## 2026-04-16T21:45:11Z  — fire #31

### State
- julia_alive: true  step: 510  loss_ckpt: 9.01e-4  β: 256.0
- binary_pct: 92.40  iso_total_pct: 0.007  gray_center_pct: 0.619
- ma20: 6.02e-4  prev_ma20: 8.63e-4  std20: 2.21e-4  iters_at_β: 23
- iters_since_doubling: 22  grad_sat_pct: 0.0  grad_avg_recent: 9.44e-6

### Diagnosis
- Phase: binarization → polish (Phase 3/4 transition) — early β=256 (final step), already descending below pre-advance floor without a spike
- Geometry: 3-panel shows raw ε saturated (normal), filtered ε smooth/organized, projected at **92.4% binary** with crisp connected structure preserved. gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces with sparse 1-px specks (mid-transit), no contiguous blobs, no concentrated noise. iso_total_pct=0.007% (1B+5W) — run-minimum class. gray_center_pct **halved** 1.27→0.619 in 22 iters at β=256 (transition zone compressing as expected). Structural identity fully preserved. grad histogram shifted noticeably leftward (mode now ~10^-5 with light tail to 10^-3) — heavy commitment.
- Trajectory: loss_trajectory.png shows β=256 vertical at iter 493. **No spike** — unlike β=32→64 (~5e-2 peak, 50×) and β=64→128 (~1.8e-2 peak, 30×), this advance produced only a small bump and is now smoothly descending into the ~5e-4 region. MA20=6.02e-4 < prev_ma20=8.63e-4 (ratio 0.70) and **below the β=128 floor** (which bottomed at 0.835e-3, fire #26). std20 collapsed from 1.26e-3 (#28-29) → 2.21e-4 (std/ma=0.37, signal dominates). grad_avg fell 4× in one fire (3.82e-5 → 9.44e-6) — pixels rapidly committing. The gentle response is consistent with geometry being already pre-binarized (84.8% pre-advance).
- Bottleneck: nothing. β=256 is doing exactly what it was supposed to — sharpening already-organized structure without disrupting it. Binary still has headroom (92.4% → target 98%+), gray_center still compressing, MA20 still descending, gradients still falling. Acting now would interrupt productive work.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~30 min since fire #29 advance_beta to 256 at 21:15Z). Substantively the right call: the β=256 stage is unfolding cleanly — no spike to recover from, descent already below prior β=128 floor, binary climbing fast, geometry pristine. Plenty of headroom before any decision is needed. Watch for: (a) binary_pct stalling near 95-99% with ma20 floor for 2-3 fires → stop_success candidate, (b) ma20 plateau at ~5-6e-4 with binary stalling → stop_plateau (no more β to advance), (c) ma20 reversal upward without spike (regression at max β) → diagnose, (d) blob formation in gray zone (wrong-basin — absent). Expect 3-6 more fires of productive descent before plateau emerges.

## 2026-04-16T22:00:10Z  — fire #32

### State
- julia_alive: true  step: 520  loss_ckpt: 3.35e-4  β: 256.0
- binary_pct: 92.401  iso_total_pct: 0.004  gray_center_pct: 0.63
- ma20: 5.73e-4  prev_ma20: 6.47e-4  std20: 1.94e-4  iters_at_β: 35
- iters_since_doubling: 34  grad_sat_pct: 0.0  grad_avg_recent: 9.26e-6

### Diagnosis
- Phase: late binarization → polish (Phase 3/4) at FINAL β=256, 34 iters since doubling
- Geometry: 3-panel shows raw ε saturated (normal), filtered ε smooth/organized, projected at 92.4% binary with crisp connected structure (waveguide-like topology preserved). gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces across the field, sparse 1-px specks (mid-transit), no contiguous blobs, no concentrated noise. **iso_total_pct=0.004% (1B+3W) — run-minimum**, down further from fire #31 (0.007). gray_center_pct stable 0.619→0.63. grad histogram centered ~10^-5 with healthy tail to 10^-2.
- Trajectory: loss_trajectory.png shows β=256 vertical at iter 493 with NO spike (unlike β=32→64 ~5e-2 peak and β=64→128 ~1.8e-2 peak). MA20 line descends smoothly below all prior β floors. MA20 sequence post-advance: 7.59e-4 (#30, 10 iters) → 6.02e-4 (#31, 22 iters) → 5.73e-4 (#32, 34 iters) — still descending ~5-11% per fire. std20=1.94e-4 (std/ma=0.34, signal dominates). loss_min_recent=3.11e-4 already below 5e-4 success threshold.
- Bottleneck: nothing actionable. β at safety cap (256), loss still descending, geometry pristine, no spike. Binary appears plateaued at 92.4% across 3 fires (#30/31/32) — pixels committed quickly in first ~10 iters of β=256, now β is polishing interface *positions* without further binarizing. The 7.6% non-binary may be the asymptotic floor at β=256 with filter_radius=5. This is consistent with healthy late-stage behavior, not failure.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~45 min since fire #29 advance to β=256 at 21:17Z, applied 16:17:03 local). Substantively right call: MA20 made meaningful progress (8.63e-4 → 5.73e-4 across two fires, ~33% reduction), loss_min 3.11e-4 already below success threshold. Acting now would interrupt productive polish at the final β. Watch for: (a) ma20 ≤ 5e-4 with binary holding ≥ 92% for 2-3 fires → **stop_success** (success criteria: binary ≥ 98% AND ma20 ≤ 5e-4 — but if binary asymptotes below 98% at β=256, stop_plateau is the practical stop); (b) ma20 floor at ~5-6e-4 with no further descent for 3+ fires → **stop_plateau**; (c) ma20 reversal upward without spike → diagnose; (d) blob formation in gray_zone (wrong-basin — absent for entire run). Expect 2-5 more fires before plateau emerges or success criteria hit.

## 2026-04-16T22:15:09Z  — fire #33

### State
- julia_alive: true  step: 530  loss_ckpt: 3.27e-4  β: 256.0
- binary_pct: 92.408  iso_total_pct: 0.004  gray_center_pct: 0.63
- ma20: 4.93e-4  prev_ma20: 6.10e-4  std20: 1.53e-4  iters_at_β: 47
- iters_since_doubling: 46  grad_sat_pct: 0.0  grad_avg_recent: 9.14e-6

### Diagnosis
- Phase: polish (Phase 4) at FINAL β=256, 46 iters since doubling
- Geometry: 3-panel shows raw ε saturated (normal), filtered ε smooth/organized, projected at 92.4% binary with crisp connected structure preserved. gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces with sparse 1-px specks (mid-transit), no contiguous blobs, no concentrated noise. iso_total_pct=0.004% (1B+3W) — **run-minimum**, unchanged from fire #32. gray_center_pct stable 0.619→0.63 across 3 fires (#31/32/33) — transition zone fully compressed. grad histogram centered ~10^-5 with healthy tail to 10^-2.
- Trajectory: loss_trajectory.png shows β=256 vertical at iter 493 with NO spike (unique among β doublings — β=32→64 ~5e-2 peak, β=64→128 ~1.8e-2 peak, β=128→256 flat). MA20 descent at β=256: 7.59e-4 (#30, 10 iters) → 6.02e-4 (#31, 22 iters) → 5.73e-4 (#32, 34 iters) → **4.93e-4 (#33, 46 iters)**. Fire-over-fire ratio ~0.81-0.95, still clean descent. std20=1.53e-4 (std/ma=0.31, signal dominates). loss_min_recent=2.88e-4 (below 5e-4 success threshold). grad_avg flat 9.14e-6 — pixels fully committed.
- Bottleneck: nothing actionable. β at safety cap (256), loss still descending ~14% per fire, geometry pristine. Binary plateau at 92.4% across 4 fires (#30/31/32/33) appears structural — asymptotic floor at β=256 with filter_radius=5, not failure. MA20 just crossed 5e-4 success loss threshold but binary 92.4% < 98% target (and binary is locked at max β, so target is unreachable by design).

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~60 min since fire #29 advance_beta to 256 at 21:15Z). Substantively the right call: the β=256 polish stage is unfolding cleanly — smooth descent, no spike, geometry pristine, gradients stable. Descent still ~14% per fire (6.10e-4 → 4.93e-4), plenty of headroom. Practical stop criterion is emerging: since binary locks at 92.4% at max β (structural), stop_success as defined (binary ≥ 98%) is unreachable — the relevant stop is **stop_plateau** when MA20 floors out. Not yet: clean descent continues. Watch for: (a) MA20 plateau at ~3-4e-4 for 2-3 fires → stop_plateau (final result); (b) MA20 ≤ 3e-4 with clean floor → stop_success in spirit (loss target hit at max β); (c) MA20 reversal upward without spike → diagnose; (d) blob formation in gray_zone → wrong-basin (absent entire run). Expect 2-5 more fires before plateau emerges.

## 2026-04-16T22:30:09Z  — fire #34

### State
- julia_alive: true  step: 550  loss_ckpt: 3.30e-4  β: 256.0
- binary_pct: 92.416  iso_total_pct: 0.004  gray_center_pct: 0.636
- ma20: 4.91e-4  prev_ma20: 5.43e-4  std20: 1.52e-4  iters_at_β: 58
- iters_since_doubling: 57  grad_sat_pct: 0.0  grad_avg_recent: 8.83e-6

### Diagnosis
- Phase: polish (Phase 4) at FINAL β=256, 57 iters since doubling — MA20 crossed 5e-4 threshold for first time; early plateau deceleration
- Geometry: 3-panel shows raw ε saturated (normal), filtered ε smooth/organized, projected at 92.4% binary with clean connected waveguide-like structure. gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces, sparse 1-px specks (mid-transit), no contiguous blobs, no concentrated noise. iso_total_pct=0.004% (1B+3W) — **run-minimum**, unchanged from fires #32 and #33 (three consecutive fires at the floor). gray_center_pct stable 0.619→0.636 across 4 fires (#31-#34). grad histogram centered ~10^-5 with healthy tail to 10^-2.5, mode tight — pixels committed.
- Trajectory: loss_trajectory.png shows β=256 vertical at iter 493 with NO spike. MA20 descent at β=256: 7.59e-4 (#30) → 6.02e-4 (#31) → 5.73e-4 (#32) → 4.93e-4 (#33) → **4.91e-4 (#34)**. **Fire-over-fire ratio collapsed from 0.86 (last fire) to 0.996 (this fire)** — clear deceleration. Within-window ratio 0.905 (MA20/prev_ma20) still shows 10% descent across most recent 20 iters, so not fully flat yet. std20=1.52e-4 (std/ma=0.31, signal dominates). loss_min_recent=2.78e-4, loss_max_recent=6.25e-3. grad_avg flat 9.14e-6 → 8.83e-6.
- Bottleneck: nothing actionable. β at safety cap (256), binary plateaued at 92.4% for 5 fires (structural asymptote at max β + filter_radius=5), loss just crossed 5e-4 threshold with fire-over-fire deceleration consistent with approach to floor. Need 1-2 more fires to distinguish "genuine plateau" (fire-over-fire ratio ~1.0 sustained, MA20 flatlines ~4.9e-4) from "slow continued polish" (MA20 resumes descent into 3-4e-4 range). Either resolves to stop_plateau or stop_success — not urgent.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~75 min since fire #29 advance_beta to 256 at 21:15Z). Substantively the right call: fire-over-fire stall (4.93 → 4.91 MA20, Δ=-0.4%) is the first plateau signal after 4 fires of clean descent, but single-fire deceleration with within-window ratio still 0.905 is insufficient for a stop call. Plateau criterion needs 2-3 fires of flat MA20. No failure signals — geometry at run-minimum cleanliness, no wrong-basin, no gradient saturation. Watch for: (a) MA20 flat at ~4.9e-4 with ratio ≈ 1.0 for 2+ more fires → **stop_plateau**; (b) MA20 drops below 4e-4 showing ongoing polish → keep holding; (c) MA20 reverses above 5.5e-4 for 2 fires → diagnose regression; (d) blob formation in gray_zone (wrong-basin — absent entire run). Loss floor crossed 5e-4 threshold; binary 92.4% < 98% (structural ceiling at max β) — practical stop is stop_plateau when floor confirms.

## 2026-04-16T22:45:09Z  — fire #35

### State
- julia_alive: true  step: 550  loss_ckpt: 3.30e-4  β: 256.0
- binary_pct: 92.416  iso_total_pct: 0.004  gray_center_pct: 0.636
- ma20: 4.66e-4  prev_ma20: 4.93e-4  std20: 1.97e-4  iters_at_β: 67
- iters_since_doubling: 66  grad_sat_pct: 0.0  grad_avg_recent: 8.40e-6

### Diagnosis
- Phase: polish (Phase 4) at FINAL β=256, 66 iters since doubling — MA20 resumed descent after fire #34's apparent stall
- Geometry: 3-panel shows raw ε saturated (normal), filtered ε smooth/organized, projected at 92.4% binary with clean connected waveguide-like structure fully preserved. gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces, sparse 1-px specks (mid-transit), no contiguous blobs, no concentrated noise. iso_total_pct=0.004% (1B+3W) — run-minimum, unchanged for **four consecutive fires** (#32/#33/#34/#35). gray_center_pct stable 0.619→0.636 across 5 fires (#31-#35) — transition zone fully compressed and stable. grad histogram centered ~10^-4.8 (≈1.6e-5) with healthy tail to 10^-3, mode tight — pixels committed and polishing.
- Trajectory: loss_trajectory.png shows β=256 vertical at iter 493 with NO spike (unique). MA20 descent sequence at β=256: 7.59e-4 (#30) → 6.02e-4 (#31) → 5.73e-4 (#32) → 4.93e-4 (#33) → 4.91e-4 (#34, stall) → **4.66e-4 (#35)**. **Fire-over-fire ratio #34→#35 = 0.949** — clean resumption of descent (ratio #33→#34 was 0.996). Within-window ratio 0.946 confirms ongoing 5% descent across most recent 20 iters. std20=1.97e-4 (std/ma=0.42). **loss_min_recent=1.63e-4** — new run floor (prior fires: 3.11e-4 at #32, 2.88e-4 at #33, 2.78e-4 at #34). grad_avg 8.83e-6 → 8.40e-6 (pixels slowly committing further).
- Bottleneck: nothing actionable. β at safety cap, geometry at run-minimum cleanliness, MA20 descending ~5% per fire with new loss-min floor each fire. Fire #34's "stall" resolved into continued descent this fire — consistent with stochastic MA20 noise on a true descending trajectory, not the onset of plateau. Binary plateau at 92.4% across 5 fires remains structural (filter_radius=5 asymptote at max β).

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~90 min since fire #29 advance_beta to 256 at 21:15Z). Substantively the right call: fire-over-fire resumed descent (0.949 ratio) + new loss_min (1.63e-4, half the fire #33 floor) rule out plateau. Polish continuing productively at the final β. No failure signals — geometry pristine, no wrong-basin, no gradient saturation, no regression. Watch for: (a) MA20 flat for 2-3 consecutive fires (ratio ≥ 0.98) → **stop_plateau** (final result); (b) MA20 crosses 3e-4 with continued descent → continue holding; (c) MA20 reversal above 5.5e-4 for 2 fires → diagnose regression; (d) blob formation in gray_zone → wrong-basin (absent entire run). Since binary ceiling 92.4% < 98% target is structural at max β, the practical terminal state is stop_plateau; not yet — expect 2-5 more fires of productive polish before floor emerges.

## 2026-04-16T23:00:10Z  — fire #36

### State
- julia_alive: true  step: 570  loss_ckpt: 4.24e-4  β: 256.0
- binary_pct: 92.41  iso_total_pct: 0.007  gray_center_pct: 0.622
- ma20: 4.89e-4  prev_ma20: 4.91e-4  std20: 2.80e-4  iters_at_β: 78
- iters_since_doubling: 77  grad_sat_pct: 0.0  grad_avg_recent: 9.13e-6

### Diagnosis
- Phase: polish (Phase 4) at FINAL β=256, 77 iters since doubling — first within-window flat MA20 signal (ratio 0.9955)
- Geometry: 3-panel shows raw ε saturated (normal), filtered ε smooth/organized, projected at 92.4% binary with clean connected waveguide-like structure fully preserved (unchanged across 6 fires). gray_zone.png textbook healthy — fine curvilinear ribbons tracing thin interfaces, sparse 1-px specks (mid-transit), no contiguous blobs, no concentrated noise. iso_total_pct=0.007% (3B+3W) — ticked up marginally from 0.004% (three fires at floor) but still run-minimum class. gray_center_pct stable 0.619→0.636→0.622 across 6 fires — transition zone fully compressed. grad histogram centered ~10^-4.8 with tail to 10^-3, mode tight.
- Trajectory: loss_trajectory.png shows β=256 vertical at iter 493 with NO spike (unique). MA20 descent at β=256: 7.59e-4 (#30) → 6.02e-4 (#31) → 5.73e-4 (#32) → 4.93e-4 (#33) → 4.91e-4 (#34) → 4.66e-4 (#35) → **4.89e-4 (#36, +5%)**. **Within-window ratio ma20/prev_ma20 = 0.9955** (essentially flat across most recent 40 iters) — first clear flat reading (fires #34 was 0.905, #35 was 0.946). std20 jumped 1.97e-4 → 2.80e-4 (std/ma=0.57, noise growing). loss_min_recent=1.63e-4 **unchanged from #35** — no new floor this fire. loss_max_recent=6.25e-3 (some large single-iter spikes in window).
- Bottleneck: nothing actionable. β at safety cap (256), binary structural plateau at 92.4%. MA20 showing FIRST flat signal (both fire-over-fire +5% within stochastic noise AND within-window ratio 0.9955) plus frozen loss_min — this is the earliest plateau signature, but not yet confirmed (single-fire reading, std20=2.80e-4 means +5% well within noise band). Plateau criterion needs 2-3 consecutive flat fires.

### Verdict
hold

### Action
none

### Notes
Cooldown long satisfied (~1h43min since fire #29 advance_beta to 256 at 21:17Z applied). Substantively right call: first flat-MA20 fire after 6 fires of clean descent at β=256, plus frozen loss_min for first time — consistent with plateau onset, but single-fire reading with std20=2.80e-4 is insufficient. One more flat fire (ratio ≥ 0.98 AND loss_min frozen) will confirm stop_plateau. No failure signals — geometry at run-minimum, no wrong-basin (6-fire clean history), no gradient saturation, no regression. Watch for: (a) MA20 ratio ≥ 0.98 for one more fire with loss_min unchanged → **stop_plateau** (final result); (b) MA20 resumes descent (<4.5e-4) or new loss_min → continue holding (fire #34→#35 already demonstrated this pattern once); (c) MA20 reversal above 5.5e-4 sustained 2 fires → diagnose regression; (d) blob formation → wrong-basin (absent entire run). Run is at the margin of stop_plateau; next fire decides.

## 2026-04-16T23:15:09Z  — fire #37

### State
- julia_alive: true  step: 570  loss_ckpt: 4.24e-4  β: 256.0
- binary_pct: 92.41  iso_total_pct: 0.007  gray_center_pct: 0.622
- ma20: 5.15e-4  prev_ma20: 4.66e-4  std20: 2.48e-4  iters_at_β: 87
- iters_since_doubling: 86  grad_sat_pct: 0.0  grad_avg_recent: 8.24e-6

### Diagnosis
- Phase: polish (Phase 4) at FINAL β=256, 86 iters since doubling — PLATEAU CONFIRMED
- Geometry: 3-panel unchanged — raw ε saturated, filtered ε smooth with identifiable waveguide-like structure, projected 92.4% binary with clean connected structure preserved for 7 consecutive fires. gray_zone.png still textbook healthy — fine curvilinear ribbons tracing thin interfaces, sparse 1-px specks, no contiguous blobs, no concentrated noise. iso_total_pct=0.007% (3B+3W, vs 0.004% for three prior fires — marginal uptick, still run-minimum class). gray_center_pct 0.622 (sequence 0.619→0.636→0.622, stable). grad histogram centered ~10^-5 with tail to 10^-3, mode tight — pixels fully committed.
- Trajectory: loss_trajectory.png shows β=256 vertical at iter 493, clear MA20 floor emerging on right edge. MA20 at β=256: 7.59e-4 (#30) → 6.02e-4 (#31) → 5.73e-4 (#32) → 4.93e-4 (#33) → 4.91e-4 (#34) → 4.66e-4 (#35) → 4.89e-4 (#36) → **5.15e-4 (#37)**. Last 4 fires in band [4.66, 5.15]×1e-4, no descent. Within-window ratio ma20/prev_ma20 = 1.106 (second consecutive non-descending fire after 0.9955 at #36). std20=2.48e-4 (std/ma=0.48, high noise). **loss_min_recent=1.63e-4 unchanged for 3rd consecutive fire** (#35/#36/#37) — no new best loss in 20+ iters = basin found. Checkpoint step=570 unchanged from fire #36 (Julia running, next save pending at ~iter 580).
- Bottleneck: nothing actionable. β at safety cap (256), binary at structural ceiling (92.4% at max β + filter_radius=5), loss at MA20 floor ~5e-4 with loss_min frozen. Pre-committed plateau criterion from fire #36 ("MA20 ratio ≥ 0.98 for one more fire with loss_min unchanged → stop_plateau") is satisfied: ratio 1.106 ≥ 0.98, loss_min 1.63e-4 unchanged. Two consecutive flat-to-rising MA20 readings plus 3-fire frozen loss_min is a stronger signal than fire #34's isolated stall (which resumed descent at #35). No lever remains — β can't advance (cap), lr change would only slow non-existent descent, filter_radius change requires restart. Run is terminal.

### Verdict
stop_plateau

### Action
control.toml:
stop = true

### Notes
Cooldown long satisfied (~2h since fire #29 advance_beta to 256 at 21:17Z applied). Terminal state at β=256: MA20≈5.15e-4, loss_min=1.63e-4, binary=92.4%, geometry pristine with identifiable waveguide-like structure, zero wrong-basin signals across 37 fires. Binary ≥ 98% success target was structurally unreachable at max β (filter_radius=5 asymptote), so stop_plateau is the practical terminal verdict. Run delivered: clean β schedule (32→64→128→256), healthy post-doubling recoveries at β=64/128, unique no-spike β=256 doubling, monotonic MA20 descent across 7 fires at final β (7.59→4.66e-4, 39% reduction), then flattening onto a ~5e-4 floor. No memory updates needed — no new patterns; this is a canonical successful-polish-then-plateau trajectory consistent with existing project_autotune_phases.md expectations.

## 2026-04-16T23:30:08Z  — fire #38

### State
- julia_alive: true (in shutdown)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752 (local CDT = 23:17:52Z)

### Diagnosis
- Phase: terminated (post-stop_plateau)
- Geometry: unchanged from fire #37 — 92.4% binary, clean waveguide-like structure, healthy interface-concentrated gray zone, iso_total_pct=0.007% (run-minimum class)
- Trajectory: Julia consumed fire #37's stop=true at iter 583, logged "Exiting at iter 583." in nohup.out. Final iters 578-582 continued in the 3.3e-4 – 5.5e-4 band consistent with the plateau diagnosis.
- Bottleneck: none — run has terminated per fire #37's plateau verdict.

### Verdict
hold

### Action
none

### Notes
Terminal fire — run already stopped cleanly. Fire #37's stop_plateau was written to control.toml and consumed by Julia (applied-file timestamp 20260416_181752 is local-time CDT = 23:17:52 UTC, matching fire #37's write window). Five total applied files across the run match the five non-hold actions taken (four β-doublings + final stop). No notification needed (fire #37 handled the PLATEAU alert). Cron may continue firing; subsequent fires will all hold while julia_alive reports transiently true during shutdown, then false — no action warranted either way.

## 2026-04-16T23:45:10Z  — fire #39

### State
- julia_alive: false (process gone — ps shows no julia/PhEnd2End)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752 (CDT 18:17:52 = UTC 23:17:52)

### Diagnosis
- Phase: terminated (post-stop_plateau, 2 fires after exit)
- Geometry: unchanged from fires #37/#38 — 92.4% binary, clean waveguide-like structure, healthy interface-concentrated gray zone, iso_total_pct=0.007% (run-minimum class). Snapshot regenerates from same checkpoint.
- Trajectory: unchanged. Loss-trajectory PNG shows clean β schedule (32→64→128→256), post-#256 tail at ~5e-4 floor with no spike at the 256 doubling, terminal MA20=5.08e-4. Final log line in nohup.out: "Exiting at iter 583." per fire #37's stop=true write.
- Bottleneck: none — run finished cleanly. Inspector's `julia_alive=true` from this fire was a stale read; ps confirms no process. No notification needed (fire #37 already sent PLATEAU; this death is the *expected* consequence of that intentional stop, not an unexpected crash).

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron should continue to be a no-op until disabled or reused for a new run. No memory updates — pattern is canonical and already captured.

## 2026-04-17T00:00:08Z  — fire #40

### State
- julia_alive: false (ps confirms no zlin julia process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752 (CDT 18:17:52 = UTC 23:17:52)

### Diagnosis
- Phase: terminated (post-stop_plateau, 3 fires after exit)
- Geometry: unchanged from #37/#38/#39 — 92.4% binary, clean waveguide-like structure, healthy interface-concentrated gray zone, iso_total_pct=0.007% (run-minimum class). Snapshot regenerates from same checkpoint each fire.
- Trajectory: unchanged. Final terminal MA20=5.08e-4, loss_min=1.63e-4. nohup.out tail still ends at "Exiting at iter 583." per fire #37's stop=true write.
- Bottleneck: none — run finished cleanly. Death already noted in fire #39 (no notification — was expected outcome of intentional stop_plateau).

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron continues to be a no-op until disabled or reused for a new run. No memory updates — pattern is canonical and already captured.

## 2026-04-17T00:15:10Z  — fire #41

### State
- julia_alive: false (ps confirms no process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 4 fires after exit)
- Geometry: unchanged — 92.4% binary, clean waveguide-like structure, iso_total_pct=0.007%. Snapshot regenerates from same checkpoint each fire.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4. nohup.out still ends at "Exiting at iter 583."
- Bottleneck: none — run finished cleanly per fire #37's stop_plateau. Death already noted at fire #39 (expected consequence of intentional stop, no notification needed).

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or reused for a new run. No memory updates.

## 2026-04-17T00:30:08Z  — fire #42

### State
- julia_alive: false (ps: only other user's unrelated run_step1.jl; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 5 fires after exit)
- Geometry: unchanged — 92.4% binary, clean waveguide-like structure, iso_total_pct=0.007%. Inspector regenerates snapshot from same checkpoint each fire.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4. nohup.out ends at "Exiting at iter 583." per fire #37's stop=true write.
- Bottleneck: none — run finished cleanly per fire #37's stop_plateau. Death already noted at fire #39 (expected consequence of intentional stop, no notification).

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or reused for a new run. No memory updates.

## 2026-04-17T00:45:10Z  — fire #43

### State
- julia_alive: false (pgrep confirms no zlin julia process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 6 fires after exit)
- Geometry: unchanged — 92.4% binary, iso_total_pct=0.007%. Snapshot regenerates from same checkpoint each fire.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly per fire #37's stop_plateau. Death already noted at fire #39.

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or reused for a new run. No memory updates.

## 2026-04-17T01:00:13Z  — fire #44

### State
- julia_alive: false (no zlin julia process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 7 fires after exit)
- Geometry: unchanged — 92.4% binary, iso_total_pct=0.007%. Snapshot regenerates from same checkpoint each fire.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly per fire #37's stop_plateau. Death already noted at fire #39.

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or reused for a new run. No memory updates.

## 2026-04-17T01:15:09Z  — fire #45

### State
- julia_alive: false (no zlin julia process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 8 fires after exit)
- Geometry: unchanged — 92.4% binary, iso_total_pct=0.007%. Snapshot regenerates from same checkpoint each fire.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly per fire #37's stop_plateau. Death already noted at fire #39.

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or reused for a new run. No memory updates.

## 2026-04-17T01:30:20Z  — fire #46

### State
- julia_alive: false (no zlin julia optimization process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 9 fires after exit)
- Geometry: unchanged — 92.4% binary, iso_total_pct=0.007%. Snapshot regenerates from same frozen checkpoint.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly per fire #37's stop_plateau. Death already noted at fire #39.

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or reused for a new run. No memory updates.

## 2026-04-17T01:45:08Z  — fire #47

### State
- julia_alive: false (pgrep: no zlin julia optimization process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 10 fires after exit)
- Geometry: unchanged — 92.4% binary, iso_total_pct=0.007%. Inspector regenerates snapshot from same frozen checkpoint.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly per fire #37's stop_plateau. Death already noted at fire #39.

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or reused for a new run. No memory updates.

## 2026-04-17T02:00:09Z  — fire #48

### State
- julia_alive: false (no zlin julia optimization process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 11 fires after exit)
- Geometry: unchanged — 92.4% binary, iso_total_pct=0.007%. Inspector regenerates snapshot from same frozen checkpoint.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly per fire #37's stop_plateau. Death already noted at fire #39.

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or reused for a new run. No memory updates.

## 2026-04-17T02:15:09Z  — fire #49

### State
- julia_alive: false (pgrep confirms no zlin julia optimization process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 12 fires after exit)
- Geometry: unchanged — 92.4% binary, iso_total_pct=0.007%. Inspector regenerates snapshot from same frozen checkpoint.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly per fire #37's stop_plateau. Death already noted at fire #39.

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or reused for a new run. No memory updates.

## 2026-04-17T02:30:09Z  — fire #50

### State
- julia_alive: false (pgrep: no zlin julia PhEnd2End process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 13 fires after exit)
- Geometry: unchanged — frozen checkpoint eps_geom_step_00580.jls.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly at fire #37 (stop_plateau). Death noted fire #39.

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or repurposed for a new run. No memory update.

## 2026-04-17T02:45:08Z  — fire #51

### State
- julia_alive: false (pgrep: no zlin julia PhEnd2End process; inspector's `true` is stale)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 14 fires after exit)
- Geometry: unchanged — frozen checkpoint eps_geom_step_00580.jls; gray pixels remain thin interface traces.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly at fire #37 (stop_plateau). Death noted fire #39.

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or repurposed for a new run. No memory update.

## 2026-04-17T03:00:09Z  — fire #52

### State
- julia_alive: false (stale inspector flag; no zlin julia PhEnd2End process)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 15 fires after exit)
- Geometry: unchanged — frozen checkpoint eps_geom_step_00580.jls.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly at fire #37 (stop_plateau).

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or repurposed. No memory update.

## 2026-04-17T03:15:10Z  — fire #53

### State
- julia_alive: false (stale inspector flag; no zlin julia PhEnd2End process)  step: 580  loss_ckpt: 3.93e-4  β: 256.0
- binary_pct: 92.403  iso_total_pct: 0.007  gray_center_pct: 0.620
- ma20: 5.08e-4  prev_ma20: 4.56e-4  std20: 2.48e-4  iters_at_β: 90
- iters_since_doubling: 89  grad_sat_pct: 0.0  grad_avg_recent: 8.48e-6
- done: true  control_pending: false  last_applied: control.toml.applied.20260416_181752

### Diagnosis
- Phase: terminated (post-stop_plateau, 16 fires after exit)
- Geometry: unchanged — frozen checkpoint eps_geom_step_00580.jls.
- Trajectory: unchanged. Terminal MA20=5.08e-4, loss_min=1.63e-4.
- Bottleneck: none — run finished cleanly at fire #37 (stop_plateau).

### Verdict
hold

### Action
none

### Notes
Routine post-termination fire. Cron remains a no-op until disabled or repurposed. No memory update.
