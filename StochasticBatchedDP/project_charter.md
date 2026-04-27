# Experiment Plan: SBDP at φ_max = 0.5, K_total = 16–32, dim(c) > 100

The most impactful configuration the framework enables.

## Goal

Demonstrate that **stochastic batched DP beats the canonical adaptive-QPE baseline
(geometric-Ramsey with Higgins-Wiseman feedback) at long horizons in the wide-prior
aliasing-floor regime**.  Specifically, at `φ_max = 0.5` (the "uninformative" prior limit
where the existing exact-DP joint-DP advantage collapses to 1.76× at K=4) and at horizons
`K_total = 16, 32` (well beyond the exact-DP ceiling), with `dim(c) > 100` continuous
parametric hardware optimized via stochastic-gradient descent, push the
**SBDP / Higgins-Wiseman MSE ratio to ≥ 5× at K_total = 16 and ≥ 10× at K_total = 32**.

The Higgins-Wiseman protocol — geometric delays `τ_k = τ_max / 2^(k-1)` with feedback
phase chosen at each step from the current posterior mean — is the gold-standard
adaptive baseline in the quantum metrology literature (Berry-Wiseman 2000;
Higgins-Berry-Wiseman-Pryde 2007 Nature).  Beating it by ≥5× at long horizon is the
substantive headline; beating PCRB-extended (the non-adaptive baseline) by orders of
magnitude follows trivially and is reported as a secondary number.

## Why this configuration is the most impactful

Three independent claims combine here, each compelling on its own:

### Claim 1: SBDP escapes the aliasing floor

The §7.6 prediction in the parent paper ("at φ_max → Φ_0/2 the adaptive margin shrinks")
was empirically confirmed at K=4: ratio drops from 11.3× (φ_max=0.1) to 1.76× (φ_max=0.5).
The cliff is the prior wrapping past one Ramsey fringe at the longest delay.

At long horizons (K=16, 32) with adaptive Higgins-Wiseman feedback, the policy has
information-theoretic room to disambiguate: at K=16 with n=10 shots/epoch and active
fringe-disambiguating actions, ~160 bits of total Ramsey information are available, vs
~8 bits needed to resolve a 2-fringe prior to 1% precision.  The bottleneck is no longer
information; it's the policy's ability to extract it.

**SBDP enables this**: exact Bellman fails at K≥5 due to count-tuple memo blow-up; SBDP
extends the framework to arbitrary K_total at the cost of receding-horizon approximation
within each K_batch.  If the receding-horizon truncation is tolerable (which we expect at
K_batch=4 since each batch can complete a coarse-to-fine localization arc), SBDP should
reproduce most of the long-horizon advantage.

**Why SBDP should beat Higgins-Wiseman specifically at φ_max=0.5**: HW's geometric
schedule starts at the longest delay `τ_max` to read the coarsest phase bit.  At
φ_max=0.5 the prior spans ~1.85 Ramsey fringes at τ_max, so HW's first measurement is
nearly aliased: `p(y=1)` is almost the same value at both fringes, contributing little
bit-level information.  HW then halves `τ` and re-measures, but several halvings are
"wasted" before τ becomes short enough to disambiguate.  Bellman-optimal policy, by
contrast, starts at a SHORT τ to disambiguate the multimodal prior, then transitions to
long τ for refinement within the selected fringe.  SBDP at K_batch=4 should reproduce
this short-then-long structure within each batch, while HW cannot.

**Expected SBDP / HW ratio**: ≥ 5× at K_total=16, ≥ 10× at K_total=32.  Beating PCRB-
extended (non-adaptive) follows trivially with much larger ratios (≥ 100× expected),
reported as a secondary check.

### Claim 2: First long-horizon Bellman-optimal joint co-design

K_total = 16–32 is well beyond the K=4–5 ceiling of exact Bellman DP.  No prior work
demonstrates Bellman-optimal joint hardware-policy co-design at these horizons (see
literature analysis in `README.md`).  SDDP, sequential Bayesian OED, and deep-RL
adaptive design all use approximations at the inner-policy level.  SBDP retains exact
Bellman within each batch.

### Claim 3: Stochastic-gradient hardware co-design at dim(c) > 100

BayesOpt struggles past 20 dims.  Mensch-Blondel-style differentiable DP doesn't address
POMDP belief-space integrals.  Deep-RL adaptive co-design exhibits the hardware-light
pathology.  **SBDP's per-sample envelope-theorem-exact gradient enables Adam to climb a
220-dim continuous hardware landscape**, which to our knowledge is the highest-dim
parametric continuous hardware optimization with exact-Bellman inner solver in the
literature.

The combination of (φ_max=0.5 wide-prior aliasing regime) × (K=16–32 long horizon) ×
(dim(c)=220) is uniquely supported by SBDP and demonstrates the framework's three
independent advantages in one experiment.

## Phases

Five phases, each with concrete deliverables and go/no-go criteria.

### Phase 0: Infrastructure (~1 week, no compute)

**Goal**: implementation framework ready for SBDP runs.

**Tasks**:
- Extend `Bellman.jl` / `BellmanThreaded.jl` to take a non-uniform prior (currently bakes
  in uniform).
- Implement R=2 phase-basis action dimension in actions; extend count-tuple memo
  accordingly.
- Implement Layers 1–7 of the parametric `c` (220 dims) with analytical-gradient closed
  forms for all 220 parameters into the Ramsey likelihood.
- Implement the SBDP gradient pipeline: per-trajectory pathwise + per-batch advantage
  score-function terms.
- Implement five baselines: PCRB-extended, geometric-Ramsey + Higgins-Wiseman feedback,
  particle-filter myopic info-gain, deep-RL (small LSTM PPO), and a no-feedback fixed
  schedule.
- Unit tests at K=4: SBDP n=1 gradient should match the existing exact-Bellman gradient
  bit-for-bit.

**Deliverable**: code passes K=4 calibration tests; baselines computable on demand.

### Phase 1: Calibration (~3 days, 200 cores)

**Goal**: confirm SBDP framework works numerically at φ=0.5.

**Setup**:
- K_total = 8, K_batch = 4, n = 2 (single batch boundary)
- dim(c) = 7 (Layer 1 only, existing Danilin)
- M = 10 trajectories per gradient step
- K_phi = 64
- Action set (j, ℓ, r) with R=2 (40 actions)
- T_outer = 100 Adam steps

**Compute estimate**: per gradient step ~5 min × T=100 = ~10 hours wall.

**Deliverables**:
- SBDP gradient direction matches finite-difference gradient at multiple test points
- Adam reaches a stationary point of V_batched
- V_batched(c_SBDP) > V_batched(c_init), monotonically over Adam iterations
- Comparison vs exact K=4 Bellman: V_K=4 ≤ V_SBDP_K=8 (longer horizon should help)

**Go/no-go**: if V_batched does not improve monotonically, debug before Phase 2.

### Phase 2: Main experiment, K_total = 16, dim(c) = 220 (~3 weeks, 380 cores)

**Goal**: demonstrate the central headline.

**Setup**:
- K_total = 16, K_batch = 4, n = 4
- dim(c) = 220 (all 7 layers)
- M = 20 trajectories per gradient step
- K_phi = 64 (compromise: accuracy ↔ speed at φ=0.5 wide prior)
- Action set (j, ℓ, r) with R=2; consider J=5 reduction if K=4 Bellman at φ=0.5 too slow
- T_outer = 200 Adam steps with learning-rate schedule

**Compute estimate**: per K=4 Bellman at φ=0.5 K_phi=64 ~3–5 min (depending on action-set
trim).  Per trajectory at n=4: ~12–20 min.  Per gradient step at M=20 with 4-way
parallelism on samples: ~1–1.5 hours.  T=200: ~10–14 days.

**Deliverables**:
- Final c* and final V_batched(c*) at φ_max=0.5, K_total=16
- Paired-MC comparison vs all five baselines at the same K_total=16, paired x_true seeds
- Headline ratio table

**Headline target**: **MSE_HW / MSE_SBDP ≥ 5× at K_total=16** (vs Higgins-Wiseman, the
canonical adaptive QPE baseline).  Secondary: MSE_PCRB / MSE_SBDP ≥ 100× (vs the
non-adaptive baseline).  The HW comparison is the substantive headline; PCRB the
sanity check.

### Phase 3: Long-horizon scaling, K_total = 32 (~3 weeks, 380 cores)

**Goal**: extend horizon to demonstrate the scaling.

**Setup**:
- K_total = 32, K_batch = 4, n = 8
- dim(c) = 220 (warm-start at Phase-2 c*)
- M = 30 trajectories per gradient step (more samples needed for variance at higher n)
- K_phi = 64
- T_outer = 100 Adam steps from warm-start

**Compute estimate**: per gradient step at n=8 vs n=4: ~2× cost. T=100: ~14 days.

**Deliverables**:
- Final c** and final V_batched(c**) at φ_max=0.5, K_total=32
- Paired-MC comparison vs baselines at K_total=32

**Headline target**: **MSE_HW / MSE_SBDP ≥ 10× at K_total=32**.  Demonstrates that SBDP's
advantage *grows* with horizon (5× at K=16 → 10× at K=32), rather than saturating.
Secondary: MSE_PCRB / MSE_SBDP ≥ 500× (the non-adaptive comparison continues to widen).

### Phase 4: Variance-reduction & ablation studies (~1 week)

**Goal**: characterize the framework's behavior.

**Tasks**:
- Sweep M ∈ {5, 10, 20, 30, 50}: measure gradient noise and Adam convergence rate
- Sweep K_batch ∈ {2, 3, 4}: measure approximation gap to exact Bellman at K_total = 16
- Compare advantage baselines: empirical mean vs leave-one-out vs local-Bellman
- Ablate: drop pathwise term, drop score-function term, see what survives
- Ablation on dim(c): use only Layers 1–3 (21 dims), 1–5 (56 dims), 1–7 (220 dims).
  Report the marginal gain per layer.

**Deliverables**:
- Variance-vs-M plot
- Approximation-gap-vs-K_batch plot
- Per-layer marginal-gain plot

### Phase 5: Writeup (~3 weeks)

**Goal**: paper draft.

**Tasks**:
- Methods section: SBDP framework + gradient derivation (largely from `.tex` already)
- Case study section: scqubit problem, parametric c layers, baselines
- Results section: Phase 2/3/4 outputs
- Discussion: connection to existing methods, limitations, extensions

## Compute budget (total)

| Phase | Days | Cores | CPU-days |
|---|---:|---:|---:|
| 0 | 7 | 0 (dev) | 0 |
| 1 | 3 | 200 | 600 |
| 2 | 18 | 380 | 6840 |
| 3 | 18 | 380 | 6840 |
| 4 | 7 | 200 | 1400 |
| 5 | 21 | 0 (writing) | 0 |
| **Total** | **74** | | **15,680** |

About 2.5 months wall-clock from Phase 0 start.  Compute envelope ~16,000 CPU-days, well
within the available envelope on this machine.

## Risk analysis

### High-impact risks

1. **K=4 Bellman at φ=0.5 K_phi=64 is still too slow.**
   *Probability*: medium.  *Impact*: blocks Phase 2.
   *Mitigation*: action-set trim (J=5, drop L=1 → 10-action × R=2 = 20 actions, 4× faster
   Bellman); custom packed hash-table memo (2–3× faster); K_phi=32 (additional 2× faster
   at known accuracy cost).  Stack of mitigations gets us 16–32× speedup; should
   suffice.

2. **SBDP gradient variance kills convergence at high dim(c).**
   *Probability*: medium.  *Impact*: makes the dim(c)=220 demonstration noisy.
   *Mitigation*: advantage baselines (already in framework); larger M (30 → 50);
   gradient clipping; layered learning rates per c-layer (smaller LR for high-dim
   pulse-library Layer 6).

3. **Approximation gap of receding-horizon SBDP at K_total=16 leaves big room vs full
   K=16 Bellman.**
   *Probability*: medium-high.  *Impact*: weakens the headline if we can't bracket the
   gap.
   *Mitigation*: this is fundamental.  Acknowledge in the paper; show the gap empirically
   in Phase 4 ablation.  The narrative becomes "SBDP is a tractable and well-specified
   approximation, with the gap quantified" rather than "SBDP is exact."  This is honest.

4. **Higgins-Wiseman R=2 is not enough action richness.**
   *Probability*: low.  *Impact*: weakens the long-horizon disambiguation argument.
   *Mitigation*: try R=4 in Phase 1 calibration, fall back to R=2 if memo blowup.

### Medium-impact risks

5. **Pulse-shape parameters (Layer 6, 160 dims) have small individual gradients,
   leaving most of those dims unconverged after T=200 Adam steps.**
   *Mitigation*: longer T at Phase 3 with warm-start; per-layer LR; ablation will
   reveal whether Layer 6 actually moves.

6. **PCRB baseline at K_total=32 is hard to compute exactly because the schedule space
   is large.**
   *Mitigation*: PCRB schedule scales with `J^K`; at K=32 with J=10 that's 10^32 schedules.
   Use BayesOpt over schedules (existing infrastructure) and report best schedule found.
   This is what the parent project already does.

7. **Deep-RL baseline doesn't converge at K_total=16-32.**
   *Mitigation*: this is *expected* — the hardware-light pathology is real.  Document
   the failure as supporting evidence for SBDP's advantage.

### Low-probability / catastrophic

8. **The receding-horizon batched policy is *worse* than K=4 exact Bellman at φ=0.5.**
   *Probability*: low (each batch can do useful work; n=4 batches give 4× more
   information than n=1).  *Impact*: kills the paper.
   *Mitigation*: Phase 1 calibration tests this directly — if K=8 SBDP is worse than
   K=4 exact, halt and re-think.

## Success criteria

The project succeeds if **all three** are achieved:

| # | Criterion | Threshold |
|---|---|---:|
| 1 | SBDP K=16 ratio at φ=0.5 vs **Higgins-Wiseman** | ≥ 5× |
| 2 | SBDP K=32 ratio at φ=0.5 vs **Higgins-Wiseman** | ≥ 10× |
| 3 | dim(c)=220 SGD converges (per-layer ablation shows Layer 6 moves) | qualitative |
| 4 | (Secondary) SBDP K=16 ratio vs PCRB-extended | ≥ 100× |

If only #1, #3, #4 succeed but #2 saturates earlier, the paper is still strong: the
saturation point itself is publishable as "SBDP saturates at K_total ≈ X for this
problem class against the canonical adaptive baseline."

If only #1 and #4 succeed: paper still publishable but at lower-tier venue (PR Applied).

If even #1 fails: re-run at φ_max=0.3 (intermediate prior, partly aliased) where the
opportunity is smaller but the demonstration is cleaner.  At φ_max=0.3 the SBDP/HW gap
should be larger because the aliasing-disambiguation advantage of Bellman is more
pronounced relative to the easy unimodal-localization regime that HW handles well.

## What this paper would say

**Title (working)**: *"Long-horizon, high-dimensional joint quantum-sensor co-design via
stochastic batched dynamic programming"*

**One-sentence summary**: *"We extend exact-Bellman joint hardware-policy co-design from
K=4 epochs to K=32 via stochastic batched DP, beating the canonical Higgins-Wiseman
adaptive QPE protocol by 5–10× on deployed Bayesian MSE for a 220-dimensional
superconducting-qubit flux sensor design at the wide-prior aliasing-floor regime where
existing exact methods are K-limited."*

**Likely venue**: PRX Quantum (best fit), NMI (high-dim ML methodology angle), or PRX
(broader physics).  The (φ=0.5, K=16-32, dim(c)=220) combination is striking enough for
any of these.

## Decision points

Before committing the full ~2.5 months:

- **Phase 0 (infrastructure)**: 1 week of coding.  No-go if Layer 6 (pulse shapes) turns
  out to require numerical Schrödinger-equation integration after all (rather than
  closed-form DRAG corrections).  In that case, downscope to Layer 1–5 (60 dims) and
  proceed.

- **Phase 1 (calibration)**: 3 days.  No-go if K=8 SBDP doesn't beat K=4 exact at φ=0.5.
  Pivot to a different prior width or re-think the receding-horizon truncation.

- **Phase 2 (main)**: 3 weeks.  Soft go/no-go at week 1: if K=4 Bellman at φ=0.5
  K_phi=64 is taking >10 min/solve, escalate the action-set trim and packed-memo work.

- **Phase 3**: 3 weeks.  Conditional on Phase 2 hitting the ≥50× threshold.  If Phase 2
  falls short of 50×, recalibrate before committing to K=32.

These are the natural off-ramps; total at-risk compute commitment is ~3 days (Phase 1)
before the first decision point.
