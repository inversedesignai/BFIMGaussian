# Stochastic Batched DP

A standalone research project on a new framework for joint hardware-policy co-design at long horizons.  Companion to (but logically independent of) the joint-DP work in the parent BFIMGaussian repository; targeted at its own paper.

## What this project is

Joint hardware-policy co-design via exact Bellman dynamic programming hits a horizon ceiling
of `K_total ≈ 4–5` because the count-tuple memoization grows superlinearly in the horizon.
This project introduces **stochastic batched dynamic programming** (SBDP), a scaling
framework that pushes the tractable joint-DP horizon to arbitrary `K_total` while
preserving exactness of inner Bellman backups within each batch and unbiasedness of
per-sample outer-hardware gradients.

The framework combines four ingredients:

1. **Batched DP**: decompose `K_total = n × K_batch` into `n` consecutive batches of length
   `K_batch`, each a tractable exact Bellman solve from its predecessor's terminal belief.

2. **Stochastic outer optimization**: at each outer-gradient step, sample `M` trajectories
   under the current batched policy; bypass the exhaustive enumeration over reachable
   batch-boundary beliefs.

3. **Per-sample sharp-max envelope-theorem gradients**: each sampled batch's local Bellman
   gradient is exact, no smoothing temperature.

4. **Per-batch advantage variance reduction**: the score-function term in the REINFORCE
   identity is paired per batch with a remaining-horizon-value baseline, reducing variance
   while preserving unbiasedness.

The resulting outer-hardware gradient is unbiased for the batched-policy value and has
wall-clock cost linear in `M` and `n`, not in the reachable-belief count.

## Why this is publishable on its own

To our knowledge the specific combination of (continuous physical hardware as outer
variable) × (exact Bellman dynamic programming as inner solver in each batch) ×
(stochastic Monte-Carlo sampling of batch-boundary trajectories) × (sharp-max
envelope-theorem gradients) × (per-batch advantage baselines) is not present in the
literature.  Pieces exist separately:

- **Stochastic Dual Dynamic Programming** (Pereira-Pinto 1991): scenario-sampled
  forward-backward sweeps for stochastic linear programs.  Closest cousin in spirit, but
  for convex programs with cutting-plane value approximations, not POMDPs with exact
  memoized Bellman.
- **Sequential Bayesian experimental design** (Huan-Marzouk 2013, Long-Marzouk-Wang
  successors): MC sampling of belief trajectories with stochastic-gradient outer updates.
  Inner policy is typically rung-3 myopic info-gain, not exact Bellman.
- **Deep Adaptive Design** (Foster-Ivanova-Rainforth 2021): variational lower bound on
  expected info gain optimized via SGD over an NN policy.  Different inner solver.
- **PBVI / SARSOP / Perseus**: sample-based POMDP solvers.  Sampling at policy-inference
  time, not at outer-hardware-gradient time.
- **Differentiable DP** (Mensch-Blondel 2018, Amos-Kolter 2017, Domke 2012):
  gradient-through-DP for outer parameter learning.  MDPs / structured prediction, not
  POMDPs with sample-based belief enumeration.

## Files

- `README.md` — this file (project overview, scope, plan)
- `StochasticBatchedDP.tex` — formal article: setup, full gradient derivation including
  detailed n>2 case, algorithm, convergence, connections to existing methods
- `StochasticBatchedDP.pdf` — compiled article (15 pages)

Forthcoming: experimental modules, scqubit case-study scripts, baseline implementations,
results.

## Demonstration target

The main demonstration application is the frequency-tunable transmon flux sensor of
Danilin, Nugent, and Weides (arXiv:2211.08344v4).  This is the same physical device as in
the parent BFIMGaussian project's joint-DP case study, chosen here because (a) the
existing exact-Bellman infrastructure provides a clean baseline at `K_total = K_batch =
4`; (b) the Ramsey likelihood and count-tuple sufficient statistic match the SBDP
framework cleanly; (c) the wide-prior multimodal regime (`φ_max → Φ_0/2`) is exactly the
regime where exact DP fails and SBDP's long-horizon disambiguation should shine.

## Scope, in three ambition tiers

Choose one for the standalone paper:

### Tier 1: Long-horizon Bellman-optimal quantum metrology (cleanest)

- Demonstrate SBDP at `K_total ∈ {8, 16, 32}` with `K_batch = 4` on the scqubit problem.
- Compare against PCRB-extended schedule, geometric-Ramsey + Higgins feedback,
  particle-filter myopic information gain, and a small deep-RL policy.
- Headline: SBDP achieves `~N×` MSE reduction over the best classical baseline at
  `K_total = N`.
- Likely home: PRX Quantum, NMI.

### Tier 2: Joint hardware-policy co-design at long horizons with high-dim continuous hardware

- Tier 1 plus: extend the action set with adaptive measurement basis `φ_ref`
  (canonical Higgins-Wiseman feedback dimension, `R = 2`).
- Extend the design vector `c` to ~30 dims via multi-junction SQUID asymmetry (~5 dims),
  multi-mode readout (~10 dims), bias-line filter parameters (~5 dims), and
  full-physics geometric/coupling unlocks (~3 dims).  All with meaningful physics impact
  (see physics-impact analysis in the parent project notes).
- Headline: first long-horizon Bellman-optimal hardware-policy co-design with continuous
  ~30-dim hardware.
- Likely home: PRX, NMI, or JMLR with a methods focus.

### Tier 3: SBDP as a general-purpose POMDP scaling primitive

- Tier 2 plus: demonstrate the framework on at least one additional application
  (e.g., adaptive radar tracking with continuous beam-pattern parameters, or adaptive
  microscopy).
- Add a theoretical convergence analysis (variance scaling, approximation gap to full
  exact `K_total` Bellman, sample complexity bounds).
- Headline: SBDP as a unifying scaling framework for joint hardware-policy co-design across
  the relaxation hierarchy.
- Likely home: NMI, JMLR, or possibly Nature Communications.

## Algorithm

```
Stochastic Batched DP for joint outer-c, inner-π optimization

Input:
  c_0 : initial outer parameter
  K_total : total horizon
  K_batch : per-batch horizon (chosen so exact Bellman is tractable)
  M : sample size per gradient step
  η : Adam learning rate
  T_outer : number of outer iterations

Initialize c ← c_0

for t = 1 ... T_outer:
    g ← 0
    for m = 1 ... M (parallel):
        # Sample one trajectory of length K_total under the batched policy at current c
        b ← prior
        record actions, observations, beliefs along the way
        for batch_id = 1 ... n = K_total / K_batch:
            (V_batch, π_batch, memo_batch) ← solve_bellman(b, K_batch, c)
            for k = 1 ... K_batch:
                a ← π_batch(b)
                y ← sample_observation(b, a, c, x ~ b)
                b ← bayes_update(b, a, y, c)
        # Per-sample value and gradient
        V_m ← terminal_reward(b, c)
        g_m ← (pathwise via reverse-mode AD over K_total belief-update chain)
              + Σ_j (advantage A_j × per-batch score function at c)
        g += g_m
    g /= M
    c ← Adam_step(c, g, η)

return c
```

The advantage `A_j` and pathwise term are derived in detail in `StochasticBatchedDP.tex`,
§5.

## Compute budget (rough estimates)

For the scqubit demonstration on existing hardware (~380 cores):

| experiment | wall-clock |
|---|---|
| SBDP `n=4` `K_total=16`, `M=20`, `K_phi=128`, `T=500`, narrow prior | ~3 days |
| SBDP `n=8` `K_total=32`, `M=20`, `K_phi=128`, `T=500`, narrow prior | ~7 days |
| SBDP `n=16` `K_total=64`, `M=30`, `K_phi=64`, `T=500`, narrow prior | ~10 days |
| Tier 2 with `dim(c) = 30` | ~4× the above (larger AD graph) |
| Wide prior (`φ_max = 0.5`) | infeasible at K_phi=128 without further engineering |

A first 1-day feasibility test at `K_total = 8`, `M = 10` suffices to confirm the
framework works numerically before committing to the larger experiments.

## Baselines for comparison

For the scqubit demonstration:

1. **Exact `K_total = 4` joint-DP** (the existing parent-project headline; serves as the
   baseline at the joint-DP horizon ceiling).
2. **PCRB-extended schedule** at the same `K_total`: `(τ_max, n_max)^K_total` at the
   PCRB-optimal `c`.  The single-level Fisher-information baseline.
3. **Geometric-Ramsey with Higgins-Wiseman feedback** at the same `K_total`: classical
   adaptive QPE baseline.  Requires `φ_ref` in the action space.
4. **Particle-filter myopic information gain** (Granade-Ferrie 2012 style): rung-3 myopic
   adaptive baseline.
5. **Deep-RL policy** (small LSTM, PPO or REINFORCE training): neural-network adaptive
   baseline.  Tests the "hardware-light pathology" claim (cf. paper §5).

Each baseline produces an MSE on the same paired-MC deployment; ratios against SBDP
provide the headline numbers.

## Convergence (sketch; see `.tex` §6)

Standard SGD theory applies.  With unbiased per-sample gradients and bounded variance, the
iterates converge in expectation to a stationary point of `V_batched(c)` at rate
`O(1/√T)`.  The estimator variance scales as `O(σ²/M)` where `σ²` depends on the spread of
batch-boundary belief values; advantage baselines reduce `σ²` substantially.  No smoothing
temperature anywhere; the only approximation is the receding-horizon truncation of the
batched policy relative to the (intractable) full `K_total` Bellman, and even that
vanishes in the `n → 1` limit (where SBDP recovers exact joint-DP at the inner Bellman
horizon ceiling).
