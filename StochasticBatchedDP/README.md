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
- Extend the design vector `c` to **220 dims** via the parametric layers listed below.
  All physics-meaningful, all fab-relevant, all with analytical gradients.
- Headline: first long-horizon Bellman-optimal hardware-policy co-design with continuous
  ~200-dim hardware optimized via stochastic gradient descent.
- Likely home: PRX, NMI, or JMLR with a methods focus.

#### Parametric `c` layers (target dim = 220)

| Layer | Description | Dim | Running |
|---|---|---:|---:|
| 1 | Core transmon (Danilin model, all 7 unpinned) | 7 | 7 |
| 2 | Asymmetric multi-junction SQUID | 4 | 11 |
| 3 | Junction array (fluxonium-like dispersion) | 10 | 21 |
| 4 | Multi-mode resonator readout (5 modes) | 15 | 36 |
| 5 | Spatial bias-coil network (10 coils) | 20 | 56 |
| 6 | Pulse-library shape parametrization | 160 | 216 |
| 7 | RF / signal-chain parameters | 4 | 220 |

#### Layer 1: Core transmon (Danilin model, all freed)

| Index | Parameter | Symbol | Role |
|---:|---|---|---|
| 1 | qubit transition frequency at Φ=0 | `f_q_max` | sets Δω in the Ramsey cosine |
| 2 | charging energy | `E_C/h` | sets anharmonicity, enters `ω_q(φ)` |
| 3 | resonator decay rate | `κ` | sets `Γ_1^cav` |
| 4 | qubit-resonator detuning | `Δ_qr` | sets dispersive coupling |
| 5 | mixing-chamber temperature | `T` | sets thermal factor `tanh(½ hf_q/kT)` |
| 6 | flux-noise amplitude | `A_Φ` | sets `B²` quadratic dephasing |
| 7 | critical-current-noise amplitude | `A_Ic` | sets `B²` quadratic dephasing |

#### Layer 2: Asymmetric multi-junction SQUID

| Index | Parameter | Symbol | Role |
|---:|---|---|---|
| 8 | junction asymmetry | `α = (I_{c,1}-I_{c,2})/(I_{c,1}+I_{c,2})` | richer `ω_q(φ)` shape; non-zero `α` shifts the sweet spot off Φ=0 |
| 9 | asymmetry phase offset | `φ_α` | additional flux degree of freedom for asymmetric loop |
| 10 | loop mutual inductance | `M` | sets `Γ_1^ind` |
| 11 | parasitic mutual inductance | `M'` | sets parasitic `Γ_1^ind` contribution |

#### Layer 3: Junction array (fluxonium-like dispersion)

A series array of `N_arr = 10` junctions in addition to the SQUID, each with its own Josephson energy. The array's net inductance and dispersion shape `ω_q(φ)` in ways that simple two-junction SQUIDs cannot. Each `E_{J,k}` is a fab-controllable junction area.

| Index | Parameter | Role |
|---:|---|---|
| 12 | `E_J,1` array junction 1 | individual junction Josephson energy |
| 13 | `E_J,2` | individual |
| ... | ... | ... |
| 21 | `E_J,10` array junction 10 | individual |

#### Layer 4: Multi-mode resonator readout

Five resonator modes coupled to the qubit via dispersive interactions; each mode contributes additively to `Γ_1^cav`. Multi-mode readout enables Purcell filtering and parallel readout.

| Index | Parameter | Symbol | Role |
|---:|---|---|---|
| 22-24 | mode 1: `ω_{r,1}, κ_1, g_1` | resonator freq, decay, coupling | sets mode-1 contribution to `Γ_1^cav` |
| 25-27 | mode 2: `ω_{r,2}, κ_2, g_2` | as above | mode 2 |
| 28-30 | mode 3: `ω_{r,3}, κ_3, g_3` | as above | mode 3 |
| 31-33 | mode 4: `ω_{r,4}, κ_4, g_4` | as above | mode 4 |
| 34-36 | mode 5: `ω_{r,5}, κ_5, g_5` | as above | mode 5 |

#### Layer 5: Spatial bias-coil network

Ten current loops at different positions that together produce the bias flux. Each loop contributes its own mutual inductance to the SQUID and has its own low-pass filter cutoff (different cable lengths, different filter stages).

| Index | Parameter | Symbol | Role |
|---:|---|---|---|
| 37-38 | coil 1: `M_1, f_{c,1}` | mutual inductance, filter cutoff | spatial-noise contribution from coil 1 |
| 39-40 | coil 2: `M_2, f_{c,2}` | as above | coil 2 |
| ... | ... | ... | ... |
| 55-56 | coil 10: `M_{10}, f_{c,10}` | as above | coil 10 |

The filter cutoffs `f_{c,k}` shape the 1/f flux-noise PSD seen by each contribution; spatial superposition of multiple coils with engineered filter profiles can sculpt the effective dephasing spectrum.

#### Layer 6: Pulse-library shape parametrization (160 dims)

The action `s_k = (j, ℓ, r)` selects one of `J × R = 10 × 2 = 20` pulse types from a calibrated library: delay index `j`, repetition `ℓ`, phase basis `r`. Repetitions `ℓ` share the same pulse shape; only `(j, r)` indexes the pulse library.

Each of the 20 library pulses is parametrized by **8 calibration parameters**, giving 160 total dims. These parameters live in `c` (set once at calibration time, used by all subsequent measurements); they are not part of the per-step action.

For pulse `(j, r)`, j ∈ {1..10}, r ∈ {0, π/2}:

| Param | Symbol | Physical role |
|---|---|---|
| 1 | `A_{j,r}` | amplitude scaling (vs nominal `π/2` rotation) — sets effective rotation angle |
| 2 | `σ_{j,r}` | Gaussian envelope width — sets pulse-shape selectivity in frequency |
| 3 | `β_{j,r}` | DRAG coefficient — first-order anharmonicity leakage suppression |
| 4 | `β2_{j,r}` | DRAG² coefficient — second-order anharmonicity correction |
| 5 | `Δω_{j,r}^\mathrm{pulse}` | linear frequency chirp — corrects for AC Stark, drift |
| 6 | `φ0_{j,r}` | static phase offset (calibration, distinct from action `φ_ref`) |
| 7 | `γ_{j,r}` | envelope skew (asymmetric Gaussian) — corrects rise/fall mismatch |
| 8 | `tc_{j,r}` | pulse center offset (timing calibration) |

Total Layer-6 dimension: `8 × 20 = 160`.

These 8 parameters per pulse modify the Ramsey signal at delay τ in analytically-tractable ways (via DRAG theory, AC-Stark corrections, and effective-rotation-angle expressions) — full numerical Schrödinger-equation pulse simulation is a future extension. The reduced-form pulse-shape model gives closed-form gradients of the Ramsey likelihood `p(y | x, τ; c)` with respect to all 8 shape parameters per pulse.

#### Layer 7: RF / signal-chain parameters

| Index | Parameter | Symbol | Role |
|---:|---|---|---|
| 217 | LO frequency offset | `Δf_LO` | sets effective drive frequency offset |
| 218 | IF mixer carrier offset | `Δf_IF` | mixer up-conversion offset |
| 219 | IQ amplitude imbalance | `δA_{IQ}` | I/Q amplitude mismatch on the drive line |
| 220 | IQ phase imbalance | `δφ_{IQ}` | I/Q phase mismatch on the drive line |

#### Action set (unchanged in dimension, but `φ_ref` adds R=2)

| Component | Symbol | Cardinality |
|---|---|---:|
| delay index | `j ∈ {1..10}` | 10 |
| repetition index | `ℓ ∈ {1..2}` | 2 |
| phase basis | `r ∈ {0, π/2}` | 2 |
| **total action set** | `(j, ℓ, r)` | **40** |

#### Why each layer is fab-relevant

- Layers 1–4 are standard tunable knobs at design time (junction E_J via area, resonator
  frequency via length, etc.).
- Layer 5 corresponds to the actual bias-line topology: filter cascades, multiple flux
  drive lines, on-chip vs off-chip bias coils.
- Layer 6 corresponds to AWG calibration parameters that experimentalists routinely tune
  via pulse-tomography campaigns (Rabi-2, Wittmann calibration, DRAG sweeps).  All 8
  per-pulse parameters are independently calibrated in modern superconducting-qubit
  experiments.
- Layer 7 corresponds to RF chain calibration done at instrument-level (LO mixer
  calibration, IQ balance).

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
