# scqubit_5d — joint-DP vs PCRB with `ω_d` as a free 5th parameter

Self-contained reproduction of the K=4, J=10, L=2, narrow-prior (φ_max=0.1)
superconducting-qubit benchmark at K_PHI=256, with the **drive frequency
`ω_d` promoted to a free optimization parameter**. The 4D version (in
`doc/scqubit_headline/`) tied `ω_d` to the single-shot Fisher-optimal flux
bias via `ω_d_fn(c) = ω_q(φ*(c), c)`. Here we treat `ω_d` as a separate
hardware-design parameter that the experimenter can tune at runtime.

## TL;DR — 5D headline

```
joint MSE_1 (5D BayesOpt) = 4.17e-6 ± 2.36e-7
PCRB  MSE_2 (5D BayesOpt) = 8.45e-4 ± 5.35e-6
ratio = 202.55     z = +156.94 σ
```

Both estimators chose **ω_d/(2π) = 1.0 GHz** (the lower bound of the
realistic transmon-drive range). That's substantially below the
single-shot heuristic value `ω_d_fn(c)` (which would have been 2.06 GHz
for joint-DP and 4.05 GHz for PCRB at their respective optima). Off-resonant
drive — large detuning `|Δ| = ω_q − ω_d` ≈ 5–10 GHz at the operating point —
turns out to be the right choice for both: joint-DP exploits the fast
fringes for adaptive disambiguation, PCRB exploits the high per-shot
Fisher info from the steep fringe slope.

## Why 5D?

The 4D framework forced `ω_d = ω_d_fn(c)`. That assumes the experimenter
parks the drive resonant with the qubit at the most flux-sensitive bias
point — a textbook single-shot Fisher recipe, but not necessarily optimal
for an adaptive-policy estimator with multimodal posteriors.

Promoting `ω_d` to a free parameter does three things:
1. **Eliminates a methodological inconsistency.** The 4D code recomputed
   `ω_d_fn(c)` between iterations but treated `ω_d` as constant during
   gradient evaluation — a hybrid that's neither cleanly "fixed `ω_d`" nor
   cleanly "chain-ruled `ω_d(c)`". With `ω_d` free, the gradient is
   unambiguously the full `dV/d(c, ω_d)`.
2. **Gives the optimizer freedom to engineer detuning.** Both joint-DP and
   PCRB end up at the lower `ω_d` bound — neither prefers the
   single-shot-resonant value.
3. **Improves the headline by ~19%** (4D: 169.9× → 5D: 202.55×).

## Box

| Parameter | Range | Units | Notes |
|---|---|---|---|
| f_q_max | [3, 12] | GHz | Maximum qubit transition frequency |
| E_C/h | [0.15, 0.4] | GHz | Charging energy in frequency units |
| κ | [0.1, 5] | MHz | Resonator decay rate |
| Δ_qr | [0.8, 5] | GHz | Qubit-resonator detuning |
| **ω_d/(2π)** | **[1, 12]** | **GHz** | **Drive frequency — FREE** |

Pinned at PAPER_BASELINE values: `temperature=40 mK`, `A_phi=1e-6`, `A_Ic=1e-6`.

The `ω_d` range is chosen to match the qubit-frequency box (`f_q_max ∈ [3,
12] GHz`), giving the optimizer freedom to put `ω_d` anywhere a typical
microwave drive can reach.

## Files

### Modules (don't run; included by the driver scripts)

| File | What it is |
|---|---|
| `ScqubitModel.jl` | Transmon physics: `omega_q`, `P1_ramsey`, `ScqubitParams`, `PAPER_BASELINE`. |
| `Belief.jl` | Belief-grid types and discretization. |
| `Baselines.jl` | Schedule enumeration helpers. |
| `Bellman.jl` | Single-threaded exact-DP Bellman recursion. |
| `BellmanThreaded.jl` | Multi-threaded exact-DP via topological sort (bit-identical). |
| `Gradient.jl` | Envelope-theorem gradient on the policy tree. |
| `GradientThreaded.jl` | Threaded ForwardDiff gradient (`Threads.@spawn` per m-branch). |
| `JointOpt.jl` | Adam state, box projection, `phi_star_fn`/`omega_d_fn` (used as a *reference* for comparison). |
| `PCRB.jl` | PCRB baseline: Fisher info + schedule enumeration. |

These are byte-identical copies of the modules in `doc/scqubit_headline/`
(verified by smoke test). The folder is self-contained — every `include`
resolves inside it.

### Driver scripts (run these in order)

| Script | What it does | Wall-clock |
|---|---|---|
| `bayesopt_joint_5d.jl` | 5D BayesOpt of joint-DP `V_adaptive(c, ω_d)` at K_PHI=256, ARD-Matérn GP + EI, 100 evals (12 random init + 88 EI). | ~50 min @ 64 threads |
| `bayesopt_pcrb_5d.jl` | 5D BayesOpt of PCRB `log_J_P(c, ω_d)` at K_PHI=256, schedule fixed to (10,2)^4. Includes post-hoc schedule re-verification. | ~1 min @ 8 threads |
| `compare_mse_5d.jl` | Paired MC deployment of both 5D BayesOpt winners at K_PHI=256, n_mc=20000. | ~1 min |
| `mma_joint_5d.jl` | Optional 5D MMA refinement starting from the BayesOpt winner — confirms the basin and (potentially) polishes `V`. | ~10-25 min |

### Tests (optional, ~3 min total)

| File | What it validates |
|---|---|
| `tests/test_bellman_threaded.jl` | `BellmanThreaded.jl` matches single-threaded `Bellman.jl` bit-exactly across multiple test sizes. |
| `tests/test_gradient_threaded.jl` | `GradientThreaded.jl` matches single-threaded ForwardDiff bit-exactly. |

These are byte-identical to the tests in `doc/scqubit_headline/tests/`.

## Reproduction

### Prerequisites

Julia ≥ 1.10 (developed on 1.12.5). Project deps (all pinned in the
repo's top-level `Project.toml`):
- ForwardDiff, Zygote, SpecialFunctions, Optim, NLSolversBase
- NLopt, BayesianOptimization, GaussianProcesses, Distributions

### Run order for the FINAL 202.55× headline

```bash
# 1. Joint-DP 5D global optimum via Bayesian optimization (~50 min)
julia --project=. -t 64 doc/scqubit_5d/bayesopt_joint_5d.jl

# 2. PCRB 5D global optimum via Bayesian optimization (~1 min, no Bellman)
julia --project=. -t 8 doc/scqubit_5d/bayesopt_pcrb_5d.jl

# 3. Paired MC deployment (~1 min)
julia --project=. -t 64 doc/scqubit_5d/compare_mse_5d.jl
```

Total wall-clock: ~52 min.

### Optional: MMA refinement of the joint-DP winner

```bash
INIT_ID=bayesopt julia --project=. -t 64 doc/scqubit_5d/mma_joint_5d.jl
```

This loads the BayesOpt 5D joint-DP winner from `results/bayesopt_joint_5d/`,
initializes 5D MMA (NLopt `:LD_MMA`) at that point, and runs up to 60 MMA
evaluations. If MMA's V_best matches BayesOpt's V_best to within ~1%, the
basin is a true local minimum. Otherwise MMA finds a deeper polished
optimum or reveals that BayesOpt's winner was an artifact.

The script computes `dV/dc` via threaded ForwardDiff (the same gradient
used in the 4D scripts) and `dV/dω_d` via central finite differences (two
extra Bellman re-solves per gradient call, since the codebase doesn't
analytically differentiate through `ω_d`). Total: 3 Bellman re-solves per
MMA gradient evaluation.

### Optional environment overrides

| Variable | Default | Notes |
|---|---|---|
| `K_PHI` | 256 | Belief-grid resolution. |
| `N_EVAL` | 100 | BayesOpt total budget per side. |
| `N_INIT` | 12 | BayesOpt random initial design. |
| `SEED` | 42 | RNG seed. |
| `MAX_EVALS` | 60 | MMA evaluation budget. |
| `XTOL_REL` | 1e-4 | MMA x-tolerance. |
| `FTOL_REL` | 1e-6 | MMA f-tolerance. |
| `MSE_N` | 20000 | Deployment MC sample count. |
| `MSE_K_PHI` | 256 | Deployment grid resolution. |

## Detailed findings

### Both estimators prefer ω_d at the lower bound

| | c_best | ω_d/(2π) | ω_d_fn(c)/(2π) | Δ (BOpt − heuristic) |
|---|---|---|---|---|
| joint-DP | (3.0, 0.15, 0.484 MHz, 4.024) | **1.000 GHz** | 2.057 GHz | −1.057 GHz |
| PCRB | (12.0, 0.4, 0.1 MHz, 5.0) | **1.000 GHz** | 4.053 GHz | −3.053 GHz |

Both deviate from the single-shot heuristic by 1-3 GHz, both in the same
direction (lower ω_d). The fact that PCRB also prefers off-resonant drive
suggests the 1-GHz lower bound may be hit because of box edges rather than
because of an interior minimum — extending the box to ω_d/(2π) > 0.5 GHz
might shift this slightly but probably wouldn't change the ratio
materially.

### c-asymmetry mirrors the 4D result

| | f_q_max | E_C/h | κ | Δ_qr | ω_d/(2π) |
|---|---|---|---|---|---|
| joint-DP | 3.0 GHz (lower) | 0.15 GHz (lower) | 0.48 MHz (interior) | 4.0 GHz (interior) | 1.0 GHz (lower) |
| PCRB | 12.0 GHz (upper) | 0.4 GHz (upper) | 0.1 MHz (lower) | 5.0 GHz (upper) | 1.0 GHz (lower) |

Joint-DP and PCRB land at *physically opposite corners* on f_q_max, E_C,
and Δ_qr, just like the 4D version. The κ choice is qualitatively
different in the 5D version — joint-DP picks an *interior* κ rather than
the upper-bound 5 MHz from 4D — but the fundamental "joint-DP wants
opposite hardware to PCRB" message stays the same.

### Headline progression — full arc across both folders

| # | Setup | joint MSE | PCRB MSE | ratio | folder |
|---|---|---|---|---|---|
| 1 | Paper headline (PAPER_BASELINE × 2) | 7.43e-5 | 8.43e-4 | 11.27× | `scqubit_headline/` |
| 2 | Naive multistart-of-2 | 3.28e-5 | 8.60e-4 | 26.19× | `scqubit_headline/` |
| 3 | Multistart-of-4 by deployment MSE | 3.01e-5 | 8.43e-4 | 28.05× | `scqubit_headline/` |
| 4 | MMA at K=256 multistart-of-4 | 8.75e-6 | 8.43e-4 | 96.42× | `scqubit_headline/` |
| 5 | BayesOpt 4D (ω_d via heuristic) | 4.98e-6 | 8.43e-4 | 169.9× | `scqubit_headline/` |
| 6 | **BayesOpt 5D (ω_d FREE)** | **4.17e-6** | **8.45e-4** | **202.55×** | **`scqubit_5d/`** |

PCRB MSE varies less than 2% across all six rows — entirely robust to
optimizer choice and ω_d treatment. Joint-DP MSE varies by 18× (7.4e-5 →
4.2e-6). The methodological story is unambiguous: **the 11.27× paper
headline understates joint-DP's true advantage by an order of magnitude;
proper global optimization gives 202×.**

## Caveats

1. **Both estimators hit the ω_d=1 GHz lower bound.** This is a sign the
   box might be too restrictive on ω_d. Re-running with ω_d/(2π) ∈ [0.3,
   12] GHz could shift the optimum further.

2. **BayesOpt at d=5 with 100 evals** — sample efficiency is OK but not
   guaranteed to find the true global. With another seed (`SEED=43, 44,
   ...`), BayesOpt might find slightly different optima. The MMA
   refinement script provides an additional check on whether BayesOpt's
   winner is a local minimum.

3. **K_PHI=256 is a deployment-grade discretization, not infinite-grid
   truth.** Going to K_PHI=512 might shift joint-DP MSE by a few percent.
   The relative joint-DP / PCRB ratio should be stable.

4. **Three pinned components** (T, A_φ, A_Ic) are still held fixed at
   PAPER_BASELINE values. Optimizing over these would likely benefit both
   estimators; the ratio probably stays similar.

5. **BayesOpt provides no gradient info.** For a paper-grade global
   optimum claim, the right pipeline is BayesOpt for global discovery →
   MMA polish at the basin (which is what `mma_joint_5d.jl` provides).
   The MMA polish script's output should be reported alongside the
   BayesOpt result for full rigor.

## What's tracked vs not

The driver scripts and modules are committed to `doc/scqubit_5d/`. The
`results/*.jls` artifacts are *not* tracked (large binary outputs, fully
reproducible from the scripts in ~1 hour wall-clock).
