# scqubit_headline

Self-contained reproduction of the K=4, J=10, L=2, narrow-prior (φ_max=0.1)
superconducting-qubit flux-sensor benchmark, comparing **joint-DP**
(Bellman-optimal adaptive measurement policy + co-designed hardware c)
against **PCRB** (joint Bayesian Cramér–Rao baseline: c + fixed schedule
optimised for prior-averaged Fisher information).

## TL;DR — final result

**With both estimators globally optimised by Bayesian optimization at the
deployment grid (K_PHI=256), joint-DP beats PCRB by 170.37× in deployed
Bayesian MSE, at z = +155.0 σ.**

```
joint MSE_1 (BayesOpt K=256) = 4.98e-6 ± 3.66e-7   c=(3.0 GHz, 0.28 GHz, 5.0 MHz, 5.0 GHz)
PCRB  MSE_2 (BayesOpt K=256) = 8.48e-4 ± 5.43e-6   c=(12.0 GHz, 0.4 GHz, 0.1 MHz, 5.0 GHz)
ratio = 170.37  z = +155.0 σ
```

This is **15.1× larger** than the original paper headline of 11.27× (which
initialised both optimizers from a hand-tuned `PAPER_BASELINE` and stopped
short of the global optimum on the joint-DP side). The folder is
self-contained: every `.jl` referenced lives here; nothing in this folder
includes anything outside it.

## Headline progression across the entire investigation

| # | Setup | joint MSE | PCRB MSE | ratio | z |
|---|---|---|---|---|---|
| 1 | Paper headline (PAPER_BASELINE × 2 init, Adam-Zygote, K=128) | 7.43e-5 | 8.43e-4 | 11.27× | +132σ |
| 2 | v3 converged at PAPER_BASELINE init (Adam-FD-threaded, lr decay) | 7.43e-5 | 8.43e-4 | 11.35× | +132σ |
| 3 | Both at naive (mid-box) init | 3.28e-5 | 8.60e-4 | 26.19× | +145σ |
| 4 | Multistart-of-4 (Adam, K=128 train), select by deployment MSE | 3.01e-5 | 8.43e-4 | 28.05× | +148σ |
| 5 | Multistart-of-4 (MMA at K=256) | 8.75e-6 | 8.43e-4 | ~96× | +148σ |
| 6 | **BayesOpt joint vs MMA PCRB at K=256** | **4.98e-6** | 8.43e-4 | 169.9× | +156σ |
| 7 | **BayesOpt joint vs BayesOpt PCRB at K=256 (final)** | **4.98e-6** | **8.48e-4** | **170.37×** | **+155σ** |

Each row is reproducible from a script in this folder (see "Files" below).
The progression reveals two distinct sources of underestimation in the
original 11.27×:

1. **Joint-DP optimization was not globally minimised.** Multiple basins
   exist; PAPER_BASELINE sits in a shallow one (MSE 7.4e-5). The
   global optimum (MSE ≈ 5e-6) is in a *physically opposite* corner of
   the parameter box.
2. **K_PHI=128 training over-estimated joint-DP V_adaptive** in some c
   regions because the coarse belief grid under-resolves multimodal
   posteriors. The "global by training V at K=128" approach picked the
   wrong c (rand_1's training V=-1.18e-5 was a grid artifact;
   deployment at K=256 gave MSE 5.5e-5, no better than the naive init).

Both issues are fixed in row 7 by (a) using **K_PHI=256 throughout**
training and deployment, and (b) using **Bayesian optimization** with
explicit GP-driven exploration instead of multi-start gradient descent.

## Files

### Modules (don't run directly; included by the driver scripts)

| File | What it is |
|---|---|
| `ScqubitModel.jl` | Transmon physics: `omega_q`, `P1_ramsey`, `ScqubitParams`, `PAPER_BASELINE`. |
| `Belief.jl` | Belief-grid types and discretization helpers. |
| `Baselines.jl` | Schedule enumeration + fixed-schedule oracle helpers. |
| `Bellman.jl` | Single-threaded exact-DP Bellman recursion. Reference; used by tests. |
| `BellmanThreaded.jl` | Multi-threaded exact-DP via topological sort. Bit-identical to `Bellman.jl` per state. |
| `Gradient.jl` | Envelope-theorem gradient on the policy tree (Zygote `grad_c_exact` + ForwardDiff `grad_c_exact_fd`). |
| `GradientThreaded.jl` | Threaded ForwardDiff gradient (`Threads.@spawn` per m-branch) — used by Adam scripts. |
| `JointOpt.jl` | Adam state, box projection, `phi_star_fn`/`omega_d_fn`, `CBox`. |
| `PCRB.jl` | PCRB baseline: schedule enumeration + ForwardDiff Adam on c. |

### Driver scripts — paths to each row of the table above

**Row 3 (naive headline, 26.19×):**
- `sweep_joint_narrow_naive.jl` — Adam joint-DP from naive init.
- `sweep_pcrb_narrow_naive.jl` — Adam PCRB from naive init.
- `compare_mse_narrow_naive.jl` — paired MC at the naive winners.

**Row 4 (multistart by deployment MSE, 28.05×):**
- `multistart_joint_adaptive.jl` — Adam joint-DP from 4 inits with REOPT_EVERY=1 + adaptive lr.
- `multistart_pcrb.jl` — Adam PCRB from same 4 inits.
- `compare_mse_multistart_global.jl` — paired MC at training-V-best (gives the misleading 15.47×).
- `compare_mse_multistart_deployment.jl` — paired MC at deployment-MSE-best (gives the honest 28.05×).

**Row 5 (MMA at K=256, ~96×):**
- `multistart_joint_mma.jl` — MMA joint-DP (NLopt :LD_MMA) at K_PHI=256, 4 inits.
- `multistart_pcrb_k256.jl` — Adam PCRB at K_PHI=256, 2 inits.
- `compare_mse_mma_k256_deployment.jl` — paired MC at the K=256 MMA winners.

**Row 7 (BayesOpt vs BayesOpt at K=256, FINAL 170×):**
- `bayesopt_joint_k256.jl` — Bayesian opt joint-DP V_adaptive at K=256 (100 evals, ARD-Matérn GP, EI).
- `bayesopt_pcrb_k256.jl` — Bayesian opt PCRB log_JP at K=256 with schedule fixed to (320 ns, n=10)^4.
- `bayesopt_pcrb_k256_full.jl` — same as above but schedule **re-enumerated at every probe** (rigorous variant; gives bit-identical answer to the fixed-schedule version, confirming the assumption).

### Auxiliary

- `optimize_joint_lbfgs.jl` — L-BFGS-B (Optim.Fminbox(LBFGS())) joint-DP, normalized [0,1]^4 box. Confirmed unsuitable for this problem (per-dim gradient skew of ~300,000:1 traps line search in f_q-only descent). Documents *why* we moved to MMA / BayesOpt.

### Tests

- `tests/test_bellman_threaded.jl` — bit-exact validation of threaded Bellman vs single-threaded.
- `tests/test_gradient_threaded.jl` — bit-exact validation of threaded ForwardDiff gradient vs serial.

## Reproduction

### Prerequisites

- Julia ≥ 1.10 (developed on 1.12.5).
- Project deps: ForwardDiff, Zygote, SpecialFunctions, Optim, NLSolversBase,
  NLopt, BayesianOptimization, GaussianProcesses, Distributions, plus
  standard library. All pinned in the repo's top-level `Project.toml`.

### Recipe for the FINAL 170× headline

```bash
# 1. Joint-DP global optimum via Bayesian optimization at K=256 (~50 min)
julia --project=. -t 64 doc/scqubit_headline/bayesopt_joint_k256.jl

# 2. PCRB global optimum via Bayesian optimization at K=256 (rigorous variant: ~2 h)
julia --project=. -t 32 doc/scqubit_headline/bayesopt_pcrb_k256_full.jl

# 3. Paired Monte Carlo deployment + ratio
julia --project=. -t 64 -e '
include("doc/scqubit_headline/ScqubitModel.jl"); include("doc/scqubit_headline/Belief.jl")
include("doc/scqubit_headline/Bellman.jl"); include("doc/scqubit_headline/BellmanThreaded.jl")
include("doc/scqubit_headline/Gradient.jl"); include("doc/scqubit_headline/JointOpt.jl")
include("doc/scqubit_headline/PCRB.jl")
using .ScqubitModel, .Belief, .Bellman, .BellmanThreaded, .Gradient, .JointOpt, .PCRB
using Printf, Random, Serialization

j = deserialize("doc/scqubit_headline/results/bayesopt_joint_k256/result.jls")
p = deserialize("doc/scqubit_headline/results/bayesopt_pcrb_k256_full/result.jls")
v1 = j.v_best;  c1 = vec_as_c(v1)
v2 = p.v_best;  c2 = vec_as_c(v2);  sched2 = p.sched_best
phi_star_fn = make_phi_star_fn()
ωd1 = omega_q(phi_star_fn(c1)[1], c1); ωd2 = omega_q(phi_star_fn(c2)[1], c2)

J_TAU = 10
TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
grid = make_grid(; K_phi=256, phi_max=0.1, tau_grid=TAU_GRID, n_grid=(1, 10))
(_, memo1, _) = solve_bellman_threaded_full(grid, 4, c1, ωd1; terminal=:mse)

rng = MersenneTwister(2026)
(MSE_1, se_1) = deployed_mse_adaptive(c1, memo1, ωd1, grid, 4; n_mc=20000, rng=rng)
rng = MersenneTwister(2026)
(MSE_2, se_2) = deployed_mse_fixed(c2, sched2, ωd2, grid; n_mc=20000, rng=rng)
ratio = MSE_2 / MSE_1
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("MSE_1 = %.4e ± %.2e\nMSE_2 = %.4e ± %.2e\nratio = %.3f\nz = %+.2f σ\n",
        MSE_1, se_1, MSE_2, se_2, ratio, z)
'
```

Total wall-clock from cold: **~3 hours** on 64 threads at K_PHI=256.
Expected output: ratio = 170.37, z = +155.01 σ.

### Recipe for the simpler "naive headline" (26.19×)

If you don't want to wait 3h for the full BayesOpt run, the older
naive-init Adam pipeline is much faster (~50 min total) and still beats
the paper's 11.27×:

```bash
julia --project=. -t 64 doc/scqubit_headline/sweep_joint_narrow_naive.jl   # ~25 min
julia --project=. -t 64 doc/scqubit_headline/sweep_pcrb_narrow_naive.jl    # ~20 min
julia --project=. -t 64 doc/scqubit_headline/compare_mse_narrow_naive.jl   # ~1 min
```

Expected output: ratio ≈ 26.19, z ≈ +145 σ.

### Optional environment overrides

| Variable | Default | Notes |
|---|---|---|
| `K_PHI` | 256 | Belief-grid resolution for both training and deployment. |
| `N_EVAL` | 100 | BayesOpt total evaluation budget per side. |
| `N_INIT` | 10 | BayesOpt random initial design points. |
| `SEED` | 42 | RNG seed for the BayesOpt initialisation. |
| `MSE_N` | 20000 | Deployment Monte-Carlo sample count. |
| `MSE_K_PHI` | 256 | Deployment belief-grid resolution. |

### Validation tests (~3 min)

```bash
julia --project=. -t 16 doc/scqubit_headline/tests/test_bellman_threaded.jl
julia --project=. -t 8  doc/scqubit_headline/tests/test_gradient_threaded.jl
```

Both should report `max |Δ| = 0.000e+00` on every test case (bit-identical
threaded vs single-threaded).

## Detailed findings

### Why the paper's 11.27× was an underestimate

The paper used Adam at K_PHI=128 from `PAPER_BASELINE` for both joint-DP
and PCRB. Two issues:

1. **Adam from `PAPER_BASELINE` falls into a shallow local basin.** The c
   barely moves (sub-1.5% of box width on three of four free dims; only
   κ moves by ~14% of init value), and the V landscape has *multiple*
   basins. The global optimum sits at f_q=3.0 GHz — 6 GHz away from
   PAPER_BASELINE's 9.0 GHz, in an opposite corner of the box.

2. **K_PHI=128 V_adaptive is biased upward** for some c regions
   because the coarse belief grid under-resolves multimodal posteriors.
   At rand_1's c, V_adaptive(K=128) = -1.18e-5 looks promising but
   V_adaptive(K=256) = -5.07e-5 — three times worse. Picking by training
   V at K=128 leads astray.

Both are fixed by (a) globally exploring the c-box (multistart helps,
BayesOpt is even better), and (b) training and deploying at the same
K_PHI=256.

### Striking c-asymmetry between joint-DP and PCRB optima

```
joint-DP optimum (BayesOpt):  f_q=3.0 GHz   E_C=0.28   κ=5.0 MHz   Δ=5.0 GHz
PCRB     optimum (BayesOpt):  f_q=12.0 GHz  E_C=0.40   κ=0.1 MHz   Δ=5.0 GHz
```

The two estimators land at *physically opposite* corners on the f_q and κ
axes:

- PCRB picks **min-κ, max-f_q** — high single-shot Fisher information,
  consistent with its Cramér–Rao foundation.
- Joint-DP picks **max-κ (the box ceiling!), min-f_q** —
  counterintuitive from a Fisher perspective, but the adaptive policy
  exploits high decoherence for fast posterior collapse on short-τ
  disambiguation epochs, then refines on later epochs. Fast collapse
  beats per-shot SNR for narrow-prior scqubit estimation.

This asymmetry is the cleanest observable consequence of the joint-DP /
PCRB methodology gap. The paper claims joint-DP and PCRB optimise
*different physical behaviour*; the BayesOpt result makes this concrete.

### PCRB is robust to init and to optimizer; joint-DP is not

Across all our PCRB runs (Adam from {paper, naive, rand_1, rand_2} init,
MMA at K=256, BayesOpt at K=256), the deployed PCRB MSE stays in the
range **8.43e-4 to 8.60e-4 — varying by less than 2%**. Even when PCRB
finds 1.43× higher J_P (BayesOpt vs MMA: log_JP 21.06 vs 20.70), the
deployed MSE doesn't move. This re-confirms the paper's claim: PCRB at
narrow prior is **aliasing-dominated**, not Fisher-info-dominated; the
optimizer choice barely matters.

Joint-DP, in contrast, has multiple basins ranging from MSE 7.4e-5
(paper local min) to 5.0e-6 (BayesOpt global). The 15× spread on the
joint-DP side is what drives the headline ratio; choosing a better
optimizer matters a lot.

### All 100 PCRB BayesOpt probes chose schedule (10, 2)^4

The rigorous PCRB BayesOpt (`bayesopt_pcrb_k256_full.jl`) re-enumerates
the optimal schedule at every probe c. Across all 100 evaluations spread
across the 4D box, the chosen schedule was always
`[(10, 2), (10, 2), (10, 2), (10, 2)]` — `τ=320 ns, n=10` repeated
4 epochs. **The fixed-schedule shortcut in `bayesopt_pcrb_k256.jl` is
empirically valid, not just a corner-cut**, and the two PCRB BayesOpt
variants found bit-identical optima.

## Caveats

1. **The "global optimum" claim is multistart + BayesOpt-best, not a
   guarantee.** With 100 BayesOpt evaluations on a 4D box and an
   ARD-Matérn GP, finding a basin we missed is unlikely but not
   impossible. The expected-improvement acquisition naturally probes
   uncertain regions; the GP confidence at convergence is tight in
   the regions that matter. A user paranoid about even-deeper basins
   could rerun with `SEED=43, 44, ...` and check.

2. **K_PHI=256 is a *deployment* truth, not an *infinite-grid* truth.**
   Going to K_PHI=512 might shift the joint-DP MSE slightly. We chose
   K_PHI=256 to balance accuracy and compute (4× faster than K=512 per
   Bellman re-solve). The relative joint-DP / PCRB ratio should be
   stable across K_PHI ≥ 256.

3. **Pinned components.** Three of seven `ScqubitParams` fields
   (temperature, A_phi, A_Ic) are held fixed at PAPER_BASELINE values
   in all runs. Optimising over these would likely shift both estimators
   together; the joint-DP / PCRB ratio is unlikely to change much.

4. **No regularization on the c box edges.** The BayesOpt joint-DP
   optimum lies at three of four box boundaries (f_q=3.0 lower, κ=5.0
   upper, Δ=5.0 upper). Widening the box (e.g., allowing κ > 5 MHz)
   could push the optimum further. The current box is the
   `realistic_box` definition used throughout — physically motivated
   but not a hard physical limit.

## What's tracked vs ignored

The driver scripts and modules are committed to `doc/scqubit_headline/`.
The `results/*.jls` artefacts are *not* tracked (large binary outputs,
fully reproducible from the scripts in ~3 hours wall-clock).

Recent commits relevant to this folder (most recent first):

- `6ad69a8` — rigorous PCRB BayesOpt with per-eval schedule enumeration.
- `7aad4fc` — K=256 + MMA + BayesOpt → 170× final headline.
- `cbbf22c` — deployment-criterion compare; reproduces 28×.
- `f4f15e9` — multistart joint-DP and PCRB; 28× by deployment MSE.
- `021075f` — skip Zygote crosscheck at K=4 in gradient test (Julia 1.12 Zygote is too slow).
- `d73602d` — naive-init reproduction (26×).
- `94324f7`, `3a1633e`, `ac9aa49` — earlier sweep scripts and threaded modules in `doc/adaptive/scqubit/`.
