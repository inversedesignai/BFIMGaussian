# scqubit_widephi — 5-D BayesOpt at the wide-prior limit (φ_max = 0.5)

Self-contained reproduction of the `scqubit_5d` pipeline at the **uninformative** prior limit
`PHI_MAX = 0.5` (vs the paper's headline `PHI_MAX = 0.1`).  Same 5-D BayesOpt setup with the
drive frequency `ω_d` as a free parameter (no envelope-theorem bias).

This experiment validates the prediction in §7.6 of the paper:

> "At φ_max → Φ_0/2 (uninformative prior) the support spans many fringes and four epochs are
> too few for either a fixed or an adaptive schedule to resolve them; both joint-DP and
> joint-PCRB converge to comparable aliasing-floor MSE and the adaptive margin shrinks."

## Layout

Modules byte-identical to `doc/scqubit_5d/` (do not edit):
- `ScqubitModel.jl`, `Belief.jl`, `Baselines.jl`
- `Bellman.jl`, `BellmanThreaded.jl`
- `Gradient.jl`, `GradientThreaded.jl`
- `JointOpt.jl`, `PCRB.jl`

Experiment scripts (single-PHI_MAX-difference variants):
- `bayesopt_joint_widephi.jl` — 5-D BayesOpt of joint-DP `V_adaptive` at `PHI_MAX=0.5`
- `bayesopt_pcrb_widephi.jl`  — 5-D BayesOpt of PCRB `log J_P` at `PHI_MAX=0.5`
- `compare_mse_widephi.jl`    — paired MC deployment at `K_PHI=256` plus exact `-V_adaptive`

Tests (mirrored from `scqubit_5d/`): `tests/test_bellman_threaded.jl`, `tests/test_gradient_threaded.jl`.

## How to run

```bash
cd doc/scqubit_widephi

# ~50 min on 64 threads
julia --threads 64 bayesopt_joint_widephi.jl 2>&1 | tee logs/bayesopt_joint_widephi.log

# ~1 min, can be run in parallel with the joint script
julia --threads 1  bayesopt_pcrb_widephi.jl  2>&1 | tee logs/bayesopt_pcrb_widephi.log

# few minutes; reads the two result.jls above
julia --threads 64 compare_mse_widephi.jl    2>&1 | tee logs/compare_mse_widephi.log
```

Environment overrides (defaults shown): `K_PHI=256`, `N_EVAL=100`, `N_INIT=10`, `SEED=42`,
`MSE_N=20000`, `MSE_K_PHI=256`.

## Differences from `scqubit_5d`

Only **one constant** changes:

```julia
const PHI_MAX = 0.5    # was 0.1 in scqubit_5d
```

The `ω_d` search box `[2π·1, 2π·12]` GHz is unchanged; this still spans the qubit frequency
range across the wider prior (since `f_q(0.49) ≈ 1.5–2 GHz` for typical hardware values).

## Caveats

1. **Cooper-pair-box singularity at φ = 0.5.**  The model has `phi_clip = 0.49` in
   `ScqubitParams`, so grid points in `[0.49, 0.5)` all behave as if at φ = 0.49.  About 2%
   of the prior mass cannot be discriminated; this introduces a small floor in the variance
   reachable by either method.  Effect is small and identical for both approaches, so it
   does not bias the joint-DP / PCRB ratio.

2. **Multimodal posterior.**  At `PHI_MAX = 0.5` the Ramsey likelihood spans many fringes
   inside the prior support.  K = 4 epochs are typically insufficient to disambiguate.
   The PCRB framework, which assumes a unimodal Gaussian posterior, is mis-specified here —
   it is **not** a valid bound for the Bayesian MSE in this regime.  The comparison should
   therefore be interpreted as "two co-design objectives with different validity envelopes",
   not as "joint-DP vs the gold-standard CRB lower bound".

3. **Schedule re-verification.**  As at `PHI_MAX = 0.1`, the `(320 ns, n=10)^4` schedule is
   the assumed PCRB optimum and is re-verified at the BayesOpt winner via
   `argmax_schedule_enumerate`.

## Expected outcome (paper prediction)

Per §7.6:
- The joint-DP / PCRB MSE ratio should be **smaller** than the 11.3× headline at `PHI_MAX=0.1`
  (or the 190× in the 5-D `scqubit_5d` analysis).
- Both methods should produce MSE near the aliasing floor `~ φ_max² / N_modes`.
- Whether the ratio falls below ~2× (the threshold the paper claims for "margin shrinks") is
  the empirical question this folder answers.
