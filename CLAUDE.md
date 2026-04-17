# BFIMGaussian

Bilevel sensor-design optimization for photonic structures. Optimizes the physical geometry (permittivity distribution) of a 2D cross-waveguide photonic device to minimize state estimation error in a Bayesian filtering framework.

## Architecture

**Bilevel optimization:**
- **Outer**: optimize permittivity `ε_geom` (design region pixels) to minimize `E[‖μ_N − x₀‖²]` via Adam or MMA
- **Inner**: at each EKF step, optimize sensor parameters `s = (φ₁, φ₂)` to maximize BFIM trace (or minimize posterior covariance trace)
- **Gradients** through inner argmin via Implicit Function Theorem (IFT) custom `rrule`

**End-to-end pipeline** (`PhEnd2End.jl:end2end`):
1. FDFD forward: `ε_geom → S-matrices` (broadband, 20 frequencies)
2. Episode gradients: `S-matrices → c → episode_loss` (parallelized via `pmap`)
3. FDFD backward: Zygote pullback through S-matrix computation

## Modules

### `SimGeomBroadBand.jl` (~1140 lines)
2D Finite-Difference Frequency-Domain (FDFD) photonic simulator.

Key functions:
- `setup_4port_sweep` — build cross-waveguide geometry, PML, ports/monitors for frequency sweep
- `calibrate_straight_waveguide` — reference amplitude calibration for S-matrix normalization
- `batch_solve` — parallel multi-frequency FDFD solve with custom rrule (Wirtinger calculus)
- `getSmatrices` — extract normalized 4x4 S-matrices and dS/dn, d2S/dn2 from FDFD solutions
- `powers_only` — lattice port powers from S-matrix Taylor expansion
- `jac_only` — analytical Jacobian d(powers)/d(vec(Dn))
- `jac_and_dirderiv_s` — Jacobian + directional derivative along lambda in s-space (avoids nested AD in IFT)
- `_lattice_freq_core` — shared S-matrix assembly + interconnection solve

Physics model: each lattice block (n_lat x n_lat) has a 4x4 S-matrix Taylor-expanded to 2nd order in Dn:
```
S(Dn) ~ S0 + dS/dn * Dn + d2S/dn2 * Dn^2
```
Blocks are interconnected via port connections; the full system is solved as `(I - S_cc P)^{-1}`.

### `BFIMGaussian.jl` (~540 lines)
BFIM-based sensor selection + EKF + IFT rrule.

Key functions:
- `get_sopt(c, mu, model)` — optimize sensor params (grid init then L-BFGS)
- `bfim_trace/grad_s/hessian_s` — BFIM objective and derivatives via ForwardDiff
- `ekf_update` — one EKF measurement update (Joseph form) with custom rrule
- `episode_loss` — N-step EKF episode, returns relative squared error sum(((mu_i - x0_i)/x0_i)^2)
- Custom rrules for `_get_sopt` (IFT) and `ekf_update` (manual matrix adjoints)

### `PosteriorCovGaussian.jl` (~339 lines)
Drop-in replacement using A-optimal criterion: minimizes `tr(Sigma_new)` instead of maximizing `tr(BFIM)`.
- `get_sopt(c, mu, Sigma, model)` — takes Sigma as additional argument
- IFT pullback also returns Sigma_bar cotangent
- No `ekf_update` custom rrule (uses Zygote's built-in)

### `PhEnd2End.jl` (~1216 lines)
Training script. Controls:
- `BFIM_MODE` env var: `adam` (default), `mma`, `test`, `none`
- `BFIM_TEST` env var: comma-separated test names or `all`
- `BFIM_FXS` env var: `0` to disable analytical fxs path

## Current Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_lat` | 2 | 2x2 lattice, dx=4, dy=8, ds=2 |
| `res` | 50 | Grid: 500x500 |
| `design region` | 300x300 | 90,000 parameters |
| `nw` | 20 | Broadband frequencies w in [5.5, 7.5] |
| `sigma^2` | 1e-10 | Measurement noise variance |
| `Dn range` | [1e-5, 1e-4] | Refractive index perturbation (positive only) |
| `N_steps` | 3 | EKF steps per episode |
| `n_episodes` | 20 | Episodes per gradient estimate |
| `alpha_r` | 0.0 | No regularization on sensor params |
| `lr` | 1e-3 | Adam learning rate |
| `resample_every` | 1 | Fresh x0/noise each iteration |
| Loss | relative squared error | sum(((mu_i - x0_i)/x0_i)^2) |

## Running

```bash
# Standard Adam optimization (20 workers)
julia -p 20 PhEnd2End.jl

# With nohup for long runs
nohup julia -p 20 PhEnd2End.jl &

# Run gradient tests
BFIM_TEST=all julia PhEnd2End.jl
BFIM_TEST=ekf,sopt_grad julia PhEnd2End.jl

# Test-only mode (no optimization)
BFIM_MODE=test BFIM_TEST=all julia PhEnd2End.jl
```

Workers are added via `addprocs(20)` inside the script (line 11). For large-scale runs (opt_result1), 300 workers were used.

## AD Workarounds (Zygote compatibility)

These are necessary due to Zygote's limitations with LAPACK and certain Julia types:

1. **`inv(A)*b` instead of `A\b`** in lattice model — Zygote can't differentiate through LAPACK `zgetrs` for complex dense systems. Safe because lattice systems are small (n_int <= ~8).

2. **Dense `Matrix{Float64}(I,n,n)` instead of `Diagonal`** — Zygote's `Diagonal` pullback is broken.

3. **Analytical `fxs` path** (`jac_and_dirderiv_s`) — avoids nested ForwardDiff duals inside the IFT pullback's Zygote reverse pass. Computes `(F, dF_lambda)` without ForwardDiff, then Zygote differentiates w.r.t. `c`.

4. **Custom `rrule` for `ekf_update`** — manual matrix adjoints avoid Zygote tracing through `inv()`, `/`, and complex matrix chains that caused gradient errors in multi-step EKF.

5. **Custom `rrule` for `batch_solve`** — Wirtinger calculus for complex FDFD adjoint. Recomputes LU factorization in pullback to avoid serializing UMFPACK C-pointers across process boundaries.

6. **No `Symmetric` wrapper** in Zygote-traced `inv()` calls — Zygote has `inv` rrule for `StridedMatrix` but not `Symmetric` (hits LAPACK foreigncall).

## Available Tests (`BFIM_TEST`)

| Test | Description |
|------|-------------|
| `sim_geom` | FDFD gradient d(norm(c)^2)/d(eps_geom) vs FD |
| `mf_f` | Model f(x,s,c) gradient w.r.t. c vs FD |
| `mf_fx` | Jacobian fx directional + full check vs FD |
| `lattice` | powers_only/jac_only/jac_and_dirderiv_s consistency + Zygote AD |
| `ekf` | EKF update gradients (fixed s, with get_sopt, multi-step) |
| `sopt_heatmap` | Visual validation of get_sopt vs grid argmax |
| `sopt_grad` | IFT rrule gradient d(get_sopt)/dc vs FD |
| `episode` | Full episode_loss gradient vs FD (multiple eps) |
| `episode_warm` | Warm-started episode (basin-stable FD) |
| `batch` | Per-episode + aggregate batch gradient check |
| `ekf_perf` | Monte Carlo EKF performance evaluation |
| `taylor_fidelity` | Taylor model vs full FDFD at various Dn |

## Physical Operating Regime

The Taylor model is only reliable for Dn <= 1e-4. SNR ~ 12 at Dn = 5e-5 (midpoint of range) with sigma^2 = 1e-10.

**Shot noise feasibility** (Section 6.3 of report.tex): At lambda=1550nm, shot-noise-limited detection requires P_in ~ 64 uW per port (~1.3 mW total) at 1us integration time. Easily achievable with standard telecom lasers.

## Optimization Results

### opt_result1 (deterministic, 200 episodes, 200 workers)
- **Adam**: 200 iters, lr=1e-3, 600 episodes (no resampling). Loss: 2.46 -> 0.0081. Smooth, monotonic convergence.
- **MMA**: 316 iters, alpha_r=1.0, 200 episodes deterministic. Loss: 2.40 -> 1.12e-4 (21,400x reduction). RMS relative error: 155% -> 1.06%.
- Used relative squared error loss.

### Earlier Adam run (documented in report.tex, not saved as opt_result)
- 1246 iters, 20 episodes/batch, resample_every=1, alpha_r=0.0, 20 workers.
- Loss: ~1.5 -> ~0.003 (MA20). Best single-iter loss 1.1e-3 at iter 1237.
- 10x higher loss than MMA, but 10x fewer episodes per gradient. Confounded by alpha_r difference.

### opt_result2 (stochastic, resample every iter, 20 workers)
- **Adam**: 4759 iters (crashed), 20 episodes with `resample_every=1`, alpha_r=0.0.
- Loss oscillates 0.0001-0.0005 with occasional spikes. Further progress beyond report.
- Crashed with Julia builtins.c `jl_f__apply_iterate` error at iter 4759.
- Uses relative squared error loss, positive Dn in [1e-5, 1e-4].

### Autotune run (Adam + density filter + β continuation, 20 workers)
- **Adam** with density filter (R=5), tanh projection, β continuation (16→32→64→128→256).
- 583 iters, 20 episodes/batch, resample_every=1, alpha_r=0.0, lr=1e-3.
- Automated β tuning via cron-driven Claude agent (40 fires over ~10 hours, 2026-04-16).
- β schedule: 16→32 (iter 241), 32→64 (iter 316), 64→128 (iter 412), 128→256 (iter 493). Stop at iter 583.
- Final MA20 loss: 5.15e-4, loss_min: 1.63e-4. Binary: 92.4% (structural ceiling at β=256 + R=5).
- Post-doubling spikes got progressively milder: 159× (β=32), 230× (β=64), 30× (β=128), none (β=256).
- Geometry pristine throughout: thin-interface gray, no wrong-basin blobs, iso_total_pct at run-min 0.004%.
- First run with density filter during optimization; validates manufacturable geometry approach.
- Monte Carlo EKF evaluation (200 episodes, 10 steps): optimized achieves 0.97% rel error vs 120% for random geometry (123x improvement). BFIM trace 622x higher. Episode loss 18,400x better.

### Key observations
- Deterministic (fixed episodes) converges smoothly; stochastic (resampled) has high variance but explores more
- MMA with alpha_r=1 achieves best results (1.12e-4); Adam with alpha_r=0 plateaus higher
- eps_geom saturates to binary {0, 1} bounds early in optimization
- Gradients are very small (avg ~1e-6) at convergence; steps ~1e-5
- With density filter + β continuation, projected binary reaches 92.4% at β=256 with R=5; 98% target is structurally unreachable at this filter radius
- Post-doubling spike severity decreases as geometry pre-commits at higher β; β=256 transition was spike-free

## File Overview

```
BFIMGaussian.jl          # BFIM sensor selection + EKF + IFT rrule
PosteriorCovGaussian.jl  # A-optimal (posterior covariance) alternative
SimGeomBroadBand.jl      # 2D FDFD physics, S-matrices, lattice model
PhEnd2End.jl             # Training script (Adam/MMA), gradient tests
TODO.md                  # Scaling improvements and future work
doc/posterior_cov_policy.tex  # LaTeX derivation of posterior cov policy
opt_result1/             # First optimization run (deterministic, 600 ep)
opt_result2/             # Second optimization run (stochastic, 20 ep)
c_nom.jls                # Cached nominal S-matrix coefficients (for tests)
.gitignore               # Ignores *.jls, checkpoints/
```

## Key Design Decisions

- **Taylor expansion for S-matrices**: 2nd-order Taylor in Dn avoids re-running FDFD for each EKF evaluation. Accurate for Dn <= 1e-4 (validated by `taylor_fidelity` test).
- **Relative squared error loss**: `sum(((mu_i - x0_i)/x0_i)^2)` instead of absolute `norm(mu - x0)^2` to handle small Dn values (1e-5 to 1e-4) where absolute error would be dominated by scale.
- **Grid initialization for get_sopt**: deterministic 20x20 grid search over [-pi, pi]^2 followed by L-BFGS refinement. Reduces basin-jumping compared to fixed cold start.
- **Wrapping s to [-pi, pi]**: BFIM is 2pi-periodic in s (via `cis`), so wrap is invisible to IFT.
- **Pre-drawn noise**: noise_bank drawn outside AD graph for reproducibility and gradient correctness.
- **pmap parallelism**: episodes distributed across workers; BLAS threads set to 1 per worker to avoid oversubscription.
