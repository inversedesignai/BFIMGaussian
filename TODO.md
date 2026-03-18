# BFIMGaussian — To-Do List

## Current status

The codebase has three modules:
- **SimGeomBroadBand.jl** — 2D FDFD physics, S-matrices, lattice scattering model
- **BFIMGaussian.jl** — BFIM-based sensor selection + EKF + IFT rrule
- **PosteriorCovGaussian.jl** — A-optimal (posterior covariance) sensor selection (drop-in replacement)
- **PhEnd2End.jl** — Training script with Adam, checkpointing, gated tests

### Done
- [x] Unified `_lattice_freq_core` (shared S-matrix assembly across powers_only/jac_only/jac_and_dirderiv_s)
- [x] Analytical `jac_and_dirderiv_s` + Zygote VJP path (avoids nested ForwardDiff in IFT pullback)
- [x] `inv(A)*b` in lattice (Zygote can't trace complex LAPACK `zgetrs`)
- [x] Dense `Matrix{Float64}(I,n,n)` in EKF (Zygote can't trace `Diagonal` pullback)
- [x] Adam optimizer with checkpointing, diagnostics, uniform x0 sampling
- [x] Gated gradient tests via `BFIM_TEST` env var
- [x] PosteriorCovGaussian module (A-optimal criterion, Σ̄ cotangent in IFT)

---

## Scaling improvements (for dy ~ 100, dx ~ 100)

### High priority

- [ ] **Woodbury identity in `ekf_update`**
  Replace the dy×dy Kalman gain computation with the information-filter form:
  ```
  Σ_new = (Σ⁻¹ + FᵀF/σ²)⁻¹        # dx×dx — O(dx³) instead of O(dy³)
  K      = Σ_new · Fᵀ / σ²
  ```
  This also eliminates the Joseph form (Σ_new falls out directly).
  Essential when dy >> dx; a constant-factor improvement when dy ~ dx.

- [ ] **Cache `fx(μ, s*, c)` between `get_sopt` and `ekf_update`**
  After sensor optimisation finds s*, `ekf_update` recomputes `model.fx(μ, s*, c)`.
  Pass the evaluated F directly into `ekf_update` to avoid a redundant dy×dx evaluation
  per EKF step. Requires changing the signature of `ekf_update` to accept an optional
  pre-computed F.

- [ ] **Gradient checkpointing through the EKF loop**
  Zygote stores N snapshots of F (dy×dx), S (dy×dy), K (dx×dy), Σ (dx×dx) per episode.
  At dy=dx=100, N=10: ~3 MB per episode; scales as O(N·dx·dy).
  Use `Zygote.checkpointed` to segment the N-step loop into √N blocks, reducing tape
  memory to O(√N·dx²) at the cost of O(√N) extra forward passes.

### Medium priority

- [ ] **Exploit sparsity in F = ∂f/∂x**
  If `model.fx` produces a sparse or block-structured Jacobian (as in the toy model),
  represent F as `SparseMatrixCSC`. All downstream operations — `bfim_trace`,
  `ekf_update`, the rrule Jacobians — become O(nnz) instead of O(dy·dx).

- [ ] **Gauss-Newton Hessian approximation in the rrule**
  `bfim_hessian_s` / `posterior_hessian_s` uses nested ForwardDiff duals: O(ds²·dy·dx).
  The exact Hessian is H = (2/σ²)[JᵀJ + Σᵢⱼ Fᵢⱼ·∂²Fᵢⱼ/∂s∂sᵀ].
  Drop the second-order term: H ≈ (2/σ²)JᵀJ where J = ∂vec(F)/∂s ∈ R^{(dy·dx)×ds}.
  Requires only first-order derivatives of F; exact when F is linear in s;
  near-exact near the optimum. Avoids nested duals entirely.

- [ ] **Cholesky-maintained covariance**
  Maintain L = chol(Σ) instead of Σ itself, updating via rank-dx Cholesky updates.
  Guarantees positive definiteness over long episodes (large N) where accumulated
  floating-point errors can make the Joseph-form Σ indefinite at dx=100.
  Pairs naturally with the Woodbury update above.

### Low priority / future work

- [ ] **Unscented / ensemble EKF**
  For highly nonlinear `f`, the EKF linearisation error may dominate estimation error.
  UKF uses 2·dx+1 sigma points; EnKF uses a small ensemble.
  Consider as an alternative to EKF when model nonlinearity is significant.

- [ ] **Enzyme.jl for composable AD**
  Zygote+ForwardDiff composition is fragile (LAPACK foreigncalls, Diagonal pullback, etc.).
  Enzyme operates at LLVM IR level and handles forward/reverse composition natively.
  Would eliminate the need for the analytical `fxs` path and manual `inv(A)*b` workarounds.

- [ ] **Port to JAX/Python**
  JAX natively composes `grad`/`jvp`/`vjp` — no closure-capture issues, no LAPACK
  foreigncall problems. Tradeoff: lose native sparse LU and Distributed `pmap`.

---

## Physics / modelling

- [ ] **Validate S-matrix unitarity** for lossless geometries (|S|² conservation)
- [ ] **Multi-step warm-starting for `get_sopt`** — currently restarts from zeros each step;
  warm-starting from previous s★ could speed convergence but may break IFT smoothness
- [ ] **Larger lattice (n_lat > 2)** — test scaling of lattice model and gradient checks
- [ ] **Broadband pulse shaping** — optimize GΔω weights jointly with ε_geom
