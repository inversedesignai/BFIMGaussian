# BFIMGaussian — To-Do List

## Scaling improvements (for dy ~ 100, dx ~ 100)

### High priority

- [ ] **Woodbury identity in `ekf_update`**
  Replace the dy×dy Kalman gain computation with the information-filter form:
  ```
  Σ_new = (Σ⁻¹ + FᵀF/σ²)⁻¹        # dx×dx LU — O(dx³) instead of O(dy³)
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
  `bfim_hessian_s` uses nested ForwardDiff duals: O(ds²·dy·dx).
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

---

## Code quality / module readiness (dev1.jl)

- [ ] **Separate library code from test/script code**
  Move `gr()`, global `model` instance, `rnrg`, and all `if test==N` blocks out of the
  library file. Place library functions in a module; keep test blocks in a separate
  `test/` or `scripts/` file.

- [ ] **Apply the same fixes from dev1 to dev2**
  dev2.jl still has stale issues carried over from before dev1 was cleaned:
  - `_get_sopt_newton` should be renamed (function uses Newton(), but naming should
    reflect purpose, not solver)
  - `αr` is still a bare global `const` — move into `ModelFunctions` struct (as in dev1)
  - Same module-readiness issues (gr(), global model, test blocks)
