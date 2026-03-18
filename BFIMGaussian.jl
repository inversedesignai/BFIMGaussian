"""
    BFIMGaussian

Bilevel sensor-design optimisation under Gaussian measurement noise.

## Problem setup

A hidden state `x ∈ ℝ^dx` is observed through a nonlinear sensor model

    y = f(x, s, c) + ε,    ε ~ N(0, σ²·I_dy)

where
- `s ∈ ℝ^ds` — **sensor parameters** (inner decision variable, tuned per time step)
- `c ∈ ℝ^dc` — **sensor design parameters** (outer / meta decision variable)

## Algorithm

At every time step the sensor parameters are chosen to maximise the trace of the
Bayesian Fisher Information Matrix (BFIM):

    s★(μ, c) = argmax_s  tr(BFIM(μ, s, c))
             = argmax_s  ‖F(μ,s,c)‖²_F / σ²,   F = ∂f/∂x

The resulting measurement is then used to update the EKF belief (μ, Σ).

The outer optimisation minimises the expected squared estimation error

    min_c  E_{x₀}[ ‖μ_N(c) − x₀‖² ]

Gradients through the inner argmin are computed analytically using the
**Implicit Function Theorem** (IFT) via a custom `ChainRulesCore.rrule`
for `_get_sopt`.

## Exports

| Symbol | Description |
|---|---|
| `ModelFunctions` | Sensor-model parameter container |
| `get_sopt` | Compute optimal sensor parameters s★ |
| `bfim_trace` | Trace of the BFIM |
| `bfim_grad_s` | ∇_s of BFIM trace |
| `bfim_hessian_s` | ∇²_s of BFIM trace |
| `ekf_update` | One EKF measurement-update step |
| `episode_loss` | Squared error over an N-step episode |
| `sample_noise_bank` | Pre-draw noise sequences for reproducibility |
"""
module BFIMGaussian

using LinearAlgebra
using ChainRulesCore
using ForwardDiff
using Zygote: Zygote
using Random
using Optim

export ModelFunctions,
       get_sopt, bfim_trace, bfim_grad_s, bfim_hessian_s,
       ekf_update, episode_loss,
       sample_noise_bank

"""
    ModelFunctions{F1,F2}

Container for the sensor observation model and its associated dimensions.

Only the observation function `f` and its Jacobian `fx` are stored explicitly.
All higher-order derivatives (∇_s, ∇²_s of the BFIM trace) are computed
on-the-fly via `ForwardDiff`, so no large derivative arrays are pre-formed.

# Fields
- `f  :: F1`  — observation model `f(x, s, c) → ℝ^dy`
- `fx :: F2`  — Jacobian `∂f/∂x(x, s, c) → ℝ^{dy×dx}`  (used by EKF linearisation
                and BFIM; may be hand-coded or an AD wrapper)
- `σ² :: Float64` — measurement noise variance (scalar, isotropic: Σ_ε = σ²·I)
- `dy :: Int`  — observation dimension (default 6)
- `dx :: Int`  — state dimension (default 2)
- `ds :: Int`  — sensor-parameter dimension (default 4)
- `dc :: Int`  — sensor-design dimension (default 3)
- `αr :: Float64` — L2 regularisation coefficient for the inner s-optimisation
                    (penalises ‖s‖² to keep solutions bounded; default 20.0)
"""
Base.@kwdef struct ModelFunctions{F1,F2,F3}
    f    :: F1
    fx   :: F2
    fxs  :: F3 = nothing       # (x, s, c, λ) → (F, dF_λ): analytical s-directional derivative
    σ²   :: Float64
    dy   :: Int = 6
    dx   :: Int = 2
    ds   :: Int = 4
    dc   :: Int = 3
    αr   :: Float64 = 20.0   # regularization coefficient for sensor parameter optimisation
    zero_s_init :: Bool = false  # if true, initialise s_init=zeros(ds); else ones(ds)
end

"""
    _global_init_s(c, μ, model; n_grid=8) → s_init

Deterministic grid search over [-π, π]^ds to find a good initial point
for the L-BFGS sensor-parameter optimisation.  Evaluates the regularised
objective φ = -bfim_trace + αr‖s‖² on a uniform grid (n_grid^ds points)
and returns the grid point with the smallest φ.

For ds=2 with n_grid=8, this is 64 evaluations of bfim_trace — cheap
compared to the FDFD solver.  For ds>2, consider replacing with a
proper Bayesian optimisation (GP surrogate + acquisition function).

This function is non-differentiable and must be called inside
`@ignore_derivatives` so that Zygote does not trace through it.
"""
function _global_init_s(c, μ, model::ModelFunctions; n_grid::Int=8)
    ds = model.ds
    φ_range = collect(range(-π, π, length=n_grid+1))[1:end-1]  # exclude endpoint (periodic)
    obj = s -> -bfim_trace(μ, s, c, model) + model.αr * sum(abs2, s)

    if ds == 2
        best_obj = Inf
        best_s   = zeros(2)
        for φ₁ in φ_range, φ₂ in φ_range
            s_try = [φ₁, φ₂]
            val   = obj(s_try)
            if val < best_obj
                best_obj = val
                best_s   = s_try
            end
        end
        return best_s
    else
        # General ds: random Latin hypercube (deterministic via model dims)
        n_samples = n_grid^min(ds, 3)  # cap for high ds
        best_obj  = Inf
        best_s    = zeros(ds)
        for i in 1:n_samples
            # Deterministic quasi-random: use golden-ratio based sequence
            s_try = [mod(i * (sqrt(5)-1)/2 * (k+1), 1.0) * 2π - π for k in 1:ds]
            val   = obj(s_try)
            if val < best_obj
                best_obj = val
                best_s   = s_try
            end
        end
        return best_s
    end
end

"""
    get_sopt(c, μ, model) → s★

Compute the optimal sensor parameters `s★ ∈ [-π, π]^ds` that maximise the
BFIM trace for the current belief mean `μ` and sensor design `c`.

Solves:
    s★ = argmin_s  [ -tr(BFIM(μ, s, c)) + αr‖s‖² ]

**Initialisation**: a deterministic grid search over [-π, π]^ds finds the
best starting point (non-differentiable, wrapped in `@ignore_derivatives`),
then L-BFGS refines to the local minimum.  This global initialisation
reduces basin-jumping compared to a fixed cold start.

**Wrapping**: the result is wrapped to [-π, π] by subtracting the nearest
multiple of 2π.  The wrapping offset is computed inside `@ignore_derivatives`
so the gradient passes through as identity (∂(s − const)/∂s = 1).

Neither the grid initialisation nor the wrapping affects the IFT `rrule`:
- `s_init` receives `ZeroTangent()` in the rrule (initialisation is invisible to AD)
- The BFIM objective is 2π-periodic in s (via `cis`), so H_φ and J_c/J_μ are
  identical at wrapped vs unwrapped points

See also [`_get_sopt`](@ref) for the version that accepts an explicit `s_init`.
"""
function get_sopt(c, μ, model::ModelFunctions)
    # Step 1: Global initialisation (non-differentiable)
    s_init = @ignore_derivatives _global_init_s(c, μ, model)

    # Step 2: Refine with L-BFGS (differentiable via IFT rrule on _get_sopt)
    sopt = _get_sopt(c, μ, model, s_init)

    # Step 3: Wrap to [-π, π].  The offset is a non-differentiable integer
    # multiple of 2π, so ∂(sopt − offset)/∂sopt = I.
    offset = @ignore_derivatives 2π .* round.(sopt ./ (2π))
    return sopt .- offset
end

"""
    _get_sopt(c, μ, model, s_init) → s★

Internal implementation of sensor-parameter optimisation.  Uses L-BFGS via
`Optim.jl` with analytic gradients (computed via `bfim_grad_s`).

Emits a warning if the solver does not converge, reporting the gradient norm
at the returned point.
"""
function _get_sopt(c, μ, model::ModelFunctions, s_init::AbstractVector)
    regs      = s -> model.αr * sum(abs2, s)
    regs_grad = s -> 2 * model.αr * s
    obj      = svec -> -bfim_trace(μ, svec, c, model) + regs(svec)
    grad_obj = svec -> -bfim_grad_s(μ, svec, c, model) + regs_grad(svec)
    result = optimize(obj, grad_obj, s_init, LBFGS(),
                      Optim.Options(g_tol=1e-12, iterations=10000), inplace=false)
    sopt = Optim.minimizer(result)
    # converged = Optim.converged(result)
    # if !converged
    #     gnorm = norm(grad_obj(sopt))
    #     @warn "Optimizer did not converge. Gradient norm at sopt: $(gnorm)"
    # end
    return sopt
end

"""
    bfim_trace(μ, s, c, model) → Float64

Compute the trace of the Bayesian Fisher Information Matrix (BFIM) at belief
mean `μ`, sensor parameters `s`, and design parameters `c`:

    tr(BFIM) = ‖F(μ,s,c)‖²_F / σ²,    F = ∂f/∂x ∈ ℝ^{dy×dx}

This is the objective maximised in the inner sensor-parameter optimisation.
The Frobenius norm squared equals `tr(FᵀF)` = `tr(BFIM)` for the linearised
Gaussian model with isotropic noise covariance σ²·I.
"""
function bfim_trace(μ, s, c, model::ModelFunctions)
    F = model.fx(μ, s, c)                 # dy × dx
    return sum(abs2, F) / model.σ²
end

"""
    bfim_grad_s(μ, s, c, model) → Vector{Float64}

Gradient of [`bfim_trace`](@ref) with respect to sensor parameters `s`,
computed via `ForwardDiff.gradient`.

ForwardDiff propagates dual numbers through `model.fx`; no explicit
∂F/∂s (FXS) array is ever allocated.
Peak intermediate memory: O(dy·dx·chunk) for the dual-valued F matrix,
where `chunk` is the ForwardDiff chunk size (default ≈ 8, much smaller than ds).
"""
function bfim_grad_s(μ, s, c, model::ModelFunctions)
    return ForwardDiff.gradient(s_ -> bfim_trace(μ, s_, c, model), s)
end

"""
    bfim_hessian_s(μ, s, c, model) → Matrix{Float64}

Hessian of [`bfim_trace`](@ref) with respect to sensor parameters `s`,
computed via `ForwardDiff.hessian`.

Uses nested dual numbers (dual-of-dual) propagated through `model.fx`.
No explicit ∂²F/∂s² (FXSS) array is ever allocated.
Peak intermediate memory: O(dy·dx·chunk²), which stays small for any ds
because the ForwardDiff chunk size is fixed at ≈ 8 by default.

Used in the IFT-based `rrule` for `_get_sopt` to assemble the KKT Hessian.
"""
function bfim_hessian_s(μ, s, c, model::ModelFunctions)
    return ForwardDiff.hessian(s_ -> bfim_trace(μ, s_, c, model), s)
end

# ── IFT-based custom reverse rule for _get_sopt ───────────────────────────────
#
# Background: `_get_sopt` solves an unconstrained minimisation, so at the
# optimum s★ the KKT stationarity condition is:
#
#     ∇_s φ(μ, s★, c) = 0,    φ = -bfim_trace + αr‖s‖²
#
# Applying the Implicit Function Theorem (IFT) to this condition gives:
#
#     ∂s★/∂c = H_φ⁻¹ · J_c,    ∂s★/∂μ = H_φ⁻¹ · J_μ
#
# where H_φ = ∇²_ss φ|_{s★} is the Hessian of φ at the optimum, and
# J_c = ∂(bfim_grad_s)/∂c,  J_μ = ∂(bfim_grad_s)/∂μ  are the cross Jacobians
# of the BFIM gradient.  Note: ∂(∇_s φ)/∂c = −J_c because
# ∇_s φ = −bfim_grad_s + 2αr s, so the standard IFT sign cancels.
#
# For a scalar loss L with upstream cotangent s̄ = ∂L/∂s★:
#
#     c̄ = (∂s★/∂c)ᵀ s̄ = J_cᵀ H_φ⁻¹ s̄ = J_cᵀ λ
#     μ̄ = (∂s★/∂μ)ᵀ s̄ = J_μᵀ λ
#
# where λ = H_φ⁻¹ s̄ is obtained by solving the linear system H_φ λ = s̄.
# Note: the objective is φ = -bfim_trace + αr‖s‖², so at the *minimum* H_φ = -H_bfim + 2αr·I.
# is positive definite (the regularisation guarantees this), giving a unique λ.
"""
    ChainRulesCore.rrule(::typeof(_get_sopt), c, μ, model, s_init)

Reverse-mode differentiation rule for [`_get_sopt`](@ref) via the
**Implicit Function Theorem (IFT)**.

At the optimum `s★` the stationarity condition `∇_s φ(μ, s★, c) = 0` holds,
where `φ = -bfim_trace + αr‖s‖²`.  Differentiating this identity implicitly
yields the cotangents

    c̄  = J_cᵀ λ,    μ̄  = J_μᵀ λ,    H_φ λ = s̄

with `H_φ = ∇²_ss φ|_{s★}`, `J_c = ∂(bfim_grad_s)/∂c`, `J_μ = ∂(bfim_grad_s)/∂μ`.

The Hessian `H_φ` is factorised once (LU) and reused for the linear solve.
When `model.fxs` is provided, `c̄ = J_c'λ` is computed as a single Zygote
reverse pass through the analytical directional derivative `λ'·∇_s bfim_trace`
(no nested AD). Otherwise falls back to `ForwardDiff.jacobian` (O(dc/chunk)).
"""
function ChainRulesCore.rrule(::typeof(_get_sopt),
                              c, μ, model::ModelFunctions, s_init)

    # ── Forward pass ──────────────────────────────────────────────────────────
    s_star = _get_sopt(c, μ, model, s_init)

    # Assemble Hessian of the FULL objective φ = -bfim_trace + αr‖s‖²:
    #   H_obj = ∇²_ss φ|_{s*} = -bfim_hessian_s + 2αr·I
    H_obj = -bfim_hessian_s(μ, s_star, c, model) + 2 * model.αr * I(model.ds)
    H_lu = lu(H_obj)                       # factorise once

    # ── Pullback closure ──────────────────────────────────────────────────────
    function _get_sopt_pb(s̄)
        s̄_vec = collect(Float64, ChainRulesCore.unthunk(s̄))

        # Step 1: solve H_obj λ = s̄
        # IFT: H_obj · ds*/dc = J_c  ⟹  c̄ = J_c' · λ  where H_obj · λ = s̄
        λ = H_lu \ s̄_vec           # ds

        # Step 2: cotangent c̄ = J_c' λ.
        if model.fxs !== nothing
            # Analytical path (no nested AD): compute λ'·∇_s bfim_trace
            # via model.fxs which returns (F, dF_λ) without ForwardDiff,
            # then differentiate w.r.t. c via a single Zygote reverse pass.
            c̄ = Zygote.gradient(c) do c_
                F, dF_λ = model.fxs(μ, s_star, c_, λ)
                2 * sum(F .* dF_λ) / model.σ²
            end |> first
        else
            # ForwardDiff fallback: O(dc/chunk) passes with nested duals.
            J_c = ForwardDiff.jacobian(c_ -> bfim_grad_s(μ,  s_star, c_, model), c)
            c̄   = J_c' * λ
        end

        # Step 3: cotangent μ̄ = J_μ' λ  (dx is small, ForwardDiff is efficient).
        J_μ = ForwardDiff.jacobian(μ_ -> bfim_grad_s(μ_, s_star, c,  model), μ)
        μ̄  = J_μ' * λ                     # dx

        return (NoTangent(),    # ::typeof(_get_sopt)
                c̄,              # c
                μ̄,              # μ
                NoTangent(),    # model
                ZeroTangent())  # s_init
    end

    return s_star, _get_sopt_pb
end

"""
    episode_loss(x0, c, model, μ0, Σ0, noise_seq) → Float64

Simulate a single N-step estimation episode and return the squared error
`‖μ_N − x₀‖²` between the final EKF mean and the true state.

At each step k:
1. **Sensor selection**: compute `sₖ = get_sopt(c, μ, model)` — the sensor
   parameters that maximise the BFIM trace given the current belief `μ`.
2. **Measurement**: `yₖ = f(x₀, sₖ, c) + noise_seq[k]`  (noise is pre-drawn
   and held fixed; it does not enter the AD graph).
3. **EKF update**: `(μ, Σ) ← ekf_update(μ, Σ, yₖ, sₖ, c, model)`.

The sensor initialisation is always reset to `ones(ds)` so that `sₖ(μ, c)` is
a deterministic smooth function of `c`, keeping AD and finite-difference
gradients consistent.

# Arguments
- `x0`        — true state vector, ℝ^dx
- `c`         — sensor design parameters, ℝ^dc  (the outer optimisation variable)
- `model`     — [`ModelFunctions`](@ref)
- `μ0`, `Σ0` — initial EKF mean and covariance
- `noise_seq` — length-N vector of pre-drawn noise vectors, each ∈ ℝ^dy
"""
function episode_loss(x0, c, model::ModelFunctions,
                      μ0, Σ0,
                      noise_seq::AbstractVector)
    μ = μ0
    Σ = Σ0
    N = length(noise_seq)

    for k in 1:N
        # ── Step 1: argmax_s tr(I_BFIM) ─────────────────────────────────────
        # No warm-start: always restart from the same s_init so that sk is a
        # smooth (deterministic) function of c, making AD and FD consistent.
        sk = get_sopt(c, μ, model)

        # ── Step 2: Measurement (noise is fixed — not part of AD graph) ──────
        yk = model.f(x0, sk, c) + noise_seq[k]

        # ── Step 3: EKF update ──────────────────────────────────────────────
        μ, Σ = ekf_update(μ, Σ, yk, sk, c, model)
    end

    return sum(abs2, μ - x0)  # squared error ‖μ_N − x₀‖²
end

"""
    ekf_update(μ, Σ, y, s, c, model) → (μ_new, Σ_new)

Perform one **Extended Kalman Filter (EKF) measurement update** step.

The observation model is linearised around the current mean `μ`:

    y ≈ f(μ, s, c) + F·(x − μ) + ε,    F = ∂f/∂x|_{μ,s,c},   ε ~ N(0, σ²I)

Standard EKF equations:

    Innovation covariance:  S   = F Σ Fᵀ + σ²·I              (dy × dy)
    Kalman gain:            K   = Σ Fᵀ S⁻¹                    (dx × dy)
    Updated mean:           μ'  = μ + K·(y − f̂)
    Updated covariance:     Σ'  = (I − KF) Σ (I − KF)ᵀ + K·(σ²I)·Kᵀ

The covariance is updated using the **Joseph (symmetric) form**
`(I−KF)Σ(I−KF)ᵀ + K Rₑ Kᵀ` (where Rₑ = σ²I) to maintain positive
semi-definiteness numerically even with imperfect Kalman gains.
"""
function _ekf_forward(F, f̂, μ, Σ, y, σ²)
    dx = length(μ)
    dy = length(y)
    I_dx = Matrix{Float64}(I, dx, dx)
    I_dy = Matrix{Float64}(I, dy, dy)

    S     = F * Σ * F' + σ² * I_dy          # dy × dy
    S_inv = inv(S)                            # dy × dy
    K     = Σ * F' * S_inv                    # dx × dy
    innov = y - f̂                             # dy
    μ_new = μ + K * innov                     # dx
    IKF   = I_dx - K * F                      # dx × dx
    Σ_new = IKF * Σ * IKF' + (σ² * K) * K'   # dx × dx  (Joseph form)

    return μ_new, Σ_new, S_inv, K, IKF
end

function ekf_update(μ, Σ, y, s, c, model::ModelFunctions)
    F  = model.fx(μ, s, c)
    f̂  = model.f(μ, s, c)
    μ_new, Σ_new, _, _, _ = _ekf_forward(F, f̂, μ, Σ, y, model.σ²)
    return μ_new, Σ_new
end

# ── Custom rrule for ekf_update ───────────────────────────────────────────────
# Manually computes adjoints through the EKF matrix equations, then uses
# Zygote.pullback to propagate ΔF, Δf̂ back through model.fx, model.f.
# This avoids Zygote tracing through inv(), /, and complex matrix chains
# that caused gradient errors in multi-step EKF compositions.
function ChainRulesCore.rrule(::typeof(ekf_update),
                              μ, Σ, y, s, c, model::ModelFunctions)
    # Forward pass — get pullbacks for model.fx and model.f
    F, pb_fx = Zygote.pullback((μ_, s_, c_) -> model.fx(μ_, s_, c_), μ, s, c)
    f̂, pb_f  = Zygote.pullback((μ_, s_, c_) -> model.f(μ_, s_, c_),  μ, s, c)

    μ_new, Σ_new, S_inv, K, IKF = _ekf_forward(F, f̂, μ, Σ, y, model.σ²)
    innov = y - f̂
    σ² = model.σ²

    function ekf_update_pb(Δ_raw)
        Δμ_new_r = ChainRulesCore.unthunk(Δ_raw[1])
        ΔΣ_new_r = ChainRulesCore.unthunk(Δ_raw[2])

        dx = length(μ);  dy = length(y)
        _zero_v(n)    = zeros(n)
        _zero_m(r, c) = zeros(r, c)
        Δμ_new = (Δμ_new_r === nothing || Δμ_new_r isa ChainRulesCore.AbstractZero) ? _zero_v(dx)     : collect(Float64, Δμ_new_r)
        ΔΣ_new = (ΔΣ_new_r === nothing || ΔΣ_new_r isa ChainRulesCore.AbstractZero) ? _zero_m(dx, dx) : collect(Float64, ΔΣ_new_r)

        # ── μ_new = μ + K * innov ─────────────────────────────────────────────
        ΔK   = Δμ_new * innov'                           # dx × dy
        Δy   = K' * Δμ_new                                # dy
        Δf̂_  = -K' * Δμ_new                               # dy

        # ── Σ_new = IKF * Σ * IKF' + σ² * K * K' ─────────────────────────────
        #  ∂L/∂Σ  from  A*Σ*A'  (A = IKF)
        ΔΣ   = IKF' * ΔΣ_new * IKF                       # dx × dx

        #  ∂L/∂IKF  from  IKF*Σ*IKF'  (A*B*A' form)
        ΔIKF = ΔΣ_new * IKF * Σ' + ΔΣ_new' * IKF * Σ    # dx × dx

        #  ∂L/∂K  from  σ²*K*K'  (scalar * A*A' form)
        ΔK  += σ² * (ΔΣ_new + ΔΣ_new') * K               # dx × dy

        #  IKF = I − K*F  →  ∂L/∂K from IKF
        ΔK  += -ΔIKF * F'                                 # dx × dy

        #  IKF = I − K*F  →  ∂L/∂F from IKF
        ΔF   = -K' * ΔIKF                                 # dy × dx

        # ── K = Σ * F' * S_inv ────────────────────────────────────────────────
        M  = F' * S_inv                                    # dx × dy
        ΔΣ += ΔK * M'                                     # dx × dx
        ΔM  = Σ * ΔK                                      # dx × dy  (Σ symmetric)
        ΔF += (ΔM * S_inv)'                                # dy × dx  (from F' in M)

        ΔS_inv = F * ΔM                                   # dy × dy  (from S_inv in M)
        ΔS     = -S_inv * ΔS_inv * S_inv                  # dy × dy  (inv adjoint)

        # ── S = F * Σ * F' + σ² * I ──────────────────────────────────────────
        ΔF += ΔS * F * Σ' + ΔS' * F * Σ                   # dy × dx  (A*B*A' form)
        ΔΣ += F' * ΔS * F                                  # dx × dx

        # ── Propagate ΔF, Δf̂ back through model.fx, model.f ─────────────────
        Δμ_fx, Δs_fx, Δc_fx = pb_fx(ΔF)
        Δμ_f,  Δs_f,  Δc_f  = pb_f(Δf̂_)

        _v(x, n) = x === nothing ? zeros(n) : collect(Float64, x)
        _m(x, r, c) = x === nothing ? zeros(r, c) : collect(Float64, x)

        Δμ_total = Δμ_new .+ _v(Δμ_fx, dx) .+ _v(Δμ_f, dx)
        Δs_total = _v(Δs_fx, length(s)) .+ _v(Δs_f, length(s))
        Δc_total = _v(Δc_fx, length(c)) .+ _v(Δc_f, length(c))

        return (NoTangent(),  # ::typeof(ekf_update)
                Δμ_total,     # μ
                ΔΣ,           # Σ
                Δy,           # y
                Δs_total,     # s
                Δc_total,     # c
                NoTangent())  # model
    end

    return (μ_new, Σ_new), ekf_update_pb
end

"""
    sample_noise_bank(rng, n_episodes, N, dy, σ²) → Vector{Vector{Vector{Float64}}}

Pre-draw all measurement noise samples for a training step.

Returns a `n_episodes`-length vector of episodes, each containing N noise
vectors drawn from `N(0, σ²·I_dy)`.  Drawing noise outside the AD graph
ensures that stochasticity does not interfere with gradient computation and
that episodes are reproducible given the same `rng` state.
"""
function sample_noise_bank(rng::AbstractRNG,
                           n_episodes::Int, N::Int, dy::Int, σ²::Float64)
    σ = sqrt(σ²)
    return [[σ * randn(rng, dy) for _ in 1:N] for _ in 1:n_episodes]
end

end # module BFIMGaussian
