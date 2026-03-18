"""
    PosteriorCovGaussian

Bilevel sensor-design optimisation under Gaussian measurement noise,
using the **expected posterior covariance** as the sensor selection criterion.

## Problem setup

A hidden state `x ∈ ℝ^dx` is observed through a nonlinear sensor model

    y = f(x, s, c) + ε,    ε ~ N(0, σ²·I_dy)

where
- `s ∈ ℝ^ds` — **sensor parameters** (inner decision variable, tuned per time step)
- `c ∈ ℝ^dc` — **sensor design parameters** (outer / meta decision variable)

## Algorithm

At every time step the sensor parameters are chosen to **minimise** the trace
of the expected posterior covariance (A-optimal design):

    s★(μ, Σ, c) = argmin_s  tr(Σ_new(μ, s, c, Σ))
                 = argmin_s  tr( (Σ⁻¹ + F'F/σ²)⁻¹ ),   F = ∂f/∂x

Unlike the BFIM criterion (which only uses F and σ²), this criterion accounts
for the current belief uncertainty Σ, focusing sensor resources on directions
where uncertainty is still large.

The resulting measurement is then used to update the EKF belief (μ, Σ).

The outer optimisation minimises the expected squared estimation error

    min_c  E_{x₀}[ ‖μ_N(c) − x₀‖² ]

Gradients through the inner argmin are computed analytically using the
**Implicit Function Theorem** (IFT) via a custom `ChainRulesCore.rrule`
for `_get_sopt`.  Because s★ now depends on Σ (in addition to μ and c),
the IFT pullback also returns a cotangent for Σ.

## Exports

| Symbol | Description |
|---|---|
| `ModelFunctions` | Sensor-model parameter container |
| `get_sopt` | Compute optimal sensor parameters s★ |
| `posterior_cov_trace` | tr(Σ_new) — the optimisation objective |
| `posterior_grad_s` | ∇_s of posterior_cov_trace |
| `posterior_hessian_s` | ∇²_s of posterior_cov_trace |
| `ekf_update` | One EKF measurement-update step |
| `episode_loss` | Squared error over an N-step episode |
| `sample_noise_bank` | Pre-draw noise sequences for reproducibility |
"""
module PosteriorCovGaussian

using LinearAlgebra
using ChainRulesCore
using ForwardDiff
using Zygote: Zygote
using Random
using Optim

export ModelFunctions,
       get_sopt, posterior_cov_trace, posterior_grad_s, posterior_hessian_s,
       ekf_update, episode_loss,
       sample_noise_bank

"""
    ModelFunctions{F1,F2,F3}

Container for the sensor observation model and its associated dimensions.

# Fields
- `f  :: F1`  — observation model `f(x, s, c) → ℝ^dy`
- `fx :: F2`  — Jacobian `∂f/∂x(x, s, c) → ℝ^{dy×dx}`
- `fxs :: F3` — (x, s, c, λ) → (F, dF_λ): analytical s-directional derivative
- `σ² :: Float64` — measurement noise variance (scalar, isotropic: Σ_ε = σ²·I)
- `dy :: Int`  — observation dimension
- `dx :: Int`  — state dimension
- `ds :: Int`  — sensor-parameter dimension
- `dc :: Int`  — sensor-design dimension
- `αr :: Float64` — L2 regularisation coefficient for the inner s-optimisation
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
    αr   :: Float64 = 20.0
    zero_s_init :: Bool = false
end

# ═══════════════════════════════════════════════════════════════════════════════
# Posterior covariance objective
# ═══════════════════════════════════════════════════════════════════════════════

"""
    posterior_cov_trace(μ, s, c, model, Σ) → Float64

Compute `tr(Σ_new)` where `Σ_new = (Σ⁻¹ + F'F/σ²)⁻¹` is the posterior
covariance after one EKF measurement update (in the linearised Gaussian
approximation).

This is the objective **minimised** in the inner sensor-parameter optimisation.
In the EKF framework, Σ_new does not depend on the actual measurement y,
so the expectation over y is trivial.
"""
function posterior_cov_trace(μ, s, c, model::ModelFunctions, Σ)
    F = model.fx(μ, s, c)                          # dy × dx
    M = Symmetric(inv(Symmetric(Σ)) + F' * F / model.σ²)   # dx × dx, posterior precision
    return tr(inv(M))
end

"""
    posterior_grad_s(μ, s, c, model, Σ) → Vector{Float64}

Gradient of [`posterior_cov_trace`](@ref) with respect to sensor parameters `s`,
computed via `ForwardDiff.gradient`.
"""
function posterior_grad_s(μ, s, c, model::ModelFunctions, Σ)
    return ForwardDiff.gradient(s_ -> posterior_cov_trace(μ, s_, c, model, Σ), s)
end

"""
    posterior_hessian_s(μ, s, c, model, Σ) → Matrix{Float64}

Hessian of [`posterior_cov_trace`](@ref) with respect to sensor parameters `s`,
computed via `ForwardDiff.hessian`.

Used in the IFT-based `rrule` for `_get_sopt` to assemble the KKT Hessian.
"""
function posterior_hessian_s(μ, s, c, model::ModelFunctions, Σ)
    return ForwardDiff.hessian(s_ -> posterior_cov_trace(μ, s_, c, model, Σ), s)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Inner sensor-parameter optimisation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    get_sopt(c, μ, Σ, model) → s★

Compute the optimal sensor parameters `s★` that minimise the posterior
covariance trace for the current belief (μ, Σ) and sensor design `c`.

Solves:
    s★ = argmin_s  [ tr(Σ_new(μ, s, c, Σ)) + αr‖s‖² ]
"""
function get_sopt(c, μ, Σ, model::ModelFunctions)
    s_init = model.zero_s_init ? zeros(model.ds) : fill(1.0, model.ds)
    return _get_sopt(c, μ, Σ, model, s_init)
end

"""
    _get_sopt(c, μ, Σ, model, s_init) → s★

Internal implementation of sensor-parameter optimisation.  Uses L-BFGS via
`Optim.jl` with analytic gradients.
"""
function _get_sopt(c, μ, Σ, model::ModelFunctions, s_init::AbstractVector)
    regs      = s -> model.αr * sum(abs2, s)
    regs_grad = s -> 2 * model.αr * s
    obj      = svec -> posterior_cov_trace(μ, svec, c, model, Σ) + regs(svec)
    grad_obj = svec -> posterior_grad_s(μ, svec, c, model, Σ) + regs_grad(svec)
    result = optimize(obj, grad_obj, s_init, LBFGS(),
                      Optim.Options(g_tol=1e-12, iterations=10000), inplace=false)
    return Optim.minimizer(result)
end

# ── IFT-based custom reverse rule for _get_sopt ───────────────────────────────
#
# At the optimum s★ the stationarity condition holds:
#
#     ∇_s φ(μ, Σ, s★, c) = 0,    φ = posterior_cov_trace + αr‖s‖²
#
# Differentiating implicitly w.r.t. θ ∈ {c, μ, vec(Σ)}:
#
#     H_φ · ∂s★/∂θ + ∂(∇_s φ)/∂θ = 0
#
# where H_φ = ∇²_ss φ|_{s★} is the Hessian at the optimum.
#
# For a scalar loss L with upstream cotangent s̄ = ∂L/∂s★:
#
#     θ̄ = (∂s★/∂θ)ᵀ s̄ = J_θᵀ λ,    H_φ λ = s̄
#
# Since φ = posterior_cov_trace + αr‖s‖²  (minimised, not negated),
# we have ∂(∇_s φ)/∂θ = ∂(posterior_grad_s)/∂θ (the regularisation
# 2αr·s does not depend on θ).  And H_φ = posterior_hessian_s + 2αr·I.
#
# Unlike BFIMGaussian, s★ now depends on Σ, so we also need Σ̄.
"""
    ChainRulesCore.rrule(::typeof(_get_sopt), c, μ, Σ, model, s_init)

Reverse-mode differentiation rule for [`_get_sopt`](@ref) via the
**Implicit Function Theorem (IFT)**.

Returns cotangents for `c`, `μ`, and `Σ`.
When `model.fxs` is provided, `c̄` is computed via a single Zygote reverse
pass through the analytical directional derivative (no nested AD).
"""
function ChainRulesCore.rrule(::typeof(_get_sopt),
                              c, μ, Σ, model::ModelFunctions, s_init)

    # ── Forward pass ──────────────────────────────────────────────────────────
    s_star = _get_sopt(c, μ, Σ, model, s_init)

    # Hessian of the FULL objective φ = posterior_cov_trace + αr‖s‖²:
    H_obj = posterior_hessian_s(μ, s_star, c, model, Σ) + 2 * model.αr * I(model.ds)
    H_lu = lu(H_obj)

    # ── Pullback closure ──────────────────────────────────────────────────────
    function _get_sopt_pb(s̄)
        s̄_vec = collect(Float64, ChainRulesCore.unthunk(s̄))

        # Step 1: solve H_obj λ = s̄
        λ = H_lu \ s̄_vec

        # Step 2: cotangent c̄ = J_c' λ.
        if model.fxs !== nothing
            # Analytical path: compute λ'·∇_s posterior_cov_trace via fxs,
            # then differentiate w.r.t. c via Zygote.
            c̄ = Zygote.gradient(c) do c_
                F, dF_λ = model.fxs(μ, s_star, c_, λ)
                Σ_inv = inv(Symmetric(Σ))
                M = Symmetric(Σ_inv + F' * F / model.σ²)
                Σ_new = inv(M)
                dM = (dF_λ' * F + F' * dF_λ) / model.σ²
                -tr(Σ_new * dM * Σ_new)
            end |> first
        else
            # ForwardDiff fallback.
            J_c = ForwardDiff.jacobian(c_ -> posterior_grad_s(μ,  s_star, c_, model, Σ), c)
            c̄   = J_c' * λ
        end

        # Step 3: cotangent μ̄ = J_μ' λ  (dx is small).
        J_μ = ForwardDiff.jacobian(μ_ -> posterior_grad_s(μ_, s_star, c,  model, Σ), μ)
        μ̄  = J_μ' * λ

        # Step 4: cotangent Σ̄ = reshape(J_Σ' λ, dx, dx).
        # Σ enters posterior_grad_s; differentiate w.r.t. vec(Σ).
        dx = model.dx
        J_Σ = ForwardDiff.jacobian(
            Σ_flat -> posterior_grad_s(μ, s_star, c, model, reshape(Σ_flat, dx, dx)),
            vec(Σ))
        Σ̄  = reshape(J_Σ' * λ, dx, dx)

        return (NoTangent(),    # ::typeof(_get_sopt)
                c̄,              # c
                μ̄,              # μ
                Σ̄,              # Σ  ← NEW
                NoTangent(),    # model
                ZeroTangent())  # s_init
    end

    return s_star, _get_sopt_pb
end

# ═══════════════════════════════════════════════════════════════════════════════
# EKF update and episode simulation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    episode_loss(x0, c, model, μ0, Σ0, noise_seq) → Float64

Simulate a single N-step estimation episode and return the squared error
`‖μ_N − x₀‖²` between the final EKF mean and the true state.

At each step k:
1. **Sensor selection**: `sₖ = get_sopt(c, μ, Σ, model)` — minimises the
   expected posterior covariance trace given the current belief (μ, Σ).
2. **Measurement**: `yₖ = f(x₀, sₖ, c) + noise_seq[k]`
3. **EKF update**: `(μ, Σ) ← ekf_update(μ, Σ, yₖ, sₖ, c, model)`
"""
function episode_loss(x0, c, model::ModelFunctions,
                      μ0, Σ0,
                      noise_seq::AbstractVector)
    μ = μ0
    Σ = Σ0
    N = length(noise_seq)

    for k in 1:N
        # ── Step 1: argmin_s tr(Σ_new) ───────────────────────────────────────
        sk = get_sopt(c, μ, Σ, model)

        # ── Step 2: Measurement (noise is fixed — not part of AD graph) ──────
        yk = model.f(x0, sk, c) + noise_seq[k]

        # ── Step 3: EKF update ──────────────────────────────────────────────
        μ, Σ = ekf_update(μ, Σ, yk, sk, c, model)
    end

    return sum(abs2, μ - x0)
end

"""
    ekf_update(μ, Σ, y, s, c, model) → (μ_new, Σ_new)

Perform one Extended Kalman Filter measurement update step (Joseph form).
Identical to BFIMGaussian.ekf_update.
"""
function ekf_update(μ, Σ, y, s, c, model::ModelFunctions)
    F  = model.fx(μ, s, c)
    f̂  = model.f(μ, s, c)

    dx = length(μ)
    dy = length(y)

    I_dy = Matrix{Float64}(I, dy, dy)
    I_dx = Matrix{Float64}(I, dx, dx)
    S = F * Σ * F' + model.σ² * I_dy
    K = Σ * F' / S

    μ_new = μ + K * (y - f̂)
    IKF   = I_dx - K * F
    Σ_new = IKF * Σ * IKF' + (model.σ² * K) * K'

    return μ_new, Σ_new
end

"""
    sample_noise_bank(rng, n_episodes, N, dy, σ²) → Vector{Vector{Vector{Float64}}}

Pre-draw all measurement noise samples for a training step.
"""
function sample_noise_bank(rng::AbstractRNG,
                           n_episodes::Int, N::Int, dy::Int, σ²::Float64)
    σ = sqrt(σ²)
    return [[σ * randn(rng, dy) for _ in 1:N] for _ in 1:n_episodes]
end

end # module PosteriorCovGaussian
