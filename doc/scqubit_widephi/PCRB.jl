"""
PCRB.jl

Fisher-information / PCRB baseline for the scqubit flux sensor.

For scalar unknown Φ ∈ [0, Φ_max] (normalized flux) observed via Bernoulli
outcomes with per-shot success probability P_|1⟩(Φ, τ; c), the per-shot Fisher
information is

    J_F(Φ, τ; c) = (∂P/∂Φ)² / [P (1 − P)].

With a schedule s = ((τ_k, n_k))_{k=1..K} (n_k shots at delay τ_k), the total
Fisher information accumulated at a given Φ is

    J_N(Φ, s, c) = Σ_k n_k J_F(Φ, τ_k, c).

The prior-averaged Bayesian Information Matrix (scalar here):

    J_P(s, c) = J_0 + (1/Φ_max) ∫₀^{Φ_max} J_N(Φ, s, c) dΦ.

We target `argmax_{c, s} log J_P(s, c)` (single-level, no belief tree).

All φ-derivatives of P_|1⟩ are computed with nested ForwardDiff, which supports
forward-over-forward dual propagation natively.

Deployed MSE of the posterior-mean estimator is evaluated by Monte Carlo.
"""
module PCRB

using Main.ScqubitModel
using Main.Belief
using Main.Gradient
using ForwardDiff
using Random
using Printf
using SpecialFunctions: loggamma
import Main.Bellman: BellmanNode

export fisher_per_shot, fisher_accumulated, bim_prior_averaged,
       argmax_schedule_enumerate, pcrb_baseline,
       deployed_mse_fixed, deployed_mse_adaptive, posterior_mean_fixed,
       log_JP_of_schedule, pcrb_history_record

# ---------------------------------------------------------------
# Fisher information — scalar Φ
# ---------------------------------------------------------------
"""
    fisher_per_shot(Φ, τ, c, ω_d) -> Float64

Per-shot Bernoulli Fisher information J_F(Φ, τ; c).  Differentiable in c.
"""
function fisher_per_shot(phi::Real, tau::Real, c::ScqubitParams, omega_d::Real)
    P = P1_ramsey(phi, tau, c, omega_d)
    dPdphi = ForwardDiff.derivative(φ_ -> P1_ramsey(φ_, tau, c, omega_d), phi)
    dPdphi^2 / (P * (1 - P) + 1e-300)
end

"""
    fisher_accumulated(Φ, schedule, c, ω_d) -> Float64

J_N(Φ, s, c) = Σ_k n_k J_F(Φ, τ_k; c) over a schedule [(τ_k, n_k)].
"""
function fisher_accumulated(phi::Real, schedule::Vector{Tuple{Float64,Int}},
                            c::ScqubitParams, omega_d::Real)
    s = zero(promote_type(typeof(phi), typeof(omega_d)))
    for (τ, n) in schedule
        s += n * fisher_per_shot(phi, τ, c, omega_d)
    end
    s
end

"""
    bim_prior_averaged(schedule, c, ω_d, phi_grid; J_0=1e-4) -> Float64

Mid-point quadrature of J_N(Φ, s, c) on phi_grid + prior-regularization floor.
"""
function bim_prior_averaged(schedule::Vector{Tuple{Float64,Int}},
                            c::ScqubitParams, omega_d::Real,
                            phi_grid::AbstractVector; J_0::Real=1e-4)
    N = length(phi_grid)
    acc = zero(typeof(omega_d))
    for phi in phi_grid
        acc += fisher_accumulated(phi, schedule, c, omega_d)
    end
    J_0 + acc / N
end

# ---------------------------------------------------------------
# Helper: convert (j, ℓ) index schedule → (τ, n) schedule on grid.
# ---------------------------------------------------------------
function index_to_taunschedule(s_idx::Vector{Tuple{Int,Int}},
                               grid::Main.Belief.Grid{J, L}) where {J, L}
    [(Float64(grid.tau_grid[j]), Int(grid.n_grid[ℓ])) for (j, ℓ) in s_idx]
end

# ---------------------------------------------------------------
# Inner schedule search: enumerate all (J·L)^K discrete schedules
# ---------------------------------------------------------------
"""
    argmax_schedule_enumerate(grid, c, ω_d, K; J_0=1e-4) -> (best_sched_idx, best_log_JP)

Enumerate every discrete schedule [(j_k, ℓ_k)]_{k=1..K}.  For each, compute
log J_P.  Return argmax.
"""
function argmax_schedule_enumerate(grid::Main.Belief.Grid{J, L},
                                   c::ScqubitParams, omega_d::Real,
                                   K::Int; J_0::Real=1e-4) where {J, L}
    JL = J * L
    best_val = -Inf
    best_idx = Vector{Tuple{Int,Int}}(undef, K)
    cur_idx  = Vector{Tuple{Int,Int}}(undef, K)
    phi_grid = grid.phi_grid
    for code in 0:(JL^K - 1)
        r = code
        for k in 1:K
            jl = r % JL
            r ÷= JL
            j = (jl ÷ L) + 1
            ℓ = (jl % L) + 1
            cur_idx[k] = (j, ℓ)
        end
        sched = index_to_taunschedule(cur_idx, grid)
        JP = bim_prior_averaged(sched, c, omega_d, phi_grid; J_0=J_0)
        lv = log(JP)
        if lv > best_val
            best_val = lv
            copyto!(best_idx, cur_idx)
        end
    end
    (copy(best_idx), best_val)
end

"""
    log_JP_of_schedule(sched_idx, grid, c, ω_d; J_0=1e-4) -> Float64

log J_P at the discrete index-schedule sched_idx.  Differentiable in c.
"""
function log_JP_of_schedule(sched_idx::Vector{Tuple{Int,Int}},
                            grid::Main.Belief.Grid{J, L},
                            c::ScqubitParams, omega_d::Real;
                            J_0::Real=1e-4) where {J, L}
    sched = index_to_taunschedule(sched_idx, grid)
    log(bim_prior_averaged(sched, c, omega_d, grid.phi_grid; J_0=J_0))
end

# ---------------------------------------------------------------
# PCRB-baseline joint optimization
# ---------------------------------------------------------------
"""
    pcrb_baseline(c0; grid, K, outer_iters, outer_lr, omega_d_fn, cbox)

Alternating inner schedule enumeration + outer Adam on c.
Returns (c_final, sched_final, history).
"""
function pcrb_baseline(c0::ScqubitParams;
                       grid::Main.Belief.Grid,
                       K_epochs::Int = 3,
                       outer_iters::Int = 200,
                       outer_lr::Float64 = 1e-3,
                       omega_d_fn = (c -> omega_q(0.442, c)),
                       cbox = nothing,
                       schedule_reopt_every::Int = 5,
                       verbose::Bool = true,
                       J_0::Real = 1e-4)
    v = c_as_vec(c0)
    # Box-width scale so Adam steps are ≈ lr * box_width (pinned → 0).
    scale = cbox === nothing ? ones(length(v)) : max.(cbox.hi .- cbox.lo, 0.0)
    state = _AdamState(length(v), outer_lr; scale=scale)
    hist = (log_JP = Float64[],
            grad_norm = Float64[],
            c_vec = Vector{Vector{Float64}}(),
            sched = Vector{Vector{Tuple{Int,Int}}}(),
            reopt_iter = Int[])
    c_cur = vec_as_c(v)
    ω_d   = omega_d_fn(c_cur)
    (sched_idx, logJP_init) = argmax_schedule_enumerate(grid, c_cur, ω_d, K_epochs; J_0=J_0)
    push!(hist.reopt_iter, 0)
    verbose && (@printf("[init] log J_P = %+.6f  sched=%s  ω_d=%.3e\n",
                       logJP_init, string(sched_idx), ω_d); flush(stdout))
    for iter in 1:outer_iters
        if (iter - 1) % schedule_reopt_every == 0 && iter > 1
            c_cur = vec_as_c(v)
            ω_d   = omega_d_fn(c_cur)
            (sched_idx, _) = argmax_schedule_enumerate(grid, c_cur, ω_d, K_epochs; J_0=J_0)
            push!(hist.reopt_iter, iter)
        end
        # outer gradient via ForwardDiff (closed-form, no MC noise)
        g = ForwardDiff.gradient(
            v_ -> log_JP_of_schedule(sched_idx, grid, vec_as_c(v_),
                                      omega_d_fn(vec_as_c(v_)); J_0=J_0),
            v)
        gn = sqrt(sum(abs2, g))
        _adam_step!(v, g, state)
        if cbox !== nothing
            for i in eachindex(v)
                v[i] = clamp(v[i], cbox.lo[i], cbox.hi[i])
            end
        end
        cur_c = vec_as_c(v)
        ω_d_cur = omega_d_fn(cur_c)
        lJP = log_JP_of_schedule(sched_idx, grid, cur_c, ω_d_cur; J_0=J_0)
        push!(hist.log_JP, lJP)
        push!(hist.grad_norm, gn)
        push!(hist.c_vec, copy(v))
        push!(hist.sched, copy(sched_idx))
        if verbose && (iter == 1 || iter % max(1, schedule_reopt_every) == 0)
            @printf("iter %4d  log J_P = %+.6f  |g|=%.3e  sched=%s\n",
                    iter, lJP, gn, string(sched_idx))
            flush(stdout)
        end
    end
    c_final = vec_as_c(v)
    (c_final, sched_idx, hist)
end

# ---------------------------------------------------------------
# Deployed MSE of posterior-mean estimator
# ---------------------------------------------------------------
"""
    posterior_mean_fixed(phi_true, sched_idx, grid, c, ω_d; rng, K_phi_post)

Simulate observations from `phi_true` under the FIXED (non-adaptive) schedule
sched_idx (K steps), form the posterior on a K_phi_post-grid, return posterior
mean estimate φ̂.
"""
function posterior_mean_fixed(phi_true::Real, sched_idx::Vector{Tuple{Int,Int}},
                              grid::Main.Belief.Grid{J, L},
                              c::ScqubitParams, omega_d::Real,
                              rng::AbstractRNG) where {J, L}
    # accumulate logb incrementally on the fine grid.
    logb = zeros(Float64, length(grid.phi_grid))
    for (j, ℓ) in sched_idx
        τ  = grid.tau_grid[j]
        n  = grid.n_grid[ℓ]
        p_true = clamp(P1_ramsey(phi_true, τ, c, omega_d), 1e-300, 1 - 1e-16)
        # sample m ~ Binomial(n, p_true)
        m = 0
        for _ in 1:n
            rand(rng) < p_true && (m += 1)
        end
        # update logb on the grid
        @inbounds for i in eachindex(logb)
            p = clamp(P1_ramsey(grid.phi_grid[i], τ, c, omega_d),
                      1e-300, 1 - 1e-16)
            logb[i] += m * log(p) + (n - m) * log1p(-p)
        end
    end
    mx = maximum(logb)
    w  = exp.(logb .- mx)
    Z  = sum(w)
    pgrid = w ./ Z
    sum(pgrid .* grid.phi_grid)
end

"""
    deployed_mse_fixed(c, sched_idx, ω_d, grid; n_mc=10_000, rng)

Prior-averaged MSE of the posterior-mean estimator under the fixed schedule.
Returns (mse, standard error).
"""
function deployed_mse_fixed(c::ScqubitParams, sched_idx::Vector{Tuple{Int,Int}},
                            omega_d::Real, grid::Main.Belief.Grid{J, L};
                            n_mc::Int=10_000,
                            rng::AbstractRNG=MersenneTwister(0)) where {J, L}
    errs = Vector{Float64}(undef, n_mc)
    for t in 1:n_mc
        phi_true = rand(rng) * grid.phi_max
        phi_hat  = posterior_mean_fixed(phi_true, sched_idx, grid, c, omega_d, rng)
        errs[t] = (phi_hat - phi_true)^2
    end
    mse = sum(errs) / n_mc
    se  = sqrt(sum((errs .- mse).^2) / ((n_mc - 1) * n_mc))
    (mse, se)
end

"""
    deployed_mse_adaptive(c, policy_memo, ω_d, grid, K_epochs; n_mc, rng)

Prior-averaged MSE of the posterior-mean estimator under the adaptive policy.
"""
function deployed_mse_adaptive(c::ScqubitParams, policy_memo::Dict,
                               omega_d::Real, grid::Main.Belief.Grid{J, L},
                               K_epochs::Int;
                               n_mc::Int=10_000,
                               rng::AbstractRNG=MersenneTwister(0)) where {J, L}
    errs = Vector{Float64}(undef, n_mc)
    counts0 = ntuple(_ -> (0, 0), J)
    for t in 1:n_mc
        phi_true = rand(rng) * grid.phi_max
        counts = counts0
        logb   = zeros(Float64, length(grid.phi_grid))
        for r in K_epochs:-1:1
            node = get(policy_memo, (counts, r), nothing)
            node === nothing && error("policy memo miss at counts=$counts r=$r")
            j, ℓ = node.action
            τ = grid.tau_grid[j]; n = grid.n_grid[ℓ]
            p_true = clamp(P1_ramsey(phi_true, τ, c, omega_d), 1e-300, 1 - 1e-16)
            m = 0
            for _ in 1:n
                rand(rng) < p_true && (m += 1)
            end
            @inbounds for i in eachindex(logb)
                p = clamp(P1_ramsey(grid.phi_grid[i], τ, c, omega_d),
                          1e-300, 1 - 1e-16)
                logb[i] += m * log(p) + (n - m) * log1p(-p)
            end
            counts = ntuple(k -> k == j ? (counts[k][1] + n, counts[k][2] + m) :
                                          counts[k], J)
        end
        mx = maximum(logb)
        w  = exp.(logb .- mx); Z = sum(w)
        pgrid = w ./ Z
        phi_hat = sum(pgrid .* grid.phi_grid)
        errs[t] = (phi_hat - phi_true)^2
    end
    mse = sum(errs) / n_mc
    se  = sqrt(sum((errs .- mse).^2) / ((n_mc - 1) * n_mc))
    (mse, se)
end

# ---------------------------------------------------------------
# Local Adam (kept local to avoid circular import with JointOpt)
# ---------------------------------------------------------------
mutable struct _AdamState
    lr::Float64; β1::Float64; β2::Float64; ϵ::Float64
    m::Vector{Float64}; v::Vector{Float64}; t::Int
    scale::Vector{Float64}
end
_AdamState(n::Int, lr; scale=ones(n)) =
    _AdamState(lr, 0.9, 0.999, 1e-8, zeros(n), zeros(n), 0, scale)
function _adam_step!(v::AbstractVector, g::AbstractVector, s::_AdamState)
    s.t += 1
    g_scaled = g .* s.scale
    s.m .= s.β1 .* s.m .+ (1 - s.β1) .* g_scaled
    s.v .= s.β2 .* s.v .+ (1 - s.β2) .* g_scaled.^2
    m_hat = s.m ./ (1 - s.β1^s.t)
    v_hat = s.v ./ (1 - s.β2^s.t)
    step_norm = s.lr .* m_hat ./ (sqrt.(v_hat) .+ s.ϵ)
    v .+= step_norm .* s.scale
    v
end

end # module
