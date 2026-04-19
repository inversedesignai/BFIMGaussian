"""
Gradient.jl

Envelope-theorem gradient ∂V_adaptive/∂c under a fixed Bellman policy π*.

At the optimal policy, Danskin's theorem gives
    dV_adaptive/dc = ∂V^{π*}/∂c
and the right-hand side is well-defined as long as we hold the policy fixed in
the memo and let only the observation probabilities and the terminal entropy
depend on c.

Two paths:
  1. `V_adaptive_policy_exact(c, memo, grid, ω_d, K)` — exact expectation over
     observation trees.  Deterministic, differentiable, low overhead for small
     K (≤ 3) and moderate n_grid.  Used by the FD gradient check.
  2. `V_adaptive_policy_mc(c, memo, grid, ω_d, K; n_traj, rng)` — Monte-Carlo
     pathwise estimator with score-function correction.  Used for larger K
     where exact enumeration is infeasible.

Both paths are Zygote-compatible (no in-place mutation, no `Dict` allocation
inside differentiated code — the memo is read-only and dict lookups return
discrete actions / integers).
"""
module Gradient

using Main.ScqubitModel
using Main.Belief
using Main.Bellman
using SpecialFunctions: loggamma
using Random
using Zygote
using Zygote: @ignore_derivatives

export V_adaptive_policy_exact,
       grad_c_exact, grad_c_mc,
       c_as_vec, vec_as_c, C_FIELD_NAMES, C_DIM,
       rollout_mc

# ---------------------------------------------------------------
# Tier-1 c ↔ Vector{Float64} translation (fixed field ordering)
# ---------------------------------------------------------------
const C_FIELD_NAMES = (:f_q_max, :E_C_over_h, :kappa, :Delta_qr,
                       :temperature, :A_phi, :A_Ic)
const C_DIM = length(C_FIELD_NAMES)

"Extract the 7 differentiable scalar fields from a ScqubitParams."
c_as_vec(c::ScqubitParams) = [c.f_q_max, c.E_C_over_h, c.kappa, c.Delta_qr,
                              c.temperature, c.A_phi, c.A_Ic]

"Build a ScqubitParams from a 7-vector, keeping the fixed-geometry defaults.
Element type inferred from `v` so Zygote / ForwardDiff can trace through it."
function vec_as_c(v::AbstractVector{T}) where {T<:Real}
    ScqubitParams{T}(
        f_q_max     = v[1],
        E_C_over_h  = v[2],
        kappa       = v[3],
        Delta_qr    = v[4],
        temperature = v[5],
        A_phi       = v[6],
        A_Ic        = v[7],
    )
end

# ---------------------------------------------------------------
# Log Binomial coefficient
# ---------------------------------------------------------------
@inline _log_binom(n::Integer, k::Integer) =
    loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

# ---------------------------------------------------------------
# Zygote-friendly posterior logb from counts (functional form).
# Avoids the in-place mutation used by Bellman._logb_from_counts.
# Returns a Vector{T} matching the eltype of `c`.
# ---------------------------------------------------------------
function _logb_from_counts_diff(counts::NTuple{J, Tuple{Int,Int}},
                                c::ScqubitParams{T}, omega_d,
                                grid::Main.Belief.Grid{J, L}) where {T, J, L}
    φs = grid.phi_grid
    N = length(φs)
    # Per-j contribution vector, sum them up.  Skip j's with n_j = 0 to avoid
    # wasted P1_ramsey evaluations (these contributions are identically zero).
    contribs = map(1:J) do j
        n_j, m_j = counts[j]
        if n_j == 0
            zeros(T, N)
        else
            τj = grid.tau_grid[j]
            ps = [clamp(P1_ramsey(φ, τj, c, omega_d), 1e-300, 1 - 1e-16)
                  for φ in φs]
            m_j .* log.(ps) .+ (n_j - m_j) .* log1p.(-ps)
        end
    end
    # J ≥ 1 always, so sum without init is well-defined.
    sum(contribs)
end

# ---------------------------------------------------------------
# Marginal observation probability from an already-computed logb.
# Differentiable in c (logb carries the c-dependence).
# ---------------------------------------------------------------
function _marg_obs_from_logb(logb::AbstractVector,
                             j::Int, n::Int, m::Int,
                             c::ScqubitParams, omega_d,
                             grid::Main.Belief.Grid{J, L}) where {J, L}
    φs = grid.phi_grid
    τj = grid.tau_grid[j]
    ps = [clamp(P1_ramsey(φ, τj, c, omega_d), 1e-300, 1 - 1e-16) for φ in φs]
    mx = maximum(logb)
    w  = exp.(logb .- mx)
    Z  = sum(w)
    log_bc = _log_binom(n, m)
    # P(m) = (1/Z) Σ_i w_i · Binom(m; n, p_i)
    bincoeff = exp(log_bc)
    terms = w .* (bincoeff .* ps.^m .* (1 .- ps).^(n - m))
    sum(terms) / Z
end

# Entropy (nats) — Zygote-friendly, matches Belief.entropy_nats.
function _entropy_nats_diff(logb::AbstractVector, dphi::Real)
    mx = maximum(logb)
    w  = exp.(logb .- mx)
    Z  = sum(w)
    p  = w ./ Z
    # entropy of discrete distribution + log(dphi) offset
    H_disc = -sum(p .* log.(clamp.(p, 1e-300, 1.0)))
    H_disc + log(dphi)
end

# Posterior variance (Zygote-friendly).  Returns E[(Φ-μ)²].
function _variance_diff(logb::AbstractVector, phi_grid::AbstractVector)
    mx = maximum(logb)
    w  = exp.(logb .- mx)
    Z  = sum(w)
    p  = w ./ Z
    μ  = sum(p .* phi_grid)
    sum(p .* (phi_grid .- μ).^2)
end

# Terminal reward at a belief leaf, differentiable in c.  :mi ⇒ −H; :mse ⇒ −Var.
function _terminal_diff(logb::AbstractVector, grid::Main.Belief.Grid, terminal::Symbol)
    if terminal === :mi
        return -_entropy_nats_diff(logb, grid.dphi)
    elseif terminal === :mse
        return -_variance_diff(logb, grid.phi_grid)
    else
        error("unknown terminal :$(terminal); use :mi or :mse")
    end
end

# ---------------------------------------------------------------
# Exact recursive value under a fixed policy
# ---------------------------------------------------------------
"""
    _value_rec_exact(c, memo, counts, r, grid, ω_d) -> value (nats)

Recursive policy-tree evaluation.  `memo[(counts, r)].action` supplies the
fixed (j, ℓ) at this node (looked up via @ignore_derivatives — non-differentiable).
At leaves (r == 0) we return `-H(b_r; c)` with b_r = posterior from counts
under current c.
"""
function _value_rec_exact(c::ScqubitParams, memo::Dict,
                          counts::NTuple{J, Tuple{Int,Int}}, r::Int,
                          grid::Main.Belief.Grid{J, L}, omega_d;
                          terminal::Symbol=:mi) where {J, L}
    logb = _logb_from_counts_diff(counts, c, omega_d, grid)
    if r == 0
        return _terminal_diff(logb, grid, terminal)
    end
    action = @ignore_derivatives begin
        node = get(memo, (counts, r), nothing)
        node === nothing ? error("policy memo miss at counts=$counts r=$r") : node.action
    end
    j, ℓ = action
    n = grid.n_grid[ℓ]
    # Sum p_m · sub-value over m = 0:n.
    # Use Julia's sum with generator — Zygote handles it fine.
    sum(0:n) do m
        p_m = _marg_obs_from_logb(logb, j, n, m, c, omega_d, grid)
        new_counts = @ignore_derivatives begin
            ntuple(k -> k == j ? (counts[k][1] + n, counts[k][2] + m) : counts[k], J)
        end
        sub = _value_rec_exact(c, memo, new_counts, r - 1, grid, omega_d; terminal=terminal)
        p_m * sub
    end
end

"""
    V_adaptive_policy_exact(c, memo, grid, ω_d, K) -> Float64

V_adaptive(c) = log(phi_max) + E_{traj ~ π*}[-H(b_K; c)] under the fixed
policy in `memo`.  Differentiable in c.
"""
function V_adaptive_policy_exact(c::ScqubitParams, memo::Dict,
                                 grid::Main.Belief.Grid{J, L}, omega_d,
                                 K::Int; terminal::Symbol=:mi) where {J, L}
    counts0 = @ignore_derivatives ntuple(_ -> (0, 0), J)
    v = _value_rec_exact(c, memo, counts0, K, grid, omega_d; terminal=terminal)
    terminal === :mi ? log(grid.phi_max) + v : v
end

"""
    grad_c_exact(c_vec, memo, grid, ω_d, K) -> Vector{Float64}

Zygote gradient of V_adaptive_policy_exact w.r.t. the 7-vector c.
"""
function grad_c_exact(c_vec::AbstractVector, memo::Dict,
                      grid::Main.Belief.Grid{J, L}, omega_d, K::Int;
                      terminal::Symbol=:mi) where {J, L}
    Zygote.gradient(v -> V_adaptive_policy_exact(vec_as_c(v), memo, grid, omega_d, K; terminal=terminal),
                    c_vec)[1]
end

# ---------------------------------------------------------------
# Monte Carlo rollout (for larger problems)
# ---------------------------------------------------------------
"""
    rollout_mc(c, memo, grid, ω_d, K, rng) -> (final_counts, log_obs_prob, phi_true)

Simulate one trajectory under the fixed policy π*:
  1. sample φ_true ~ Uniform[0, phi_max]
  2. for k = 1..K: action ← memo; draw m_k ~ Binomial(n_k, P₁(φ_true, τ_k; c))
  3. accumulate log P(obs | φ_true, c) along the path and track final counts.

The draws are SAMPLED at the current c but the returned quantities (counts and
log_obs_prob) are used in `grad_c_mc` to form the pathwise + score-function
estimator.
"""
function rollout_mc(c::ScqubitParams, memo::Dict,
                    grid::Main.Belief.Grid{J, L}, omega_d, K::Int,
                    rng::AbstractRNG) where {J, L}
    counts = ntuple(_ -> (0, 0), J)
    φ_true = rand(rng) * grid.phi_max
    log_obs = 0.0
    for r in K:-1:1
        node = get(memo, (counts, r), nothing)
        node === nothing && error("policy memo miss at counts=$counts r=$r")
        j, ℓ = node.action
        n = grid.n_grid[ℓ]
        τ = grid.tau_grid[j]
        p_true = clamp(P1_ramsey(φ_true, τ, c, omega_d), 1e-300, 1 - 1e-16)
        # sample m ~ Binomial(n, p_true) by n Bernoulli draws (small n here)
        m = 0
        for _ in 1:n
            rand(rng) < p_true && (m += 1)
        end
        log_obs += _log_binom(n, m) + m*log(p_true) + (n - m)*log1p(-p_true)
        counts = ntuple(k -> k == j ? (counts[k][1] + n, counts[k][2] + m) : counts[k], J)
    end
    (counts, log_obs, φ_true)
end

"""
    grad_c_mc(c, memo, grid, ω_d, K; n_traj, rng) -> Vector{Float64}

Pathwise Monte Carlo estimate of ∂V_adaptive/∂c at the fixed policy.

Estimator:
    ∇ = E_{traj ~ π*}[ ∂(-H(b_K; c))/∂c   +   (-H(b_K; c)) · ∂ log L(traj|c)/∂c ]
where log L is the observation log-probability along the trajectory.  The first
term is a pathwise contribution (posterior depends on c); the second is the
REINFORCE correction for the trajectory distribution.

Returns the 7-vector estimate and the estimated standard error (per component).
"""
function grad_c_mc(c_vec::AbstractVector, memo::Dict,
                   grid::Main.Belief.Grid{J, L}, omega_d_of_c::Function,
                   K::Int; n_traj::Int=1000, rng::AbstractRNG=MersenneTwister(0),
                   phi_star_of_c::Function=(_c)->0.442) where {J, L}
    c_base = vec_as_c(c_vec)
    ω_d_base = omega_d_of_c(c_base)
    grads = Vector{Vector{Float64}}(undef, n_traj)
    fvals = Vector{Float64}(undef, n_traj)
    for t in 1:n_traj
        (counts, log_obs_base, φ_true) = rollout_mc(c_base, memo, grid, ω_d_base, K, rng)
        # --- pathwise term: ∂/∂c [ -H(b_K(counts, c)) ] ---
        path_grad = Zygote.gradient(c_vec) do v
            cc = vec_as_c(v)
            ω_d = omega_d_of_c(cc)
            logb = _logb_from_counts_diff(counts, cc, ω_d, grid)
            -_entropy_nats_diff(logb, grid.dphi)
        end[1]
        # --- score-function term: f · ∂ log L(traj|c)/∂c ---
        # log L = Σ_k log Binom(m_k; n_k, P₁(φ_true, τ_k; c))
        # We only need ∂ log L/∂c evaluated at (counts, φ_true, c_base).
        # Accumulate log L as a differentiable function of c, then Zygote.
        # Since `counts` encodes (n_k, m_k) at each delay j, log L is a function
        # of the summed (n_j, m_j) pairs at the TRUE flux φ_true.
        score_grad = Zygote.gradient(c_vec) do v
            cc = vec_as_c(v)
            ω_d = omega_d_of_c(cc)
            logL = zero(eltype(v))
            for j in 1:J
                n_j, m_j = counts[j]
                n_j == 0 && continue
                p = clamp(P1_ramsey(φ_true, grid.tau_grid[j], cc, ω_d),
                          1e-300, 1 - 1e-16)
                logL += m_j * log(p) + (n_j - m_j) * log1p(-p)
            end
            logL
        end[1]
        # value f(traj, c_base) = -H(b_K(counts, c_base)) evaluated non-diff
        ω_d = ω_d_base
        logb_base = _logb_from_counts_diff(counts, c_base, ω_d, grid)
        f_val = -_entropy_nats_diff(logb_base, grid.dphi)
        grads[t] = path_grad .+ f_val .* score_grad
        fvals[t] = f_val
    end
    # mean and per-component std error
    g_mat = reduce(hcat, grads)           # 7 × n_traj
    g_mean = vec(sum(g_mat; dims=2) ./ n_traj)
    g_var  = vec(sum((g_mat .- g_mean) .^ 2; dims=2) ./ (n_traj - 1))
    g_se   = sqrt.(g_var ./ n_traj)
    (g_mean, g_se, fvals)
end

end # module
