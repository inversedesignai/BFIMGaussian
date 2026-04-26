"""
Bellman.jl

Exact backward-induction DP for the scqubit sensor POMDP.

State = (counts_tuple, k_remaining).  Memoization by that key.  The value
function is W(b, r) = max over K-r actions already taken; reward =
-H(b_final) once r = 0.  V_adaptive(c) = W(prior, K) + ln(phi_max).

`solve_bellman_full` returns the full optimal-policy memo, useful for the
gradient phase: at runtime we look up π*(b) by counts-tuple only.
"""
module Bellman

using Main.ScqubitModel
using Main.Belief
using SpecialFunctions: loggamma

export solve_bellman, solve_bellman_full, V_adaptive, V_adaptive_and_policy,
       BellmanNode, expected_logpm

@inline _log_binom(n::Integer, k::Integer) =
    loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

# ---------------------------------------------------------------
# Precompute log p and log(1-p) on the Φ-grid for every (j, i).
# Returns (logp[J, K_Φ], log1mp[J, K_Φ]).
# ---------------------------------------------------------------
function _logp_tables(grid::Main.Belief.Grid{J, L},
                      c::ScqubitParams, omega_d) where {J, L}
    N = length(grid.phi_grid)
    logp   = Array{Float64}(undef, J, N)
    log1mp = Array{Float64}(undef, J, N)
    for j in 1:J, i in 1:N
        p = clamp(P1_ramsey(grid.phi_grid[i], grid.tau_grid[j], c, omega_d),
                  1e-300, 1 - 1e-16)
        logp[j, i]   = log(p)
        log1mp[j, i] = log1p(-p)
    end
    (logp, log1mp)
end

# ---------------------------------------------------------------
# posterior logb from counts (exact, differentiable in c via ScqubitModel).
# For the DP we use Float64 cached logp / log1mp tables; differentiability is
# deferred to the gradient phase which rebuilds logb from counts directly.
# ---------------------------------------------------------------
function _logb_from_counts(counts::NTuple{J, Tuple{Int,Int}},
                           logp::AbstractMatrix,
                           log1mp::AbstractMatrix) where {J}
    Nphi = size(logp, 2)
    logb = zeros(Float64, Nphi)
    for j in 1:J
        n_j, m_j = counts[j]
        n_j == 0 && continue
        @inbounds for i in 1:Nphi
            logb[i] += m_j * logp[j, i] + (n_j - m_j) * log1mp[j, i]
        end
    end
    logb
end

function _entropy_from_logb(logb::Vector{Float64}, dphi::Float64)
    m = maximum(logb)
    w = exp.(logb .- m)
    Z = sum(w)
    p = w ./ Z
    H_disc = -sum(p[i] > 0 ? p[i] * log(p[i]) : 0.0 for i in eachindex(p))
    H_disc + log(dphi)
end

"Posterior variance Var_b(Φ) = E[Φ²] − E[Φ]² given logb on grid. Φ₀² units."
function _variance_from_logb(logb::Vector{Float64}, phi_grid::Vector{Float64})
    m = maximum(logb)
    w = exp.(logb .- m)
    Z = sum(w)
    p = w ./ Z
    μ  = sum(p .* phi_grid)
    σ2 = sum(p .* (phi_grid .- μ).^2)
    σ2
end

"Terminal reward at a belief leaf. `terminal` ∈ (:mi, :mse). MI uses −H (plus dphi offset so V_adaptive = ln(phi_max) − E[H]); MSE uses −Var (so V_adaptive = −E[Var_post] = −Bayesian MSE of the post-mean estimator)."
function _terminal_reward(logb::Vector{Float64}, grid::Main.Belief.Grid, terminal::Symbol)
    if terminal === :mi
        return -_entropy_from_logb(logb, grid.dphi)
    elseif terminal === :mse
        return -_variance_from_logb(logb, grid.phi_grid)
    else
        error("unknown terminal reward :$(terminal); use :mi or :mse")
    end
end

# ---------------------------------------------------------------
# Marginal observation prob P(m|b, j, n) = Σ_i p_grid_i Binom(m;n,p_ji)
# using pre-normalized p_grid from logb (numerically stable log-sum-exp).
# ---------------------------------------------------------------
function _marg_obs(logb::Vector{Float64}, j::Int, n::Int, m::Int,
                   logp::AbstractMatrix, log1mp::AbstractMatrix)
    mx_lb = maximum(logb)
    w = exp.(logb .- mx_lb)
    Z = sum(w)
    log_bc = _log_binom(n, m)
    s = 0.0
    @inbounds for i in eachindex(w)
        s += w[i] * exp(log_bc + m * logp[j, i] + (n - m) * log1mp[j, i])
    end
    s / Z
end

# ---------------------------------------------------------------
# Core recursion with memoization
# ---------------------------------------------------------------
"""
    BellmanNode

Value-and-action at a belief node.  action = (j, ℓ) for the optimal delay and
repetition index at this node.
"""
struct BellmanNode
    value::Float64
    action::Tuple{Int, Int}        # (0, 0) at terminal
end

"""
    solve_bellman(b0_counts, K, grid, c, ω_d) -> (W, memo::Dict{(counts, r) => BellmanNode})

Solve the Bellman DP starting from the prior (counts all zero) with K steps
remaining.  Returns the value W and the memo dictionary.
"""
function solve_bellman(K::Int,
                       grid::Main.Belief.Grid{J, L},
                       c::ScqubitParams, omega_d;
                       terminal::Symbol=:mi) where {J, L}
    (logp, log1mp) = _logp_tables(grid, c, omega_d)
    memo = Dict{Tuple{NTuple{J, Tuple{Int,Int}}, Int}, BellmanNode}()
    counts0 = ntuple(_ -> (0, 0), J)
    val = _bellman_rec(counts0, K, grid, logp, log1mp, memo, terminal)
    (val.value, memo)
end

function _bellman_rec(counts::NTuple{J, Tuple{Int,Int}},
                      r::Int,
                      grid::Main.Belief.Grid{J, L},
                      logp::AbstractMatrix,
                      log1mp::AbstractMatrix,
                      memo::Dict,
                      terminal::Symbol=:mi) where {J, L}
    key = (counts, r)
    hit = get(memo, key, nothing)
    hit === nothing || return hit
    logb = _logb_from_counts(counts, logp, log1mp)
    if r == 0
        val = _terminal_reward(logb, grid, terminal)
        node = BellmanNode(val, (0, 0))
        memo[key] = node
        return node
    end
    best_val = -Inf
    best_act = (0, 0)
    for j in 1:J, ℓ in 1:L
        n = grid.n_grid[ℓ]
        val = 0.0
        for m in 0:n
            # marginal obs prob
            p_m = _marg_obs(logb, j, n, m, logp, log1mp)
            if p_m < 1e-20
                continue
            end
            # successor counts
            new_counts = ntuple(k -> k == j ?
                (counts[k][1] + n, counts[k][2] + m) :
                counts[k], J)
            sub = _bellman_rec(new_counts, r - 1, grid, logp, log1mp, memo, terminal)
            val += p_m * sub.value
        end
        if val > best_val
            best_val = val
            best_act = (j, ℓ)
        end
    end
    node = BellmanNode(best_val, best_act)
    memo[key] = node
    node
end

# ---------------------------------------------------------------
# V_adaptive(c)
# ---------------------------------------------------------------
"""
    V_adaptive(grid, K, c, ω_d) -> Float64

W₁(prior) + ln(phi_max).  The prior is uniform on [0, phi_max].
"""
function V_adaptive(grid::Main.Belief.Grid{J, L}, K::Int,
                    c::ScqubitParams, omega_d;
                    terminal::Symbol=:mi) where {J, L}
    (W, _memo) = solve_bellman(K, grid, c, omega_d; terminal=terminal)
    # For :mi we add ln(phi_max) so V_adaptive = I(Φ;y) in nats;
    # for :mse we return −E[Var_post], the (negative) Bayesian MSE.
    terminal === :mi ? (W + log(grid.phi_max)) : W
end

"""
    V_adaptive_and_policy(grid, K, c, ω_d) -> (V, memo)

Returns the adaptive value PLUS the full policy memo so downstream code can
run rollouts under π*.
"""
function V_adaptive_and_policy(grid::Main.Belief.Grid{J, L}, K::Int,
                               c::ScqubitParams, omega_d;
                               terminal::Symbol=:mi) where {J, L}
    (W, memo) = solve_bellman(K, grid, c, omega_d; terminal=terminal)
    v = terminal === :mi ? W + log(grid.phi_max) : W
    (v, memo)
end

"""
    solve_bellman_full(grid, K, c, ω_d) -> (V_adaptive, memo, statistics)

Convenience wrapper with statistics on memo size.
"""
function solve_bellman_full(grid::Main.Belief.Grid{J, L}, K::Int,
                            c::ScqubitParams, omega_d;
                            terminal::Symbol=:mi) where {J, L}
    t0 = time()
    (W, memo) = solve_bellman(K, grid, c, omega_d; terminal=terminal)
    elapsed = time() - t0
    stats = (memo_size = length(memo), elapsed = elapsed)
    v = terminal === :mi ? W + log(grid.phi_max) : W
    (v, memo, stats)
end

# ---------------------------------------------------------------
# Policy lookup API
# ---------------------------------------------------------------
"""
    policy_action(memo, counts, r_remaining) -> (j, ℓ) or (0, 0) if r=0 or miss
"""
function policy_action(memo::Dict,
                       counts::NTuple{J, Tuple{Int,Int}},
                       r::Int) where {J}
    node = get(memo, (counts, r), nothing)
    node === nothing ? (0, 0) : node.action
end

export policy_action

end # module
