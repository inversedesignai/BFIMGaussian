"""
Baselines.jl

Exact enumeration of V_oracle(φ) and V_fixed(c) for the scqubit problem.

Conventions:
  V_oracle(φ, c) = ln(phi_max) − min_schedule E_{obs|φ, schedule}[H(b_K)]
  V_fixed(c)    = ln(phi_max) − min_schedule E_φ E_{obs|φ, schedule}[H(b_K)]

where `schedule = ((j_1, ℓ_1), ..., (j_K, ℓ_K))` is a fixed (non-adaptive)
sequence of (delay-index, repetition-index) pairs.

The enumeration scales as (J·L)^K × <∏(n+1)> × K_Φ and is intended for small
K (≤ 4) and moderate J·L (≤ 24).  For K=3, J=4, L=2 and K_Φ=128 this runs in
seconds per c on a single core.
"""
module Baselines

using Main.ScqubitModel
using Main.Belief
import Main.Belief: posterior_logb, entropy_nats, belief_density

export enumerate_schedules, Phi_value, V_oracle, V_fixed,
       V_oracle_mean, Schedule

const Schedule = Vector{Tuple{Int,Int}}   # (j, ℓ) pairs

"All (J·L)^K schedules (as Schedule vectors)."
function enumerate_schedules(grid::Main.Belief.Grid{J, L}, K::Int) where {J, L}
    JL = J * L
    schedules = Vector{Schedule}(undef, JL^K)
    for idx in 1:JL^K
        s = Vector{Tuple{Int,Int}}(undef, K)
        r = idx - 1
        for k in 1:K
            jl = r % JL
            r = r ÷ JL
            j = (jl ÷ L) + 1
            ℓ = (jl % L) + 1
            s[k] = (j, ℓ)
        end
        schedules[idx] = s
    end
    schedules
end

# ---------------------------------------------------------------
# Log-likelihood grid caches
# ---------------------------------------------------------------
"""
    logp_cache(grid, c, ω_d) -> (logp, log1mp) arrays of size (J, K_Φ)

Precompute log P_|1⟩(φ_i, τ_j) and log(1 - P_|1⟩) over the Φ-grid for every
delay j.  Saves a factor of ~∏(n+1) × K over recomputing inside each enumeration.
"""
function logp_cache(grid::Main.Belief.Grid{J, L},
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
# Φ_value(φ, schedule, c) = E_{obs|φ, schedule}[ln(phi_max) - H(b_K)]
# Enumerate all observation trajectories.
# ---------------------------------------------------------------
"""
Enumerate observation tuples `(m_1, ..., m_K)` where m_k ∈ 0:n_{ℓ_k}.
`nks = [n_1, ..., n_K]`. Runs a depth-first walk to keep logb updates
minimal and avoid reconstructing from counts every step.
"""
function Phi_value(phi_idx::Int, schedule::Schedule,
                   grid::Main.Belief.Grid{J, L},
                   nks::Vector{Int},
                   logp_full::AbstractMatrix,
                   log1mp_full::AbstractMatrix) where {J, L}
    # For a specific φ (the true flux at index phi_idx on the grid),
    # the likelihood of each observation m_k is Binomial(n_k, P_|1⟩(φ, τ_k)).
    # We enumerate (m_1,...,m_K) in a tree walk, maintaining logb over grid
    # and accumulating P(obs | φ) × (ln phi_max - H(b_K)).
    K = length(schedule)
    Nphi = length(grid.phi_grid)
    logb = zeros(Float64, Nphi)
    # running product of Binomial pmf values at true φ
    log_prob_obs = 0.0
    # log binomial coefficient running sum
    val = Ref(0.0)
    # log(phi_max) is added back outside
    # precompute log p_true and log(1-p_true) at the true flux
    log_p_true = Vector{Float64}(undef, K)
    log_1mp_true = Vector{Float64}(undef, K)
    for k in 1:K
        j = schedule[k][1]
        log_p_true[k] = logp_full[j, phi_idx]
        log_1mp_true[k] = log1mp_full[j, phi_idx]
    end
    # recursive walk
    function walk(k::Int, logb::Vector{Float64}, log_prob_obs::Float64)
        if k > K
            # compute posterior entropy
            H = entropy_nats(logb, grid.dphi)
            val[] += exp(log_prob_obs) * (log(grid.phi_max) - H)
            return
        end
        j, ℓ = schedule[k]
        n_k = nks[k]
        for m_k in 0:n_k
            # contribution at true φ: log P(m_k | n_k, p_true_k) =
            #   log C(n,m) + m log p + (n-m) log(1-p)
            log_binom = _log_binom(n_k, m_k)
            log_p = log_binom + m_k * log_p_true[k] + (n_k - m_k) * log_1mp_true[k]
            # incremental posterior update across the grid
            # logb_new[i] = logb[i] + m_k log p_ji + (n_k - m_k) log(1-p_ji)
            logb_new = copy(logb)
            @inbounds for i in 1:Nphi
                logb_new[i] += m_k * logp_full[j, i] +
                               (n_k - m_k) * log1mp_full[j, i]
            end
            walk(k + 1, logb_new, log_prob_obs + log_p)
        end
    end
    walk(1, logb, 0.0)
    val[]
end

# Stable lgamma-based binomial
using SpecialFunctions: loggamma
@inline _log_binom(n::Integer, k::Integer) =
    loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

# ---------------------------------------------------------------
# V_oracle(φ, c) and V_fixed(c)
# ---------------------------------------------------------------
"""
    V_oracle(phi_idx, grid, K, c, ω_d, schedules, logp, log1mp)

V_oracle at a given φ-grid index.  Returns (value, best_schedule).
"""
function V_oracle(phi_idx::Int,
                  grid::Main.Belief.Grid{J, L},
                  K::Int,
                  c::ScqubitParams, omega_d,
                  schedules::Vector{Schedule},
                  logp::AbstractMatrix, log1mp::AbstractMatrix) where {J, L}
    best_val = -Inf
    best_sched = schedules[1]
    for s in schedules
        nks = [grid.n_grid[s[k][2]] for k in 1:K]
        v = Phi_value(phi_idx, s, grid, nks, logp, log1mp)
        if v > best_val
            best_val  = v
            best_sched = s
        end
    end
    (best_val, best_sched)
end

"""
    V_fixed(grid, K, c, ω_d, schedules, logp, log1mp)

V_fixed = max over schedules of (1/K_Φ) Σ_i Phi_value(i, schedule, ...).
Returns (value, best_schedule, Phi_per_phi::Vector{Float64}).
"""
function V_fixed(grid::Main.Belief.Grid{J, L},
                 K::Int,
                 c::ScqubitParams, omega_d,
                 schedules::Vector{Schedule},
                 logp::AbstractMatrix, log1mp::AbstractMatrix) where {J, L}
    Nphi = length(grid.phi_grid)
    best_val = -Inf
    best_sched = schedules[1]
    best_phis = zeros(Nphi)
    for s in schedules
        nks = [grid.n_grid[s[k][2]] for k in 1:K]
        phi_vals = zeros(Nphi)
        for i in 1:Nphi
            phi_vals[i] = Phi_value(i, s, grid, nks, logp, log1mp)
        end
        mean_val = sum(phi_vals) / Nphi
        if mean_val > best_val
            best_val = mean_val
            best_sched = s
            best_phis = phi_vals
        end
    end
    (best_val, best_sched, best_phis)
end

"""
    V_oracle_mean(grid, K, c, ω_d, schedules, logp, log1mp) -> Float64

Average of V_oracle(φ_i) over the Φ-grid.  Used for E[IG] identity.
Threaded over the Φ-grid.
"""
function V_oracle_mean(grid::Main.Belief.Grid{J, L},
                       K::Int,
                       c::ScqubitParams, omega_d,
                       schedules::Vector{Schedule},
                       logp::AbstractMatrix, log1mp::AbstractMatrix) where {J, L}
    Nphi = length(grid.phi_grid)
    vals = Vector{Float64}(undef, Nphi)
    Threads.@threads for i in 1:Nphi
        vals[i] = V_oracle(i, grid, K, c, omega_d, schedules, logp, log1mp)[1]
    end
    sum(vals) / Nphi
end

"""
    V_oracle_allphi(grid, K, c, ω_d, schedules, logp, log1mp)

V_oracle(φ_i) and best schedule for every φ_i on the grid, threaded.
Returns (values::Vector{Float64}, best_schedules::Vector{Schedule}).
"""
function V_oracle_allphi(grid::Main.Belief.Grid{J, L},
                         K::Int,
                         c::ScqubitParams, omega_d,
                         schedules::Vector{Schedule},
                         logp::AbstractMatrix,
                         log1mp::AbstractMatrix) where {J, L}
    Nphi = length(grid.phi_grid)
    vals = Vector{Float64}(undef, Nphi)
    bests = Vector{Schedule}(undef, Nphi)
    Threads.@threads for i in 1:Nphi
        vals[i], bests[i] = V_oracle(i, grid, K, c, omega_d, schedules, logp, log1mp)
    end
    (vals, bests)
end

export V_oracle_allphi

end # module
