"""
Belief.jl

Belief over the unknown normalized flux φ ∈ [0, 0.5] represented on a uniform
K_Φ-grid.  Stores:
  - logb::Vector{T}                 pointwise log-density (unnormalized), K_Φ long
  - counts::NTuple{J, Tuple{Int,Int}}    sufficient statistics for memoization

The counts tuple is a lossless summary of all past observations: the Bernoulli
product likelihood depends only on the total (n_j, m_j) at each delay τ_j, not
on the order.  This makes counts a canonical memoization key for the Bellman DP.

Non-mutating throughout — every update returns a new Belief, safe for Zygote.
"""
module Belief

using Main.ScqubitModel
import Main.ScqubitModel: P1_ramsey, ScqubitParams

export Grid, FluxBelief, make_grid, prior_belief, posterior_logb,
       update_belief, entropy_nats, marg_obs_prob, belief_density

# ---------------------------------------------------------------
# Grid description
# ---------------------------------------------------------------
"""
    Grid{J}

Immutable description of the discretization.
  - phi_grid : Vector{Float64}, length K_Φ, values in (0, phi_max].
              Uses mid-points so boundary singularities are excluded.
  - dphi    : grid spacing.
  - tau_grid: NTuple{J, Float64}, delay options.
  - n_grid  : NTuple{L, Int}, repetition options.
"""
struct Grid{J, L}
    phi_grid::Vector{Float64}
    dphi::Float64
    phi_max::Float64
    tau_grid::NTuple{J, Float64}
    n_grid::NTuple{L, Int}
end

function make_grid(; K_phi::Int=256, phi_max::Float64=0.49,
                     tau_grid::NTuple{J, Float64}=Tuple(10e-9 .* 2.0 .^ (0:5)),
                     n_grid::NTuple{L, Int}=(1, 3, 10, 30)) where {J, L}
    # mid-point rule: avoid φ=0 and φ=phi_max exactly.
    dphi = phi_max / K_phi
    phi_grid = collect(range(dphi/2, phi_max - dphi/2; length=K_phi))
    Grid{J, L}(phi_grid, dphi, phi_max, tau_grid, n_grid)
end

# ---------------------------------------------------------------
# FluxBelief
# ---------------------------------------------------------------
"""
    FluxBelief{J, T}

Belief at a node of the Bellman tree. `counts[j] = (n_j, m_j)` are cumulative
shots/heads at delay index j.
"""
struct FluxBelief{J, T<:Real}
    logb::Vector{T}                        # K_Φ long
    counts::NTuple{J, Tuple{Int, Int}}
end

"Uniform prior over the grid (all logb entries equal)."
function prior_belief(grid::Grid{J, L}) where {J, L}
    logb = zeros(Float64, length(grid.phi_grid))
    counts = ntuple(_ -> (0, 0), J)
    FluxBelief{J, Float64}(logb, counts)
end

# ---------------------------------------------------------------
# Posterior log-density from counts tuple
# ---------------------------------------------------------------
"""
    posterior_logb(grid, counts, c, omega_d; env_phi=nothing)

Recompute logb from the counts tuple (sufficient statistic).  This is fully
differentiable w.r.t. c.  Prefer over incremental update when recomputing on
a fresh c during gradient evaluation.
"""
function posterior_logb(grid::Grid{J, L}, counts::NTuple{J, Tuple{Int, Int}},
                        c::ScqubitParams, omega_d;
                        env_phi=nothing) where {J, L}
    N = length(grid.phi_grid)
    logb = zeros(eltype_base(c), N)       # element type inferred from c
    for j in 1:J
        n_j, m_j = counts[j]
        if n_j == 0
            continue
        end
        τj = grid.tau_grid[j]
        for i in 1:N
            p = P1_ramsey(grid.phi_grid[i], τj, c, omega_d; env_phi=env_phi)
            p = clamp(p, 1e-300, 1 - 1e-16)
            logb[i] += m_j * log(p) + (n_j - m_j) * log1p(-p)
        end
    end
    logb
end

# ---------------------------------------------------------------
# Vectorized, Zygote-friendly variant (no mutation). Returns the logb and the
# corresponding FluxBelief.
# ---------------------------------------------------------------
function posterior_logb_functional(grid::Grid{J, L}, counts::NTuple{J, Tuple{Int, Int}},
                                   c::ScqubitParams, omega_d;
                                   env_phi=nothing) where {J, L}
    N = length(grid.phi_grid)
    φs = grid.phi_grid
    # Start from zero log-density (prior uniform); fold in each delay contribution.
    contribs = map(1:J) do j
        n_j, m_j = counts[j]
        if n_j == 0
            zeros(eltype_base(c), N)
        else
            τj = grid.tau_grid[j]
            ps = [clamp(P1_ramsey(φ, τj, c, omega_d; env_phi=env_phi),
                        1e-300, 1 - 1e-16) for φ in φs]
            m_j .* log.(ps) .+ (n_j - m_j) .* log1p.(-ps)
        end
    end
    sum(contribs; init=zeros(eltype_base(c), N))
end

eltype_base(c::ScqubitParams{T}) where {T} = T

# ---------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------
"""
    update_belief(b, j, n_shots, m_heads, grid, c, omega_d; env_phi=nothing)

Return a fresh FluxBelief after adding n_shots observations at delay τ_j, of
which m_heads were |1⟩.
"""
function update_belief(b::FluxBelief{J, T},
                      j::Int, n_shots::Int, m_heads::Int,
                      grid::Grid{J, L}, c::ScqubitParams, omega_d;
                      env_phi=nothing) where {J, L, T}
    τj = grid.tau_grid[j]
    φs = grid.phi_grid
    ps = [clamp(P1_ramsey(φ, τj, c, omega_d; env_phi=env_phi),
                1e-300, 1 - 1e-16) for φ in φs]
    Δlog = m_heads .* log.(ps) .+ (n_shots - m_heads) .* log1p.(-ps)
    new_logb = b.logb .+ Δlog
    # update counts at position j
    new_counts = ntuple(k -> k == j ?
        (b.counts[k][1] + n_shots, b.counts[k][2] + m_heads) :
        b.counts[k], J)
    FluxBelief{J, promote_type(T, eltype(ps))}(new_logb, new_counts)
end

# ---------------------------------------------------------------
# Entropy (nats)
# H(b) = -∫ b(φ) log b(φ) dφ, with b a density on [0, phi_max].
# Discretized: let p_i be normalized probabilities summing to 1 across grid
# points, b_i = p_i / dphi the density.  Then
#   H = -Σ_i p_i log (p_i / dphi) = -Σ_i p_i log p_i + log dphi.
# The entropy of the uniform prior is log(phi_max), as expected.
# ---------------------------------------------------------------
"""
    entropy_nats(logb, dphi) -> Float64

Shannon entropy in nats of the density represented by `logb` (pointwise
unnormalized log-density) on a uniform grid of spacing `dphi`.
"""
function entropy_nats(logb::AbstractVector, dphi::Real)
    m = maximum(logb)
    w = exp.(logb .- m)
    Z = sum(w)
    p = w ./ Z
    # entropy of discrete distribution p, plus the log-spacing offset so the
    # continuous entropy is recovered.
    H_disc = -sum(p .* log.(clamp.(p, 1e-300, 1.0)))
    H_disc + log(dphi)
end

entropy_nats(b::FluxBelief, grid::Grid) = entropy_nats(b.logb, grid.dphi)

"""
    belief_density(b, grid) -> Vector{Float64}

Normalized density on the grid, summing to 1/dphi along the grid (so that
sum(b) * dphi == 1).
"""
function belief_density(b::FluxBelief, grid::Grid)
    m = maximum(b.logb)
    w = exp.(b.logb .- m)
    Z = sum(w) * grid.dphi
    w ./ Z
end

# ---------------------------------------------------------------
# Marginal observation probability P(m | b, τ_j, n)
# ---------------------------------------------------------------
"""
    marg_obs_prob(b, j, n_shots, m_heads, grid, c, omega_d; env_phi=nothing)

P(m = m_heads | n_shots, τ_j, c) under belief b.  Uses log-sum-exp for
numerical stability.
"""
function marg_obs_prob(b::FluxBelief{J, T},
                       j::Int, n_shots::Int, m_heads::Int,
                       grid::Grid{J, L}, c::ScqubitParams, omega_d;
                       env_phi=nothing) where {J, L, T}
    τj = grid.tau_grid[j]
    φs = grid.phi_grid
    ps = [clamp(P1_ramsey(φ, τj, c, omega_d; env_phi=env_phi),
                1e-300, 1 - 1e-16) for φ in φs]
    # binomial log-probability per grid point, combined with b_i:
    # P(m) = Σ_i b_i * C(n,m) p_i^m (1-p_i)^(n-m) · dphi  (density × dphi = prob mass)
    log_binom_coef = _log_binom(n_shots, m_heads)
    # normalize belief first
    m_lb = maximum(b.logb)
    w = exp.(b.logb .- m_lb)
    Z = sum(w)
    p_grid = w ./ Z          # probabilities per grid point (discrete), sum = 1
    log_lik = m_heads .* log.(ps) .+ (n_shots - m_heads) .* log1p.(-ps) .+ log_binom_coef
    # P(m) = Σ_i p_grid_i * exp(log_lik_i)
    # numerically stable via log-sum-exp of (log p_grid_i + log_lik_i)
    log_integrand = log.(clamp.(p_grid, 1e-300, 1.0)) .+ log_lik
    mx = maximum(log_integrand)
    exp(mx) * sum(exp.(log_integrand .- mx))
end

# Stable log binomial coefficient via lgamma.
@inline _log_binom(n::Integer, k::Integer) =
    _lgamma(n + 1) - _lgamma(k + 1) - _lgamma(n - k + 1)

# Use SpecialFunctions for lgamma if available; fall back to log∘gamma.
# We rely on SpecialFunctions; if not, users can install it.
using SpecialFunctions: loggamma
@inline _lgamma(x) = loggamma(x)

end # module
