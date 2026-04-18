"""
Joint geometry-policy sweep for the radar case study.

Replaces the top-hat beam of compute_ig.jl / compute_dp.jl with a
Gaussian beam of continuous beamwidth w:

    G_w(α) = exp(−α² / (2 w²))    (α wrapped to [−π, π])
    p(y=1 | z, a; w) = p_FA + (p_D − p_FA) · G_w(θ_a − φ(z))

The geometry variable c = w is continuous in (0, π].  We scan w on a
grid and, for each w, compute:
    V_oracle(w)   = max over sensor sequences of Φ(x=1, s, w)
                    (constant over x by D_K symmetry; use x = 1)
    V_fixed(w)    = max over s of (1/K) Σ_x Φ(x, s, w)
                    (enumerate over (s_2,s_3,s_4) with s_1 = 1 fixed
                     — orbit-representative reduction, 16x speedup)
    V_adaptive(w) = exact Bellman DP from the uniform prior
    E[IG](w)      = V_oracle − V_fixed          (EVPI upper bound)
    realized gain = V_adaptive − V_fixed        (adaptive advantage)
    saturation    = realized / E[IG]

Then we find w* = argmax V_adaptive(w) and verify the envelope-theorem
gradient at the optimum.

Runtime target: about 1 minute on a laptop for a 20-point sweep.
"""

using Printf
using Serialization

const K = 16
const N = 4
const p_D = 0.9
const p_FA = 0.05
const LNK = log(K)

φ(z) = 2π * (z - 1) / K

# wrap angle to [-π, π]
function wrap(α)
    α = mod(α + π, 2π) - π
    α
end

# Gaussian beam gain at angular offset α
gain(α, w) = exp(-α^2 / (2 * w^2))

# per-cell detection probability for pointing at cell a with beamwidth w
function p_detect(z, a, w)
    α = wrap(φ(a) - φ(z))
    g = gain(α, w)
    p_FA + (p_D - p_FA) * g
end

# --- posterior/entropy ---
function posterior(s::Vector{Int}, y::Vector{Int}, w::Float64)
    logb = zeros(K)
    for z in 1:K, k in 1:N
        pk = p_detect(z, s[k], w)
        logb[z] += log(y[k] == 1 ? pk : 1 - pk)
    end
    m = maximum(logb)
    b = exp.(logb .- m)
    b ./= sum(b)
    b
end

function entropy(b)
    H = 0.0
    for z in 1:K
        H -= b[z] > 0 ? b[z] * log(b[z]) : 0.0
    end
    H
end

function py_given_xs(y::Vector{Int}, x::Int, s::Vector{Int}, w::Float64)
    p = 1.0
    for k in 1:N
        pk = p_detect(x, s[k], w)
        p *= (y[k] == 1 ? pk : 1 - pk)
    end
    p
end

function Φ_value(x::Int, s::Vector{Int}, w::Float64)
    val = 0.0
    for bits in 0:(2^N - 1)
        y = [((bits >> (k-1)) & 1) for k in 1:N]
        pyx = py_given_xs(y, x, s, w)
        if pyx > 0
            val += pyx * (LNK - entropy(posterior(s, y, w)))
        end
    end
    val
end

# --- oracle (symmetry-reduced: use x = 1 only) ---
function V_oracle_w(w::Float64)
    best = -Inf
    for s1 in 1:K, s2 in 1:K, s3 in 1:K, s4 in 1:K
        v = Φ_value(1, [s1, s2, s3, s4], w)
        if v > best
            best = v
        end
    end
    best
end

# --- fixed (symmetry-reduced: fix s_1 = 1; search (s_2, s_3, s_4) ---
function V_fixed_w(w::Float64)
    best = -Inf
    best_s = [1, 1, 1, 1]
    for s2 in 1:K, s3 in 1:K, s4 in 1:K
        s = [1, s2, s3, s4]
        E = 0.0
        for x in 1:K
            E += Φ_value(x, s, w) / K
        end
        if E > best
            best = E
            best_s = copy(s)
        end
    end
    (best, best_s)
end

# --- Bellman DP (as in compute_dp.jl) ---
function bayes_update(b, a, y, w)
    b_new = zeros(K)
    for z in 1:K
        pk = p_detect(z, a, w)
        b_new[z] = b[z] * (y == 1 ? pk : 1 - pk)
    end
    s = sum(b_new)
    if s > 0
        b_new ./= s
    end
    (b_new, s)
end

function bellman(b::Vector{Float64}, r::Int, w::Float64)
    if r == 0
        return LNK - entropy(b)
    end
    best = -Inf
    for a in 1:K
        val = 0.0
        for y in 0:1
            (b_new, p_y) = bayes_update(b, a, y, w)
            if p_y > 0
                val += p_y * bellman(b_new, r - 1, w)
            end
        end
        if val > best
            best = val
        end
    end
    best
end

V_adaptive_w(w::Float64) = bellman(ones(K) / K, N, w)

# --- sweep ---
function sweep(w_grid)
    n = length(w_grid)
    V_or = zeros(n)
    V_ad = zeros(n)
    V_fx = zeros(n)
    s_fx = [zeros(Int, N) for _ in 1:n]
    t0 = time()
    for (i, w) in enumerate(w_grid)
        t1 = time()
        V_or[i] = V_oracle_w(w)
        (V_fx[i], s_fx[i]) = V_fixed_w(w)
        V_ad[i] = V_adaptive_w(w)
        @printf("  w=%.4f  V_oracle=%.4f  V_adaptive=%.4f  V_fixed=%.4f  (%.1f s)\n",
                w, V_or[i], V_ad[i], V_fx[i], time() - t1)
    end
    @printf("total sweep time: %.1f s\n", time() - t0)
    (V_or = V_or, V_ad = V_ad, V_fx = V_fx, s_fx = s_fx)
end

# main
println("Parameters: K=$K, N=$N, p_D=$p_D, p_FA=$p_FA")
println("Gaussian beam: G_w(α) = exp(-α²/(2w²))\n")

# Grid for coarse sweep
w_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
          0.50, 0.60, 0.70, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, π] .* 1.0

println("Running geometry-policy sweep over $(length(w_grid)) values of w ∈ [0.05, π]...\n")
results = sweep(w_grid)

# summary
println("\n" * "="^76)
println("SWEEP RESULTS (all values in nats; saturation = realized gain / EVPI)")
println("="^76)
@printf("%-8s  %-10s  %-10s  %-10s  %-10s  %-10s  %-8s\n",
        "w", "V_oracle", "V_adaptive", "V_fixed", "E[IG]", "realized", "sat%")
for (i, w) in enumerate(w_grid)
    EIG = results.V_or[i] - results.V_fx[i]
    realized = results.V_ad[i] - results.V_fx[i]
    sat = EIG > 1e-10 ? 100*realized/EIG : 0.0
    @printf("%-8.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-8.1f\n",
            w, results.V_or[i], results.V_ad[i], results.V_fx[i], EIG, realized, sat)
end

# joint optimum
(_, i_ad_star) = findmax(results.V_ad)
(_, i_fx_star) = findmax(results.V_fx)
println("\n" * "="^76)
@printf("Joint optimum for V_adaptive: w* = %.4f, V_adaptive* = %.4f nats\n",
        w_grid[i_ad_star], results.V_ad[i_ad_star])
@printf("Joint optimum for V_fixed:    w* = %.4f, V_fixed*    = %.4f nats\n",
        w_grid[i_fx_star], results.V_fx[i_fx_star])

open("joint_results.jls", "w") do io
    serialize(io, (; w_grid, results))
end
println("Saved to joint_results.jls")
