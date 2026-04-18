"""
Within-top-hat-family sweep over integer beam-width m ∈ {1, ..., K}.

Companion to compute_joint.jl (Gaussian family).  For each integer m:
    V_oracle(m)   = max over sensor sequences of Φ(x=1, s, m)   (D_K symmetry)
    V_fixed(m)    = max over (s_2, s_3, s_4) with s_1 = 1 of (1/K) Σ_x Φ(x, s, m)
    V_adaptive(m) = exact Bellman DP from the uniform prior
    E[IG](m)      = V_oracle − V_fixed
    realized(m)   = V_adaptive − V_fixed
    saturation    = realized / E[IG]

Identifies m* = argmax V_adaptive(m) within the top-hat family.
"""

using Printf
using Serialization

const K = 16
const N = 4
const p_D = 0.9
const p_FA = 0.05
const LNK = log(K)

function make_beam_coverage(m::Int)
    coverage = [falses(K) for _ in 1:K]
    half = div(m, 2)
    for a in 1:K, z in 1:K
        d = minimum((mod(z - a, K), mod(a - z, K)))
        coverage[a][z] = (m == 1 ? d == 0 :
                          (d <= half - 1) || (m % 2 == 1 && d == half))
    end
    coverage
end

p_detect(z, a, cov) = cov[a][z] ? p_D : p_FA

function posterior(s::Vector{Int}, y::Vector{Int}, cov)
    logb = zeros(K)
    for z in 1:K, k in 1:N
        pk = p_detect(z, s[k], cov)
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

function py_given_xs(y::Vector{Int}, x::Int, s::Vector{Int}, cov)
    p = 1.0
    for k in 1:N
        pk = p_detect(x, s[k], cov)
        p *= (y[k] == 1 ? pk : 1 - pk)
    end
    p
end

function Φ_value(x::Int, s::Vector{Int}, cov)
    val = 0.0
    for bits in 0:(2^N - 1)
        y = [((bits >> (k-1)) & 1) for k in 1:N]
        pyx = py_given_xs(y, x, s, cov)
        if pyx > 0
            val += pyx * (LNK - entropy(posterior(s, y, cov)))
        end
    end
    val
end

function V_oracle_m(cov)
    best = -Inf
    for s1 in 1:K, s2 in 1:K, s3 in 1:K, s4 in 1:K
        v = Φ_value(1, [s1, s2, s3, s4], cov)
        if v > best
            best = v
        end
    end
    best
end

function V_fixed_m(cov)
    best = -Inf
    best_s = [1, 1, 1, 1]
    for s2 in 1:K, s3 in 1:K, s4 in 1:K
        s = [1, s2, s3, s4]
        E = 0.0
        for x in 1:K
            E += Φ_value(x, s, cov) / K
        end
        if E > best
            best = E
            best_s = copy(s)
        end
    end
    (best, best_s)
end

function bayes_update(b, a, y, cov)
    b_new = zeros(K)
    for z in 1:K
        pk = p_detect(z, a, cov)
        b_new[z] = b[z] * (y == 1 ? pk : 1 - pk)
    end
    s = sum(b_new)
    if s > 0
        b_new ./= s
    end
    (b_new, s)
end

function bellman(b::Vector{Float64}, r::Int, cov)
    if r == 0
        return LNK - entropy(b)
    end
    best = -Inf
    for a in 1:K
        val = 0.0
        for y in 0:1
            (b_new, p_y) = bayes_update(b, a, y, cov)
            if p_y > 0
                val += p_y * bellman(b_new, r - 1, cov)
            end
        end
        if val > best
            best = val
        end
    end
    best
end

V_adaptive_m(cov) = bellman(ones(K) / K, N, cov)

function sweep(m_grid)
    n = length(m_grid)
    V_or = zeros(n)
    V_ad = zeros(n)
    V_fx = zeros(n)
    s_fx = [zeros(Int, N) for _ in 1:n]
    t0 = time()
    for (i, m) in enumerate(m_grid)
        t1 = time()
        cov = make_beam_coverage(m)
        V_or[i] = V_oracle_m(cov)
        (V_fx[i], s_fx[i]) = V_fixed_m(cov)
        V_ad[i] = V_adaptive_m(cov)
        @printf("  m=%2d  V_oracle=%.4f  V_adaptive=%.4f  V_fixed=%.4f  (%.1f s)\n",
                m, V_or[i], V_ad[i], V_fx[i], time() - t1)
    end
    @printf("total sweep time: %.1f s\n", time() - t0)
    (V_or = V_or, V_ad = V_ad, V_fx = V_fx, s_fx = s_fx)
end

println("Parameters: K=$K, N=$N, p_D=$p_D, p_FA=$p_FA")
println("Top-hat beam: m cells per lobe\n")

m_grid = collect(1:K)

println("Running within-top-hat-family sweep over m ∈ {1, ..., $K}...\n")
results = sweep(m_grid)

println("\n" * "="^76)
println("SWEEP RESULTS (top-hat family; all values in nats)")
println("="^76)
@printf("%-4s  %-10s  %-10s  %-10s  %-10s  %-10s  %-8s\n",
        "m", "V_oracle", "V_adaptive", "V_fixed", "E[IG]", "realized", "sat%")
for (i, m) in enumerate(m_grid)
    EIG = results.V_or[i] - results.V_fx[i]
    realized = results.V_ad[i] - results.V_fx[i]
    sat = EIG > 1e-10 ? 100*realized/EIG : 0.0
    @printf("%-4d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-8.1f\n",
            m, results.V_or[i], results.V_ad[i], results.V_fx[i], EIG, realized, sat)
end

(_, i_ad_star) = findmax(results.V_ad)
(_, i_fx_star) = findmax(results.V_fx)
println("\n" * "="^76)
@printf("In-family optimum for V_adaptive: m* = %d, V_adaptive* = %.4f nats\n",
        m_grid[i_ad_star], results.V_ad[i_ad_star])
@printf("In-family optimum for V_fixed:    m* = %d, V_fixed*    = %.4f nats\n",
        m_grid[i_fx_star], results.V_fx[i_fx_star])

open("joint_m_results.jls", "w") do io
    serialize(io, (; m_grid, results))
end
println("Saved to joint_m_results.jls")
