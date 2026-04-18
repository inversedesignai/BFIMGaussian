"""
Exact finite-horizon Bellman DP for the radar beam-search POMDP.

Uses direct backward induction on beliefs: we do not invoke Sondik
alpha-vectors because the terminal reward Φ = ln K − H(b) is nonlinear
in the belief; but the belief space reachable from the uniform prior
under any policy is finite and enumerable by simulating (action, obs)
branches.

Computes:
- V_adaptive = W_1(prior) for both regimes
- the optimal adaptive policy as a tree (action at each observation
  history)
- the adaptive advantage V_adaptive − V_fixed per regime
- comparison with the EVPI upper bound E[IG]
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

function belief_entropy(b)
    H = 0.0
    for z in 1:K
        H -= b[z] > 0 ? b[z] * log(b[z]) : 0.0
    end
    H
end

# Bayesian update: given belief b and observation y under action a, return new belief.
# Also returns the marginal P(y | b, a).
function bayes_update(b, a, y, cov)
    b_new = zeros(K)
    for z in 1:K
        p_z_y = (y == 1 ? p_detect(z, a, cov) : 1 - p_detect(z, a, cov))
        b_new[z] = b[z] * p_z_y
    end
    p_y = sum(b_new)
    if p_y > 0
        b_new ./= p_y
    end
    (b_new, p_y)
end

# Exact backward-induction Bellman DP.
# bellman(b, r, coverage) returns (W, best_action)
# where W = max over policies of expected (ln K − H(b_final)) starting from b
# with r steps remaining.  For r = 0, W = ln K − H(b) and action = 0.
function bellman(b::Vector{Float64}, r::Int, cov, depth::Int=0, memo=nothing)
    if r == 0
        return (LNK - belief_entropy(b), 0)
    end
    # try each action
    best_val = -Inf
    best_act = 0
    for a in 1:K
        val = 0.0
        for y in 0:1
            (b_new, p_y) = bayes_update(b, a, y, cov)
            if p_y > 0
                sub, _ = bellman(b_new, r - 1, cov, depth + 1, memo)
                val += p_y * sub
            end
        end
        if val > best_val
            best_val = val
            best_act = a
        end
    end
    (best_val, best_act)
end

# Build the optimal policy tree explicitly (so we can inspect the action structure).
# Tree is a dict:
#   key = tuple of observations (y_1, ..., y_{k-1})
#   value = action taken at that node
# Plus the value at each leaf.
function build_policy_tree(cov)
    tree = Dict{NTuple{N,Int}, Int}()    # complete history (filled with -1 before step k)
    actions = Dict{Vector{Int}, Int}()    # variable-length history → action
    values = Dict{Vector{Int}, Float64}() # history → expected remaining-value
    function recurse(b, history, r)
        if r == 0
            values[copy(history)] = LNK - belief_entropy(b)
            return LNK - belief_entropy(b)
        end
        best_val = -Inf
        best_act = 0
        best_sub_vals = (0.0, 0.0)
        for a in 1:K
            val = 0.0
            sub_vals = (0.0, 0.0)
            for y in 0:1
                (b_new, p_y) = bayes_update(b, a, y, cov)
                push!(history, y)
                sub = recurse(b_new, history, r - 1)
                pop!(history)
                if p_y > 0
                    val += p_y * sub
                    sub_vals = (y == 0 ? (sub, sub_vals[2]) : (sub_vals[1], sub))
                end
            end
            if val > best_val
                best_val = val
                best_act = a
                best_sub_vals = sub_vals
            end
        end
        actions[copy(history)] = best_act
        values[copy(history)] = best_val
        best_val
    end
    V = recurse(ones(K) / K, Int[], N)
    (V, actions, values)
end

# ---------- main ----------
function run_dp(m::Int, label::String)
    println("\n" * "="^72)
    println("POMDP Bellman DP — regime: $label (m = $m cells/beam)")
    println("="^72)

    cov = make_beam_coverage(m)
    println("Beam pointing at cell 1 covers: $(findall(cov[1]))")

    prior = ones(K) / K
    t0 = time()
    V_ad, root_action = bellman(prior, N, cov)
    t_bellman = time() - t0
    @printf("\nV_adaptive = W_1(prior) = %.6f nats\n", V_ad)
    @printf("Optimal action at root = cell %d\n", root_action)
    @printf("Bellman solve time: %.2f s\n", t_bellman)

    # build full policy tree for inspection
    t0 = time()
    (V_ad2, actions, values) = build_policy_tree(cov)
    t_tree = time() - t0
    @printf("\nPolicy-tree construction: %.2f s (V check = %.6f)\n", t_tree, V_ad2)

    # inspect first few levels
    println("\nOptimal policy tree (first 2 levels):")
    @printf("  step 1 (no obs):          action = cell %d\n", actions[Int[]])
    for y1 in 0:1
        h = [y1]
        @printf("  step 2 (y_1 = %d):        action = cell %d\n", y1, actions[h])
        for y2 in 0:1
            h = [y1, y2]
            @printf("    step 3 (y_1=%d, y_2=%d):   action = cell %d\n", y1, y2, actions[h])
        end
    end

    return (V_ad = V_ad, actions = actions, values = values)
end

println("Parameters: K=$K, N=$N, p_D=$p_D, p_FA=$p_FA")

narrow_dp = run_dp(1, "NARROW (m = 1)")
wide_dp   = run_dp(div(K, 2), "WIDE (m = K/2 = $(div(K,2)))")

# Compare with V_fixed and E[IG] from compute_ig.jl
ig_data = deserialize("ig_results.jls")

println("\n" * "="^72)
println("FINAL COMPARISON")
println("="^72)

function show_regime(label, V_or_array, V_fx, E_IG, V_ad)
    V_or = V_or_array[1]  # constant over x
    println("\n$label regime:")
    @printf("  V_oracle   = %.6f nats\n", V_or)
    @printf("  V_adaptive = %.6f nats\n", V_ad)
    @printf("  V_fixed    = %.6f nats\n", V_fx)
    @printf("  E[IG] = V_oracle − V_fixed          = %.6f nats (EVPI upper bound)\n", E_IG)
    @printf("  V_adaptive − V_fixed (realized gain) = %.6f nats\n", V_ad - V_fx)
    @printf("  EVPI saturation fraction             = %.1f%%\n",
            100 * (V_ad - V_fx) / E_IG)
end

show_regime("NARROW", ig_data.narrow.V_or, ig_data.narrow.V_fx,
            ig_data.narrow.V_or[1] - ig_data.narrow.V_fx, narrow_dp.V_ad)
show_regime("WIDE", ig_data.wide.V_or, ig_data.wide.V_fx,
            ig_data.wide.V_or[1] - ig_data.wide.V_fx, wide_dp.V_ad)

open("dp_results.jls", "w") do io
    serialize(io, (; narrow = narrow_dp, wide = wide_dp))
end
println("\nSaved to dp_results.jls")
