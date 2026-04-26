"""
GradientThreaded.jl

Threaded policy-tree evaluation for the envelope-theorem gradient. Used
by the joint-DP outer optimization at K ≥ 4 where the Zygote-based
`grad_c_exact` is the per-iter bottleneck.

Two functions:
  V_adaptive_policy_exact_threaded(c, memo, grid, omega_d, K; terminal,
                                   parallel_depth)
      Same return value as `Gradient.V_adaptive_policy_exact`. Spawns
      `Threads.@spawn` tasks at each recursion level up to depth
      `parallel_depth` (default K-1, so the policy tree's first K-1
      levels are parallel; the deepest level runs sequentially in each
      task to amortise spawn overhead).

  grad_c_exact_fd_threaded(c_vec, memo, grid, omega_d, K; terminal,
                            parallel_depth)
      ForwardDiff gradient w.r.t. the 7-vector `c`. Drop-in replacement
      for `Gradient.grad_c_exact_fd`. Threading is purely across the
      policy tree's outcome branches; the Duals are immutable so spawned
      tasks need no synchronisation.

Why ForwardDiff and not Zygote: Zygote uses a global mutable tape that
is not thread-safe across `@spawn`. ForwardDiff propagates Dual numbers
through the same recursion, fully immutably, so each task computes its
subtree's Dual value independently.
"""
module GradientThreaded

using Main.ScqubitModel
using Main.Belief
using Main.Bellman
using Main.Gradient: _logb_from_counts_diff, _marg_obs_from_logb,
                     _terminal_diff, c_as_vec, vec_as_c, C_DIM
using ForwardDiff
using Base.Threads: @spawn

export V_adaptive_policy_exact_threaded, grad_c_exact_fd_threaded

# ---------------------------------------------------------------
# Threaded policy-tree evaluation
# ---------------------------------------------------------------
"""
    _value_rec_threaded(c, memo, counts, r, grid, omega_d; terminal, parallel_depth)

Same recursion as `Gradient._value_rec_exact` but with `Threads.@spawn`
tasks at each m-branch when `r > parallel_depth`. Below that depth the
recursion runs sequentially so spawn overhead doesn't dominate.

`memo` is read-only (the policy is fixed), and the Dual return values
are immutable, so the recursion is thread-safe.
"""
function _value_rec_threaded(c::ScqubitParams, memo::Dict,
                              counts::NTuple{J, Tuple{Int,Int}}, r::Int,
                              grid::Main.Belief.Grid{J, L}, omega_d;
                              terminal::Symbol=:mi,
                              parallel_depth::Int=0) where {J, L}
    logb = _logb_from_counts_diff(counts, c, omega_d, grid)
    if r == 0
        return _terminal_diff(logb, grid, terminal)
    end
    node = get(memo, (counts, r), nothing)
    node === nothing && error("policy memo miss at counts=$counts r=$r")
    action = node.action
    j, ℓ = action
    n = grid.n_grid[ℓ]

    if r > parallel_depth
        # Parallel: spawn a task per m-branch
        tasks = Vector{Task}(undef, n + 1)
        for (i, m) in enumerate(0:n)
            tasks[i] = @spawn _branch_value(c, memo, counts, j, ℓ, n, m,
                                            logb, r, grid, omega_d;
                                            terminal=terminal,
                                            parallel_depth=parallel_depth)
        end
        # `fetch` blocks; collected sums are over Duals, fully associative-stable
        # under the same ordering as the single-threaded version.
        s = fetch(tasks[1])
        for i in 2:length(tasks)
            s += fetch(tasks[i])
        end
        return s
    else
        # Sequential subtree
        s = _branch_value(c, memo, counts, j, ℓ, n, 0, logb, r, grid, omega_d;
                          terminal=terminal, parallel_depth=parallel_depth)
        for m in 1:n
            s += _branch_value(c, memo, counts, j, ℓ, n, m, logb, r, grid, omega_d;
                               terminal=terminal, parallel_depth=parallel_depth)
        end
        return s
    end
end

@inline function _branch_value(c::ScqubitParams, memo::Dict,
                               counts::NTuple{J, Tuple{Int,Int}},
                               j::Int, ℓ::Int, n::Int, m::Int,
                               logb, r::Int,
                               grid::Main.Belief.Grid{J, L}, omega_d;
                               terminal::Symbol=:mi,
                               parallel_depth::Int=0) where {J, L}
    p_m = _marg_obs_from_logb(logb, j, n, m, c, omega_d, grid)
    new_counts = ntuple(k -> k == j ?
                        (counts[k][1] + n, counts[k][2] + m) :
                        counts[k], J)
    sub = _value_rec_threaded(c, memo, new_counts, r - 1, grid, omega_d;
                              terminal=terminal, parallel_depth=parallel_depth)
    p_m * sub
end

"""
    V_adaptive_policy_exact_threaded(c, memo, grid, omega_d, K; terminal,
                                      parallel_depth)

V_adaptive(c) under the fixed policy in `memo`, evaluated with parallel
recursion. Returns a scalar (Float64 or Dual depending on `c`'s eltype).

`parallel_depth` defaults to K-1: the top K-1 levels of the policy tree
are parallel, the deepest level runs sequentially within each task.
Setting it to 0 gives full-depth parallelism; setting it to K-1 disables
threading entirely.
"""
function V_adaptive_policy_exact_threaded(c::ScqubitParams, memo::Dict,
                                          grid::Main.Belief.Grid{J, L}, omega_d,
                                          K::Int;
                                          terminal::Symbol=:mi,
                                          parallel_depth::Int=K-1) where {J, L}
    counts0 = ntuple(_ -> (0, 0), J)
    v = _value_rec_threaded(c, memo, counts0, K, grid, omega_d;
                             terminal=terminal, parallel_depth=parallel_depth)
    terminal === :mi ? log(grid.phi_max) + v : v
end

"""
    grad_c_exact_fd_threaded(c_vec, memo, grid, omega_d, K; terminal,
                              parallel_depth) -> Vector{Float64}

ForwardDiff gradient of V_adaptive_policy_exact_threaded w.r.t. the
7-vector c. Threading happens inside the Dual-typed evaluation: each
spawned task computes its subtree's contribution to the Dual return.
"""
function grad_c_exact_fd_threaded(c_vec::AbstractVector, memo::Dict,
                                   grid::Main.Belief.Grid{J, L}, omega_d, K::Int;
                                   terminal::Symbol=:mi,
                                   parallel_depth::Int=K-1) where {J, L}
    ForwardDiff.gradient(
        v -> V_adaptive_policy_exact_threaded(vec_as_c(v), memo, grid, omega_d, K;
                                              terminal=terminal,
                                              parallel_depth=parallel_depth),
        c_vec)
end

end # module
