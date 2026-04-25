"""
BellmanThreaded.jl

Multi-threaded version of `Bellman.solve_bellman` using a topological-sort
approach: enumerate reachable `(counts, r)` states layer-by-layer (parallel
within each layer with per-thread Sets that are merged afterwards), then
solve backwards from r=0 to r=K (parallel within each layer, lock-free
because layer r reads layer r-1 which is fully populated by the time
layer r begins).

Per-state arithmetic is bit-identical to `Bellman.jl` — the in-place
buffer routines preserve the same accumulation order (`logb_into!` matches
`_logb_from_counts`, `marg_obs_into!` matches `_marg_obs` with `Z = sum(w)`
using Julia's pairwise summation). Floating-point determinism: the value
function is identical to single-threaded Bellman (validated in
tests/test_bellman_threaded.jl).

Public API:
    solve_bellman_threaded(K, grid, c, omega_d; terminal=:mi, threshold=1e-20)
        -> (W, memo)
    solve_bellman_threaded_full(grid, K, c, omega_d; terminal=:mi)
        -> (V_adaptive, memo, stats)

The `memo` Dict has the same keys/values as `Bellman.solve_bellman`'s.
"""
module BellmanThreaded

using Main.ScqubitModel
using Main.Belief
using SpecialFunctions: loggamma
using Base.Threads: @threads, nthreads, threadid, maxthreadid

import Main.Bellman: BellmanNode, _logp_tables

export solve_bellman_threaded, solve_bellman_threaded_full

# ---------------------------------------------------------------
# In-place helpers — same arithmetic as Bellman.jl
# ---------------------------------------------------------------
@inline _log_binom(n::Integer, k::Integer) =
    loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

"In-place logb construction. Mutates buf to match `_logb_from_counts(counts, logp, log1mp)`."
@inline function logb_into!(buf::Vector{Float64},
                            counts::NTuple{J, Tuple{Int,Int}},
                            logp::AbstractMatrix,
                            log1mp::AbstractMatrix) where {J}
    Nphi = size(logp, 2)
    fill!(buf, 0.0)
    for j in 1:J
        n_j, m_j = counts[j]
        n_j == 0 && continue
        @inbounds for i in 1:Nphi
            buf[i] += m_j * logp[j, i] + (n_j - m_j) * log1mp[j, i]
        end
    end
    buf
end

"In-place marg_obs. Mutates w_buf. Matches `_marg_obs` exactly (same Z = sum(w) and same s-accumulation order)."
@inline function marg_obs_into!(w_buf::Vector{Float64},
                                logb::Vector{Float64},
                                j::Int, n::Int, m::Int,
                                logp::AbstractMatrix,
                                log1mp::AbstractMatrix)
    mx = maximum(logb)
    @inbounds for i in eachindex(logb)
        w_buf[i] = exp(logb[i] - mx)
    end
    Z = sum(w_buf)  # Julia pairwise sum — must match _marg_obs
    log_bc = _log_binom(n, m)
    s = 0.0
    @inbounds for i in eachindex(w_buf)
        s += w_buf[i] * exp(log_bc + m * logp[j, i] + (n - m) * log1mp[j, i])
    end
    s / Z
end

"Match `_entropy_from_logb`."
function _entropy_from_logb(logb::Vector{Float64}, dphi::Float64)
    m = maximum(logb)
    w = exp.(logb .- m)
    Z = sum(w)
    p = w ./ Z
    H_disc = -sum(p[i] > 0 ? p[i] * log(p[i]) : 0.0 for i in eachindex(p))
    H_disc + log(dphi)
end

"Match `_variance_from_logb`."
function _variance_from_logb(logb::Vector{Float64}, phi_grid::Vector{Float64})
    m = maximum(logb)
    w = exp.(logb .- m)
    Z = sum(w)
    p = w ./ Z
    μ  = sum(p .* phi_grid)
    σ2 = sum(p .* (phi_grid .- μ).^2)
    σ2
end

@inline function _terminal_reward(logb::Vector{Float64}, grid::Main.Belief.Grid, terminal::Symbol)
    if terminal === :mi
        return -_entropy_from_logb(logb, grid.dphi)
    elseif terminal === :mse
        return -_variance_from_logb(logb, grid.phi_grid)
    else
        error("unknown terminal reward :$(terminal); use :mi or :mse")
    end
end

# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------
"""
    solve_bellman_threaded(K, grid, c, omega_d; terminal=:mi, threshold=1e-20)
        -> (W, memo::Dict{Tuple{Counts, Int}, BellmanNode})

Topological-sort multi-threaded Bellman. Drop-in replacement for
`Bellman.solve_bellman` with the same return signature. Uses
`Threads.nthreads()` workers; falls back to sequential semantics when
nthreads()==1.
"""
function solve_bellman_threaded(K::Int,
                                grid::Main.Belief.Grid{J, L},
                                c::ScqubitParams, omega_d;
                                terminal::Symbol=:mi,
                                threshold::Real=1e-20) where {J, L}
    (logp, log1mp) = _logp_tables(grid, c, omega_d)
    Nphi = length(grid.phi_grid)
    # Size per-thread buffers by maxthreadid() (covers GC/interactive threads
    # that may appear in `threadid()` even though `nthreads()` doesn't count them).
    n_thr_buf = maxthreadid()
    counts0 = ntuple(_ -> (0, 0), J)
    Counts_T = typeof(counts0)

    # Storage: structure-of-arrays for (counts, r, node) per state.
    counts_arr = Counts_T[]
    r_arr      = Int[]
    nodes_arr  = BellmanNode[]
    counts_to_idx = Dict{Tuple{Counts_T, Int}, Int}()

    # Layer book-keeping: layers[i] = (lo, hi) range of indices for the i-th
    # forward layer.  Layer 1 = root only; layer K+1 = terminal states (r=0).
    layers = Tuple{Int,Int}[]

    # Seed root.
    push!(counts_arr, counts0)
    push!(r_arr, K)
    push!(nodes_arr, BellmanNode(0.0, (0, 0)))
    counts_to_idx[(counts0, K)] = 1
    push!(layers, (1, 1))

    # ---------------- FORWARD PASS: enumerate reachable states ----------------
    for layer_step in 1:K
        prev_lo, prev_hi = layers[end]
        n_in_prev = prev_hi - prev_lo + 1
        new_r = K - layer_step

        # Per-thread accumulators (sized by maxthreadid)
        local_sets = [Set{Counts_T}() for _ in 1:n_thr_buf]
        thr_logb   = [zeros(Float64, Nphi) for _ in 1:n_thr_buf]
        thr_w      = [zeros(Float64, Nphi) for _ in 1:n_thr_buf]

        @threads :static for rel in 1:n_in_prev
            tid = threadid()
            buf_logb = thr_logb[tid]
            buf_w    = thr_w[tid]
            local_set = local_sets[tid]
            global_idx = prev_lo + rel - 1
            cnt = counts_arr[global_idx]

            logb_into!(buf_logb, cnt, logp, log1mp)

            for j in 1:J, ℓ in 1:L
                n = grid.n_grid[ℓ]
                for m in 0:n
                    p_m = marg_obs_into!(buf_w, buf_logb, j, n, m, logp, log1mp)
                    p_m < threshold && continue
                    new_cnt = ntuple(k -> k == j ?
                                     (cnt[k][1] + n, cnt[k][2] + m) :
                                     cnt[k], J)
                    push!(local_set, new_cnt)
                end
            end
        end

        # Merge per-thread Sets and append new states to global storage.
        layer_start = length(counts_arr) + 1
        merged = Set{Counts_T}()
        for s in local_sets
            union!(merged, s)
        end
        # Pre-grow vectors to avoid repeated re-allocation
        nnew = length(merged)
        sizehint!(counts_arr, length(counts_arr) + nnew)
        sizehint!(r_arr,      length(r_arr)      + nnew)
        sizehint!(nodes_arr,  length(nodes_arr)  + nnew)
        sizehint!(counts_to_idx, length(counts_to_idx) + nnew)
        for cnt in merged
            push!(counts_arr, cnt)
            push!(r_arr, new_r)
            push!(nodes_arr, BellmanNode(0.0, (0, 0)))
            counts_to_idx[(cnt, new_r)] = length(counts_arr)
        end
        push!(layers, (layer_start, length(counts_arr)))
    end

    # ---------------- BACKWARD PASS: solve states in reverse layer order ----------------
    # layers[end] = terminals (r=0). Process them first; then layers[end-1] (r=1); ...
    # finally layers[1] (root, r=K).
    for li in length(layers):-1:1
        lo, hi = layers[li]
        n_in_layer = hi - lo + 1

        thr_logb = [zeros(Float64, Nphi) for _ in 1:n_thr_buf]
        thr_w    = [zeros(Float64, Nphi) for _ in 1:n_thr_buf]

        @threads :static for rel in 1:n_in_layer
            tid = threadid()
            buf_logb = thr_logb[tid]
            buf_w    = thr_w[tid]
            global_idx = lo + rel - 1
            cnt   = counts_arr[global_idx]
            r_val = r_arr[global_idx]

            logb_into!(buf_logb, cnt, logp, log1mp)

            if r_val == 0
                val = _terminal_reward(buf_logb, grid, terminal)
                nodes_arr[global_idx] = BellmanNode(val, (0, 0))
            else
                best_val = -Inf
                best_act = (0, 0)
                for j in 1:J, ℓ in 1:L
                    n = grid.n_grid[ℓ]
                    val = 0.0
                    for m in 0:n
                        p_m = marg_obs_into!(buf_w, buf_logb, j, n, m, logp, log1mp)
                        p_m < threshold && continue
                        new_cnt = ntuple(k -> k == j ?
                                         (cnt[k][1] + n, cnt[k][2] + m) :
                                         cnt[k], J)
                        child_idx = counts_to_idx[(new_cnt, r_val - 1)]
                        val += p_m * nodes_arr[child_idx].value
                    end
                    if val > best_val
                        best_val = val
                        best_act = (j, ℓ)
                    end
                end
                nodes_arr[global_idx] = BellmanNode(best_val, best_act)
            end
        end
    end

    # ---------------- Build standard Dict-keyed memo ----------------
    memo = Dict{Tuple{Counts_T, Int}, BellmanNode}()
    sizehint!(memo, length(counts_arr))
    for (key, idx) in counts_to_idx
        memo[key] = nodes_arr[idx]
    end

    W = nodes_arr[1].value  # root is at index 1
    (W, memo)
end

"""
    solve_bellman_threaded_full(grid, K, c, omega_d; terminal=:mi)
        -> (V_adaptive, memo, stats)

Wrapper that mirrors `Bellman.solve_bellman_full`'s signature. Adds memo
size, elapsed wall-clock, and the active thread count to `stats`.
"""
function solve_bellman_threaded_full(grid::Main.Belief.Grid{J, L}, K::Int,
                                     c::ScqubitParams, omega_d;
                                     terminal::Symbol=:mi) where {J, L}
    t0 = time()
    (W, memo) = solve_bellman_threaded(K, grid, c, omega_d; terminal=terminal)
    elapsed = time() - t0
    stats = (memo_size = length(memo),
             elapsed   = elapsed,
             n_threads = nthreads())
    v = terminal === :mi ? W + log(grid.phi_max) : W
    (v, memo, stats)
end

end # module
