# Validation + benchmark for GradientThreaded.jl.
#
# Correctness: V_adaptive_policy_exact_threaded must match
# Gradient.V_adaptive_policy_exact and the gradient must match
# Gradient.grad_c_exact_fd to within tight floating-point tolerance.
#
# Why approximate (not bit-exact) match: the threaded recursion sums
# `s += fetch(tasks[i])` in deterministic order, but inside ForwardDiff's
# Dual arithmetic the partial-derivative components reorder identically,
# so values match. Cross-check uses tight ≈ tolerance.

using Printf
using Test

include(joinpath(@__DIR__, "..", "ScqubitModel.jl"))
include(joinpath(@__DIR__, "..", "Belief.jl"))
include(joinpath(@__DIR__, "..", "Bellman.jl"))
include(joinpath(@__DIR__, "..", "Gradient.jl"))
include(joinpath(@__DIR__, "..", "GradientThreaded.jl"))
using .ScqubitModel, .Belief, .Bellman, .Gradient, .GradientThreaded

println("Threads.nthreads() = ", Threads.nthreads())
println()

function run_one(; K::Int, K_PHI::Int, J::Int, L::Int, terminal::Symbol,
                  c::ScqubitParams, label::String, parallel_depth::Int=-1)
    println("─"^72)
    println("$(label)  (parallel_depth=$(parallel_depth == -1 ? K-1 : parallel_depth))")
    println("─"^72)
    if J == 4
        tau_grid = ntuple(k -> 20e-9 * 2.0^(k-1), J)
    else
        tau_grid = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J-1)), J)
    end
    n_grid = L == 1 ? (10,) : (1, 10)
    grid = make_grid(; K_phi=K_PHI, phi_max=0.1, tau_grid=tau_grid, n_grid=n_grid)
    ω_d = omega_q(0.442, c)

    pd = parallel_depth == -1 ? K - 1 : parallel_depth

    # Build policy memo once
    (V_b, memo, _) = solve_bellman_full(grid, K, c, ω_d; terminal=terminal)

    # ----- forward value: single-threaded ref vs threaded -----
    v_ref = V_adaptive_policy_exact(c, memo, grid, ω_d, K; terminal=terminal)
    v_thr = V_adaptive_policy_exact_threaded(c, memo, grid, ω_d, K; terminal=terminal,
                                              parallel_depth=pd)
    @printf("  V_ref = %.10e\n", v_ref)
    @printf("  V_thr = %.10e   |Δ| = %.3e\n", v_thr, abs(v_ref - v_thr))
    @test abs(v_ref - v_thr) < 1e-12 * max(abs(v_ref), 1.0)

    # ----- gradient: ForwardDiff ref vs threaded ForwardDiff -----
    c_vec = c_as_vec(c)
    t0 = time()
    g_ref = grad_c_exact_fd(c_vec, memo, grid, ω_d, K; terminal=terminal)
    t_ref_fd = time() - t0
    t0 = time()
    g_thr = grad_c_exact_fd_threaded(c_vec, memo, grid, ω_d, K; terminal=terminal,
                                     parallel_depth=pd)
    t_thr_fd = time() - t0
    @printf("  grad_fd_ref:        %.2fs\n", t_ref_fd)
    @printf("  grad_fd_threaded:   %.2fs   speedup = %.2fx\n",
            t_thr_fd, t_ref_fd / t_thr_fd)
    rel_err = maximum(abs.(g_ref .- g_thr) ./ max.(abs.(g_ref), 1e-30))
    @printf("  max rel-err in gradient: %.3e\n", rel_err)
    @test rel_err < 1e-10

    # ----- compare against Zygote-based grad_c_exact for sanity -----
    t0 = time()
    g_zyg = grad_c_exact(c_vec, memo, grid, ω_d, K; terminal=terminal)
    t_zyg = time() - t0
    rel_err_zyg = maximum(abs.(g_ref .- g_zyg) ./ max.(abs.(g_ref), 1e-30))
    @printf("  grad_zygote:        %.2fs    | rel-err vs grad_fd_ref: %.3e\n",
            t_zyg, rel_err_zyg)
    println()
end

# Small case: K=2 J=4 L=2 K_PHI=16
run_one(K=2, K_PHI=16, J=4, L=2, terminal=:mi,
        c=PAPER_BASELINE, label="K=2 K_PHI=16 J=4 L=2 :mi")
run_one(K=2, K_PHI=16, J=4, L=2, terminal=:mse,
        c=PAPER_BASELINE, label="K=2 K_PHI=16 J=4 L=2 :mse")

# Medium: K=3 J=4 L=2 K_PHI=32
run_one(K=3, K_PHI=32, J=4, L=2, terminal=:mse,
        c=PAPER_BASELINE, label="K=3 K_PHI=32 J=4 L=2 :mse")

# Production-ish: K=4 J=10 L=2 K_PHI=64 (smaller K_PHI for test speed)
run_one(K=4, K_PHI=64, J=10, L=2, terminal=:mse,
        c=PAPER_BASELINE, label="K=4 K_PHI=64 J=10 L=2 :mse")

# Headline-size: K=4 J=10 L=2 K_PHI=128
run_one(K=4, K_PHI=128, J=10, L=2, terminal=:mse,
        c=PAPER_BASELINE, label="K=4 K_PHI=128 J=10 L=2 :mse (headline-size)")

println("All gradient correctness tests passed.")
