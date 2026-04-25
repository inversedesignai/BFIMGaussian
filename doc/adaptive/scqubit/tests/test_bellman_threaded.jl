# Validation + benchmark for BellmanThreaded.jl
#
# Correctness invariants we test:
#   (1) Root value W matches single-threaded `Bellman.solve_bellman` exactly
#       (per-state arithmetic is bit-identical).
#   (2) Memo dictionaries are equal: same keys, same BellmanNode values
#       (both .value and .action).
#   (3) Holds for both terminal=:mi and terminal=:mse.
#   (4) Holds across multiple geometries (PAPER_BASELINE and a perturbed c).
#   (5) Holds at multiple K and K_PHI sizes.
#
# Benchmark:
#   K=4, J=10, L=2, K_PHI=128 (production size). Single-threaded vs threaded.
#
# Run with:
#   julia --project=. -t 1   doc/adaptive/scqubit/tests/test_bellman_threaded.jl
#   julia --project=. -t 8   doc/adaptive/scqubit/tests/test_bellman_threaded.jl
#   julia --project=. -t 32  doc/adaptive/scqubit/tests/test_bellman_threaded.jl

using Printf
using Test

include(joinpath(@__DIR__, "..", "ScqubitModel.jl"))
include(joinpath(@__DIR__, "..", "Belief.jl"))
include(joinpath(@__DIR__, "..", "Bellman.jl"))
include(joinpath(@__DIR__, "..", "BellmanThreaded.jl"))
using .ScqubitModel
using .Belief
using .Bellman
using .BellmanThreaded

println("Threads.nthreads() = ", Threads.nthreads())
println()

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
function compare_memos(memo_a::Dict, memo_b::Dict; label::String="")
    keys_a = keys(memo_a); keys_b = keys(memo_b)
    n_a, n_b = length(keys_a), length(keys_b)
    @printf("  [%s] memo sizes: a=%d  b=%d  (Δ=%d)\n", label, n_a, n_b, n_a - n_b)
    @test n_a == n_b
    sym_diff = setdiff(keys_a, keys_b) ∪ setdiff(keys_b, keys_a)
    @test isempty(sym_diff)
    n_value_mismatch = 0
    n_action_mismatch = 0
    max_value_abs_diff = 0.0
    for k in keys_a
        a = memo_a[k]; b = memo_b[k]
        if a.value != b.value
            n_value_mismatch += 1
            max_value_abs_diff = max(max_value_abs_diff, abs(a.value - b.value))
        end
        if a.action != b.action
            n_action_mismatch += 1
        end
    end
    @printf("  [%s] value mismatches:  %d  (max |Δ| = %.3e)\n",
            label, n_value_mismatch, max_value_abs_diff)
    @printf("  [%s] action mismatches: %d\n", label, n_action_mismatch)
    @test n_value_mismatch == 0
    @test n_action_mismatch == 0
end

function run_one_case(; K::Int, K_PHI::Int, J::Int, L::Int, terminal::Symbol,
                       c::ScqubitParams, label::String)
    println("─"^72)
    println(label)
    println("─"^72)
    if J == 4
        tau_grid = ntuple(k -> 20e-9 * 2.0^(k-1), J)
    else
        tau_grid = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J-1)), J)
    end
    n_grid = L == 1 ? (10,) : (1, 10)
    grid = make_grid(; K_phi=K_PHI, phi_max=0.1, tau_grid=tau_grid, n_grid=n_grid)
    ω_d = omega_q(0.442, c)

    # Single-threaded reference
    t0 = time()
    (W_ref, memo_ref) = solve_bellman(K, grid, c, ω_d; terminal=terminal)
    t_ref = time() - t0
    @printf("  single-threaded:  W=%.10e  memo=%d  %.2fs\n",
            W_ref, length(memo_ref), t_ref)

    # Threaded
    t0 = time()
    (W_thr, memo_thr) = solve_bellman_threaded(K, grid, c, ω_d; terminal=terminal)
    t_thr = time() - t0
    @printf("  threaded (%d):     W=%.10e  memo=%d  %.2fs   (speedup %.2fx)\n",
            Threads.nthreads(), W_thr, length(memo_thr), t_thr, t_ref / t_thr)

    # ----- Tests -----
    @test W_ref == W_thr
    compare_memos(memo_ref, memo_thr; label="$(label) memo")
    println()
    (W_ref, t_ref, t_thr)
end

# ---------------------------------------------------------------
# Small case: K=2, J=4, L=2, K_PHI=16, terminal=:mi
# ---------------------------------------------------------------
run_one_case(K=2, K_PHI=16, J=4, L=2, terminal=:mi,
             c=PAPER_BASELINE, label="K=2 K_PHI=16 J=4 L=2 :mi")

# Same case, :mse terminal
run_one_case(K=2, K_PHI=16, J=4, L=2, terminal=:mse,
             c=PAPER_BASELINE, label="K=2 K_PHI=16 J=4 L=2 :mse")

# Perturbed c
c_p = ScqubitParams(f_q_max=8.5e9, E_C_over_h=0.27e9, kappa=0.4e6,
                    Delta_qr=2.5e9, temperature=PAPER_BASELINE.temperature,
                    A_phi=PAPER_BASELINE.A_phi, A_Ic=PAPER_BASELINE.A_Ic)
run_one_case(K=2, K_PHI=16, J=4, L=2, terminal=:mi,
             c=c_p, label="K=2 K_PHI=16 J=4 L=2 :mi (perturbed c)")

# ---------------------------------------------------------------
# Medium case: K=3, J=4, L=2, K_PHI=32
# ---------------------------------------------------------------
run_one_case(K=3, K_PHI=32, J=4, L=2, terminal=:mi,
             c=PAPER_BASELINE, label="K=3 K_PHI=32 J=4 L=2 :mi")
run_one_case(K=3, K_PHI=32, J=4, L=2, terminal=:mse,
             c=PAPER_BASELINE, label="K=3 K_PHI=32 J=4 L=2 :mse")

# ---------------------------------------------------------------
# Production-shape: K=4, J=10, L=2, K_PHI=64, terminal=:mse
# (smaller K_PHI than headline run for test speed; same shape).
# ---------------------------------------------------------------
run_one_case(K=4, K_PHI=64, J=10, L=2, terminal=:mse,
             c=PAPER_BASELINE, label="K=4 K_PHI=64 J=10 L=2 :mse (prod-shape)")

# Headline-size benchmark (correctness + timing): K=4 K_PHI=128 J=10 L=2
run_one_case(K=4, K_PHI=128, J=10, L=2, terminal=:mse,
             c=PAPER_BASELINE, label="K=4 K_PHI=128 J=10 L=2 :mse (headline-size)")

println("All correctness tests passed.")
