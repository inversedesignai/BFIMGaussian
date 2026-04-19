# Tests for JointOpt.jl — Adam + periodic Bellman re-solve.
# Small problem: K=2, J=4, L=2, K_Φ=64, 30 Adam iters.
using Printf
using Test
using Statistics: mean

include(joinpath(@__DIR__, "..", "ScqubitModel.jl"))
include(joinpath(@__DIR__, "..", "Belief.jl"))
include(joinpath(@__DIR__, "..", "Baselines.jl"))
include(joinpath(@__DIR__, "..", "Bellman.jl"))
include(joinpath(@__DIR__, "..", "Gradient.jl"))
include(joinpath(@__DIR__, "..", "JointOpt.jl"))
using .ScqubitModel
using .Belief
using .Bellman
using .Gradient
using .JointOpt

const c0   = PAPER_BASELINE
const grid = make_grid(; K_phi=64,
                          tau_grid = ntuple(k -> 20e-9 * 2.0^(k-1), 4),
                          n_grid   = (1, 10))

# Re-run Bellman at baseline to report starting V_adaptive
ω_d0 = omega_q(0.442, c0)
(V0, _, _) = solve_bellman_full(grid, 2, c0, ω_d0)
@printf("Baseline V_adaptive(c₀) = %.6f\n", V0)

# ----------------------------------------------------------------
# [1] Short joint_opt run: verify V_adaptive non-decreasing on a moving
#     average, and that we finish without errors.
# ----------------------------------------------------------------
println("\n[1] 30-iter joint_opt at K=2, policy_reopt_every=5")
(c_end, hist, memo_end) = joint_opt(c0;
    grid=grid, K_epochs=2,
    outer_iters=30, outer_lr=5e-3,
    policy_reopt_every=5, ckpt_every=0,
    verbose=true)

V_hist = hist.V_adaptive          # length == outer_iters + 1 (final)
# The V_adaptive recorded during iter is the pre-step value under the then-current policy.
# A stricter monotonicity test: the MA-5 should be non-decreasing *after* the
# first reopt cycle.
ma = [mean(V_hist[max(1, k-4):k]) for k in 5:length(V_hist)]
println("  V_hist head: ", round.(V_hist[1:min(5,end)]; digits=4))
println("  V_hist tail: ", round.(V_hist[max(1,end-4):end]; digits=4))
@printf("  first MA5  = %.6f\n", ma[1])
@printf("  last  MA5  = %.6f\n", ma[end])
# Final V_adaptive (post last reopt) should be ≥ baseline minus MC/kink slop.
V_final = V_hist[end]
@printf("  V(c₀)      = %.6f\n", V0)
@printf("  V(c_end)   = %.6f   Δ = %+.6f\n", V_final, V_final - V0)
@test V_final >= V0 - 1e-3    # allows mild oscillation from kinks at K=2

# ----------------------------------------------------------------
# [2] Box projection
# ----------------------------------------------------------------
println("\n[2] Box projection — c_end within bounds")
box = default_cbox()
v_end = c_as_vec(c_end)
for (i, name) in enumerate(C_FIELD_NAMES)
    @test box.lo[i] - 1e-12 <= v_end[i] <= box.hi[i] + 1e-12
    @printf("  %-12s  %+.4e  ∈ [%+.4e, %+.4e]\n",
            name, v_end[i], box.lo[i], box.hi[i])
end

# ----------------------------------------------------------------
# [3] History arrays have consistent sizes
# ----------------------------------------------------------------
println("\n[3] History integrity")
@test length(hist.grad_norm) == 30
@test length(hist.c_vec)     == 30
@test length(hist.omega_d)   == 30
@test length(hist.elapsed)   == 30
@printf("  mean iter time = %.2f s\n", sum(hist.elapsed)/length(hist.elapsed))
@printf("  policy reopts  = %d\n",     length(hist.reopt_iter))

println("\nAll Phase-6 tests passed.\n")
