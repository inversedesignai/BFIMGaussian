#=
compare_mse_phimax.jl — Sweep phi_max to find regime where adaptive beats PCRB.

At small phi_max (narrow prior), no aliasing → PCRB Fisher-optimal.
At phi_max≈0.49 (wide prior), severe aliasing → both near prior variance.
Sweet spot in between where adaptive can disambiguate but PCRB can't.
=#
using Printf, Random
include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt, .PCRB

const K_EPOCHS = 4
const J_TAU = 10
const N_MC = 20000
const K_PHI_POST = 256

TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
N_GRID = (1, 10)

c = PAPER_BASELINE
phi_star_fn = make_phi_star_fn()
ωd = omega_q(phi_star_fn(c)[1], c)

for phi_max in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.49]
    @printf("\n=== phi_max=%.3f ===\n", phi_max)
    flush(stdout)
    grid = make_grid(; K_phi=K_PHI_POST, phi_max=phi_max, tau_grid=TAU_GRID, n_grid=N_GRID)

    t0 = time()
    (V, memo, st) = solve_bellman_full(grid, K_EPOCHS, c, ωd; terminal=:mse)
    @printf("  V_train=%.4e memo=%d %.1fs\n", V, st.memo_size, time()-t0)
    flush(stdout)

    rng = MersenneTwister(2026)
    (MSE_ad, se_ad) = deployed_mse_adaptive(c, memo, ωd, grid, K_EPOCHS; n_mc=N_MC, rng=rng)
    @printf("  MSE_adaptive = %.4e ± %.2e\n", MSE_ad, se_ad)
    flush(stdout)

    # PCRB: enumerate optimal schedule
    (sched_best, _) = argmax_schedule_enumerate(grid, c, ωd, K_EPOCHS; J_0=1e-4)
    rng = MersenneTwister(2026)
    (MSE_pc, se_pc) = deployed_mse_fixed(c, sched_best, ωd, grid; n_mc=N_MC, rng=rng)
    @printf("  PCRB sched = %s\n", string(sched_best))
    @printf("  MSE_pcrb = %.4e ± %.2e\n", MSE_pc, se_pc)
    @printf("  ratio = %.4f  z = %+.2f\n", MSE_pc/MSE_ad,
            (MSE_pc - MSE_ad)/sqrt(se_ad^2 + se_pc^2))
    flush(stdout)
    memo = nothing; GC.gc()
end
