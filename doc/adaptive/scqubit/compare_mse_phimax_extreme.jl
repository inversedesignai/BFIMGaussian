#=
compare_mse_phimax_extreme.jl — Extend sweep to phi_max < 0.1 and K ∈ {3, 5}.

Goal: see how the joint-DP vs PCRB gap scales in the extreme narrow-prior
regime and whether the gap persists for K=3 and K=5 at phi_max=0.1.
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

const N_MC = 20000
const K_PHI_POST = 256

TAU_GRID_J10 = ntuple(k -> 10e-9 * (32.0)^((k-1)/9), 10)
N_GRID = (1, 10)

c = PAPER_BASELINE
phi_star_fn = make_phi_star_fn()
ωd = omega_q(phi_star_fn(c)[1], c)

# Sweep phi_max ∈ {0.03, 0.05, 0.08} at K=4 J=10 L=2
for phi_max in [0.03, 0.05, 0.08]
    @printf("\n=== K=4 J=10 phi_max=%.3f ===\n", phi_max); flush(stdout)
    grid = make_grid(; K_phi=K_PHI_POST, phi_max=phi_max, tau_grid=TAU_GRID_J10, n_grid=N_GRID)
    t0 = time()
    (V, memo, st) = solve_bellman_full(grid, 4, c, ωd; terminal=:mse)
    @printf("  V_train=%.4e memo=%d %.1fs\n", V, st.memo_size, time()-t0)
    flush(stdout)

    rng = MersenneTwister(2026)
    (MSE_ad, se_ad) = deployed_mse_adaptive(c, memo, ωd, grid, 4; n_mc=N_MC, rng=rng)
    @printf("  MSE_adaptive = %.4e ± %.2e\n", MSE_ad, se_ad); flush(stdout)

    (sched_best, _) = argmax_schedule_enumerate(grid, c, ωd, 4; J_0=1e-4)
    rng = MersenneTwister(2026)
    (MSE_pc, se_pc) = deployed_mse_fixed(c, sched_best, ωd, grid; n_mc=N_MC, rng=rng)
    @printf("  MSE_pcrb = %.4e ± %.2e  sched=%s\n", MSE_pc, se_pc, string(sched_best))
    @printf("  ratio = %.4f  z=%+.2f  prior_var=%.4e\n",
            MSE_pc/MSE_ad, (MSE_pc - MSE_ad)/sqrt(se_ad^2 + se_pc^2), phi_max^2/12)
    flush(stdout)
    memo = nothing; GC.gc()
end

# Cross-check at K=3 and K=5 with phi_max=0.1
for (K, J_TAU, tau_grid) in [
        (3, 10, TAU_GRID_J10),
        (5, 6, ntuple(k -> 10e-9 * (32.0)^((k-1)/5), 6)),
    ]
    @printf("\n=== K=%d J=%d phi_max=0.100 ===\n", K, J_TAU); flush(stdout)
    grid = make_grid(; K_phi=K_PHI_POST, phi_max=0.1, tau_grid=tau_grid, n_grid=N_GRID)
    t0 = time()
    (V, memo, st) = solve_bellman_full(grid, K, c, ωd; terminal=:mse)
    @printf("  V_train=%.4e memo=%d %.1fs\n", V, st.memo_size, time()-t0)
    flush(stdout)

    rng = MersenneTwister(2026)
    (MSE_ad, se_ad) = deployed_mse_adaptive(c, memo, ωd, grid, K; n_mc=N_MC, rng=rng)
    @printf("  MSE_adaptive = %.4e ± %.2e\n", MSE_ad, se_ad); flush(stdout)

    (sched_best, _) = argmax_schedule_enumerate(grid, c, ωd, K; J_0=1e-4)
    rng = MersenneTwister(2026)
    (MSE_pc, se_pc) = deployed_mse_fixed(c, sched_best, ωd, grid; n_mc=N_MC, rng=rng)
    @printf("  MSE_pcrb = %.4e ± %.2e  sched=%s\n", MSE_pc, se_pc, string(sched_best))
    @printf("  ratio = %.4f  z=%+.2f\n",
            MSE_pc/MSE_ad, (MSE_pc - MSE_ad)/sqrt(se_ad^2 + se_pc^2))
    flush(stdout)
    memo = nothing; GC.gc()
end
