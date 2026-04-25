#=
compare_mse_K3_baseline.jl — Sanity check: K=3 fine grid (J=20, L=2) baseline.
Trains DP at different K_PHI values, deploys at K_PHI=256.  Shows how sensitive
the adaptive policy is to the training-grid resolution for posterior variance.
=#
using Printf, Random, Serialization, Dates

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt, .PCRB

const N_MC = parse(Int, get(ENV, "MSE_N", "20000"))
const K_PHI_POST = 256
const K_EPOCHS = parse(Int, get(ENV, "K_EPOCHS", "4"))
const J_TAU = parse(Int, get(ENV, "J_TAU", "10"))

TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
N_GRID = (1, 10)

c = PAPER_BASELINE
phi_star_fn = make_phi_star_fn()
ωd = omega_q(phi_star_fn(c)[1], c)

@printf("Sanity: K=%d J=%d L=2, PAPER_BASELINE c\n", K_EPOCHS, J_TAU)
flush(stdout)

sched_pcrb = [(J_TAU, 2) for _ in 1:K_EPOCHS]

# Deploy grid: K_PHI=256
grid_post = make_grid(; K_phi=K_PHI_POST, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)

# Try training grids at K_PHI ∈ {64, 128, 256}.
for K_PHI_TRAIN in [64, 128, 256]
    grid_tr = make_grid(; K_phi=K_PHI_TRAIN, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)
    @printf("\n[K_PHI_TRAIN=%d] Building policy memo...\n", K_PHI_TRAIN)
    flush(stdout)
    t0 = time()
    (V_tr, memo_tr, st_tr) = solve_bellman_full(grid_tr, K_EPOCHS, c, ωd; terminal=:mse)
    @printf("  V_train=%.4e memo=%d %.1fs\n", V_tr, st_tr.memo_size, time()-t0)
    flush(stdout)

    # To deploy the TRAINED policy on a finer grid, we need to replay actions:
    # just follow policy_action(memo_tr, counts, r).  The posterior mean estimator
    # uses the deploy grid (256).
    rng = MersenneTwister(2026); t0 = time()
    (MSE_1, se_1) = deployed_mse_adaptive(c, memo_tr, ωd, grid_post, K_EPOCHS; n_mc=N_MC, rng=rng)
    @printf("  MSE̅_adaptive = %.4e ± %.2e  (%.1fs)\n", MSE_1, se_1, time()-t0)
    flush(stdout)
    GC.gc()
end

# PCRB fixed schedule — does not depend on training grid.
rng = MersenneTwister(2026); t0 = time()
(MSE_pcrb, se_pcrb) = deployed_mse_fixed(c, sched_pcrb, ωd, grid_post; n_mc=N_MC, rng=rng)
@printf("\n[PCRB schedule] MSE̅_pcrb = %.4e ± %.2e  (%.1fs)\n", MSE_pcrb, se_pcrb, time()-t0)
flush(stdout)
