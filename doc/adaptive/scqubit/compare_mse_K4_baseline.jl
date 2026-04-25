#=
compare_mse_K4_baseline.jl — Quick baseline comparison at K=4 WITHOUT
optimizing c.  Uses PAPER_BASELINE for both adaptive (via K=4 DP) and
fixed-schedule PCRB (4×(320ns, 10shots) — the known PCRB optimum).

This sets a lower bound on the joint-DP vs PCRB gap — since neither
c is optimized, both policies compete on the same footing.
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
const K_PHI_POST = parse(Int, get(ENV, "MSE_K_PHI", "256"))
const K_EPOCHS = parse(Int, get(ENV, "K_EPOCHS", "4"))
const J_TAU = parse(Int, get(ENV, "J_TAU", "10"))

TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
N_GRID = (1, 10)

grid = make_grid(; K_phi=K_PHI_POST, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)
phi_star_fn = make_phi_star_fn()

c = PAPER_BASELINE
ωd = omega_q(phi_star_fn(c)[1], c)
@printf("Config: K=%d, J=%d, L=2, K_Φ=%d, c=PAPER_BASELINE (f_q=%.2f GHz)\n",
        K_EPOCHS, J_TAU, K_PHI_POST, c.f_q_max/1e9)
flush(stdout)

# PCRB optimal schedule at PAPER_BASELINE — pick (τ_max, n_max) at every epoch
sched_pcrb = [(J_TAU, 2) for _ in 1:K_EPOCHS]

println("\n[1/3] Building adaptive policy memo...")
t0 = time()
(V_ad, memo_ad, st_ad) = solve_bellman_full(grid, K_EPOCHS, c, ωd; terminal=:mse)
@printf("  V_adaptive = %.4e  memo=%d  %.1fs\n", V_ad, st_ad.memo_size, time()-t0)
flush(stdout)

println("\n[2/3] Deploying adaptive policy (MC)...")
rng = MersenneTwister(2026); t0 = time()
(MSE_1, se_1) = deployed_mse_adaptive(c, memo_ad, ωd, grid, K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE̅_adaptive = %.4e ± %.2e  (%.1fs)\n", MSE_1, se_1, time()-t0)
flush(stdout)

println("\n[3/3] Deploying PCRB schedule (MC)...")
rng = MersenneTwister(2026); t0 = time()
(MSE_2, se_2) = deployed_mse_fixed(c, sched_pcrb, ωd, grid; n_mc=N_MC, rng=rng)
@printf("  MSE̅_pcrb     = %.4e ± %.2e  (%.1fs)\n", MSE_2, se_2, time()-t0)

println("\n" * "="^72)
@printf("BASELINE COMPARISON at K=%d, J=%d (no c-optimization)\n", K_EPOCHS, J_TAU)
println("-"^72)
@printf("  MSE̅_adaptive  = %.4e ± %.2e\n", MSE_1, se_1)
@printf("  MSE̅_pcrb      = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  ratio MSE̅_pcrb/MSE̅_adaptive = %.4f\n", MSE_2/MSE_1)
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("  z = %+.2f σ\n", z)
println("="^72)
