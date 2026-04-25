#=
compare_mse_narrow_pcrbopt.jl — MSE comparison at phi_max=0.1 with PCRB-optimized c.

Uses:
- Joint-DP: PAPER_BASELINE c (no joint c-opt)
- PCRB: c and schedule from pcrb_narrow optimization

Tests the scenario where PCRB gets to pick ITS best c while joint-DP
stays at paper baseline. Expect: gap may grow because PCRB-optimized c
is Fisher-optimal, not MSE-optimal.
=#
using Printf, Random, Serialization

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt, .PCRB

const N_MC = 40000
const K_PHI = 256
const PHI_MAX = 0.1

pcrb = deserialize(joinpath(@__DIR__, "results", "pcrb_narrow", "final.jls"))
imax = argmax(pcrb.history.log_JP)
v = pcrb.history.c_vec[imax]
c_pcrb = ScqubitParams(f_q_max=v[1], E_C_over_h=v[2], kappa=v[3], Delta_qr=v[4],
                       temperature=v[5], A_phi=v[6], A_Ic=v[7])
sched_pcrb = pcrb.history.sched[imax]

c_joint = PAPER_BASELINE  # Joint stays at baseline
TAU_GRID = pcrb.TAU_GRID
N_GRID = pcrb.N_GRID
K_EPOCHS = pcrb.K_EPOCHS

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)
phi_star_fn = make_phi_star_fn()
ωd_joint = omega_q(phi_star_fn(c_joint)[1], c_joint)
ωd_pcrb  = omega_q(phi_star_fn(c_pcrb)[1], c_pcrb)

@printf("scqubit narrow-prior with PCRB-optimized c comparison\n")
@printf("c_joint (PAPER_BASELINE): f_q=%.2f GHz, κ=%.1e Hz, Δ=%.2f GHz\n",
        c_joint.f_q_max/1e9, c_joint.kappa, c_joint.Delta_qr/1e9)
@printf("c_pcrb  (PCRB-optimized): f_q=%.2f GHz, κ=%.1e Hz, Δ=%.2f GHz\n",
        c_pcrb.f_q_max/1e9, c_pcrb.kappa, c_pcrb.Delta_qr/1e9)
@printf("PCRB sched: %s\n", string(sched_pcrb))
flush(stdout)

println("\n[1/3] Building joint-DP policy (at c_joint)...")
t0 = time()
(V_ad, memo_ad, st_ad) = solve_bellman_full(grid, K_EPOCHS, c_joint, ωd_joint; terminal=:mse)
@printf("  V_adaptive = %.4e  %.1fs\n", V_ad, time()-t0)
flush(stdout)

println("\n[2/3] Deploying joint-DP (c_joint)...")
rng = MersenneTwister(2026); t0 = time()
(MSE_1, se_1) = deployed_mse_adaptive(c_joint, memo_ad, ωd_joint, grid, K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE̅₁ = %.4e ± %.2e  (%.1fs)\n", MSE_1, se_1, time()-t0)
flush(stdout)

println("\n[3/3] Deploying PCRB (c_pcrb, sched_pcrb)...")
rng = MersenneTwister(2026); t0 = time()
(MSE_2, se_2) = deployed_mse_fixed(c_pcrb, sched_pcrb, ωd_pcrb, grid; n_mc=N_MC, rng=rng)
@printf("  MSE̅₂ = %.4e ± %.2e  (%.1fs)\n", MSE_2, se_2, time()-t0)
flush(stdout)

JP2 = exp(log_JP_of_schedule(sched_pcrb, grid, c_pcrb, ωd_pcrb; J_0=1e-4))
crb = 1 / JP2

println("\n" * "="^72)
@printf("HEADLINE (phi_max=%.3f, joint at BASELINE c, PCRB at OPTIMIZED c)\n", PHI_MAX)
println("-"^72)
@printf("  MSE̅₁ (joint DP @ baseline c)      = %.4e ± %.2e\n", MSE_1, se_1)
@printf("  MSE̅₂ (PCRB @ PCRB-optimized c,s)  = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  1/J_P (CRB bound at c_pcrb)        = %.4e\n", crb)
@printf("  ratio MSE̅₂/MSE̅₁                 = %.4f\n", MSE_2/MSE_1)
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("  z-score                             = %+.2f σ\n", z)
println("="^72)
