#=
compare_mse.jl — deploy both designs, compute MSE̅₁, MSE̅₂ and the headline ratio.

Loads:
  results/joint/final.jls    (joint-DP design c₁*, policy memo π*)
  results/pcrb/final.jls     (PCRB baseline c₂*, schedule s*)

Output:
  results/compare_mse.jls containing MSEs, SE, PCRB bound, ratio.

Environment:
  MSE_N        — number of MC trials per design (default 20_000).
  MSE_K_PHI    — posterior grid resolution for deployment (default 256).
=#
using Printf
using Random
using Serialization
using Dates

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel
using .Belief
using .Bellman
using .Gradient
using .JointOpt
using .PCRB

const N_MC       = parse(Int, get(ENV, "MSE_N",     "20000"))
const K_PHI_POST = parse(Int, get(ENV, "MSE_K_PHI", "256"))

println("compare_mse.jl  —  n_mc=$(N_MC), posterior K_Φ=$K_PHI_POST")

joint_path = joinpath(@__DIR__, "results", "joint", "final.jls")
pcrb_path  = joinpath(@__DIR__, "results", "pcrb",  "final.jls")
isfile(joint_path) || error("Missing $joint_path — run sweep_joint.jl first.")
isfile(pcrb_path)  || error("Missing $pcrb_path — run sweep_pcrb.jl first.")

joint = deserialize(joint_path)
pcrb  = deserialize(pcrb_path)

# Rebuild the same grid used in the sweeps (matching J, L, τ, n).
grid = make_grid(; K_phi=K_PHI_POST, phi_max=0.49,
                   tau_grid=joint.TAU_GRID, n_grid=joint.N_GRID)

# Recompute ω_d at c₁* and c₂* following the same operating-point convention.
phi_star_fn = make_phi_star_fn()
omega_d_1 = omega_q(phi_star_fn(joint.c_final)[1], joint.c_final)
omega_d_2 = omega_q(phi_star_fn(pcrb.c_final)[1],  pcrb.c_final)

println("\n[Deploy joint-DP design]  computing adaptive-policy MSE…")
# Re-solve Bellman at c₁* to get the optimal-policy memo (the saved memo was
# computed on a different Φ-grid if K_PHI_POST ≠ saved K_PHI).
(V1_check, memo1, st1) = solve_bellman_full(grid, joint.K_EPOCHS, joint.c_final, omega_d_1)
@printf("  Re-solve Bellman at c₁*:  V_adaptive=%.4f  memo=%d  %.2f s\n",
        V1_check, st1.memo_size, st1.elapsed)

rng = MersenneTwister(2026)
t0 = time()
(MSE_1, se_1) = deployed_mse_adaptive(joint.c_final, memo1, omega_d_1, grid,
                                      joint.K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE̅₁ (joint-DP) = %.4e ± %.2e      %.1f s\n", MSE_1, se_1, time()-t0)

println("\n[Deploy PCRB-baseline design]  computing fixed-schedule MSE…")
t0 = time()
rng = MersenneTwister(2026)   # same seed for paired MC comparison
(MSE_2, se_2) = deployed_mse_fixed(pcrb.c_final, pcrb.sched_final,
                                   omega_d_2, grid; n_mc=N_MC, rng=rng)
@printf("  MSE̅₂ (PCRB baseline) = %.4e ± %.2e      %.1f s\n", MSE_2, se_2, time()-t0)

# PCRB bound at c₂*, s₂*
JP2 = exp(log_JP_of_schedule(pcrb.sched_final, grid, pcrb.c_final, omega_d_2; J_0=1e-4))
pcrb_bound = 1 / JP2

println("\n" * "="^70)
println("HEADLINE COMPARISON (MSE of posterior-mean estimator, prior-averaged)")
println("-"^70)
@printf("  MSE̅₁ (joint DP)       = %.4e ± %.2e  nats²/Φ₀²\n", MSE_1, se_1)
@printf("  MSE̅₂ (PCRB baseline)  = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  1/J_P(c₂*, s₂*)       = %.4e  (CRB lower bound)\n", pcrb_bound)
@printf("  ratio MSE̅₂ / MSE̅₁    = %.3f\n", MSE_2 / MSE_1)
# z-score that MSE̅₂ > MSE̅₁
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("  z = (MSE̅₂ - MSE̅₁)/σ  = %+.2f σ\n", z)
println("="^70)

# Sanity: PCRB bound not violated
if MSE_2 < pcrb_bound - 1e-10
    @warn @sprintf("PCRB bound violation? MSE̅₂=%.3e < 1/J_P=%.3e — check estimator bias.",
                   MSE_2, pcrb_bound)
end

# Save
outdir = joinpath(@__DIR__, "results")
out_path = joinpath(outdir, "compare_mse.jls")
open(out_path, "w") do io
    serialize(io, (; MSE_1, se_1, MSE_2, se_2, pcrb_bound, ratio=MSE_2/MSE_1,
                     c_1_star=joint.c_final, c_2_star=pcrb.c_final,
                     sched_2_star=pcrb.sched_final,
                     N_MC=N_MC, K_PHI_POST=K_PHI_POST,
                     omega_d_1, omega_d_2, timestamp=now()))
end
println("Saved comparison to $out_path")
