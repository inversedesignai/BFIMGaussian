#=
compare_mse_geom_mse.jl — MSE comparison for the geom-only MSE-terminal run.
Uses joint_geom_mse/final.jls (terminal=:mse) and the existing pcrb_geom/final.jls.
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
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt, .PCRB

const N_MC       = parse(Int, get(ENV, "MSE_N",     "20000"))
const K_PHI_POST = parse(Int, get(ENV, "MSE_K_PHI", "256"))

joint_path = joinpath(@__DIR__, "results", "joint_geom_mse", "final.jls")
pcrb_path  = joinpath(@__DIR__, "results", "pcrb_geom", "final.jls")
isfile(joint_path) || error("Missing $joint_path")
isfile(pcrb_path)  || error("Missing $pcrb_path")

joint = deserialize(joint_path)
pcrb  = deserialize(pcrb_path)

grid = make_grid(; K_phi=K_PHI_POST, phi_max=0.49,
                   tau_grid=joint.TAU_GRID, n_grid=joint.N_GRID)

phi_star_fn = make_phi_star_fn()

# Best-c-seen: joint uses -Var so argmax is still valid (largest -Var = smallest Var).
# Restrict to the range that has a corresponding c_vec entry (joint_opt appends
# an extra V_adaptive entry at the end without a paired c_vec).
n_joint = length(joint.history.c_vec)
i_best_joint = argmax(joint.history.V_adaptive[1:n_joint])
v_best_joint = joint.history.c_vec[i_best_joint]
c_best_joint = ScqubitParams(f_q_max=v_best_joint[1], E_C_over_h=v_best_joint[2],
                             kappa=v_best_joint[3], Delta_qr=v_best_joint[4],
                             temperature=v_best_joint[5], A_phi=v_best_joint[6], A_Ic=v_best_joint[7])
@printf("  joint best @ iter %d: -E[Var]=%.4e\n", i_best_joint, joint.history.V_adaptive[i_best_joint])

i_best_pcrb = argmax(pcrb.history.log_JP)
v_best_pcrb = pcrb.history.c_vec[i_best_pcrb]
c_best_pcrb = ScqubitParams(f_q_max=v_best_pcrb[1], E_C_over_h=v_best_pcrb[2],
                            kappa=v_best_pcrb[3], Delta_qr=v_best_pcrb[4],
                            temperature=v_best_pcrb[5], A_phi=v_best_pcrb[6], A_Ic=v_best_pcrb[7])
sched_best_pcrb = pcrb.history.sched[i_best_pcrb]

omega_d_1 = omega_q(phi_star_fn(c_best_joint)[1], c_best_joint)
omega_d_2 = omega_q(phi_star_fn(c_best_pcrb)[1],  c_best_pcrb)

println("\n[Deploy joint-DP (terminal=:mse, geom-only)]")
(V1_check, memo1, st1) = solve_bellman_full(grid, joint.K_EPOCHS, c_best_joint, omega_d_1; terminal=:mse)
@printf("  Re-solve Bellman (:mse) at c₁*:  -E[Var]=%.4e  memo=%d  %.2fs\n",
        V1_check, st1.memo_size, st1.elapsed)

rng = MersenneTwister(2026)
t0 = time()
(MSE_1, se_1) = deployed_mse_adaptive(c_best_joint, memo1, omega_d_1, grid,
                                      joint.K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE̅₁ (joint-DP MSE-terminal) = %.4e ± %.2e   %.1fs\n", MSE_1, se_1, time()-t0)

println("\n[Deploy PCRB-baseline (geom-only)]")
rng = MersenneTwister(2026)
(MSE_2, se_2) = deployed_mse_fixed(c_best_pcrb, sched_best_pcrb,
                                   omega_d_2, grid; n_mc=N_MC, rng=rng)
@printf("  MSE̅₂ (PCRB baseline) = %.4e ± %.2e\n", MSE_2, se_2)

JP2 = exp(log_JP_of_schedule(sched_best_pcrb, grid, c_best_pcrb, omega_d_2; J_0=1e-4))
pcrb_bound = 1 / JP2

println("\n" * "="^72)
println("HEADLINE (MSE-TERMINAL, GEOM-ONLY)")
println("-"^72)
@printf("  MSE̅₁ (joint DP, MSE-terminal)    = %.4e ± %.2e\n", MSE_1, se_1)
@printf("  MSE̅₂ (PCRB baseline)             = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  1/J_P(c₂*, s₂*) (CRB)            = %.4e\n", pcrb_bound)
@printf("  ratio MSE̅₂ / MSE̅₁              = %.4f\n", MSE_2 / MSE_1)
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("  z = (MSE̅₂ - MSE̅₁)/σ            = %+.2f σ\n", z)
println("="^72)

out_path = joinpath(@__DIR__, "results", "compare_mse_geom_mse.jls")
open(out_path, "w") do io
    serialize(io, (; MSE_1, se_1, MSE_2, se_2, pcrb_bound, ratio=MSE_2/MSE_1,
                     c_1_star=c_best_joint, c_2_star=c_best_pcrb,
                     sched_2_star=sched_best_pcrb,
                     i_best_joint, i_best_pcrb,
                     N_MC=N_MC, K_PHI_POST=K_PHI_POST,
                     omega_d_1, omega_d_2, timestamp=now()))
end
println("Saved to $out_path")
