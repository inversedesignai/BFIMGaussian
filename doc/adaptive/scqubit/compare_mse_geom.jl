#=
compare_mse_geom.jl — MSE comparison for the geom-only Adam runs.

Loads results/joint_geom/final.jls, results/pcrb_geom/final.jls.
Saves results/compare_mse_geom.jls.
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

println("compare_mse_geom.jl  —  n_mc=$(N_MC), posterior K_Φ=$K_PHI_POST")

joint_path = joinpath(@__DIR__, "results", "joint_geom", "final.jls")
pcrb_path  = joinpath(@__DIR__, "results", "pcrb_geom",  "final.jls")
isfile(joint_path) || error("Missing $joint_path — run sweep_joint_geom.jl first.")
isfile(pcrb_path)  || error("Missing $pcrb_path — run sweep_pcrb_geom.jl first.")

joint = deserialize(joint_path)
pcrb  = deserialize(pcrb_path)

grid = make_grid(; K_phi=K_PHI_POST, phi_max=0.49,
                   tau_grid=joint.TAU_GRID, n_grid=joint.N_GRID)

phi_star_fn = make_phi_star_fn()

# Pick best-c-seen from history (c_final may be at an Adam overshoot).
# Joint-DP: max V_adaptive. PCRB: max log_JP.
i_best_joint = argmax(joint.history.V_adaptive)
v_best_joint = joint.history.c_vec[i_best_joint]
c_best_joint = ScqubitParams(f_q_max=v_best_joint[1], E_C_over_h=v_best_joint[2],
                             kappa=v_best_joint[3], Delta_qr=v_best_joint[4],
                             temperature=v_best_joint[5], A_phi=v_best_joint[6], A_Ic=v_best_joint[7])
@printf("  joint best @ iter %d: V_adaptive=%.4f\n", i_best_joint, joint.history.V_adaptive[i_best_joint])

i_best_pcrb = argmax(pcrb.history.log_JP)
v_best_pcrb = pcrb.history.c_vec[i_best_pcrb]
c_best_pcrb = ScqubitParams(f_q_max=v_best_pcrb[1], E_C_over_h=v_best_pcrb[2],
                            kappa=v_best_pcrb[3], Delta_qr=v_best_pcrb[4],
                            temperature=v_best_pcrb[5], A_phi=v_best_pcrb[6], A_Ic=v_best_pcrb[7])
sched_best_pcrb = pcrb.history.sched[i_best_pcrb]
@printf("  pcrb  best @ iter %d: log_JP=%.4f\n", i_best_pcrb, pcrb.history.log_JP[i_best_pcrb])

omega_d_1 = omega_q(phi_star_fn(c_best_joint)[1], c_best_joint)
omega_d_2 = omega_q(phi_star_fn(c_best_pcrb)[1],  c_best_pcrb)

println("\n[Deploy joint-DP design (geom-only, best-c-seen)]  computing adaptive-policy MSE…")
(V1_check, memo1, st1) = solve_bellman_full(grid, joint.K_EPOCHS, c_best_joint, omega_d_1)
@printf("  Re-solve Bellman at c₁*:  V_adaptive=%.4f  memo=%d  %.2f s\n",
        V1_check, st1.memo_size, st1.elapsed)

rng = MersenneTwister(2026)
t0 = time()
(MSE_1, se_1) = deployed_mse_adaptive(c_best_joint, memo1, omega_d_1, grid,
                                      joint.K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE̅₁ (joint-DP) = %.4e ± %.2e      %.1f s\n", MSE_1, se_1, time()-t0)

println("\n[Deploy PCRB-baseline design (geom-only, best-c-seen)]  computing fixed-schedule MSE…")
t0 = time()
rng = MersenneTwister(2026)
(MSE_2, se_2) = deployed_mse_fixed(c_best_pcrb, sched_best_pcrb,
                                   omega_d_2, grid; n_mc=N_MC, rng=rng)
@printf("  MSE̅₂ (PCRB baseline) = %.4e ± %.2e      %.1f s\n", MSE_2, se_2, time()-t0)

JP2 = exp(log_JP_of_schedule(sched_best_pcrb, grid, c_best_pcrb, omega_d_2; J_0=1e-4))
pcrb_bound = 1 / JP2

println("\n" * "="^70)
println("HEADLINE (GEOM-ONLY): MSE of posterior-mean estimator, prior-averaged")
println("-"^70)
@printf("  MSE̅₁ (joint DP)       = %.4e ± %.2e  nats²/Φ₀²\n", MSE_1, se_1)
@printf("  MSE̅₂ (PCRB baseline)  = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  1/J_P(c₂*, s₂*)       = %.4e  (CRB lower bound)\n", pcrb_bound)
@printf("  ratio MSE̅₂ / MSE̅₁    = %.3f\n", MSE_2 / MSE_1)
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("  z = (MSE̅₂ - MSE̅₁)/σ  = %+.2f σ\n", z)
println("="^70)

if MSE_2 < pcrb_bound - 1e-10
    @warn @sprintf("PCRB bound violation? MSE̅₂=%.3e < 1/J_P=%.3e", MSE_2, pcrb_bound)
end

outdir = joinpath(@__DIR__, "results")
out_path = joinpath(outdir, "compare_mse_geom.jls")
open(out_path, "w") do io
    serialize(io, (; MSE_1, se_1, MSE_2, se_2, pcrb_bound, ratio=MSE_2/MSE_1,
                     c_1_star=c_best_joint, c_2_star=c_best_pcrb,
                     sched_2_star=sched_best_pcrb,
                     i_best_joint, i_best_pcrb,
                     N_MC=N_MC, K_PHI_POST=K_PHI_POST,
                     omega_d_1, omega_d_2, timestamp=now()))
end
println("Saved to $out_path")
