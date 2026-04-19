#=
compare_mse_rich.jl — MSE comparison with J=6, L=4 (plan defaults) and realistic box.
Uses joint_rich/final.jls (terminal=:mse) and pcrb_rich/final.jls.
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

joint_path = joinpath(@__DIR__, "results", "joint_rich", "final.jls")
pcrb_path  = joinpath(@__DIR__, "results", "pcrb_rich",  "final.jls")
isfile(joint_path) || error("Missing $joint_path — run sweep_joint_rich.jl first")
isfile(pcrb_path)  || error("Missing $pcrb_path")

joint = deserialize(joint_path)
pcrb  = deserialize(pcrb_path)

grid = make_grid(; K_phi=K_PHI_POST, phi_max=0.49,
                   tau_grid=joint.TAU_GRID, n_grid=joint.N_GRID)

phi_star_fn = make_phi_star_fn()

# Best-c-seen (restrict to range with c_vec)
nj = length(joint.history.c_vec)
i_best_joint = argmax(joint.history.V_adaptive[1:nj])
v_best_joint = joint.history.c_vec[i_best_joint]
c_best_joint = ScqubitParams(f_q_max=v_best_joint[1], E_C_over_h=v_best_joint[2],
                             kappa=v_best_joint[3], Delta_qr=v_best_joint[4],
                             temperature=v_best_joint[5], A_phi=v_best_joint[6], A_Ic=v_best_joint[7])

i_best_pcrb = argmax(pcrb.history.log_JP)
v_best_pcrb = pcrb.history.c_vec[i_best_pcrb]
c_best_pcrb = ScqubitParams(f_q_max=v_best_pcrb[1], E_C_over_h=v_best_pcrb[2],
                            kappa=v_best_pcrb[3], Delta_qr=v_best_pcrb[4],
                            temperature=v_best_pcrb[5], A_phi=v_best_pcrb[6], A_Ic=v_best_pcrb[7])
sched_best_pcrb = pcrb.history.sched[i_best_pcrb]

@printf("\nc_best_joint: f_q=%.3fGHz E_C=%.3f κ=%.2fMHz Δ=%.3fGHz\n",
        c_best_joint.f_q_max/1e9, c_best_joint.E_C_over_h/1e9,
        c_best_joint.kappa/1e6, c_best_joint.Delta_qr/1e9)
@printf("c_best_pcrb:  f_q=%.3fGHz E_C=%.3f κ=%.2fMHz Δ=%.3fGHz  sched=%s\n",
        c_best_pcrb.f_q_max/1e9, c_best_pcrb.E_C_over_h/1e9,
        c_best_pcrb.kappa/1e6, c_best_pcrb.Delta_qr/1e9, string(sched_best_pcrb))

omega_d_1 = omega_q(phi_star_fn(c_best_joint)[1], c_best_joint)
omega_d_2 = omega_q(phi_star_fn(c_best_pcrb)[1],  c_best_pcrb)

# joint_rich's terminal; default :mse but support either
terminal = get(joint, :terminal, :mse)

println("\n[Deploy joint-DP (terminal=$(terminal), rich, realistic box)]")
(V1_check, memo1, st1) = solve_bellman_full(grid, joint.K_EPOCHS, c_best_joint, omega_d_1; terminal=terminal)
@printf("  Re-solve Bellman at c₁*: V=%.4e  memo=%d  %.2fs\n",
        V1_check, st1.memo_size, st1.elapsed)

rng = MersenneTwister(2026)
t0 = time()
(MSE_1, se_1) = deployed_mse_adaptive(c_best_joint, memo1, omega_d_1, grid,
                                      joint.K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE̅₁ (joint-DP) = %.4e ± %.2e  %.1fs\n", MSE_1, se_1, time()-t0)

println("\n[Deploy PCRB-baseline (rich, realistic box)]")
rng = MersenneTwister(2026)
(MSE_2, se_2) = deployed_mse_fixed(c_best_pcrb, sched_best_pcrb,
                                   omega_d_2, grid; n_mc=N_MC, rng=rng)
@printf("  MSE̅₂ (PCRB) = %.4e ± %.2e\n", MSE_2, se_2)

JP2 = exp(log_JP_of_schedule(sched_best_pcrb, grid, c_best_pcrb, omega_d_2; J_0=1e-4))
pcrb_bound = 1 / JP2

println("\n" * "="^72)
println("HEADLINE (J=6, L=4, REALISTIC BOX, MSE-TERMINAL)")
println("-"^72)
@printf("  MSE̅₁ (joint DP)       = %.4e ± %.2e\n", MSE_1, se_1)
@printf("  MSE̅₂ (PCRB baseline)  = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  1/J_P(c₂*, s₂*) (CRB) = %.4e\n", pcrb_bound)
@printf("  ratio MSE̅₂ / MSE̅₁   = %.4f\n", MSE_2 / MSE_1)
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("  z = (MSE̅₂-MSE̅₁)/σ   = %+.2f σ\n", z)
println("="^72)

out_path = joinpath(@__DIR__, "results", "compare_mse_rich.jls")
open(out_path, "w") do io
    serialize(io, (; MSE_1, se_1, MSE_2, se_2, pcrb_bound, ratio=MSE_2/MSE_1,
                     c_1_star=c_best_joint, c_2_star=c_best_pcrb,
                     sched_2_star=sched_best_pcrb,
                     i_best_joint, i_best_pcrb,
                     N_MC=N_MC, K_PHI_POST=K_PHI_POST,
                     omega_d_1, omega_d_2, timestamp=now()))
end
println("Saved to $out_path")
