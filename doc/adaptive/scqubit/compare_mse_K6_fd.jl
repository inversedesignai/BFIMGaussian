#=
compare_mse_K6_fd.jl — MSE comparison for K=6, J=4, L=2 (ForwardDiff joint).
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

joint = deserialize(joinpath(@__DIR__, "results", "joint_K6_fd", "final.jls"))
pcrb  = deserialize(joinpath(@__DIR__, "results", "pcrb_K6",  "final.jls"))

grid = make_grid(; K_phi=K_PHI_POST, phi_max=0.49,
                   tau_grid=joint.TAU_GRID, n_grid=joint.N_GRID)
phi_star_fn = make_phi_star_fn()

nj = length(joint.history.c_vec)
i_best_joint = argmax(joint.history.V_adaptive[1:nj])
v = joint.history.c_vec[i_best_joint]
c_bj = ScqubitParams(f_q_max=v[1], E_C_over_h=v[2], kappa=v[3], Delta_qr=v[4],
                     temperature=v[5], A_phi=v[6], A_Ic=v[7])

i_best_pcrb = argmax(pcrb.history.log_JP)
v = pcrb.history.c_vec[i_best_pcrb]
c_bp = ScqubitParams(f_q_max=v[1], E_C_over_h=v[2], kappa=v[3], Delta_qr=v[4],
                     temperature=v[5], A_phi=v[6], A_Ic=v[7])
sched_bp = pcrb.history.sched[i_best_pcrb]

ωd1 = omega_q(phi_star_fn(c_bj)[1], c_bj)
ωd2 = omega_q(phi_star_fn(c_bp)[1], c_bp)
@printf("c_joint (FD): f_q=%.3f E_C=%.3f kappa=%.1e Delta=%.3f\n",
        c_bj.f_q_max/1e9, c_bj.E_C_over_h/1e9, c_bj.kappa, c_bj.Delta_qr/1e9)
@printf("c_pcrb:  f_q=%.3f E_C=%.3f kappa=%.1e Delta=%.3f sched=%s\n",
        c_bp.f_q_max/1e9, c_bp.E_C_over_h/1e9, c_bp.kappa, c_bp.Delta_qr/1e9, string(sched_bp))
flush(stdout)

println("\n[Deploy joint-DP (MSE terminal, K=6, J=4, FD)]")
t0 = time()
(V1, memo1, st1) = solve_bellman_full(grid, joint.K_EPOCHS, c_bj, ωd1; terminal=:mse)
@printf("  Re-solve Bellman at c₁*: V=%.4e memo=%d %.1fs\n", V1, st1.memo_size, time()-t0)
flush(stdout)

rng = MersenneTwister(2026); t0 = time()
(MSE_1, se_1) = deployed_mse_adaptive(c_bj, memo1, ωd1, grid, joint.K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE̅₁ = %.4e ± %.2e  (%.1fs)\n", MSE_1, se_1, time()-t0)
flush(stdout)

println("\n[Deploy PCRB baseline]")
rng = MersenneTwister(2026); t0 = time()
(MSE_2, se_2) = deployed_mse_fixed(c_bp, sched_bp, ωd2, grid; n_mc=N_MC, rng=rng)
@printf("  MSE̅₂ = %.4e ± %.2e  (%.1fs)\n", MSE_2, se_2, time()-t0)
flush(stdout)

JP2 = exp(log_JP_of_schedule(sched_bp, grid, c_bp, ωd2; J_0=1e-4))
crb = 1 / JP2

println("\n" * "="^72)
println("HEADLINE (K=6, J=4, L=2, realistic box, MSE-terminal, FD gradient)")
println("-"^72)
@printf("  MSE̅₁ (joint DP)       = %.4e ± %.2e\n", MSE_1, se_1)
@printf("  MSE̅₂ (PCRB baseline)  = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  1/J_P (CRB bound)     = %.4e\n", crb)
@printf("  ratio MSE̅₂/MSE̅₁    = %.4f\n", MSE_2/MSE_1)
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("  z-score               = %+.2f σ\n", z)
println("="^72)
flush(stdout)

open(joinpath(@__DIR__, "results", "compare_mse_K6_fd.jls"), "w") do io
    serialize(io, (; MSE_1, se_1, MSE_2, se_2, pcrb_bound=crb, ratio=MSE_2/MSE_1,
                     c_1_star=c_bj, c_2_star=c_bp, sched_2_star=sched_bp,
                     N_MC, K_PHI_POST, omega_d_1=ωd1, omega_d_2=ωd2, timestamp=now()))
end
println("Saved.")
