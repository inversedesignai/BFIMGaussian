#=
compare_mse_tier2.jl — tier-2 (11-D c) MSE comparison: joint-DP vs joint-PCRB.
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

const N_MC       = parse(Int, get(ENV, "MSE_N",     "20000"))
const K_PHI_POST = parse(Int, get(ENV, "MSE_K_PHI", "256"))

function vec_as_c_t2(v::AbstractVector{T}) where {T<:Real}
    ScqubitParams{T}(
        f_q_max=v[1], E_C_over_h=v[2], kappa=v[3], Delta_qr=v[4],
        temperature=v[5], A_phi=v[6], A_Ic=v[7],
        M=v[8], Mprime=v[9], C_qg=v[10], C_c=v[11])
end

joint = deserialize(joinpath(@__DIR__, "results", "joint_tier2", "final.jls"))
pcrb  = deserialize(joinpath(@__DIR__, "results", "pcrb_tier2",  "final.jls"))

grid = make_grid(; K_phi=K_PHI_POST, phi_max=0.49,
                   tau_grid=joint.TAU_GRID, n_grid=joint.N_GRID)

nj = length(joint.history.c_vec)
i_best_joint = argmax(joint.history.V_adaptive[1:nj])
v_bj = joint.history.c_vec[i_best_joint]
c_bj = vec_as_c_t2(v_bj)

i_best_pcrb = argmax(pcrb.history.log_JP)
v_bp = pcrb.history.c_vec[i_best_pcrb]
c_bp = vec_as_c_t2(v_bp)
sched_bp = pcrb.history.sched[i_best_pcrb]

phi_star_fn = make_phi_star_fn()
ωd1 = omega_q(phi_star_fn(c_bj)[1], c_bj)
ωd2 = omega_q(phi_star_fn(c_bp)[1], c_bp)

println("\nc_best_joint (tier-2):")
@printf("  f_q=%.3fGHz E_C=%.3f κ=%.2eMHz Δ=%.3fGHz  M=%.2e M'=%.2e C_qg=%.2e C_c=%.2e\n",
        c_bj.f_q_max/1e9, c_bj.E_C_over_h/1e9, c_bj.kappa/1e6, c_bj.Delta_qr/1e9,
        c_bj.M, c_bj.Mprime, c_bj.C_qg, c_bj.C_c)
println("c_best_pcrb (tier-2):")
@printf("  f_q=%.3fGHz E_C=%.3f κ=%.2eMHz Δ=%.3fGHz  M=%.2e M'=%.2e C_qg=%.2e C_c=%.2e sched=%s\n",
        c_bp.f_q_max/1e9, c_bp.E_C_over_h/1e9, c_bp.kappa/1e6, c_bp.Delta_qr/1e9,
        c_bp.M, c_bp.Mprime, c_bp.C_qg, c_bp.C_c, string(sched_bp))

println("\n[Deploy joint-DP (tier-2, MSE terminal)]")
t0 = time()
(V1, memo1, st1) = solve_bellman_full(grid, joint.K_EPOCHS, c_bj, ωd1; terminal=:mse)
@printf("  Re-solve Bellman at c₁*: V=%.4e memo=%d %.1fs\n", V1, st1.memo_size, time()-t0)

rng = MersenneTwister(2026)
t0 = time()
(MSE_1, se_1) = deployed_mse_adaptive(c_bj, memo1, ωd1, grid, joint.K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE̅₁ = %.4e ± %.2e  (%.1fs)\n", MSE_1, se_1, time()-t0)

println("\n[Deploy PCRB baseline (tier-2)]")
rng = MersenneTwister(2026); t0 = time()
(MSE_2, se_2) = deployed_mse_fixed(c_bp, sched_bp, ωd2, grid; n_mc=N_MC, rng=rng)
@printf("  MSE̅₂ = %.4e ± %.2e  (%.1fs)\n", MSE_2, se_2, time()-t0)

JP2 = exp(log_JP_of_schedule(sched_bp, grid, c_bp, ωd2; J_0=1e-4))
crb = 1 / JP2

println("\n" * "="^72)
println("HEADLINE (TIER-2, 11-D c, J=20, L=2, K=3, MSE-terminal)")
println("-"^72)
@printf("  MSE̅₁ (joint DP, tier-2)       = %.4e ± %.2e\n", MSE_1, se_1)
@printf("  MSE̅₂ (PCRB baseline, tier-2)  = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  1/J_P (CRB bound)              = %.4e\n", crb)
@printf("  ratio MSE̅₂/MSE̅₁             = %.4f\n", MSE_2 / MSE_1)
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("  z-score                        = %+.2f σ\n", z)
println("="^72)
println("(Tier-1 headline for reference: ratio 1.134 at z=+11.54σ)")

open(joinpath(@__DIR__, "results", "compare_mse_tier2.jls"), "w") do io
    serialize(io, (; MSE_1, se_1, MSE_2, se_2, pcrb_bound=crb, ratio=MSE_2/MSE_1,
                     c_1_star=c_bj, c_2_star=c_bp, sched_2_star=sched_bp,
                     N_MC, K_PHI_POST, omega_d_1=ωd1, omega_d_2=ωd2, timestamp=now()))
end
println("Saved.")
