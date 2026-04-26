#=
compare_mse_multistart_global.jl — paired Monte Carlo deployment of the
global-best joint-DP and PCRB optima from the multistart experiments.

Reads:
  results/joint_multistart/_summary.jls  (selects argmax V_best)
  results/pcrb_multistart/_summary.jls   (selects argmax log_JP_best)

Writes:
  results/compare_mse_multistart_global.jls
=#
using Printf, Random, Serialization, Dates

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "BellmanThreaded.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel, .Belief, .Bellman, .BellmanThreaded, .Gradient, .JointOpt, .PCRB

const N_MC = parse(Int, get(ENV, "MSE_N", "20000"))
const K_PHI_POST = parse(Int, get(ENV, "MSE_K_PHI", "256"))
const J_TAU = 10
const TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID = (1, 10)
const PHI_MAX = 0.1
const K_EPOCHS = 4

# ---- Pick global-best joint-DP ----
joint_summary = deserialize(joinpath(@__DIR__, "results", "joint_multistart", "_summary.jls"))
i_g_joint = joint_summary.i_global
joint_results = joint_summary.all_results
println("Joint-DP multistart summary:")
for r in joint_results
    @printf("  %-10s V_best=%+.6e\n", r.init_id, r.V_best)
end
println("  Global best: $(joint_results[i_g_joint].init_id)")
joint_winner_id = joint_results[i_g_joint].init_id
joint_winner = deserialize(joinpath(@__DIR__, "results", "joint_multistart", "$(joint_winner_id).jls"))
v_bj = joint_winner.v_best
c_bj = ScqubitParams(f_q_max=v_bj[1], E_C_over_h=v_bj[2], kappa=v_bj[3], Delta_qr=v_bj[4],
                     temperature=v_bj[5], A_phi=v_bj[6], A_Ic=v_bj[7])

# ---- Pick global-best PCRB ----
pcrb_summary = deserialize(joinpath(@__DIR__, "results", "pcrb_multistart", "_summary.jls"))
i_g_pcrb = pcrb_summary.i_global
pcrb_results = pcrb_summary.all_results
println("\nPCRB multistart summary:")
for r in pcrb_results
    @printf("  %-10s log_JP_best=%+.6f\n", r.init_id, r.logJP_best)
end
println("  Global best: $(pcrb_results[i_g_pcrb].init_id)")
pcrb_winner_id = pcrb_results[i_g_pcrb].init_id
pcrb_winner = deserialize(joinpath(@__DIR__, "results", "pcrb_multistart", "$(pcrb_winner_id).jls"))
v_bp = pcrb_winner.v_best
c_bp = ScqubitParams(f_q_max=v_bp[1], E_C_over_h=v_bp[2], kappa=v_bp[3], Delta_qr=v_bp[4],
                     temperature=v_bp[5], A_phi=v_bp[6], A_Ic=v_bp[7])
sched_bp = pcrb_winner.sched_best

# ---- Set up deployment grid + operating points ----
grid = make_grid(; K_phi=K_PHI_POST, phi_max=PHI_MAX,
                   tau_grid=TAU_GRID, n_grid=N_GRID)
phi_star_fn = make_phi_star_fn()
ωd1 = omega_q(phi_star_fn(c_bj)[1], c_bj)
ωd2 = omega_q(phi_star_fn(c_bp)[1], c_bp)

@printf("\nc_joint*: f_q=%.4f E_C=%.4f κ=%.4f MHz Δ=%.4f  (init=%s, V_best=%.4e)\n",
        c_bj.f_q_max/1e9, c_bj.E_C_over_h/1e9, c_bj.kappa/1e6, c_bj.Delta_qr/1e9,
        joint_winner_id, joint_winner.V_best)
@printf("c_pcrb*:  f_q=%.4f E_C=%.4f κ=%.4f MHz Δ=%.4f  sched=%s  (init=%s, log_JP=%.4f)\n",
        c_bp.f_q_max/1e9, c_bp.E_C_over_h/1e9, c_bp.kappa/1e6, c_bp.Delta_qr/1e9,
        string(sched_bp), pcrb_winner_id, pcrb_winner.logJP_best)
flush(stdout)

# ---- Deploy joint-DP ----
println("\n[Deploy joint-DP, K_PHI=$K_PHI_POST]")
t0 = time()
(V1, memo1, st1) = solve_bellman_threaded_full(grid, K_EPOCHS, c_bj, ωd1; terminal=:mse)
@printf("  Re-solve V=%.4e memo=%d %.1fs (%d threads)\n",
        V1, st1.memo_size, time()-t0, st1.n_threads); flush(stdout)
rng = MersenneTwister(2026); t0 = time()
(MSE_1, se_1) = deployed_mse_adaptive(c_bj, memo1, ωd1, grid, K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE̅₁ = %.4e ± %.2e   (%.1fs)\n", MSE_1, se_1, time()-t0); flush(stdout)

# ---- Deploy PCRB ----
println("\n[Deploy PCRB, K_PHI=$K_PHI_POST]")
rng = MersenneTwister(2026); t0 = time()
(MSE_2, se_2) = deployed_mse_fixed(c_bp, sched_bp, ωd2, grid; n_mc=N_MC, rng=rng)
@printf("  MSE̅₂ = %.4e ± %.2e   (%.1fs)\n", MSE_2, se_2, time()-t0); flush(stdout)

JP2 = exp(log_JP_of_schedule(sched_bp, grid, c_bp, ωd2; J_0=1e-4))
crb = 1 / JP2

println("\n" * "="^72)
@printf("HEADLINE — multistart-global (K=4, J=10, L=2, phi_max=%.3f)\n", PHI_MAX)
@printf("  joint-DP best init: %s   PCRB best init: %s\n",
        joint_winner_id, pcrb_winner_id)
println("-"^72)
@printf("  MSE̅₁ (joint-DP)    = %.4e ± %.2e\n", MSE_1, se_1)
@printf("  MSE̅₂ (PCRB)        = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  CRB lower bound     = %.4e\n", crb)
@printf("  ratio MSE̅₂/MSE̅₁   = %.3f\n", MSE_2/MSE_1)
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("  z-score             = %+.2f σ\n", z)
println("="^72); flush(stdout)

open(joinpath(@__DIR__, "results", "compare_mse_multistart_global.jls"), "w") do io
    serialize(io, (; MSE_1, se_1, MSE_2, se_2, pcrb_bound=crb,
                     ratio=MSE_2/MSE_1, z,
                     c_1_star=c_bj, c_2_star=c_bp, sched_2_star=sched_bp,
                     joint_winner_id, pcrb_winner_id,
                     joint_results, pcrb_results,
                     N_MC, K_PHI_POST, PHI_MAX,
                     omega_d_1=ωd1, omega_d_2=ωd2,
                     timestamp=now()))
end
println("Saved.")
