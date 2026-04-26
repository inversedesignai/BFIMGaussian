#=
compare_mse_5d.jl — paired Monte Carlo deployment of the 5-D BayesOpt
joint-DP and PCRB winners (where ω_d is a free parameter, not derived
from c via single-shot Fisher logic).

Reads:
  results/bayesopt_joint_5d/result.jls
  results/bayesopt_pcrb_5d/result.jls

Writes:
  results/compare_mse_5d.jls
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

const N_MC       = parse(Int, get(ENV, "MSE_N", "20000"))
const K_PHI_POST = parse(Int, get(ENV, "MSE_K_PHI", "256"))
const J_TAU      = 10
const TAU_GRID   = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID     = (1, 10)
const PHI_MAX    = 0.1
const K_EPOCHS   = 4
const TWO_PI     = 2π

println("compare_mse_5d.jl  (5-D BayesOpt: ω_d free)"); flush(stdout)
println("Threads: $(Threads.nthreads())"); flush(stdout)

j = deserialize(joinpath(@__DIR__, "results", "bayesopt_joint_5d", "result.jls"))
p = deserialize(joinpath(@__DIR__, "results", "bayesopt_pcrb_5d",  "result.jls"))

v1  = j.v_best;  c1 = vec_as_c(v1);  ωd1 = j.omega_d_best
v2  = p.v_best;  c2 = vec_as_c(v2);  ωd2 = p.omega_d_best
sched2 = p.fixed_schedule

@printf("c_joint*  : f_q=%.4f  E_C=%.4f  κ=%.4f MHz  Δ=%.4f   V_train=%.4e\n",
        c1.f_q_max/1e9, c1.E_C_over_h/1e9, c1.kappa/1e6, c1.Delta_qr/1e9, j.V_best)
@printf("ω_d_joint*/(2π) = %.4f GHz   (would-be ω_d_fn at c1: %.4f GHz)\n",
        ωd1/TWO_PI/1e9, j.omega_d_fn_at_best/TWO_PI/1e9)
@printf("c_pcrb*   : f_q=%.4f  E_C=%.4f  κ=%.4f MHz  Δ=%.4f   log_JP=%.4f\n",
        c2.f_q_max/1e9, c2.E_C_over_h/1e9, c2.kappa/1e6, c2.Delta_qr/1e9, p.logJP_best)
@printf("ω_d_pcrb*/(2π)  = %.4f GHz   (would-be ω_d_fn at c2: %.4f GHz)   sched=%s\n",
        ωd2/TWO_PI/1e9, p.omega_d_fn_at_best/TWO_PI/1e9, string(sched2))
flush(stdout)

grid = make_grid(; K_phi=K_PHI_POST, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)

println("\n[Deploy joint-DP at K_PHI=$K_PHI_POST]")
t = time()
(V1, memo1, _) = solve_bellman_threaded_full(grid, K_EPOCHS, c1, ωd1; terminal=:mse)
@printf("  Re-solve V=%.4e  %.1fs\n", V1, time()-t)
rng = MersenneTwister(2026); t = time()
(MSE_1, se_1) = deployed_mse_adaptive(c1, memo1, ωd1, grid, K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE_1 = %.4e ± %.2e  (%.1fs)\n", MSE_1, se_1, time()-t)
flush(stdout)

println("\n[Deploy PCRB at K_PHI=$K_PHI_POST]")
rng = MersenneTwister(2026); t = time()
(MSE_2, se_2) = deployed_mse_fixed(c2, sched2, ωd2, grid; n_mc=N_MC, rng=rng)
@printf("  MSE_2 = %.4e ± %.2e  (%.1fs)\n", MSE_2, se_2, time()-t)
flush(stdout)

JP2 = exp(log_JP_of_schedule(sched2, grid, c2, ωd2; J_0=1e-4))
crb = 1/JP2
ratio = MSE_2 / MSE_1
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)

println("\n" * "="^72)
println("HEADLINE 5-D — both BayesOpt with ω_d free, K_PHI=256")
println("-"^72)
@printf("  MSE_1 (joint-DP, 5-D BayesOpt) = %.4e ± %.2e\n", MSE_1, se_1)
@printf("  MSE_2 (PCRB,     5-D BayesOpt) = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  CRB lower bound                = %.4e\n", crb)
@printf("  ratio MSE_2/MSE_1              = %.3f\n", ratio)
@printf("  z-score                        = %+.2f σ\n", z)
println("="^72); flush(stdout)

open(joinpath(@__DIR__, "results", "compare_mse_5d.jls"), "w") do io
    serialize(io, (; MSE_1, se_1, MSE_2, se_2, ratio, z, pcrb_bound=crb,
                     c_1_star=c1, c_2_star=c2,
                     omega_d_1=ωd1, omega_d_2=ωd2,
                     omega_d_fn_at_c1=j.omega_d_fn_at_best,
                     omega_d_fn_at_c2=p.omega_d_fn_at_best,
                     sched_2_star=sched2,
                     N_MC, K_PHI_POST, PHI_MAX,
                     timestamp=now()))
end
println("Saved.")
