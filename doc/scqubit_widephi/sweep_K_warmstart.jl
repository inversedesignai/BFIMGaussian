#=
sweep_K_warmstart.jl — sweep horizon K = 4, 5, 6 with the K=4 BayesOpt
winners (joint and PCRB) held fixed.  Isolates the "K alone" effect on the
joint-DP / PCRB ratio at phi_max = 0.5, without re-optimizing geometry.

This is a feasibility-bounded experiment.  Full BayesOpt at K >= 5 is too
costly because at phi_max = 0.5 the multimodal posterior prevents
count-tuple memo coalescing, so each Bellman eval blows up to ~hours of
wallclock and ~TB of memory.  Warm-starting at the K=4 winner geometry
sidesteps this: we do one Bellman solve per K (joint), and one log_JP
per K (PCRB), then deploy via MC.

Reads:
  results/bayesopt_joint_widephi_K4/result.jls
  results/bayesopt_pcrb_widephi_K4/result.jls

Writes:
  results/sweep_K_warmstart.jls
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

const PHI_MAX  = 0.5
const J_TAU    = 10
const TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID   = (1, 10)
const TWO_PI   = 2π
const K_VALUES = parse.(Int, split(get(ENV, "K_VALUES", "4,5,6"), ","))
const K_PHI    = parse(Int, get(ENV, "K_PHI", "256"))
const N_MC     = parse(Int, get(ENV, "MSE_N", "20000"))

println("sweep_K_warmstart.jl  (PHI_MAX=$(PHI_MAX); K_VALUES=$(K_VALUES))"); flush(stdout)
println("Threads: $(Threads.nthreads())"); flush(stdout)

j4 = deserialize(joinpath(@__DIR__, "results", "bayesopt_joint_widephi_K4", "result.jls"))
p4 = deserialize(joinpath(@__DIR__, "results", "bayesopt_pcrb_widephi_K4",  "result.jls"))

v1  = j4.v_best;  c1 = vec_as_c(v1);  ωd1 = j4.omega_d_best
v2  = p4.v_best;  c2 = vec_as_c(v2);  ωd2 = p4.omega_d_best

@printf("c_joint*  : f_q=%.4f  E_C=%.4f  κ=%.4f MHz  Δ=%.4f   ω_d/(2π)=%.4f GHz\n",
        c1.f_q_max/1e9, c1.E_C_over_h/1e9, c1.kappa/1e6, c1.Delta_qr/1e9, ωd1/TWO_PI/1e9)
@printf("c_pcrb*   : f_q=%.4f  E_C=%.4f  κ=%.4f MHz  Δ=%.4f   ω_d/(2π)=%.4f GHz\n",
        c2.f_q_max/1e9, c2.E_C_over_h/1e9, c2.kappa/1e6, c2.Delta_qr/1e9, ωd2/TWO_PI/1e9)
flush(stdout)

results = Vector{NamedTuple}(undef, length(K_VALUES))
grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)

for (idx, K) in enumerate(K_VALUES)
    @printf("\n%s\n=== K = %d ===\n%s\n", "="^60, K, "="^60); flush(stdout)

    # Joint: solve Bellman at K with the K=4 winner geometry
    @printf("[joint K=%d] solving Bellman (K_phi=%d)...\n", K, K_PHI); flush(stdout)
    t0 = time()
    (V1, memo1, _) = solve_bellman_threaded_full(grid, K, c1, ωd1; terminal=:mse)
    bell_t = time() - t0
    @printf("  V_adaptive = %.6e   memo_size=%d   %.1f min\n", V1, length(memo1), bell_t/60); flush(stdout)

    # MC deployment
    @printf("[joint K=%d] MC deployment (n_mc=%d)...\n", K, N_MC); flush(stdout)
    rng = MersenneTwister(2026); t0 = time()
    (MSE_1, se_1) = deployed_mse_adaptive(c1, memo1, ωd1, grid, K; n_mc=N_MC, rng=rng)
    @printf("  MC MSE_1 = %.4e ± %.2e   (%.1f s)\n", MSE_1, se_1, time()-t0); flush(stdout)

    # PCRB: at this K, the optimal schedule is (10, 2)^K (n=10 at longest tau, K times)
    pcrb_sched = [(J_TAU, 2) for _ in 1:K]
    log_JP_2 = log_JP_of_schedule(pcrb_sched, grid, c2, ωd2; J_0=1e-4)
    @printf("[pcrb K=%d] log_JP=%.4f   schedule=%s\n", K, log_JP_2, string(pcrb_sched)); flush(stdout)

    @printf("[pcrb K=%d] MC deployment...\n", K); flush(stdout)
    rng = MersenneTwister(2026); t0 = time()
    (MSE_2, se_2) = deployed_mse_fixed(c2, pcrb_sched, ωd2, grid; n_mc=N_MC, rng=rng)
    @printf("  MC MSE_2 = %.4e ± %.2e   (%.1f s)\n", MSE_2, se_2, time()-t0); flush(stdout)

    ratio_mc    = MSE_2 / MSE_1
    ratio_exact = MSE_2 / (-V1)
    z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)

    @printf("\n  --- K=%d summary ---\n", K)
    @printf("    -V_adaptive (exact)            = %.4e\n", -V1)
    @printf("    MC MSE_1 (joint-DP)            = %.4e ± %.2e\n", MSE_1, se_1)
    @printf("    MC MSE_2 (PCRB)                = %.4e ± %.2e\n", MSE_2, se_2)
    @printf("    ratio MC: MSE_2 / MSE_1        = %.3f\n", ratio_mc)
    @printf("    ratio exact: MSE_2 / (-V)      = %.3f\n", ratio_exact)
    @printf("    z-score                        = %+.2f σ\n", z)
    flush(stdout)

    results[idx] = (; K, V1, mmse_exact=-V1,
                     MSE_1, se_1, MSE_2, se_2, ratio_mc, ratio_exact, z,
                     memo_size=length(memo1), bell_t, log_JP_2, pcrb_sched)

    # Drop large memo before next K to free memory
    memo1 = nothing
    GC.gc()
end

println("\n", "="^72)
println("WARM-START K SWEEP SUMMARY (PHI_MAX=$(PHI_MAX), c fixed at K=4 winners)")
println("-"^72)
@printf("%4s  %12s  %12s  %12s  %8s  %8s\n",
        "K", "-V (exact)", "MSE_1 (MC)", "MSE_2 (MC)", "ratio_mc", "z-score")
for r in results
    @printf("%4d  %12.4e  %12.4e  %12.4e  %8.3f  %+8.2f σ\n",
            r.K, r.mmse_exact, r.MSE_1, r.MSE_2, r.ratio_mc, r.z)
end
println("="^72); flush(stdout)

open(joinpath(@__DIR__, "results", "sweep_K_warmstart.jls"), "w") do io
    serialize(io, (; results, K_VALUES, PHI_MAX, K_PHI,
                     c_joint=c1, c_pcrb=c2, omega_d_joint=ωd1, omega_d_pcrb=ωd2,
                     N_MC, timestamp=now()))
end
println("Saved.")
