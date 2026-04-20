#=
compare_mse_narrow_baseline.jl — Clean headline for narrow-prior regime.

K=4, J=10, L=2, phi_max=0.1, PAPER_BASELINE c (no c-optimization).

Expected: adaptive joint-DP beats PCRB by ~8× (ratio ≥ 8.0).

This is the cleanest demonstration that the joint-DP advantage emerges in
regimes where PCRB's Fisher-optimal schedule suffers from multi-modal
posterior bias.  At the full phi_max=0.49 prior, both policies are near
prior variance; narrowing the prior to [0, 0.1] exposes the regime where
adaptive disambiguation matters decisively.
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

const K_EPOCHS = 4
const J_TAU = 10
const PHI_MAX = 0.1
const N_MC = 40000  # double the default for tighter error bars
const K_PHI = 256   # same grid for training and deployment

TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
N_GRID = (1, 10)

c = PAPER_BASELINE
phi_star_fn = make_phi_star_fn()
ωd = omega_q(phi_star_fn(c)[1], c)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)

@printf("scqubit narrow-prior headline (BASELINE c)\n")
@printf("K=%d J=%d L=%d K_Φ=%d phi_max=%.3f N_MC=%d\n",
        K_EPOCHS, J_TAU, length(N_GRID), K_PHI, PHI_MAX, N_MC)
@printf("c = PAPER_BASELINE: f_q=%.2f GHz, E_C/h=%.3f GHz, κ=%.2e Hz, Δ=%.2f GHz\n",
        c.f_q_max/1e9, c.E_C_over_h/1e9, c.kappa, c.Delta_qr/1e9)
@printf("ω_d = %.4e rad/s  (at φ*=%.4f)\n", ωd, phi_star_fn(c)[1])
@printf("prior variance = %.4e\n", PHI_MAX^2/12)
flush(stdout)

println("\n[1/4] Building adaptive policy memo at K_PHI=$K_PHI...")
t0 = time()
(V_ad, memo_ad, st_ad) = solve_bellman_full(grid, K_EPOCHS, c, ωd; terminal=:mse)
@printf("  V_adaptive (grid-variance) = %.4e  memo=%d  %.1fs\n",
        V_ad, st_ad.memo_size, time()-t0)
flush(stdout)

println("\n[2/4] Deploying adaptive policy (MC)...")
rng = MersenneTwister(2026); t0 = time()
(MSE_1, se_1) = deployed_mse_adaptive(c, memo_ad, ωd, grid, K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE̅_adaptive = %.4e ± %.2e  (%.1fs)\n", MSE_1, se_1, time()-t0)
flush(stdout)

println("\n[3/4] Finding optimal PCRB schedule (enumerate)...")
t0 = time()
(sched_pcrb, logJP_pcrb) = argmax_schedule_enumerate(grid, c, ωd, K_EPOCHS; J_0=1e-4)
@printf("  best schedule: %s  log J_P=%.4f  (%.1fs)\n",
        string(sched_pcrb), logJP_pcrb, time()-t0)
flush(stdout)

println("\n[4/4] Deploying PCRB schedule (MC)...")
rng = MersenneTwister(2026); t0 = time()
(MSE_2, se_2) = deployed_mse_fixed(c, sched_pcrb, ωd, grid; n_mc=N_MC, rng=rng)
@printf("  MSE̅_pcrb = %.4e ± %.2e  (%.1fs)\n", MSE_2, se_2, time()-t0)
flush(stdout)

crb = 1 / exp(logJP_pcrb)

println("\n" * "="^72)
println("HEADLINE")
println("-"^72)
@printf("  prior variance         = %.4e\n", PHI_MAX^2/12)
@printf("  MSE̅_adaptive (joint DP) = %.4e ± %.2e\n", MSE_1, se_1)
@printf("  MSE̅_pcrb (baseline)     = %.4e ± %.2e\n", MSE_2, se_2)
@printf("  1/J_P (CRB bound)       = %.4e\n", crb)
@printf("  ratio MSE̅_pcrb/MSE̅_adaptive = %.4f\n", MSE_2/MSE_1)
z = (MSE_2 - MSE_1) / sqrt(se_1^2 + se_2^2)
@printf("  z-score                 = %+.2f σ\n", z)
@printf("  gap (%%)                = %.1f%%\n", 100 * (MSE_2/MSE_1 - 1.0))
println("="^72)
flush(stdout)

open(joinpath(@__DIR__, "results", "compare_mse_narrow_baseline.jls"), "w") do io
    serialize(io, (; MSE_1, se_1, MSE_2, se_2, pcrb_bound=crb, ratio=MSE_2/MSE_1,
                     c=c, sched_pcrb,
                     N_MC, K_PHI, PHI_MAX, omega_d=ωd, timestamp=now()))
end
println("Saved.")
