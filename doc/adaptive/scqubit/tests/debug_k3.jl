# Diagnostic: at K=3, verify V_adaptive ≤ V_oracle_mean per φ.
using Printf

include(joinpath(@__DIR__, "..", "ScqubitModel.jl"))
include(joinpath(@__DIR__, "..", "Belief.jl"))
include(joinpath(@__DIR__, "..", "Baselines.jl"))
include(joinpath(@__DIR__, "..", "Bellman.jl"))
using .ScqubitModel
using .Belief
using .Baselines
using .Bellman

c0 = PAPER_BASELINE
grid = make_grid(; K_phi=64,
    tau_grid = ntuple(k -> 20e-9 * 2.0^(k-1), 4),
    n_grid   = (1, 10))
ω_d = omega_q(0.442, c0)

# V_adaptive + policy memo
(Vad, memo, stats) = solve_bellman_full(grid, 3, c0, ω_d)
@printf("V_adaptive(K=3) = %.8f nats (memo=%d)\n", Vad, stats.memo_size)

# V_oracle per φ via enumeration
schedules = enumerate_schedules(grid, 3)
(logp, log1mp) = Baselines.logp_cache(grid, c0, ω_d)
@printf("# schedules = %d\n", length(schedules))
(V_or_vals, _) = V_oracle_allphi(grid, 3, c0, ω_d, schedules, logp, log1mp)
V_or_mean = sum(V_or_vals) / length(V_or_vals)
@printf("V_oracle_mean(K=3) = %.8f  (per-φ: min=%.4f max=%.4f)\n",
        V_or_mean, minimum(V_or_vals), maximum(V_or_vals))

# Per-φ V_adaptive: evaluate by MC (sample obs trajectories under policy and
# compute mean −H(b_K) + log(phi_max))
using Random
Random.seed!(42)
function V_adaptive_at_phi(phi_idx::Int, memo, grid, c, omega_d, K::Int, n_traj::Int)
    phi_true = grid.phi_grid[phi_idx]
    J, L = length(grid.tau_grid), length(grid.n_grid)
    total = 0.0
    for _ in 1:n_traj
        counts = ntuple(_ -> (0, 0), J)
        for k in K:-1:1
            node = memo[(counts, k)]
            (j, ℓ) = node.action
            τj = grid.tau_grid[j]
            n  = grid.n_grid[ℓ]
            p = clamp(P1_ramsey(phi_true, τj, c, omega_d), 1e-300, 1 - 1e-16)
            # sample m ~ Binomial(n, p)
            m = 0
            for _ in 1:n
                m += rand() < p ? 1 : 0
            end
            counts = ntuple(kk -> kk == j ?
                (counts[kk][1] + n, counts[kk][2] + m) :
                counts[kk], J)
        end
        # final posterior entropy
        logb = Bellman._logb_from_counts(counts, Baselines.logp_cache(grid, c, omega_d)...)
        H = Bellman._entropy_from_logb(logb, grid.dphi)
        total += log(grid.phi_max) - H
    end
    total / n_traj
end

println("\nPer-φ comparison (V_adaptive_at_φ via 1000 MC rollouts vs V_oracle(φ)):")
violation_count = 0
for phi_idx in (1, 8, 16, 32, 48, 56, 64)
    V_ad_phi = V_adaptive_at_phi(phi_idx, memo, grid, c0, ω_d, 3, 2000)
    V_or_phi = V_or_vals[phi_idx]
    diff = V_ad_phi - V_or_phi
    flag = diff > 0.02 ? "  VIOLATION!" : ""
    @printf("  φ_idx=%3d  φ=%.4f   V_ad=%.4f  V_or=%.4f  Δ=%+.4f%s\n",
            phi_idx, grid.phi_grid[phi_idx], V_ad_phi, V_or_phi, diff, flag)
    if diff > 0.05
        violation_count += 1
    end
end
println("\nTotal violations beyond MC noise: $violation_count")
