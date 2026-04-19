# Tests for Bellman.jl — exact DP, monotonicity, brute-force cross-check at K=2.
using Printf
using Test

include(joinpath(@__DIR__, "..", "ScqubitModel.jl"))
include(joinpath(@__DIR__, "..", "Belief.jl"))
include(joinpath(@__DIR__, "..", "Baselines.jl"))
include(joinpath(@__DIR__, "..", "Bellman.jl"))
using .ScqubitModel
using .Belief
using .Baselines
using .Bellman

const c0 = PAPER_BASELINE
const grid = make_grid(; K_phi=64,
    tau_grid = ntuple(k -> 20e-9 * 2.0^(k-1), 4),   # J=4
    n_grid   = (1, 10))                              # L=2
const ω_d = omega_q(0.442, c0)

# ----------------------------------------------------------------
# [1] Bellman solve at K=2 and stats
# ----------------------------------------------------------------
println("\n[1] Bellman solve at K=2")
(Vadp_K2, memo_K2, stats_K2) = solve_bellman_full(grid, 2, c0, ω_d)
@printf("  V_adaptive(K=2) = %.6f nats  (memo size = %d, %.3f s)\n",
        Vadp_K2, stats_K2.memo_size, stats_K2.elapsed)

# ----------------------------------------------------------------
# [2] Compare Bellman (K=2) with fixed-schedule V_fixed — Bellman ≥ V_fixed
# ----------------------------------------------------------------
println("\n[2] V_adaptive ≥ V_fixed at K=2")
schedules = enumerate_schedules(grid, 2)
(logp, log1mp) = Baselines.logp_cache(grid, c0, ω_d)
(Vfx, s_star, _) = V_fixed(grid, 2, c0, ω_d, schedules, logp, log1mp)
@printf("  V_fixed   = %.6f nats\n", Vfx)
@printf("  V_adaptive= %.6f nats   (Δ = %+.6e)\n", Vadp_K2, Vadp_K2 - Vfx)
@test Vadp_K2 >= Vfx - 1e-9

# ----------------------------------------------------------------
# [3] Brute-force cross-check at K=2
#     V_adaptive_K=2 = max_{a_1} E_{m_1|a_1}[ max_{a_2} E_{m_2|a_1,m_1,a_2}[ -H(b_2) ] ] + log phi_max
# Enumerate explicitly.
# ----------------------------------------------------------------
println("\n[3] Brute-force verification at K=2")
function brute_force_K2(grid, c, omega_d, logp, log1mp)
    J, L = length(grid.tau_grid), length(grid.n_grid)
    Nphi = length(grid.phi_grid)
    # Start from counts = (0,0,0,0) for J delays
    counts0 = ntuple(_ -> (0, 0), J)
    logb0 = Bellman._logb_from_counts(counts0, logp, log1mp)
    # enumerate first action (j1, ℓ1)
    best_outer = -Inf
    for j1 in 1:J, ℓ1 in 1:L
        n1 = grid.n_grid[ℓ1]
        val_outer = 0.0
        for m1 in 0:n1
            p_m1 = Bellman._marg_obs(logb0, j1, n1, m1, logp, log1mp)
            p_m1 < 1e-20 && continue
            # successor belief after step 1
            counts1 = ntuple(k -> k == j1 ? (n1, m1) : (0, 0), J)
            logb1 = Bellman._logb_from_counts(counts1, logp, log1mp)
            # max over second action
            best_inner = -Inf
            for j2 in 1:J, ℓ2 in 1:L
                n2 = grid.n_grid[ℓ2]
                val_inner = 0.0
                for m2 in 0:n2
                    p_m2 = Bellman._marg_obs(logb1, j2, n2, m2, logp, log1mp)
                    p_m2 < 1e-20 && continue
                    counts2 = ntuple(k -> (counts1[k][1] + (k==j2 ? n2 : 0),
                                           counts1[k][2] + (k==j2 ? m2 : 0)), J)
                    logb2 = Bellman._logb_from_counts(counts2, logp, log1mp)
                    H = Bellman._entropy_from_logb(logb2, grid.dphi)
                    val_inner += p_m2 * (-H)
                end
                best_inner = max(best_inner, val_inner)
            end
            val_outer += p_m1 * best_inner
        end
        best_outer = max(best_outer, val_outer)
    end
    best_outer + log(grid.phi_max)
end
V_brute = brute_force_K2(grid, c0, ω_d, logp, log1mp)
@printf("  V_brute_K2 = %.10f  V_bellman_K2 = %.10f   |Δ| = %.3e\n",
        V_brute, Vadp_K2, abs(V_brute - Vadp_K2))
@test isapprox(V_brute, Vadp_K2; atol=1e-9)

# ----------------------------------------------------------------
# [4] Monotonicity in K: V_adaptive(K+1) ≥ V_adaptive(K)
# ----------------------------------------------------------------
println("\n[4] Monotonicity in K")
Vs = Dict{Int, Float64}()
for K in 1:3
    (V, _, s) = solve_bellman_full(grid, K, c0, ω_d)
    @printf("  K=%d  V_adaptive = %.6f   memo = %d  %.3f s\n",
            K, V, s.memo_size, s.elapsed)
    Vs[K] = V
end
@test Vs[1] <= Vs[2] + 1e-9
@test Vs[2] <= Vs[3] + 1e-9

# ----------------------------------------------------------------
# [5] V_adaptive ≤ mean V_oracle (oracle is upper bound on E over trajectories)
# ----------------------------------------------------------------
println("\n[5] V_adaptive ≤ mean V_oracle at K=2")
(V_or, _) = V_oracle_allphi(grid, 2, c0, ω_d, schedules, logp, log1mp)
meanV_or = sum(V_or) / length(V_or)
@printf("  mean V_oracle = %.6f   V_adaptive(K=2) = %.6f\n",
        meanV_or, Vadp_K2)
@test Vadp_K2 <= meanV_or + 1e-9

println("\nAll Phase-4 tests passed.\n")
