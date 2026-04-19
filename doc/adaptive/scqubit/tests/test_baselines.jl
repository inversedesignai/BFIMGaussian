# Tests for Baselines.jl — V_oracle, V_fixed enumeration + identities.
using Printf
using Test

include(joinpath(@__DIR__, "..", "ScqubitModel.jl"))
include(joinpath(@__DIR__, "..", "Belief.jl"))
include(joinpath(@__DIR__, "..", "Baselines.jl"))
using .ScqubitModel
using .Belief
using .Baselines

const c0 = PAPER_BASELINE
# Start with a small problem size to keep enumeration fast: K=2, J=4, L=2,
# n_grid=(1, 10), K_Φ=64.  V_oracle/V_fixed enumeration = 8^2 = 64 schedules.
const K_test = 2
const grid = make_grid(; K_phi=64,
    tau_grid = ntuple(k -> 20e-9 * 2.0^(k-1), 4),     # J=4 delays
    n_grid   = (1, 10))                                # L=2
const ω_d = omega_q(0.442, c0)

println("Problem size: K=$K_test, J=$(length(grid.tau_grid)), L=$(length(grid.n_grid)), K_Φ=$(length(grid.phi_grid))")
schedules = enumerate_schedules(grid, K_test)
println("  # schedules = $(length(schedules))")

println("\n[1] Precompute logp cache")
@time (logp, log1mp) = Baselines.logp_cache(grid, c0, ω_d)
println("  size(logp) = $(size(logp))")

println("\n[2] V_fixed(c₀)  (enumerate over all schedules, mean over Φ)")
@time (Vfx, s_star, phi_per_phi) = V_fixed(grid, K_test, c0, ω_d, schedules, logp, log1mp)
@printf("  V_fixed = %.6f nats\n", Vfx)
@printf("  best-fixed schedule = %s\n", string(s_star))

println("\n[3] V_oracle per φ on the grid")
V_or = zeros(length(grid.phi_grid))
best_s_or = Vector{Baselines.Schedule}(undef, length(grid.phi_grid))
@time for i in 1:length(grid.phi_grid)
    V_or[i], best_s_or[i] = V_oracle(i, grid, K_test, c0, ω_d, schedules, logp, log1mp)
end
@printf("  V_oracle mean over Φ = %.6f nats\n", sum(V_or)/length(V_or))
@printf("  V_oracle min / max    = %.6f / %.6f\n", minimum(V_or), maximum(V_or))

# ----------------------------------------------------------------
# [4] Identity check
# ----------------------------------------------------------------
println("\n[4] Identity: E[IG] = mean(V_oracle) - V_fixed")
# IG(φ) = V_oracle(φ) - Φ_value(φ, s_fixed_star)
mean_Vor = sum(V_or) / length(V_or)
E_IG = mean_Vor - Vfx
# Recompute Φ_value at s_star for each φ to check
phi_at_star = zeros(length(grid.phi_grid))
let nks = [grid.n_grid[s_star[k][2]] for k in 1:K_test]
    for i in 1:length(grid.phi_grid)
        phi_at_star[i] = Baselines.Phi_value(i, s_star, grid, nks, logp, log1mp)
    end
end
mean_phi_at_star = sum(phi_at_star) / length(phi_at_star)
@printf("  mean(V_oracle)        = %.10f\n", mean_Vor)
@printf("  V_fixed               = %.10f\n", Vfx)
@printf("  mean Φ(φ, s★_fixed)   = %.10f   (should equal V_fixed by definition)\n",
        mean_phi_at_star)
@printf("  E[IG] = mean_Vor - Vfx= %.10f\n", E_IG)
@test isapprox(mean_phi_at_star, Vfx; atol=1e-10)

# ----------------------------------------------------------------
# [5] IG(φ) ≥ 0 per φ
# ----------------------------------------------------------------
println("\n[5] IG(φ) = V_oracle(φ) - Φ(φ, s★_fixed) ≥ 0 per φ")
IG = V_or .- phi_at_star
@printf("  min(IG) = %+.6e  max(IG) = %+.6e\n", minimum(IG), maximum(IG))
@test minimum(IG) > -1e-9

# ----------------------------------------------------------------
# [6] V_oracle(φ) ≥ Φ(φ, s★_fixed) per φ (trivially by above)
# ----------------------------------------------------------------
println("\n[6] V_oracle(φ) ≥ 0 per φ  (entropy decrease is non-negative)")
@printf("  min(V_oracle) = %+.6f\n", minimum(V_or))
# V_oracle ≥ 0 because posterior entropy can't exceed prior entropy (Jensen's).
# But wait: V_oracle = E[-H(b_K)] + log(phi_max) = log(phi_max) - E[H(b_K)].
# If E[H(b_K)] > log(phi_max), V_oracle < 0. That can happen if H(b_K) computed
# with grid clipping (phi_clip) gives posterior entropy slightly above the
# prior's -log(1/phi_max) = log(phi_max).  We'll allow small numerical slop.
@test minimum(V_or) > -1e-6

println("\nAll Phase-3 tests passed.\n")
