# Tests for Belief.jl — grid, log-density, update, entropy, marginal obs prob.
using Printf
using Test

include(joinpath(@__DIR__, "..", "ScqubitModel.jl"))
include(joinpath(@__DIR__, "..", "Belief.jl"))
using .ScqubitModel
using .Belief

const c0 = PAPER_BASELINE
const grid = make_grid(; K_phi=256, phi_max=0.49)
const ω_d = omega_q(0.442, c0)

# ----------------------------------------------------------------
# 1. Prior entropy = log(phi_max) in nats
# ----------------------------------------------------------------
println("\n[1] Prior entropy = log(phi_max)")
b0 = prior_belief(grid)
H0 = entropy_nats(b0, grid)
H_expected = log(grid.phi_max)
@printf("  H(prior) = %.10f  expected log(phi_max) = %.10f  |Δ| = %.3e\n",
        H0, H_expected, abs(H0 - H_expected))
@test isapprox(H0, H_expected; atol=1e-10)

# ----------------------------------------------------------------
# 2. Posterior narrows after an informative measurement
# ----------------------------------------------------------------
println("\n[2] Posterior entropy decreases after a non-trivial measurement")
# pick τ = τ_opt(0.442) scale ~ 5e-7 s, n=10 shots, m=5 heads
j_idx = 5           # tau_grid[5] = 10e-9 * 2^4 = 160 ns
n = 10; m = 5
b1 = update_belief(b0, j_idx, n, m, grid, c0, ω_d)
H1 = entropy_nats(b1, grid)
@printf("  H(b0) = %.6f  H(b1) = %.6f  Δ = %+.6f\n", H0, H1, H1 - H0)
@test H1 < H0

# ----------------------------------------------------------------
# 3. Order independence of count-statistic representation
# ----------------------------------------------------------------
println("\n[3] Order-invariance: two updates in either order yield same logb")
j1, n1, m1 = 3, 4, 2
j2, n2, m2 = 5, 6, 3
bA = update_belief(b0, j1, n1, m1, grid, c0, ω_d)
bA = update_belief(bA, j2, n2, m2, grid, c0, ω_d)
bB = update_belief(b0, j2, n2, m2, grid, c0, ω_d)
bB = update_belief(bB, j1, n1, m1, grid, c0, ω_d)
max_diff = maximum(abs.(bA.logb .- bB.logb))
@printf("  max |Δlogb| across orderings = %.3e\n", max_diff)
@test max_diff < 1e-12
@test bA.counts == bB.counts

# ----------------------------------------------------------------
# 4. Re-computing logb from counts matches incremental logb
# ----------------------------------------------------------------
println("\n[4] posterior_logb(counts) matches incremental logb")
# chain a few updates at various indices
let btmp = prior_belief(grid)
    for (j, nn, mm) in ((1, 3, 1), (4, 8, 5), (6, 2, 0), (2, 12, 7))
        btmp = update_belief(btmp, j, nn, mm, grid, c0, ω_d)
    end
    global b_chain = btmp
end
logb_from_counts = Belief.posterior_logb(grid, b_chain.counts, c0, ω_d)
max_diff = maximum(abs.(logb_from_counts .- b_chain.logb))
@printf("  max |Δ logb| = %.3e  counts=%s\n", max_diff, string(b_chain.counts))
@test max_diff < 1e-12

# ----------------------------------------------------------------
# 5. Marginal observation probabilities sum to 1
# ----------------------------------------------------------------
println("\n[5] Σ_m P(m | b, τ_j, n) = 1 for every (b, j, n)")
worst_val = let w = 0.0
    for b_ in (b0, b1, b_chain), (j, nn) in ((1, 3), (3, 10), (5, 30))
        ptotal = 0.0
        for m in 0:nn
            ptotal += marg_obs_prob(b_, j, nn, m, grid, c0, ω_d)
        end
        w = max(w, abs(ptotal - 1.0))
    end
    w
end
@printf("  max |Σ P(m) - 1| over tested cases = %.3e\n", worst_val)
@test worst_val < 1e-9

# ----------------------------------------------------------------
# 6. Density sums to 1/dphi
# ----------------------------------------------------------------
println("\n[6] belief_density integrates to 1 on the grid")
for b_ in (b0, b1, b_chain)
    dens = belief_density(b_, grid)
    integ = sum(dens) * grid.dphi
    @printf("  ∫ b(φ) dφ = %.10f\n", integ)
    @test isapprox(integ, 1.0; atol=1e-12)
end

println("\nAll Phase-2 tests passed.\n")
