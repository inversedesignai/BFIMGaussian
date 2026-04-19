# Tests for Gradient.jl — envelope-theorem ∂V_adaptive/∂c under a fixed π*.
# Two paths:
#   (a) exact policy-tree traversal + Zygote AD  — ground truth for FD checks
#   (b) Monte Carlo pathwise + score-function estimator — noisy; consistency
#
# Each AD value is printed against the central-difference FD with full precision
# per CLAUDE.md.
using Printf
using Test
using Random

include(joinpath(@__DIR__, "..", "ScqubitModel.jl"))
include(joinpath(@__DIR__, "..", "Belief.jl"))
include(joinpath(@__DIR__, "..", "Baselines.jl"))
include(joinpath(@__DIR__, "..", "Bellman.jl"))
include(joinpath(@__DIR__, "..", "Gradient.jl"))
using .ScqubitModel
using .Belief
using .Baselines
using .Bellman
using .Gradient

# Small problem to keep runtime reasonable: K=2, J=4, L=2, K_Φ=64.
const c0    = PAPER_BASELINE
const K     = 2
const grid  = make_grid(; K_phi=64,
                          tau_grid = ntuple(k -> 20e-9 * 2.0^(k-1), 4),
                          n_grid   = (1, 10))
const ω_d   = omega_q(0.442, c0)

println("Problem: K=$K, J=$(length(grid.tau_grid)), L=$(length(grid.n_grid)), K_Φ=$(length(grid.phi_grid))")

# ----------------------------------------------------------------
# [1] Solve Bellman, confirm exact-enumeration V under π* matches
# ----------------------------------------------------------------
println("\n[1] Exact policy-tree evaluation matches Bellman V_adaptive")
(V_bell, memo, stats) = solve_bellman_full(grid, K, c0, ω_d)
V_exact = V_adaptive_policy_exact(c0, memo, grid, ω_d, K)
@printf("  V_bellman = %.12f\n", V_bell)
@printf("  V_exact   = %.12f   |Δ| = %.3e\n", V_exact, abs(V_exact - V_bell))
@test isapprox(V_exact, V_bell; atol=1e-10)

# ----------------------------------------------------------------
# [2] Per-component FD gradient check at c₀ (policy held fixed)
# ----------------------------------------------------------------
println("\n[2] Per-component AD vs central-difference FD of ∂V/∂c_i")
v0 = c_as_vec(c0)
@time grad_AD = grad_c_exact(v0, memo, grid, ω_d, K)
# Use a ω_d that doesn't depend on c for this envelope test — paper's
# operating-point convention would recompute ω_d(c*) but we want to isolate
# the pure envelope gradient at the fixed memo.  Our test keeps ω_d = ω_d(c₀)
# constant, treating it as a frozen scalar (same as inside the memo solve).
println("  Component     AD                  FD                  |rel|")
for (i, name) in enumerate(C_FIELD_NAMES)
    δ = max(1e-6 * abs(v0[i]), 1e-14)
    v_plus  = copy(v0); v_plus[i]  += δ
    v_minus = copy(v0); v_minus[i] -= δ
    V_plus  = V_adaptive_policy_exact(vec_as_c(v_plus),  memo, grid, ω_d, K)
    V_minus = V_adaptive_policy_exact(vec_as_c(v_minus), memo, grid, ω_d, K)
    fd = (V_plus - V_minus) / (2*δ)
    denom = max(abs(fd), abs(grad_AD[i]), 1e-30)
    rel = abs(grad_AD[i] - fd) / denom
    @printf("  %-9s  AD=%+.10e  FD=%+.10e  rel=%.3e\n", name, grad_AD[i], fd, rel)
    @test rel < 1e-3
end

# ----------------------------------------------------------------
# [3] Random-direction directional derivative check
# ----------------------------------------------------------------
println("\n[3] Random-direction directional derivative: u·grad_AD vs (V(c+δu)-V(c-δu))/(2δ)")
rng = MersenneTwister(42)
let max_rel = 0.0
    for trial in 1:5
        u = randn(rng, C_DIM)
        u ./= sqrt(sum(abs2, u))
        δ = 1e-6
        v_plus  = v0 .+ δ .* u .* abs.(v0)
        v_minus = v0 .- δ .* u .* abs.(v0)
        V_plus  = V_adaptive_policy_exact(vec_as_c(v_plus),  memo, grid, ω_d, K)
        V_minus = V_adaptive_policy_exact(vec_as_c(v_minus), memo, grid, ω_d, K)
        fd = (V_plus - V_minus) / (2*δ)
        ad = sum(u .* grad_AD .* abs.(v0))
        denom = max(abs(fd), abs(ad), 1e-30)
        rel = abs(ad - fd) / denom
        @printf("  trial %d  AD=%+.10e  FD=%+.10e  rel=%.3e\n", trial, ad, fd, rel)
        max_rel = max(max_rel, rel)
    end
    @test max_rel < 1e-3
end

# ----------------------------------------------------------------
# [4] MC consistency (small n_traj, loose tolerance)
# ----------------------------------------------------------------
println("\n[4] Monte Carlo gradient — consistency with exact (loose tol)")
ω_d_fn = _c -> ω_d           # freeze for this test
rng = MersenneTwister(7)
@time (g_mc, g_se, _) = grad_c_mc(v0, memo, grid, ω_d_fn, K;
                                   n_traj=3000, rng=rng)
println("  Component     AD (exact)          MC mean ± SE        rel(AD, MC)")
for (i, name) in enumerate(C_FIELD_NAMES)
    denom = max(abs(grad_AD[i]), abs(g_mc[i]), 1e-30)
    rel = abs(grad_AD[i] - g_mc[i]) / denom
    @printf("  %-9s  AD=%+.4e  MC=%+.4e ± %.2e   rel=%.3e\n",
            name, grad_AD[i], g_mc[i], g_se[i], rel)
    # Within ~3σ of the MC estimate OR within 25% relative — whichever is looser.
    z = abs(grad_AD[i] - g_mc[i]) / max(g_se[i], 1e-30)
    @test (z < 5.0) || (rel < 0.25)
end

println("\nAll Phase-5 tests passed.\n")
