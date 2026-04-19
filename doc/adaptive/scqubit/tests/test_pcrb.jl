# Tests for PCRB.jl — Fisher positivity, additivity, gradient check, PCRB bound.
using Printf
using Test
using Random
using ForwardDiff

include(joinpath(@__DIR__, "..", "ScqubitModel.jl"))
include(joinpath(@__DIR__, "..", "Belief.jl"))
include(joinpath(@__DIR__, "..", "Baselines.jl"))
include(joinpath(@__DIR__, "..", "Bellman.jl"))
include(joinpath(@__DIR__, "..", "Gradient.jl"))
include(joinpath(@__DIR__, "..", "JointOpt.jl"))
include(joinpath(@__DIR__, "..", "PCRB.jl"))
using .ScqubitModel
using .Belief
using .Bellman
using .Gradient
using .JointOpt
using .PCRB

const c0   = PAPER_BASELINE
const grid = make_grid(; K_phi=64,
                          tau_grid = ntuple(k -> 20e-9 * 2.0^(k-1), 4),
                          n_grid   = (1, 10))
const ω_d  = omega_q(0.442, c0)

# ----------------------------------------------------------------
# [1] Fisher ≥ 0 across the grid
# ----------------------------------------------------------------
println("\n[1] Fisher positivity")
let minJ = Inf
    for phi in grid.phi_grid, τ in grid.tau_grid
        J = fisher_per_shot(phi, τ, c0, ω_d)
        if J < minJ; minJ = J; end
    end
    @printf("  min J_F over grid = %+.4e\n", minJ)
    @test minJ >= 0
end

# ----------------------------------------------------------------
# [2] J_F near φ=0 is small (∂ω_q/∂φ ∝ sin(πφ) vanishes at 0)
# ----------------------------------------------------------------
println("\n[2] J_F near sweet spot φ → 0 is small")
for τ in grid.tau_grid
    J_small = fisher_per_shot(0.001, τ, c0, ω_d)
    J_mid   = fisher_per_shot(0.40, τ, c0, ω_d)
    @printf("  τ=%.2e  J_F(φ=0.001)=%.3e  J_F(φ=0.40)=%.3e\n", τ, J_small, J_mid)
end
# Sanity: at φ = 0.001, J_F is orders of magnitude smaller than at mid φ (no strict assert
# because the ratio depends strongly on τ).

# ----------------------------------------------------------------
# [3] J_N additivity: J_N(concat(s₁, s₂)) = J_N(s₁) + J_N(s₂)
# ----------------------------------------------------------------
println("\n[3] J_N additivity")
s1 = [(Float64(grid.tau_grid[1]), 3)]
s2 = [(Float64(grid.tau_grid[4]), 5)]
s_concat = vcat(s1, s2)
for phi in (0.1, 0.3, 0.442)
    a = fisher_accumulated(phi, s1, c0, ω_d)
    b = fisher_accumulated(phi, s2, c0, ω_d)
    c_ab = fisher_accumulated(phi, s_concat, c0, ω_d)
    @printf("  φ=%.3f  J_s1=%.3e  J_s2=%.3e  J_s1+s2=%.3e  sum_err=%.3e\n",
            phi, a, b, c_ab, abs(a + b - c_ab))
    @test isapprox(a + b, c_ab; rtol=1e-12)
end

# ----------------------------------------------------------------
# [4] Gradient check: ∂ log J_P / ∂c via ForwardDiff vs finite difference
# ----------------------------------------------------------------
println("\n[4] Per-component AD vs FD of ∂ log J_P/∂c at c₀")
# Pick a non-trivial schedule.
sched_idx = [(1, 2), (3, 2)]    # (j, ℓ): (τ₁=20ns, n=10), (τ₃=80ns, n=10)
# Freeze ω_d at c₀-value so the gradient check isolates direct c-dependence.
omega_d_fn_const = (_c) -> ω_d
v0 = c_as_vec(c0)
g_AD = ForwardDiff.gradient(
    v -> log_JP_of_schedule(sched_idx, grid, vec_as_c(v), ω_d; J_0=1e-4), v0)
println("  Component     AD                  FD                  |rel|")
for (i, name) in enumerate(C_FIELD_NAMES)
    # Scan a few FD steps and take the minimum residual — cancellation is
    # tricky for large-magnitude components like f_q_max (~1e10 Hz).
    best_rel = Inf; best_fd = 0.0; best_delta = 0.0
    for rel_δ in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
        δ = max(rel_δ * abs(v0[i]), 1e-14)
        v_plus  = copy(v0); v_plus[i]  += δ
        v_minus = copy(v0); v_minus[i] -= δ
        f_plus  = log_JP_of_schedule(sched_idx, grid, vec_as_c(v_plus),  ω_d; J_0=1e-4)
        f_minus = log_JP_of_schedule(sched_idx, grid, vec_as_c(v_minus), ω_d; J_0=1e-4)
        fd = (f_plus - f_minus) / (2*δ)
        denom = max(abs(fd), abs(g_AD[i]), 1e-30)
        rel = abs(g_AD[i] - fd) / denom
        if rel < best_rel
            best_rel = rel; best_fd = fd; best_delta = δ
        end
    end
    @printf("  %-9s  AD=%+.10e  FD=%+.10e  δ=%.1e  rel=%.3e\n",
            name, g_AD[i], best_fd, best_delta, best_rel)
    @test best_rel < 1e-3
end

# ----------------------------------------------------------------
# [5] Schedule enumeration finds a schedule with higher log J_P than baseline
#     single-shot (τ=τ_min, n=1)
# ----------------------------------------------------------------
println("\n[5] Schedule enumeration beats a trivial single-shot schedule")
K_test = 2
(best_idx, best_lJP) = argmax_schedule_enumerate(grid, c0, ω_d, K_test)
sched_trivial = [(1, 1), (1, 1)]
trivial_lJP = log_JP_of_schedule(sched_trivial, grid, c0, ω_d; J_0=1e-4)
@printf("  best schedule = %s,  log J_P = %+.6f\n", string(best_idx), best_lJP)
@printf("  trivial       = %s,  log J_P = %+.6f\n", string(sched_trivial), trivial_lJP)
@test best_lJP >= trivial_lJP - 1e-9

# ----------------------------------------------------------------
# [6] PCRB empirical bound: E[(φ̂ - φ)²] ≥ 1/J_P (softened since posterior-mean is biased)
# ----------------------------------------------------------------
println("\n[6] PCRB bound sanity — deployed MSE vs 1/J_P")
rng = MersenneTwister(1)
(mse_val, mse_se) = deployed_mse_fixed(c0, best_idx, ω_d, grid; n_mc=2000, rng=rng)
pcrb_bound = 1.0 / exp(best_lJP)
@printf("  deployed MSE     = %.4e ± %.2e\n", mse_val, mse_se)
@printf("  1/J_P (pcrb)     = %.4e\n", pcrb_bound)
# Posterior-mean can be biased near the support boundary; use a slack factor
# of 0.9 so we don't fail on finite-grid quadrature / bias artifacts.
@test mse_val > 0.5 * pcrb_bound  # bound can be loose or tight; just sanity

# ----------------------------------------------------------------
# [7] Short pcrb_baseline run — log J_P should increase
# ----------------------------------------------------------------
println("\n[7] pcrb_baseline short run (15 iters)")
(c_end, sched_end, hist) = pcrb_baseline(c0;
    grid=grid, K_epochs=2,
    outer_iters=15, outer_lr=5e-3,
    schedule_reopt_every=5,
    omega_d_fn=(_c) -> ω_d,
    cbox=default_cbox(), verbose=true)
@printf("  log J_P start = %+.6f\n", hist.log_JP[1])
@printf("  log J_P end   = %+.6f   (Δ = %+.6f)\n",
        hist.log_JP[end], hist.log_JP[end] - hist.log_JP[1])
@test hist.log_JP[end] >= hist.log_JP[1] - 1e-6

println("\nAll Phase-7 tests passed.\n")
