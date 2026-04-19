# Tests for ScqubitModel.jl — forward model, rates, Ramsey likelihood.
# Run:  julia --project=. doc/adaptive/scqubit/tests/test_model.jl
using Printf
using Test
using ForwardDiff

include(joinpath(@__DIR__, "..", "ScqubitModel.jl"))
using .ScqubitModel

c0 = PAPER_BASELINE
const φmax = 0.49

function report(label, expected, actual; tol=0.0, relative::Bool=false)
    abs_err = abs(expected - actual)
    if relative && abs(expected) > 1e-30
        rel_err = abs_err / abs(expected)
        @printf("  %-48s expected=%+.12e  actual=%+.12e  |Δ|=%.3e  rel=%.3e\n",
                label, expected, actual, abs_err, rel_err)
        return rel_err
    else
        @printf("  %-48s expected=%+.12e  actual=%+.12e  |Δ|=%.3e\n",
                label, expected, actual, abs_err)
        return abs_err
    end
end

# ----------------------------------------------------------------
# 1. ω_q at the sweet spot and at the far edge
# ----------------------------------------------------------------
println("\n[1] ω_q at boundaries")
ωq_at_0 = omega_q(0.0, c0)
expected_0 = 2π * c0.f_q_max
@test isapprox(ωq_at_0, expected_0; rtol=1e-14)
report("ω_q(φ=0)", expected_0, ωq_at_0; relative=true)

# At φ = 0.5 (clipped to φmax=0.49): should NOT be -2π·E_C/h exactly because of clipping.
ωq_half = omega_q(0.5, c0)
println(@sprintf("  ω_q(φ=0.5) [clipped to %.2f] = %+.6e rad/s", c0.phi_clip, ωq_half))

# ----------------------------------------------------------------
# 2. FD verification of derivatives d/dφ, d²/dφ², d/dIc
# ----------------------------------------------------------------
println("\n[2] Analytical derivative vs finite differences on ω_q")
δ = 1e-6
for phi in (0.05, 0.20, 0.442, 0.47)
    d_fd  = (omega_q(phi + δ, c0) - omega_q(phi - δ, c0)) / (2δ)
    d_an  = domega_q_dphi(phi, c0)
    relerr = abs(d_an - d_fd) / abs(d_fd)
    @printf("  φ=%.3f  dω/dφ  analytic=%+.10e  FD=%+.10e  rel=%.2e\n",
            phi, d_an, d_fd, relerr)
    @test relerr < 1e-6

    d2_fd = (omega_q(phi + δ, c0) - 2*omega_q(phi, c0) + omega_q(phi - δ, c0)) / δ^2
    d2_an = d2omega_q_dphi2(phi, c0)
    relerr2 = abs(d2_an - d2_fd) / abs(d2_fd)
    @printf("  φ=%.3f  d²ω/dφ² an=%+.10e  FD=%+.10e  rel=%.2e\n",
            phi, d2_an, d2_fd, relerr2)
    @test relerr2 < 1e-4
end

# ----------------------------------------------------------------
# 3. A, B are non-negative for all tested φ
# ----------------------------------------------------------------
println("\n[3] A(φ), B(φ) sign & magnitude sanity")
for phi in (0.001, 0.05, 0.20, 0.442, 0.48)
    A = A_coef(phi, c0)
    B = B_coef(phi, c0)
    γcav = Gamma1_cav(phi, c0); γind = Gamma1_ind(phi, c0); γcap = Gamma1_cap(phi, c0)
    @printf("  φ=%.3f  A=%+.4e  B=%+.4e  Γcav=%.3e Γind=%.3e Γcap=%.3e (Hz)\n",
            phi, A, B, γcav, γind, γcap)
    @test A >= 0 && B >= 0 && γcav >= 0 && γind >= 0 && γcap >= 0
end

# ----------------------------------------------------------------
# 4. P_|1⟩ bounds
# ----------------------------------------------------------------
println("\n[4] P_|1⟩ range")
# At τ = 0, no decoherence, cos(0)=1, thermal can be <1; but P should be
# in (0, 1] regardless.
for (phi, tau) in ((0.0, 0.0), (0.1, 0.0), (0.3, 1e-7), (0.442, 5e-7),
                   (0.48, 1e-6))
    ω_d = omega_q(0.442, c0)
    p = P1_ramsey(phi, tau, c0, ω_d)
    @printf("  φ=%.3f  τ=%.2e  P_|1⟩=%.6f\n", phi, tau, p)
    @test 0.0 <= p <= 1.0
end

# At τ = 0 and include_thermal=false, P_|1⟩ must be exactly 1 (0.5 + 0.5·1·1·1).
for phi in (0.0, 0.1, 0.3, 0.48)
    ω_d = omega_q(0.442, c0)
    p = P1_ramsey(phi, 0.0, c0, ω_d; include_thermal=false)
    @test isapprox(p, 1.0; atol=1e-14)
end
println("  P_|1⟩(τ=0, include_thermal=false) = 1.0 exactly  ✓")

# ----------------------------------------------------------------
# 5. Optimal flux point — reproduce paper's Φ* ≈ 0.442·Φ₀ for c₀
# ----------------------------------------------------------------
println("\n[5] Optimal sensing flux (Eq. 12: argmax over (φ, τ) of |∂P₁/∂φ|)")
# Grid search: for each φ ∈ [0.05, 0.485], maximize |∂P₁/∂φ|_amp over τ.
# The envelope amplitude is (Eq. 15):
#   |∂P₁/∂φ|_amp = (τ/2) · thermal · e^{-Aτ-B²τ²} · |∂ω_q/∂φ|
# Maximize over τ: close-form τ_opt satisfies Aτ + 2B²τ² = 1 (paper Eq. 16
# gives τ_opt = (-A + √(A² + 8B²))/(4B²)).  We use Eq. 16 directly.
function sensitivity_amp(phi, c)
    A = A_coef(phi, c)
    B = B_coef(phi, c)
    τopt = (-A + sqrt(A^2 + 8*B^2)) / (4*B^2)
    th = thermal_factor(phi, c)
    dωdφ = abs(domega_q_dphi(phi, c))
    (τopt/2) * th * exp(-A*τopt - B^2 * τopt^2) * dωdφ
end

φ_grid = range(0.05, 0.485; length=800)
amps = sensitivity_amp.(φ_grid, Ref(c0))
(_, imax) = findmax(amps)
φ_star = φ_grid[imax]
@printf("  grid-argmax Φ*/Φ₀ = %.4f  (paper: 0.442)  |Δ| = %.4f\n",
        φ_star, abs(φ_star - 0.442))
@test abs(φ_star - 0.442) < 0.02   # ± 0.02 tolerance

# ----------------------------------------------------------------
# 6. ForwardDiff through P1_ramsey w.r.t. a specific c-component
# ----------------------------------------------------------------
println("\n[6] ForwardDiff of P1_ramsey w.r.t. each c-component")
c_as_vec(c) = [c.f_q_max, c.E_C_over_h, c.kappa, c.Delta_qr, c.temperature, c.A_phi, c.A_Ic]
vec_as_c(v) = ScqubitParams(f_q_max=v[1], E_C_over_h=v[2], kappa=v[3],
                            Delta_qr=v[4], temperature=v[5], A_phi=v[6], A_Ic=v[7])
v0 = c_as_vec(c0)
# Pick a non-stationary test point: off the optimal-sensing flux and with ω_d
# chosen so Δω·τ is not at a cos extremum. This maximizes the FD signal-to-noise.
phi_test = 0.35
tau_test = 2.0e-7
ω_d = omega_q(0.30, c0)              # intentionally mistuned off phi_test
f = v -> P1_ramsey(phi_test, tau_test, vec_as_c(v), ω_d)
p0 = f(v0)
@printf("  (φ, τ, ω_d) = (%.3f, %.2e, %.6e rad/s);   P1 = %.6f\n",
        phi_test, tau_test, ω_d, p0)
grad_AD = ForwardDiff.gradient(f, v0)
println("  c-component gradient AD vs FD (rel step = 1e-5):")
for (i, name) in enumerate(("f_q_max","E_C/h","κ","Δ_qr","T","A_Φ","A_Ic"))
    δc = max(1e-5 * abs(v0[i]), 1e-14)
    v_plus = copy(v0); v_plus[i] += δc
    v_minus= copy(v0); v_minus[i] -= δc
    fd_i = (f(v_plus) - f(v_minus)) / (2*δc)
    denom = max(abs(fd_i), abs(grad_AD[i]), 1e-30)
    rel = abs(grad_AD[i] - fd_i) / denom
    @printf("    %-8s  AD=%+.6e  FD=%+.6e  rel=%.2e\n", name, grad_AD[i], fd_i, rel)
    @test rel < 1e-3
end

println("\nAll Phase-1 tests passed.\n")
