"""
ScqubitModel

Closed-form Ramsey likelihood and decoherence rates for the Danilin-Nugent-Weides
frequency-tunable transmon flux sensor (arXiv:2211.08344v4).

All frequencies stored as linear-Hz in ScqubitParams. Internal formulas convert
to angular (rad/s) via 2π where required. The flux coordinate is the normalized
φ = Φ_ext / Φ₀ ∈ [0, 0.5]. Working in normalized flux keeps the rate formulas
independent of the explicit value of Φ₀ (the derivations in scqubit_model.tex
§Eq. 8, Eq. 10 verify this).

Cross-checked against paper equations:
  Eq. 1 : f_q(f_q_max, Φ) = (f_q_max + E_C/h)√(cos(πΦ/Φ₀)) - E_C/h
  Eq. 2 : Γ_1^cav = κ · g_01² / Δ²
  Eq. 3 : g_01 = β·e·√(Z₀/h)·(ω_q + Δ)·√((h f_q_max + E_C)/E_C)·cos^(1/4)(πΦ/Φ₀)
  Eq. 4 : Γ_1^ind = (M² + M'²)·ω_q² / (L_J · Z₀)
  Eq. 5 : L_J = 2 E_C / (e²·(ω_q_max_plus)²·cos(πΦ/Φ₀))   (ω_q_max_plus = 2π(f_q_max+E_C/h))
  Eq. 6 : Γ_1^cap = ω_q² Z₀ C_c² / C_qg
  Eq. 8 : Flux-noise factor — linear-in-t and quadratic-in-t contributions
  Eq. 9 : ∂ω_q/∂I_c = 4 E_C √(cos(πΦ/Φ₀)) / ((E_C + h f_q_max)·e)
  Eq. 10: Critical-current contribution to B² — exp(-t² π² γ² C² cos(πφ))
  Eq. 17: Ramsey fringe with thermal factor

The Ramsey envelope A, B (scqubit_model.tex Eqs. 3):
  A = (Γ_1^cav + Γ_1^ind + Γ_1^cap)/2 + π² A_Φ² |∂²ω_q/∂Φ²|
  B² = A_Φ² |∂ω_q/∂Φ|² + A_Ic² |∂ω_q/∂I_c|²
"""
module ScqubitModel

using LinearAlgebra

export ScqubitParams, PAPER_BASELINE,
       omega_q, domega_q_dphi, d2omega_q_dphi2, domega_q_dIc,
       g01, L_J, Gamma1_cav, Gamma1_ind, Gamma1_cap,
       A_coef, B_coef, thermal_factor,
       P1_ramsey, log_binom_like,
       I_critical, h_planck, e_charge, kB, Phi0

# ------- physical constants (SI) -------
const h_planck = 6.62607015e-34   # J·s
const e_charge = 1.602176634e-19  # C
const kB       = 1.380649e-23     # J/K
const Phi0     = h_planck / (2*e_charge)   # superconducting flux quantum, Wb

# ------- tier-1 design vector c -------
# All frequencies LINEAR (Hz), detuning Δ_qr LINEAR (Hz).
# Noise amplitudes as dimensionless fractions: A_Φ = (noise amplitude)/Φ₀,
#                                              A_Ic = (noise amplitude)/I_c.
Base.@kwdef struct ScqubitParams{T<:Real}
    f_q_max::T                   # Hz   maximal qubit transition frequency
    E_C_over_h::T                # Hz   charging energy in frequency units E_C/h
    kappa::T                     # Hz   resonator decay (linear)
    Delta_qr::T                  # Hz   qubit-resonator detuning (linear)
    temperature::T               # K    mixing-chamber temperature
    A_phi::T                     # dim-less  flux-noise amplitude / Φ₀
    A_Ic::T                      # dim-less  critical-current-noise amplitude / I_c
    # fixed-geometry constants (defaults from Danilin et al. Table 1 / App. A)
    beta::Float64     = 0.03             # xy-line voltage divider
    Z0::Float64       = 50.0             # Ω
    M::Float64        = 2.08e-12         # H (Biot–Savart, SQUID-loop mutual)
    Mprime::Float64   = 0.22e-12         # H (parasitic mutual to full x-mon)
    C_qg::Float64     = 76e-15           # F  (x-mon to ground plane)
    C_c::Float64      = 0.2e-15          # F  (x-mon to xy control line)
    phi_clip::Float64 = 0.49             # clip |φ| ≤ phi_clip near Cooper-pair-box degeneracy
end

# Paper's baseline c (Table 1, Fig. 2 central value)
const PAPER_BASELINE = ScqubitParams(
    f_q_max   = 9.0e9,
    E_C_over_h= 0.254e9,
    kappa     = 0.5e6,
    Delta_qr  = 2.0e9,
    temperature = 40e-3,
    A_phi     = 1.0e-6,
    A_Ic      = 1.0e-6,
)

# ---------------------------------------------------------------
# primitive geometric quantities
# ---------------------------------------------------------------

@inline _clip_phi(phi, pmax) = min(max(phi, -pmax), pmax)

"Cosine of πφ, clipped to avoid the Cooper-pair-box singularity at φ = 0.5."
@inline function cos_pi_phi(phi, c::ScqubitParams)
    φ = _clip_phi(phi, c.phi_clip)
    cos(π * φ)
end

"Linear qubit frequency (Hz) f_q(φ; c) = (f_q_max + E_C/h)√(cos πφ) - E_C/h."
@inline function f_q(phi, c::ScqubitParams)
    C = c.f_q_max + c.E_C_over_h
    cφ = cos_pi_phi(phi, c)
    C * sqrt(cφ) - c.E_C_over_h
end

"Angular qubit frequency ω_q(φ; c) in rad/s."
omega_q(phi, c::ScqubitParams) = 2π * f_q(phi, c)

"Angular qubit frequency at φ=0, shifted by E_C/h: used in L_J and g_01.
ω_q_max_plus = 2π(f_q_max + E_C/h) (rad/s)."
@inline omega_q_max_plus(c::ScqubitParams) = 2π * (c.f_q_max + c.E_C_over_h)

# Closed-form derivatives (so we stay ForwardDiff/Zygote-free internally).
"∂ω_q/∂φ (units rad/s per unit of normalized flux φ)."
function domega_q_dphi(phi, c::ScqubitParams)
    φ = _clip_phi(phi, c.phi_clip)
    C = c.f_q_max + c.E_C_over_h
    u = π * φ
    -π^2 * C * sin(u) / sqrt(cos(u))
end

"∂²ω_q/∂φ² (units rad/s per φ²)."
function d2omega_q_dphi2(phi, c::ScqubitParams)
    φ = _clip_phi(phi, c.phi_clip)
    C = c.f_q_max + c.E_C_over_h
    u = π * φ
    cu = cos(u)
    -π^3 * C * (1 + cu^2) / (2 * cu^(3/2))
end

"∂ω_q/∂I_c at (φ,c). Paper Eq. 9 (rad/s / Ampere)."
function domega_q_dIc(phi, c::ScqubitParams)
    φ = _clip_phi(phi, c.phi_clip)
    # 4 E_C √cos / ((E_C + h f_q_max) e), paper gives this for angular ω_q.
    EC = c.E_C_over_h * h_planck
    num = 4 * EC * sqrt(cos(π * φ))
    den = (EC + h_planck * c.f_q_max) * e_charge
    num / den
end

"Critical current I_c = 2π E_J / Φ₀, with E_J = (h f_q_max + E_C)²/(8 E_C) (transmon approx)."
function I_critical(c::ScqubitParams)
    EC = c.E_C_over_h * h_planck
    EJ = (h_planck * c.f_q_max + EC)^2 / (8 * EC)
    2π * EJ / Phi0
end

# ---------------------------------------------------------------
# Eq. 5: Josephson inductance L_J (Henries)
# L_J = 2 E_C / (e² · (ω_q_max_plus)² · cos(πφ))
# ---------------------------------------------------------------
function L_J(phi, c::ScqubitParams)
    EC = c.E_C_over_h * h_planck
    ω_plus = omega_q_max_plus(c)
    2 * EC / (e_charge^2 * ω_plus^2 * cos_pi_phi(phi, c))
end

# ---------------------------------------------------------------
# Eq. 3: qubit-resonator coupling g_01 (rad/s)
# g_01 = β e √(Z₀/h) (ω_q + Δ) √((h f_q_max + E_C)/E_C) cos^(1/4)(πφ)
# ---------------------------------------------------------------
function g01(phi, c::ScqubitParams)
    EC = c.E_C_over_h * h_planck
    ωq = omega_q(phi, c)
    Δ_rad = 2π * c.Delta_qr
    ratio = (h_planck * c.f_q_max + EC) / EC
    c.beta * e_charge * sqrt(c.Z0 / h_planck) * (ωq + Δ_rad) *
        sqrt(ratio) * cos_pi_phi(phi, c)^(1/4)
end

# ---------------------------------------------------------------
# Relaxation rates (units 1/s)
# Convention: the κ, Δ in Eq. 2 are treated as angular. The paper
# quotes κ, Δ in linear-Hz; convert before use.
# ---------------------------------------------------------------
function Gamma1_cav(phi, c::ScqubitParams)
    κ_rad = 2π * c.kappa
    Δ_rad = 2π * c.Delta_qr
    g = g01(phi, c)
    κ_rad * g^2 / Δ_rad^2
end

function Gamma1_ind(phi, c::ScqubitParams)
    ωq = omega_q(phi, c)
    Lj = L_J(phi, c)
    (c.M^2 + c.Mprime^2) * ωq^2 / (Lj * c.Z0)
end

function Gamma1_cap(phi, c::ScqubitParams)
    ωq = omega_q(phi, c)
    ωq^2 * c.Z0 * c.C_c^2 / c.C_qg
end

# ---------------------------------------------------------------
# Ramsey-envelope coefficients A, B (units 1/s and 1/s respectively)
# ---------------------------------------------------------------
"A(φ; c) = (Γ1_cav + Γ1_ind + Γ1_cap)/2 + π² A_Φ² |∂²ω_q/∂Φ²|.  Rate, 1/s."
function A_coef(phi, c::ScqubitParams)
    Γ_total = Gamma1_cav(phi, c) + Gamma1_ind(phi, c) + Gamma1_cap(phi, c)
    d2ω = abs(d2omega_q_dphi2(phi, c))
    0.5 * Γ_total + π^2 * c.A_phi^2 * d2ω
end

"B(φ; c) = √( A_Φ² |∂ω_q/∂Φ|² + A_Ic² |∂ω_q/∂I_c|² ).  Rate-ish, 1/s."
function B_coef(phi, c::ScqubitParams)
    dω_dφ = domega_q_dphi(phi, c)          # rad/s per φ; equal numerically to
    #   A_Φ² |∂ω_q/∂Φ|² if A_Φ is stored as Φ₀-fraction (derivation in docstring).
    dω_dIc = domega_q_dIc(phi, c)          # rad/s per Ampere
    Ic = I_critical(c)
    # Physical A_Ic amplitude is c.A_Ic * I_c (Amperes).
    Aphy_Ic = c.A_Ic * Ic
    sqrt(c.A_phi^2 * dω_dφ^2 + Aphy_Ic^2 * dω_dIc^2)
end

# ---------------------------------------------------------------
# Thermal factor (Eq. 17): (e^{hf_q/kT}-1)/(e^{hf_q/kT}+1).
# Evaluated at the actual flux φ (so the likelihood depends on it).
# ---------------------------------------------------------------
function thermal_factor(phi, c::ScqubitParams)
    fq = f_q(phi, c)
    x = h_planck * fq / (kB * c.temperature)
    tanh(0.5 * x)    # = (e^x-1)/(e^x+1)
end

# ---------------------------------------------------------------
# Ramsey likelihood P_|1⟩(φ, τ; c)  (Eq. 13 / 17)
# ---------------------------------------------------------------
"""
    P1_ramsey(phi, tau, c, phi_star, omega_d; include_thermal=true)

Expected |1⟩-population after a Ramsey sequence of delay τ (s) at flux φ,
with drive frequency ω_d (rad/s). phi_star is the operating-point flux (used
only to compute the decoherence envelope A, B if we want to keep them constant
over the posterior; default below evaluates A,B at φ itself).
"""
function P1_ramsey(phi, tau, c::ScqubitParams, omega_d;
                   include_thermal::Bool=true, env_phi=nothing)
    # Decoherence envelope evaluated at env_phi (typically φ_star) or at phi.
    envφ = env_phi === nothing ? phi : env_phi
    A = A_coef(envφ, c)
    B = B_coef(envφ, c)
    Δω = omega_q(phi, c) - omega_d
    thermal = include_thermal ? thermal_factor(phi, c) : 1.0
    env = exp(-A*tau - B^2 * tau^2)
    0.5 + 0.5 * thermal * env * cos(Δω * tau)
end

"""
    log_binom_like(phi, tau, n, m, c, omega_d; include_thermal=true, env_phi=nothing)

Log-likelihood of observing m |1⟩-outcomes in n repeated Ramsey shots.
"""
function log_binom_like(phi, tau, n::Integer, m::Integer, c::ScqubitParams, omega_d;
                        include_thermal::Bool=true, env_phi=nothing)
    p = P1_ramsey(phi, tau, c, omega_d;
                  include_thermal=include_thermal, env_phi=env_phi)
    # Clamp to avoid log(0) in edge cases (p extremely close to 0 or 1).
    p = clamp(p, 1e-300, 1 - 1e-16)
    m * log(p) + (n - m) * log1p(-p)
end

end # module
