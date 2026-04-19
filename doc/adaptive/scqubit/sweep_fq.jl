#=
sweep_fq.jl

Within-family 1-D sweep over f_q_max ∈ [2, 20] GHz, reproducing the paper's
Fig. 2 analog.  For each f_q_max, we:
  (a) hold all other c components at baseline;
  (b) compute V_oracle_mean, V_fixed, V_adaptive at horizon K
      and the implied EVPI saturation;
  (c) also record the grid-argmax Φ* from the single-shot sensitivity formula
      (paper Eq. 12) for direct comparison with Fig. 2.

Outputs saved to results/sweep_fq.jls.

Runtime: a few minutes at K=3, K_Φ=64, J=4, L=2 using threads.
=#
using Printf
using Serialization
using Dates

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))   # needed by PCRB
include(joinpath(@__DIR__, "JointOpt.jl"))   # needed by PCRB
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel
using .Belief
using .Baselines
using .Bellman
using .PCRB

println("sweep_fq.jl — within-family sweep over f_q_max")
println("Threads available: $(Threads.nthreads())")

# ---- configuration ----
const K_epochs = 3
const K_phi    = 64
const tau_grid = ntuple(k -> 20e-9 * 2.0^(k-1), 4)   # J = 4
const n_grid   = (1, 10)                              # L = 2
const f_q_max_list = (2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 15.0, 18.0, 20.0) .* 1e9   # Hz

# -------- single-shot optimal-flux calculation (paper Eq. 12) --------
"Grid-argmax Φ*/Φ₀ of |∂P₁/∂φ|_amp(φ) using Eq. 16 τ_opt."
function phi_star_of_fq(c::ScqubitParams; n_grid_phi::Int=400)
    φ_grid = range(0.05, 0.485; length=n_grid_phi)
    best = -Inf
    best_phi = 0.442
    for phi in φ_grid
        A = A_coef(phi, c)
        B = B_coef(phi, c)
        τopt = (-A + sqrt(A^2 + 8*B^2)) / (4*B^2)
        th = thermal_factor(phi, c)
        dωdφ = abs(domega_q_dphi(phi, c))
        amp = (τopt/2) * th * exp(-A*τopt - B^2 * τopt^2) * dωdφ
        if amp > best
            best = amp
            best_phi = phi
        end
    end
    (best_phi, best)
end

# -------- main sweep --------
baseline = PAPER_BASELINE
results = Vector{NamedTuple}(undef, length(f_q_max_list))
total_t0 = time()
for (i, fq) in enumerate(f_q_max_list)
    t0 = time()
    c = ScqubitParams(
        f_q_max    = fq,
        E_C_over_h = baseline.E_C_over_h,
        kappa      = baseline.kappa,
        Delta_qr   = baseline.Delta_qr,
        temperature= baseline.temperature,
        A_phi      = baseline.A_phi,
        A_Ic       = baseline.A_Ic,
    )
    (phi_star, amp_star) = phi_star_of_fq(c)
    ω_d = omega_q(phi_star, c)

    grid = make_grid(; K_phi=K_phi, phi_max=0.49,
                       tau_grid=tau_grid, n_grid=n_grid)
    schedules = enumerate_schedules(grid, K_epochs)
    (logp, log1mp) = Baselines.logp_cache(grid, c, ω_d)

    (Vfx, s_fx, _) = V_fixed(grid, K_epochs, c, ω_d, schedules, logp, log1mp)
    Vor_mean       = V_oracle_mean(grid, K_epochs, c, ω_d, schedules, logp, log1mp)
    (Vad, memo, st)= solve_bellman_full(grid, K_epochs, c, ω_d)

    # PCRB framework: argmax_s log J_P(s, c) via enumeration on same grid
    (pcrb_sched, log_JP_star) = argmax_schedule_enumerate(grid, c, ω_d, K_epochs)
    JP_star    = exp(log_JP_star)
    pcrb_bound = 1.0 / JP_star    # MSE lower bound (Φ₀² units since Φ ∈ [0, Φ₀/2])

    E_IG   = Vor_mean - Vfx
    realized = Vad - Vfx
    sat    = E_IG > 1e-10 ? 100 * realized / E_IG : 0.0
    elapsed = time() - t0

    @printf("  f_q_max=%4.1f GHz  Φ*=%.4f  V_or=%.4f  V_ad=%.4f  V_fx=%.4f  E[IG]=%.4f  sat=%.1f%%  logJP*=%+.3f  1/JP*=%.2e  %.1f s\n",
            fq/1e9, phi_star, Vor_mean, Vad, Vfx, E_IG, sat, log_JP_star, pcrb_bound, elapsed)
    results[i] = (; f_q_max=fq, phi_star=phi_star, amp_star=amp_star,
                    V_oracle_mean=Vor_mean, V_adaptive=Vad, V_fixed=Vfx,
                    E_IG=E_IG, realized=realized, saturation=sat,
                    best_fixed_schedule=s_fx, memo_size=st.memo_size,
                    log_JP_star=log_JP_star, pcrb_sched=pcrb_sched,
                    pcrb_bound=pcrb_bound,
                    elapsed=elapsed)
end
@printf("\nTotal sweep time: %.1f s\n", time() - total_t0)

# -------- summary --------
println("\n" * "="^110)
@printf("%-10s  %-8s  %-8s  %-8s  %-8s  %-8s  %-6s  %-10s  %-10s\n",
        "f_q_max", "Φ*", "V_or", "V_ad", "V_fx", "E[IG]", "sat%", "logJP*", "1/JP*")
println("-"^110)
for r in results
    @printf("%-10.1f  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %-6.1f  %+10.4f  %10.3e\n",
            r.f_q_max/1e9, r.phi_star, r.V_oracle_mean, r.V_adaptive, r.V_fixed,
            r.E_IG, r.saturation, r.log_JP_star, r.pcrb_bound)
end

# Identify in-family optima (both frameworks)
(best_Vad, iad)      = findmax([r.V_adaptive for r in results])
(best_Vfx_val, ifx)  = findmax([r.V_fixed for r in results])
(best_amp, iamp)     = findmax([r.amp_star for r in results])
(best_logJP, ipcrb)  = findmax([r.log_JP_star for r in results])
println("\n" * "="^110)
@printf("Φ framework    V_adaptive optimum:    f_q_max = %.1f GHz, V_ad   = %.4f nats\n",
        results[iad].f_q_max/1e9, best_Vad)
@printf("Φ framework    V_fixed optimum:       f_q_max = %.1f GHz, V_fx   = %.4f nats\n",
        results[ifx].f_q_max/1e9, best_Vfx_val)
@printf("PCRB framework log J_P* optimum:      f_q_max = %.1f GHz, logJP* = %.4f\n",
        results[ipcrb].f_q_max/1e9, best_logJP)
@printf("Single-shot sensitivity optimum:      f_q_max = %.1f GHz, Φ* = %.4f\n",
        results[iamp].f_q_max/1e9, results[iamp].phi_star)
println("Paper's in-family optimum:           f_q_max ≈ 9 GHz, Φ* ≈ 0.442")

outdir = joinpath(@__DIR__, "results")
isdir(outdir) || mkpath(outdir)
out_path = joinpath(outdir, "sweep_fq.jls")
open(out_path, "w") do io
    serialize(io, (; f_q_max_list, results, K_epochs, K_phi, tau_grid, n_grid,
                     timestamp=now()))
end
println("Saved to $out_path")
