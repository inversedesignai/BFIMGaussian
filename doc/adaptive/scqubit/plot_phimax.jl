#=
plot_phimax.jl — Visualize the ratio MSE_pcrb / MSE_adaptive as a function of
prior width phi_max.  Shows the regime-dependence of adaptive dominance:
dramatic in the narrow-prior regime, shrinking as the prior widens and
finally flipping for phi_max > ~0.35 (PCRB slightly wins in the wide-prior
regime because adaptive's DP policy over-commits to short-τ schedules).
=#
using Printf
using Plots

# Data from phimax_sweep + phimax_extreme (K=4, J=10, L=2, PAPER_BASELINE c,
# K_PHI=256, N_MC=20000, paired seed).
data = [
    # (phi_max, MSE_adaptive, MSE_pcrb, se_ad, se_pc)
    (0.030, 5.4678e-07, 6.9988e-05, 2.96e-08, 4.58e-07),
    (0.050, 2.6491e-06, 2.0366e-04, 2.02e-07, 1.28e-06),
    (0.080, 4.2970e-05, 5.3465e-04, 1.71e-06, 3.34e-06),
    (0.100, 1.0297e-04, 8.4429e-04, 3.29e-06, 5.34e-06),
    (0.150, 5.6889e-04, 1.8967e-03, 1.04e-05, 1.20e-05),
    (0.200, 2.3834e-03, 3.3839e-03, 2.90e-05, 2.16e-05),
    (0.250, 4.2798e-03, 5.3316e-03, 4.75e-05, 3.41e-05),
    (0.300, 7.3292e-03, 7.7074e-03, 7.34e-05, 4.97e-05),
    (0.400, 1.5370e-02, 1.3657e-02, 1.45e-04, 8.77e-05),
    (0.490, 2.5329e-02, 2.0428e-02, 2.20e-04, 1.30e-04),
]

phi_max = [d[1] for d in data]
mse_ad  = [d[2] for d in data]
mse_pc  = [d[3] for d in data]
se_ad   = [d[4] for d in data]
se_pc   = [d[5] for d in data]
ratio   = mse_pc ./ mse_ad
prior_var = phi_max .^ 2 ./ 12

# --- MSE vs phi_max (log-log) ---
p1 = plot(phi_max, mse_ad; yscale=:log10, xscale=:log10,
          label="joint-DP", marker=:circle, lw=2, color=:royalblue,
          xlabel="prior width φ_max", ylabel="deployed MSE",
          title="MSE vs prior width (K=4, J=10, PAPER_BASELINE)")
plot!(p1, phi_max, mse_pc; label="PCRB baseline", marker=:square, lw=2, color=:crimson)
plot!(p1, phi_max, prior_var; label="prior variance", ls=:dash, color=:gray, lw=1.5)

# --- Ratio vs phi_max (log-y) ---
p2 = plot(phi_max, ratio; yscale=:log10,
          label="MSE_pcrb / MSE_adaptive", marker=:circle, lw=2, color=:seagreen,
          xlabel="prior width φ_max", ylabel="ratio (log scale)",
          title="Adaptive-over-PCRB advantage")
hline!(p2, [1.0]; ls=:dash, label="parity", color=:black, lw=1)
hline!(p2, [1.5]; ls=:dot, label="50%-gap target", color=:gray, lw=1)

# Annotate the headline point
annotate!(p2, 0.10, 8.29, text("8.3× @ φ_max=0.1", :red, :bottom, 9))

fig = plot(p1, p2; layout=(2,1), size=(700, 800))
out = joinpath(@__DIR__, "results", "phimax_plot.png")
savefig(fig, out)
println("Saved $out")
