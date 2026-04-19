#=
plot_results.jl — Phase 9 plotting.

Loads results/sweep_fq.jls, results/joint/final.jls, results/pcrb/final.jls,
results/compare_mse.jls and generates plots for the writeup.

Output:
  results/plot_sweep_fq.png       — 6-panel f_q_max sweep (Φ + PCRB frameworks)
  results/plot_adam_history.png   — convergence curves for joint-DP and PCRB
  results/plot_mse_comparison.png — bar chart MSE̅₁ vs MSE̅₂ vs 1/J_P
=#
using Printf
using Serialization
using Plots

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel
using .Belief
using .PCRB

gr()
default(fontfamily="sans-serif", linewidth=1.8, grid=true, frame=:box)

resdir = joinpath(@__DIR__, "results")

# ---------------- sweep_fq ----------------
sweep_fq_path = joinpath(resdir, "sweep_fq.jls")
if isfile(sweep_fq_path)
    sw = deserialize(sweep_fq_path)
    fq_list = [r.f_q_max / 1e9 for r in sw.results]
    phi_star = [r.phi_star for r in sw.results]
    V_or     = [r.V_oracle_mean for r in sw.results]
    V_ad     = [r.V_adaptive for r in sw.results]
    V_fx     = [r.V_fixed for r in sw.results]
    E_IG     = [r.E_IG for r in sw.results]
    logJP    = [r.log_JP_star for r in sw.results]
    pcrb_bd  = [r.pcrb_bound for r in sw.results]
    sat      = [r.saturation for r in sw.results]

    p1 = plot(fq_list, [V_or V_ad V_fx]; label=["V_oracle_mean" "V_adaptive" "V_fixed"],
              xlabel="f_q_max (GHz)", ylabel="value (nats)", title="Φ framework")
    vline!(p1, [9.0]; color=:gray, linestyle=:dash, label="paper (9 GHz)")

    p2 = plot(fq_list, E_IG; label="E[IG]", xlabel="f_q_max (GHz)",
              ylabel="nats", title="ignorance gap E[IG]")
    vline!(p2, [9.0]; color=:gray, linestyle=:dash, label="")

    p3 = plot(fq_list, phi_star; label="Φ*/Φ₀", xlabel="f_q_max (GHz)",
              ylabel="Φ*/Φ₀", title="paper sensitivity Φ* (Eq. 12)")
    hline!(p3, [0.442]; color=:gray, linestyle=:dash, label="paper (0.442)")
    vline!(p3, [9.0]; color=:gray, linestyle=:dash, label="")

    p4 = plot(fq_list, logJP; label="log J_P*", xlabel="f_q_max (GHz)",
              ylabel="log J_P*", title="PCRB framework: log J_P(s*, c)")
    vline!(p4, [9.0]; color=:gray, linestyle=:dash, label="")

    p5 = plot(fq_list, pcrb_bd; label="1/J_P*", xlabel="f_q_max (GHz)",
              ylabel="1/J_P* (Φ₀² units)", yscale=:log10,
              title="PCRB MSE lower bound")
    vline!(p5, [9.0]; color=:gray, linestyle=:dash, label="")

    p6 = plot(fq_list, sat; label="saturation %", xlabel="f_q_max (GHz)",
              ylabel="%", title="adaptive saturation V_realized/E[IG]")
    hline!(p6, [100.0]; color=:gray, linestyle=:dash, label="")

    plt = plot(p1, p2, p3, p4, p5, p6; layout=(3,2), size=(1100, 1000),
               plot_title="Within-family sweep over f_q_max")
    savefig(plt, joinpath(resdir, "plot_sweep_fq.png"))
    println("Saved plot_sweep_fq.png")
else
    @warn "no sweep_fq.jls found — skipping"
end

# ---------------- joint Adam history ----------------
joint_path = joinpath(resdir, "joint", "final.jls")
pcrb_path  = joinpath(resdir, "pcrb",  "final.jls")

if isfile(joint_path) && isfile(pcrb_path)
    jo = deserialize(joint_path)
    pc = deserialize(pcrb_path)

    pj1 = plot(jo.history.V_adaptive; xlabel="Adam iter", ylabel="V_adaptive (nats)",
               title="joint-DP: V_adaptive", legend=false)
    pj2 = plot(jo.history.grad_norm; xlabel="Adam iter", ylabel="‖∇_c V‖",
               yscale=:log10, title="joint-DP: gradient norm", legend=false)
    pp1 = plot(pc.history.log_JP; xlabel="Adam iter", ylabel="log J_P",
               title="PCRB baseline: log J_P", legend=false)
    pp2 = plot(pc.history.grad_norm; xlabel="Adam iter", ylabel="‖∇_c log J_P‖",
               yscale=:log10, title="PCRB baseline: gradient norm", legend=false)
    plt2 = plot(pj1, pj2, pp1, pp2; layout=(2,2), size=(1000, 700),
                plot_title="Adam convergence")
    savefig(plt2, joinpath(resdir, "plot_adam_history.png"))
    println("Saved plot_adam_history.png")
else
    @warn "missing final.jls files — skipping Adam history plot"
end

# ---------------- MSE comparison ----------------
cmp_path = joinpath(resdir, "compare_mse.jls")
if isfile(cmp_path)
    cmp = deserialize(cmp_path)
    vals = [cmp.MSE_1, cmp.MSE_2, cmp.pcrb_bound]
    errs = [cmp.se_1,  cmp.se_2,  0.0]
    labs = ["MSE̅₁ (joint DP)", "MSE̅₂ (PCRB baseline)", "1/J_P (CRB)"]
    colors = [:steelblue, :tomato, :gray]
    plt3 = bar(labs, vals; yerror=errs, color=colors, legend=false,
               ylabel="MSE (Φ₀² units)", yscale=:log10,
               title=@sprintf("Deployed MSE  —  ratio MSE̅₂/MSE̅₁ = %.3f",
                              cmp.MSE_2 / cmp.MSE_1))
    savefig(plt3, joinpath(resdir, "plot_mse_comparison.png"))
    println("Saved plot_mse_comparison.png")
else
    @warn "no compare_mse.jls — skipping bar chart"
end

println("\nAll plots saved to $resdir")
