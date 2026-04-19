#=
plot_results_geom.jl — plots for the geom-only (restricted-c) experiment.
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
using .ScqubitModel, .Belief, .PCRB

gr()
default(fontfamily="sans-serif", linewidth=1.5, grid=true, frame=:box)
resdir = joinpath(@__DIR__, "results")

joint_path = joinpath(resdir, "joint_geom", "final.jls")
pcrb_path  = joinpath(resdir, "pcrb_geom",  "final.jls")
cmp_path   = joinpath(resdir, "compare_mse_geom.jls")

if isfile(joint_path) && isfile(pcrb_path)
    jo = deserialize(joint_path)
    pc = deserialize(pcrb_path)

    # Top row: joint-DP V_adaptive and c-trajectory
    p1 = plot(jo.history.V_adaptive; xlabel="Adam iter", ylabel="V_adaptive (nats)",
              title="joint-DP (geom-only): V_adaptive", legend=false)
    hline!(p1, [jo.history.V_adaptive[1]]; color=:gray, linestyle=:dash, label="c₀")
    scatter!(p1, [argmax(jo.history.V_adaptive)], [maximum(jo.history.V_adaptive)];
             color=:red, markersize=6, label="best")

    fq_hist_j = [cv[1] / 1e9 for cv in jo.history.c_vec]
    p2 = plot(fq_hist_j; xlabel="Adam iter", ylabel="f_q_max (GHz)",
              title="joint-DP (geom-only): f_q_max trajectory", legend=false)

    # Bottom row: PCRB log_JP and c-trajectory
    p3 = plot(pc.history.log_JP; xlabel="Adam iter", ylabel="log J_P",
              title="PCRB (geom-only): log J_P", legend=false)
    hline!(p3, [pc.history.log_JP[1]]; color=:gray, linestyle=:dash, label="c₀")
    scatter!(p3, [argmax(pc.history.log_JP)], [maximum(pc.history.log_JP)];
             color=:red, markersize=6, label="best")

    fq_hist_p = [cv[1] / 1e9 for cv in pc.history.c_vec]
    p4 = plot(fq_hist_p; xlabel="Adam iter", ylabel="f_q_max (GHz)",
              title="PCRB (geom-only): f_q_max trajectory", legend=false)

    plt = plot(p1, p2, p3, p4; layout=(2,2), size=(1100, 750),
               plot_title="Geom-only Adam: joint-DP (top) vs PCRB (bottom)")
    savefig(plt, joinpath(resdir, "plot_adam_geom.png"))
    println("Saved plot_adam_geom.png")
else
    @warn "missing geom final.jls files"
end

if isfile(cmp_path)
    cmp = deserialize(cmp_path)
    vals = [cmp.MSE_1, cmp.MSE_2, cmp.pcrb_bound]
    errs = [cmp.se_1,  cmp.se_2,  0.0]
    labs = ["MSE̅₁ (joint DP)", "MSE̅₂ (PCRB baseline)", "1/J_P (CRB)"]
    colors = [:steelblue, :tomato, :gray]
    plt3 = bar(labs, vals; yerror=errs, color=colors, legend=false,
               ylabel="MSE (Φ₀² units)", yscale=:log10,
               title=@sprintf("Geom-only deployed MSE  —  ratio MSE̅₂/MSE̅₁ = %.3f",
                              cmp.MSE_2 / cmp.MSE_1))
    savefig(plt3, joinpath(resdir, "plot_mse_comparison_geom.png"))
    println("Saved plot_mse_comparison_geom.png")
end
