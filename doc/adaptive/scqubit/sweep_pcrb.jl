#=
sweep_pcrb.jl — full 7-D PCRB-baseline optimization from the paper baseline.

Single-level outer Adam on c, inner schedule enumeration at every c_reopt.
Outputs:
  results/pcrb/final.jls   — final (c_final, sched_final, history)
=#
using Printf
using Serialization
using Dates

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel
using .Belief
using .Bellman
using .Gradient
using .JointOpt
using .PCRB

println("sweep_pcrb.jl — PCRB-baseline Adam starting from PAPER_BASELINE")
println("Threads: $(Threads.nthreads())")

# --- configuration ---
const K_EPOCHS       = 3
const K_PHI          = 64
const TAU_GRID       = ntuple(k -> 20e-9 * 2.0^(k-1), 4)
const N_GRID         = (1, 10)
const OUTER_ITERS    = parse(Int, get(ENV, "PCRB_ITERS", "300"))
const OUTER_LR       = parse(Float64, get(ENV, "PCRB_LR", "2e-3"))
const REOPT_EVERY    = parse(Int, get(ENV, "PCRB_REOPT", "20"))

grid = make_grid(; K_phi=K_PHI, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)

println(@sprintf("Config: K=%d  K_Φ=%d  J=%d  L=%d  iters=%d  lr=%.1e  reopt=%d",
                 K_EPOCHS, K_PHI, length(TAU_GRID), length(N_GRID),
                 OUTER_ITERS, OUTER_LR, REOPT_EVERY))

t0 = time()
# For PCRB we keep ω_d(c) adaptive: rebuild at each c via the paper sensitivity formula.
omega_d_fn = make_omega_d_fn()
(c_final, sched_final, hist) = pcrb_baseline(PAPER_BASELINE;
    grid=grid, K_epochs=K_EPOCHS,
    outer_iters=OUTER_ITERS, outer_lr=OUTER_LR,
    schedule_reopt_every=REOPT_EVERY,
    omega_d_fn=omega_d_fn,
    cbox=default_cbox(),
    verbose=true)
@printf("\nTotal elapsed: %.1f min\n", (time() - t0) / 60)

# --- save ---
outdir = joinpath(@__DIR__, "results", "pcrb")
isdir(outdir) || mkpath(outdir)
out_path = joinpath(outdir, "final.jls")
open(out_path, "w") do io
    serialize(io, (; c_final, v_final=c_as_vec(c_final),
                     sched_final, history=hist,
                     K_EPOCHS, K_PHI, TAU_GRID, N_GRID,
                     timestamp=now(), baseline=PAPER_BASELINE))
end
println("Saved final to $out_path")

# --- summary ---
lJP_start = hist.log_JP[1]
lJP_end   = hist.log_JP[end]
println("\n" * "="^60)
@printf("log J_P(c₀, s*₀)     = %+.6f\n", lJP_start)
@printf("log J_P(c_final, s*) = %+.6f\n", lJP_end)
@printf("Δ                    = %+.6f  (J_P ratio = %.3f)\n",
        lJP_end - lJP_start, exp(lJP_end - lJP_start))
println("Final sched (indices): ", sched_final)
println("Final c:")
v = c_as_vec(c_final)
for (i, name) in enumerate(C_FIELD_NAMES)
    @printf("  %-12s = %+.4e\n", name, v[i])
end
