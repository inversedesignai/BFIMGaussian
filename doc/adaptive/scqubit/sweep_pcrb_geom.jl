#=
sweep_pcrb_geom.jl — PCRB-baseline Adam restricted to the 4 geometric/circuit dims.
Noise amplitudes T, A_φ, A_Ic pinned at paper baseline.
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

println("sweep_pcrb_geom.jl — PCRB baseline, geometric dims only")
println("Threads: $(Threads.nthreads())")

const K_EPOCHS       = 3
const K_PHI          = 64
const TAU_GRID       = ntuple(k -> 20e-9 * 2.0^(k-1), 4)
const N_GRID         = (1, 10)
const OUTER_ITERS    = parse(Int, get(ENV, "PCRB_ITERS", "400"))
const OUTER_LR       = parse(Float64, get(ENV, "PCRB_LR", "2e-3"))
const REOPT_EVERY    = parse(Int, get(ENV, "PCRB_REOPT", "20"))

function geom_only_box(bl::ScqubitParams)
    lo = [ 1.0e9,   0.1e9,  0.01e6,   0.5e9,   bl.temperature,   bl.A_phi,   bl.A_Ic]
    hi = [30.0e9,   1.0e9,  10.0e6,  10.0e9,   bl.temperature,   bl.A_phi,   bl.A_Ic]
    CBox(lo, hi)
end

grid = make_grid(; K_phi=K_PHI, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)
box  = geom_only_box(PAPER_BASELINE)

println(@sprintf("Config: K=%d  K_Φ=%d  J=%d  L=%d  iters=%d  lr=%.1e  reopt=%d",
                 K_EPOCHS, K_PHI, length(TAU_GRID), length(N_GRID),
                 OUTER_ITERS, OUTER_LR, REOPT_EVERY))
println("Pinned: T=$(PAPER_BASELINE.temperature), A_phi=$(PAPER_BASELINE.A_phi), A_Ic=$(PAPER_BASELINE.A_Ic)")

t0 = time()
omega_d_fn = make_omega_d_fn()
(c_final, sched_final, hist) = pcrb_baseline(PAPER_BASELINE;
    grid=grid, K_epochs=K_EPOCHS,
    outer_iters=OUTER_ITERS, outer_lr=OUTER_LR,
    schedule_reopt_every=REOPT_EVERY,
    omega_d_fn=omega_d_fn,
    cbox=box,
    verbose=true)
@printf("\nTotal elapsed: %.1f min\n", (time() - t0) / 60)

outdir = joinpath(@__DIR__, "results", "pcrb_geom")
isdir(outdir) || mkpath(outdir)
out_path = joinpath(outdir, "final.jls")
open(out_path, "w") do io
    serialize(io, (; c_final, v_final=c_as_vec(c_final),
                     sched_final, history=hist,
                     K_EPOCHS, K_PHI, TAU_GRID, N_GRID,
                     timestamp=now(), baseline=PAPER_BASELINE,
                     fixed_components=(:temperature, :A_phi, :A_Ic)))
end
println("Saved final to $out_path")

lJP_start = hist.log_JP[1]
lJP_end   = hist.log_JP[end]
println("\n" * "="^60)
@printf("log J_P(c₀, s*₀)     = %+.6f\n", lJP_start)
@printf("log J_P(c_final, s*) = %+.6f\n", lJP_end)
@printf("Δ                    = %+.6f  (J_P ratio = %.3f)\n",
        lJP_end - lJP_start, exp(lJP_end - lJP_start))
println("Final sched (indices): ", sched_final)
println("Final c (pinned: T, A_φ, A_Ic):")
v = c_as_vec(c_final)
for (i, name) in enumerate(C_FIELD_NAMES)
    pinned = i in (5, 6, 7) ? " [pinned]" : ""
    @printf("  %-12s = %+.4e%s\n", name, v[i], pinned)
end
