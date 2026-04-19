#=
sweep_pcrb_rich.jl — PCRB baseline at J=6, L=4 (plan defaults), geom-only.
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
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt, .PCRB

println("sweep_pcrb_rich.jl — J=6, L=4, geom-only")
println("Threads: $(Threads.nthreads())")

const K_EPOCHS    = 3
const K_PHI       = 64
const TAU_GRID    = ntuple(k -> 10e-9 * 2.0^(k-1), 6)
const N_GRID      = (1, 3, 10, 30)
const OUTER_ITERS = parse(Int, get(ENV, "PCRB_ITERS", "300"))
const OUTER_LR    = parse(Float64, get(ENV, "PCRB_LR", "5e-3"))
const REOPT_EVERY = parse(Int, get(ENV, "PCRB_REOPT", "1"))

function realistic_box(bl::ScqubitParams)
    lo = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    hi = [12.0e9,   0.4e9,   5.0e6,   5.0e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    CBox(lo, hi)
end
geom_only_box(bl) = realistic_box(bl)

grid = make_grid(; K_phi=K_PHI, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)
box  = geom_only_box(PAPER_BASELINE)

println(@sprintf("Config: K=%d K_Φ=%d J=%d L=%d iters=%d lr=%.1e reopt=%d",
                 K_EPOCHS, K_PHI, length(TAU_GRID), length(N_GRID),
                 OUTER_ITERS, OUTER_LR, REOPT_EVERY))

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

outdir = joinpath(@__DIR__, "results", "pcrb_rich")
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

imax = argmax(hist.log_JP)
@printf("log J_P best @ iter %d = %.4f  (init %.4f)\n", imax, hist.log_JP[imax], hist.log_JP[1])
println("best sched: ", hist.sched[imax])
names = (:f_q_max, :E_C_over_h, :kappa, :Delta_qr, :temperature, :A_phi, :A_Ic)
for (n, v) in zip(names, hist.c_vec[imax])
    @printf("  %-12s = %+.4e\n", n, v)
end
