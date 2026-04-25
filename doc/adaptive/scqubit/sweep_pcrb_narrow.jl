#=
sweep_pcrb_narrow.jl — PCRB baseline at K=4, J=10, L=2, phi_max=0.1.
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

println("sweep_pcrb_narrow.jl — K=4, J=10, L=2, phi_max=0.1")
println("Threads: $(Threads.nthreads())"); flush(stdout)

const K_EPOCHS    = 4
const K_PHI       = 128
const J_TAU       = 10
const PHI_MAX     = 0.1
const TAU_GRID    = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID      = (1, 10)
const OUTER_ITERS = parse(Int, get(ENV, "PCRB_ITERS", "150"))
const OUTER_LR    = parse(Float64, get(ENV, "PCRB_LR", "5e-3"))
const REOPT_EVERY = parse(Int, get(ENV, "PCRB_REOPT", "2"))

function realistic_box(bl::ScqubitParams)
    lo = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    hi = [12.0e9,   0.4e9,   5.0e6,   5.0e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    CBox(lo, hi)
end

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)
box  = realistic_box(PAPER_BASELINE)

println(@sprintf("Config: K=%d K_Φ=%d J=%d L=%d iters=%d lr=%.1e reopt=%d phi_max=%.3f",
                 K_EPOCHS, K_PHI, length(TAU_GRID), length(N_GRID),
                 OUTER_ITERS, OUTER_LR, REOPT_EVERY, PHI_MAX))
flush(stdout)

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
flush(stdout)

outdir = joinpath(@__DIR__, "results", "pcrb_narrow")
isdir(outdir) || mkpath(outdir)
open(joinpath(outdir, "final.jls"), "w") do io
    serialize(io, (; c_final, v_final=c_as_vec(c_final),
                     sched_final, history=hist,
                     K_EPOCHS, K_PHI, PHI_MAX, TAU_GRID, N_GRID,
                     timestamp=now(), baseline=PAPER_BASELINE))
end
println("Saved.")

imax = argmax(hist.log_JP)
@printf("log J_P best @ iter %d = %.4f\n", imax, hist.log_JP[imax])
println("best sched: ", hist.sched[imax])
println("best c: ", hist.c_vec[imax])
