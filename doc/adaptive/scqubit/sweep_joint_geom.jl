#=
sweep_joint_geom.jl — joint-DP Adam restricted to the 4 geometric/circuit dims.

Noise amplitudes and temperature (A_φ, A_Ic, T) are PINNED at paper baseline
values via a tight box (lo = hi).  Adam can only move (f_q_max, E_C/h, κ, Δ_qr).

This experiment removes the "noise-reduction escape" observed in sweep_joint.jl
(where both optimizers converged to the same c because all three noise variables
saturated the lower box bound).
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
using .ScqubitModel
using .Belief
using .Bellman
using .Gradient
using .JointOpt

println("sweep_joint_geom.jl — joint-DP Adam, geometric dims only")
println("Threads: $(Threads.nthreads())")

# --- configuration ---
const K_EPOCHS       = 3
const K_PHI          = 64
const TAU_GRID       = ntuple(k -> 20e-9 * 2.0^(k-1), 4)
const N_GRID         = (1, 10)
const OUTER_ITERS    = parse(Int, get(ENV, "JOINT_ITERS", "200"))
const OUTER_LR       = parse(Float64, get(ENV, "JOINT_LR", "2e-3"))
const REOPT_EVERY    = parse(Int, get(ENV, "JOINT_REOPT", "10"))

# --- box: pin T, A_phi, A_Ic at paper baseline ---
function geom_only_box(bl::ScqubitParams)
    # Indices in c_as_vec: 1=f_q_max, 2=E_C_over_h, 3=kappa, 4=Delta_qr,
    #                     5=temperature, 6=A_phi, 7=A_Ic
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
(c_final, hist, memo_final) = joint_opt(PAPER_BASELINE;
    grid=grid, K_epochs=K_EPOCHS,
    outer_iters=OUTER_ITERS, outer_lr=OUTER_LR,
    policy_reopt_every=REOPT_EVERY,
    ckpt_every=50,
    ckpt_dir=joinpath(@__DIR__, "results", "joint_geom"),
    cbox=box,
    verbose=true)
@printf("\nTotal elapsed: %.1f min\n", (time() - t0) / 60)

out_path = joinpath(@__DIR__, "results", "joint_geom", "final.jls")
isdir(dirname(out_path)) || mkpath(dirname(out_path))
open(out_path, "w") do io
    serialize(io, (; c_final, v_final=c_as_vec(c_final),
                     history=hist, memo_final, K_EPOCHS, K_PHI,
                     TAU_GRID, N_GRID, timestamp=now(),
                     baseline=PAPER_BASELINE,
                     fixed_components=(:temperature, :A_phi, :A_Ic)))
end
println("Saved final to $out_path")

V_start = hist.V_adaptive[1]
V_end   = hist.V_adaptive[end]
println("\n" * "="^60)
@printf("V_adaptive(c₀)      = %.6f nats\n", V_start)
@printf("V_adaptive(c_final) = %.6f nats\n", V_end)
@printf("Δ                   = %+.6f nats  (%.3fx relative)\n",
        V_end - V_start, V_end / V_start)
println("Final c (pinned: T, A_φ, A_Ic):")
v = c_as_vec(c_final)
for (i, name) in enumerate(C_FIELD_NAMES)
    pinned = i in (5, 6, 7) ? " [pinned]" : ""
    @printf("  %-12s = %+.4e%s\n", name, v[i], pinned)
end
