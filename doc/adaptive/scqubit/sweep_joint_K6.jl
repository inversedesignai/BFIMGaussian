#=
sweep_joint_K6.jl — K=6 horizon, J=4, L=2. Long-horizon config.

Memo at K=6 J=4 L=2 = 4.1M entries (measured at baseline c, V=-3.76e-3).
PCRB enumeration: (4*2)^6 = 262k schedules. Tractable both sides.
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
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt

println("sweep_joint_K6.jl — K=6, J=4, L=2, MSE terminal")
flush(stdout)

const K_EPOCHS    = 6
const K_PHI       = 64
const J_TAU       = 4
const TAU_GRID    = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID      = (1, 10)
const OUTER_ITERS = parse(Int, get(ENV, "JOINT_ITERS", "30"))
const OUTER_LR    = parse(Float64, get(ENV, "JOINT_LR", "5e-4"))
const REOPT_EVERY = parse(Int, get(ENV, "JOINT_REOPT", "5"))

function realistic_box(bl::ScqubitParams)
    lo = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    hi = [12.0e9,   0.4e9,   5.0e6,   5.0e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    CBox(lo, hi)
end

grid = make_grid(; K_phi=K_PHI, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)
box  = realistic_box(PAPER_BASELINE)

println(@sprintf("Config: K=%d K_Φ=%d J=%d L=%d iters=%d lr=%.1e reopt=%d",
                 K_EPOCHS, K_PHI, length(TAU_GRID), length(N_GRID),
                 OUTER_ITERS, OUTER_LR, REOPT_EVERY))
println("τ grid (ns): ", round.(collect(TAU_GRID) .* 1e9; digits=1))
flush(stdout)

t_probe = time()
(V0, memo0, st0) = solve_bellman_full(grid, K_EPOCHS, PAPER_BASELINE, omega_q(0.442, PAPER_BASELINE); terminal=:mse)
@printf("Probe Bellman at c₀: V=%.4e, memo=%d, %.2fs\n", V0, st0.memo_size, time()-t_probe)
flush(stdout)
memo0 = nothing
GC.gc()

t0 = time()
(c_final, hist, memo_final) = joint_opt(PAPER_BASELINE;
    grid=grid, K_epochs=K_EPOCHS,
    outer_iters=OUTER_ITERS, outer_lr=OUTER_LR,
    policy_reopt_every=REOPT_EVERY,
    ckpt_every=10,
    ckpt_dir=joinpath(@__DIR__, "results", "joint_K6"),
    cbox=box,
    terminal=:mse,
    verbose=true)
@printf("\nTotal elapsed: %.1f min\n", (time() - t0) / 60)
flush(stdout)

out_path = joinpath(@__DIR__, "results", "joint_K6", "final.jls")
isdir(dirname(out_path)) || mkpath(dirname(out_path))
open(out_path, "w") do io
    serialize(io, (; c_final, v_final=c_as_vec(c_final),
                     history=hist, K_EPOCHS, K_PHI,
                     TAU_GRID, N_GRID, timestamp=now(),
                     baseline=PAPER_BASELINE,
                     fixed_components=(:temperature, :A_phi, :A_Ic),
                     terminal=:mse))
end
println("Saved to $out_path")

nj = length(hist.c_vec)
va = hist.V_adaptive[1:nj]
ibest = argmax(va)
@printf("-E[Var_post](c₀)   = %.4e\n", va[1])
@printf("-E[Var_post](best) = %.4e  (iter %d; Var=%.4e)\n", va[ibest], ibest, -va[ibest])
