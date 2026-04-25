#=
sweep_joint_narrow_v2.jl — converged rerun of sweep_joint_narrow.jl.

Same K=4, J=10, L=2, phi_max=0.1, MSE-terminal Bellman, same realistic_box,
same PAPER_BASELINE init, but smaller lr and longer horizon to reach a
stationary point (the original run with lr=5e-4 / iters=25 overshot the
V-peak via Adam momentum after iter ~11).

Defaults:
  JOINT_LR    = 1e-4   (5x smaller than v1)
  JOINT_ITERS = 100    (4x longer)
  JOINT_REOPT = 5      (same)

Output: results/joint_narrow_v2/{ckpt_NNNNNN.jls, final.jls}
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

println("sweep_joint_narrow_v2.jl — K=4, J=10, L=2, phi_max=0.1, MSE terminal (CONVERGED)")
println("Threads: $(Threads.nthreads())"); flush(stdout)

const K_EPOCHS    = 4
const K_PHI       = 128
const J_TAU       = 10
const PHI_MAX     = 0.1
const TAU_GRID    = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID      = (1, 10)
const OUTER_ITERS = parse(Int, get(ENV, "JOINT_ITERS", "100"))
const OUTER_LR    = parse(Float64, get(ENV, "JOINT_LR", "1e-4"))
const REOPT_EVERY = parse(Int, get(ENV, "JOINT_REOPT", "5"))

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
println("τ grid (ns): ", round.(collect(TAU_GRID) .* 1e9; digits=1))
flush(stdout)

t_probe = time()
(V0, memo0, st0) = solve_bellman_full(grid, K_EPOCHS, PAPER_BASELINE,
                                      omega_q(0.442, PAPER_BASELINE); terminal=:mse)
@printf("Probe Bellman at c₀: V=%.4e, memo=%d, %.2fs\n", V0, st0.memo_size, time()-t_probe)
flush(stdout)
memo0 = nothing; GC.gc()

t0 = time()
(c_final, hist, memo_final) = joint_opt(PAPER_BASELINE;
    grid=grid, K_epochs=K_EPOCHS,
    outer_iters=OUTER_ITERS, outer_lr=OUTER_LR,
    policy_reopt_every=REOPT_EVERY,
    ckpt_every=10,
    ckpt_dir=joinpath(@__DIR__, "results", "joint_narrow_v2"),
    cbox=box,
    terminal=:mse,
    verbose=true)
@printf("\nTotal elapsed: %.1f min\n", (time() - t0) / 60)
flush(stdout)

out_path = joinpath(@__DIR__, "results", "joint_narrow_v2", "final.jls")
isdir(dirname(out_path)) || mkpath(dirname(out_path))
open(out_path, "w") do io
    serialize(io, (; c_final, v_final=c_as_vec(c_final),
                     history=hist, K_EPOCHS, K_PHI, PHI_MAX,
                     TAU_GRID, N_GRID, timestamp=now(),
                     baseline=PAPER_BASELINE,
                     fixed_components=(:temperature, :A_phi, :A_Ic),
                     terminal=:mse))
end
println("Saved to $out_path")

nj = length(hist.c_vec)
va = hist.V_adaptive[1:nj]
ibest = argmax(va)
@printf("V at init           = %.4e\n", va[1])
@printf("V at best (iter %d) = %.4e (improvement %.2f%% vs init)\n",
        ibest, va[ibest], 100*(va[ibest]-va[1])/abs(va[1]))
@printf("V at final          = %.4e\n", va[end])
println("\nLast 5 V values: ", va[max(1, end-4):end])
println("Last 5 |grad|:   ", hist.grad_norm[max(1, end-4):end])
println("reopt iters: ", hist.reopt_iter)
