#=
sweep_joint_geom_mse.jl — joint-DP with MSE terminal reward (−Var_post).

Differs from sweep_joint_geom.jl only in `terminal=:mse`:
Bellman uses −posterior variance as the leaf value instead of −entropy.
This makes the adaptive policy directly MSE-optimal for the post-mean
estimator, addressing the §9 finding that MI-maximization ≠ MSE-minimization
at K=3.
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

println("sweep_joint_geom_mse.jl — joint-DP (terminal=:mse), geom-only")
println("Threads: $(Threads.nthreads())")

const K_EPOCHS    = 3
const K_PHI       = 64
const TAU_GRID    = ntuple(k -> 20e-9 * 2.0^(k-1), 4)
const N_GRID      = (1, 10)
const OUTER_ITERS = parse(Int, get(ENV, "JOINT_ITERS", "150"))
const OUTER_LR    = parse(Float64, get(ENV, "JOINT_LR", "5e-4"))
const REOPT_EVERY = parse(Int, get(ENV, "JOINT_REOPT", "5"))

function geom_only_box(bl::ScqubitParams)
    lo = [ 1.0e9,   0.1e9,  0.01e6,   0.5e9,   bl.temperature,   bl.A_phi,   bl.A_Ic]
    hi = [30.0e9,   1.0e9,  10.0e6,  10.0e9,   bl.temperature,   bl.A_phi,   bl.A_Ic]
    CBox(lo, hi)
end

grid = make_grid(; K_phi=K_PHI, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)
box  = geom_only_box(PAPER_BASELINE)

println(@sprintf("Config: K=%d  K_Φ=%d  J=%d  L=%d  iters=%d  lr=%.1e  reopt=%d  terminal=:mse",
                 K_EPOCHS, K_PHI, length(TAU_GRID), length(N_GRID),
                 OUTER_ITERS, OUTER_LR, REOPT_EVERY))

t0 = time()
(c_final, hist, memo_final) = joint_opt(PAPER_BASELINE;
    grid=grid, K_epochs=K_EPOCHS,
    outer_iters=OUTER_ITERS, outer_lr=OUTER_LR,
    policy_reopt_every=REOPT_EVERY,
    ckpt_every=50,
    ckpt_dir=joinpath(@__DIR__, "results", "joint_geom_mse"),
    cbox=box,
    terminal=:mse,
    verbose=true)
@printf("\nTotal elapsed: %.1f min\n", (time() - t0) / 60)

out_path = joinpath(@__DIR__, "results", "joint_geom_mse", "final.jls")
isdir(dirname(out_path)) || mkpath(dirname(out_path))
open(out_path, "w") do io
    serialize(io, (; c_final, v_final=c_as_vec(c_final),
                     history=hist, memo_final, K_EPOCHS, K_PHI,
                     TAU_GRID, N_GRID, timestamp=now(),
                     baseline=PAPER_BASELINE,
                     fixed_components=(:temperature, :A_phi, :A_Ic),
                     terminal=:mse))
end
println("Saved final to $out_path")

# hist.V_adaptive is now -E[Var_post] (the MSE objective, negated for maximization)
@printf("-E[Var_post](c₀)      = %.4e\n", hist.V_adaptive[1])
@printf("-E[Var_post](c_final) = %.4e\n", hist.V_adaptive[end])
ibest = argmax(hist.V_adaptive)
@printf("best iter %d: -E[Var_post] = %.4e  (⇔ E[Var_post] = %.4e)\n",
        ibest, hist.V_adaptive[ibest], -hist.V_adaptive[ibest])
println("Best c:")
names = (:f_q_max, :E_C_over_h, :kappa, :Delta_qr, :temperature, :A_phi, :A_Ic)
for (n, v) in zip(names, hist.c_vec[ibest])
    @printf("  %-12s = %+.4e\n", n, v)
end
