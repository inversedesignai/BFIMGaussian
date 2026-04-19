#=
sweep_joint.jl — full 7-D Adam joint-DP optimization from the paper baseline.

Outputs:
  results/joint/ckpt_*.jls   — periodic checkpoints
  results/joint/final.jls    — final (c_final, history, memo_final)
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

println("sweep_joint.jl — joint DP Adam starting from PAPER_BASELINE")
println("Threads: $(Threads.nthreads())")

# --- configuration ---
const K_EPOCHS       = 3
const K_PHI          = 64
const TAU_GRID       = ntuple(k -> 20e-9 * 2.0^(k-1), 4)  # J = 4
const N_GRID         = (1, 10)                              # L = 2
const OUTER_ITERS    = parse(Int, get(ENV, "JOINT_ITERS", "150"))
const OUTER_LR       = parse(Float64, get(ENV, "JOINT_LR", "2e-3"))
const REOPT_EVERY    = parse(Int, get(ENV, "JOINT_REOPT", "10"))

grid = make_grid(; K_phi=K_PHI, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)

println(@sprintf("Config: K=%d  K_Φ=%d  J=%d  L=%d  iters=%d  lr=%.1e  reopt=%d",
                 K_EPOCHS, K_PHI, length(TAU_GRID), length(N_GRID),
                 OUTER_ITERS, OUTER_LR, REOPT_EVERY))

t0 = time()
(c_final, hist, memo_final) = joint_opt(PAPER_BASELINE;
    grid=grid, K_epochs=K_EPOCHS,
    outer_iters=OUTER_ITERS, outer_lr=OUTER_LR,
    policy_reopt_every=REOPT_EVERY,
    ckpt_every=50,
    ckpt_dir=joinpath(@__DIR__, "results", "joint"),
    verbose=true)
@printf("\nTotal elapsed: %.1f min\n", (time() - t0) / 60)

# --- save final ---
out_path = joinpath(@__DIR__, "results", "joint", "final.jls")
open(out_path, "w") do io
    serialize(io, (; c_final, v_final=c_as_vec(c_final),
                     history=hist, memo_final, K_EPOCHS, K_PHI,
                     TAU_GRID, N_GRID, timestamp=now(),
                     baseline=PAPER_BASELINE))
end
println("Saved final to $out_path")

# --- summary ---
V_start = hist.V_adaptive[1]
V_end   = hist.V_adaptive[end]
println("\n" * "="^60)
@printf("V_adaptive(c₀)      = %.6f nats\n", V_start)
@printf("V_adaptive(c_final) = %.6f nats\n", V_end)
@printf("Δ                   = %+.6f nats  (%.2fx relative)\n",
        V_end - V_start, V_end / V_start)
println("Final c:")
v = c_as_vec(c_final)
for (i, name) in enumerate(C_FIELD_NAMES)
    @printf("  %-12s = %+.4e\n", name, v[i])
end
