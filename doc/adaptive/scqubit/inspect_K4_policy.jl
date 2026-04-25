#=
inspect_K4_policy.jl — Inspect the DP-optimal policy at various K_PHI.
Prints the first few actions from root for each K_PHI setting.
=#
using Printf
include("/home/zlin/BFIMGaussian/doc/adaptive/scqubit/ScqubitModel.jl")
include("/home/zlin/BFIMGaussian/doc/adaptive/scqubit/Belief.jl")
include("/home/zlin/BFIMGaussian/doc/adaptive/scqubit/Bellman.jl")
include("/home/zlin/BFIMGaussian/doc/adaptive/scqubit/Gradient.jl")
include("/home/zlin/BFIMGaussian/doc/adaptive/scqubit/JointOpt.jl")
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt

const K_EPOCHS = 4
const J_TAU = 10
const TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID = (1, 10)

c = PAPER_BASELINE
phi_star_fn = make_phi_star_fn()
ωd = omega_q(phi_star_fn(c)[1], c)

for K_PHI in [64, 256]
    grid = make_grid(; K_phi=K_PHI, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)
    @printf("\n=== K_PHI=%d ===\n", K_PHI)
    flush(stdout)
    (V, memo, st) = solve_bellman_full(grid, K_EPOCHS, c, ωd; terminal=:mse)
    @printf("V_train=%.4e memo=%d %.1fs\n", V, st.memo_size, st.elapsed)
    flush(stdout)

    # Print root action
    counts0 = ntuple(_ -> (0, 0), J_TAU)
    root_node = memo[(counts0, K_EPOCHS)]
    (j0, l0) = root_node.action
    @printf("Root action: τ[%d]=%.1fns, n=%d (V=%.4e)\n",
            j0, TAU_GRID[j0]*1e9, N_GRID[l0], root_node.value)

    # Print all reachable second-level actions (after one observation at root)
    τ0 = TAU_GRID[j0]
    n0 = N_GRID[l0]
    println("\nFirst-level branch actions (one root obs, then r=K-1):")
    for m0 in 0:n0
        counts1 = ntuple(k -> k == j0 ? (n0, m0) : (0, 0), J_TAU)
        node1 = get(memo, (counts1, K_EPOCHS-1), nothing)
        if node1 !== nothing
            (j1, l1) = node1.action
            @printf("  obs %d/%d ⇒ τ[%d]=%.1fns n=%d (V=%.4e)\n",
                    m0, n0, j1, TAU_GRID[j1]*1e9, N_GRID[l1], node1.value)
        end
    end
    flush(stdout)
    GC.gc()
end
