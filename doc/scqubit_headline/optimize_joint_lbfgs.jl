#=
optimize_joint_lbfgs.jl — deterministic L-BFGS-B (Optim.Fminbox(LBFGS()))
for joint-DP. Optimises over the 4 free `c` dimensions
(f_q_max, E_C/h, κ, Δ_qr) with the realistic_box constraints. Every V/grad
evaluation re-solves the Bellman policy at the current c so the gradient is
the true envelope-theorem gradient (no stale-policy bias).

Init is selected via env var `INIT_ID`:
    paper     - PAPER_BASELINE
    naive     - mid-box (the previous headline init)
    rand_N    - uniform-random sample from the box, seed N

Output: results/joint_lbfgs/<INIT_ID>.jls

Wall-clock per restart: roughly 30-50 min on 64 threads
(L-BFGS line search ≈ 2-3 V evals per outer iter, ≈ 30 iters; each V eval
is a Bellman re-solve ≈ 32 s threaded).
=#
using Printf, Serialization, Dates, LinearAlgebra, Random
using Optim
using NLSolversBase: only_fg!

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "BellmanThreaded.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "GradientThreaded.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
using .ScqubitModel, .Belief, .Bellman, .BellmanThreaded, .Gradient, .GradientThreaded, .JointOpt

println("optimize_joint_lbfgs.jl"); flush(stdout)
println("Threads: $(Threads.nthreads())"); flush(stdout)

const K_EPOCHS = 4
const K_PHI    = 128
const J_TAU    = 10
const PHI_MAX  = 0.1
const TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID   = (1, 10)
const PARALLEL_DEPTH = 1

# Free indices in the 7-vector c (rest are pinned to PAPER_BASELINE)
const FREE = [1, 2, 3, 4]   # f_q_max, E_C/h, κ, Δ_qr
const PINNED_VALS = (PAPER_BASELINE.temperature, PAPER_BASELINE.A_phi, PAPER_BASELINE.A_Ic)

# Box bounds in physical units, restricted to the 4 free dims
const LO = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9 ]
const HI = [12.0e9,   0.4e9,   5.0e6,   5.0e9 ]

function init_c_from_id(id::AbstractString)
    if id == "paper"
        return PAPER_BASELINE
    elseif id == "naive"
        return ScqubitParams(
            f_q_max=7.5e9, E_C_over_h=0.275e9, kappa=2.55e6, Delta_qr=2.9e9,
            temperature=PAPER_BASELINE.temperature,
            A_phi=PAPER_BASELINE.A_phi, A_Ic=PAPER_BASELINE.A_Ic,
        )
    elseif startswith(id, "rand_")
        seed = parse(Int, replace(id, "rand_" => ""))
        rng = MersenneTwister(seed)
        x = LO .+ (HI .- LO) .* rand(rng, 4)
        return ScqubitParams(
            f_q_max=x[1], E_C_over_h=x[2], kappa=x[3], Delta_qr=x[4],
            temperature=PAPER_BASELINE.temperature,
            A_phi=PAPER_BASELINE.A_phi, A_Ic=PAPER_BASELINE.A_Ic,
        )
    else
        error("unknown INIT_ID=$id")
    end
end

const INIT_ID = get(ENV, "INIT_ID", "naive")
init_c = init_c_from_id(INIT_ID)
@printf("INIT_ID=%s  ->  f_q=%.4f GHz  E_C=%.4f GHz  κ=%.4f MHz  Δ=%.4f GHz\n",
        INIT_ID, init_c.f_q_max/1e9, init_c.E_C_over_h/1e9,
        init_c.kappa/1e6, init_c.Delta_qr/1e9)
flush(stdout)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)
phi_star_fn = make_phi_star_fn()
omega_d_fn  = make_omega_d_fn(; phi_star_fn=phi_star_fn)

# History of every (V, grad_norm, c, time) as Optim calls fg!
const HIST = (V_adaptive=Float64[], grad_norm=Float64[],
              c_vec=Vector{Vector{Float64}}(), elapsed=Float64[],
              memo_size=Int[], omega_d=Float64[])

# Build full 7-vector from 4 free
function full_c_vec(x_free::AbstractVector)
    [x_free[1], x_free[2], x_free[3], x_free[4],
     PINNED_VALS[1], PINNED_VALS[2], PINNED_VALS[3]]
end

# fg! returns -V (Optim minimizes) and -grad
# L-BFGS-B operates on a NORMALIZED [0,1]^4 parameter space so the gradient
# magnitudes across components (each measured in different physical units, with
# very different scales — Hz for f_q vs Hz for κ) are made comparable.
# z = (x - LO) / (HI - LO), x = LO + (HI - LO) .* z
# Chain rule: ∂V/∂z_i = (HI_i - LO_i) * ∂V/∂x_i  →  raw box-scaled gradient.
const SCALE = HI .- LO

function fg!(F, G, z)
    t0 = time()
    x_free = LO .+ SCALE .* z
    v = full_c_vec(x_free)
    c = vec_as_c(v)
    ω_d = omega_d_fn(c)
    (V, memo, st) = solve_bellman_threaded_full(grid, K_EPOCHS, c, ω_d; terminal=:mse)
    push!(HIST.V_adaptive, V); push!(HIST.memo_size, st.memo_size)
    push!(HIST.omega_d, ω_d); push!(HIST.c_vec, copy(v))
    if G !== nothing
        g_full = grad_c_exact_fd_threaded(v, memo, grid, ω_d, K_EPOCHS;
                                          terminal=:mse, parallel_depth=PARALLEL_DEPTH)
        # Box-scaled gradient in normalized space, negated (minimize -V)
        for (i, idx) in enumerate(FREE)
            G[i] = -g_full[idx] * SCALE[i]
        end
        push!(HIST.grad_norm, norm(g_full[FREE] .* SCALE))
    else
        push!(HIST.grad_norm, NaN)
    end
    push!(HIST.elapsed, time() - t0)
    nfeval = length(HIST.V_adaptive)
    @printf("[fg eval %3d]  V=%.6e  |g_norm|=%.3e  c=[%7.4f, %7.4f, %7.4f MHz, %7.4f] %.1fs\n",
            nfeval, V, isnan(HIST.grad_norm[end]) ? 0.0 : HIST.grad_norm[end],
            x_free[1]/1e9, x_free[2]/1e9, x_free[3]/1e6, x_free[4]/1e9,
            HIST.elapsed[end])
    flush(stdout)
    return -V
end

# Initial point in normalized [0,1]^4 space
x0_phys = clamp.([init_c.f_q_max, init_c.E_C_over_h, init_c.kappa, init_c.Delta_qr],
                  LO, HI)
z0 = (x0_phys .- LO) ./ SCALE
@printf("\nLaunching L-BFGS-B (Fminbox(LBFGS())) on 4 free dims, normalized [0,1]^4\n")
@printf("Box: f_q∈[%.1f,%.1f] GHz  E_C∈[%.2f,%.2f]  κ∈[%.1f,%.1f] MHz  Δ∈[%.1f,%.1f] GHz\n",
        LO[1]/1e9, HI[1]/1e9, LO[2]/1e9, HI[2]/1e9,
        LO[3]/1e6, HI[3]/1e6, LO[4]/1e9, HI[4]/1e9)
@printf("Init z = %s  (physical = %s)\n", string(z0), string(x0_phys))
flush(stdout)

t_run0 = time()
result = optimize(only_fg!(fg!), zeros(4), ones(4), z0,
                  Fminbox(LBFGS()),
                  Optim.Options(iterations = 50,
                                show_trace = true,
                                store_trace = true,
                                g_tol = 1e-4,
                                outer_iterations = 5))
elapsed = time() - t_run0

z_final = Optim.minimizer(result)
x_final = LO .+ SCALE .* z_final
v_final = full_c_vec(x_final)
c_final = vec_as_c(v_final)
V_final = -Optim.minimum(result)

# Best across history (in case Fminbox went past the optimum)
i_best = argmax(HIST.V_adaptive)
v_best = HIST.c_vec[i_best]
V_best = HIST.V_adaptive[i_best]

println("\n" * "="^72)
@printf("L-BFGS-B done  (%.1f min total, %d V evaluations)\n",
        elapsed/60, length(HIST.V_adaptive))
@printf("converged   = %s\n", Optim.converged(result))
@printf("V_final     = %.6e   c=[%.4f, %.4f, %.4f MHz, %.4f]\n",
        V_final, x_final[1]/1e9, x_final[2]/1e9, x_final[3]/1e6, x_final[4]/1e9)
@printf("V_best (i=%d) = %.6e   c=[%.4f, %.4f, %.4f MHz, %.4f]\n", i_best,
        V_best, v_best[1]/1e9, v_best[2]/1e9, v_best[3]/1e6, v_best[4]/1e9)
println("="^72); flush(stdout)

outdir = joinpath(@__DIR__, "results", "joint_lbfgs")
isdir(outdir) || mkpath(outdir)
out_path = joinpath(outdir, "$(INIT_ID).jls")
open(out_path, "w") do io
    serialize(io, (; result, c_final, v_final, V_final,
                     c_best = vec_as_c(v_best), v_best, V_best,
                     i_best, hist = HIST,
                     INIT_ID, init_c,
                     K_EPOCHS, K_PHI, PHI_MAX, TAU_GRID, N_GRID,
                     elapsed, timestamp = now(),
                     terminal = :mse,
                     fixed_components = (:temperature, :A_phi, :A_Ic)))
end
println("Saved to $out_path")
