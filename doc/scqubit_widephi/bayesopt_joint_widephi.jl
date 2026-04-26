#=
bayesopt_joint_widephi.jl — 5-D Bayesian optimization of joint-DP V_adaptive
at the WIDE-prior limit phi_max = 0.5 (the "uninformative" limit referenced
in §7.6 of the paper).

Mirrors doc/scqubit_5d/bayesopt_joint_5d.jl with PHI_MAX=0.5 instead of 0.1.
ω_d is a FREE 5th parameter (not derived via single-shot Fisher heuristic).

The Cooper-pair-box singularity at φ = 0.5 is handled by the model's
phi_clip = 0.49 default; grid points in [0.49, 0.5) all behave as if at
φ = 0.49.  The information-content floor this introduces is real and
appears in the comparison.

Output: results/bayesopt_joint_widephi/result.jls
Wall-clock: ~50 min on 64 threads.
=#
using Printf, Serialization, Dates, Random
using BayesianOptimization, GaussianProcesses, Distributions

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "BellmanThreaded.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
using .ScqubitModel, .Belief, .Bellman, .BellmanThreaded, .Gradient, .JointOpt

println("bayesopt_joint_widephi.jl  (phi_max = 0.5)"); flush(stdout)
println("Threads: $(Threads.nthreads())"); flush(stdout)

const K_EPOCHS = parse(Int, get(ENV, "K_EPOCHS", "4"))
const K_PHI    = parse(Int, get(ENV, "K_PHI", "256"))
const K_TAG    = "K$(K_EPOCHS)"
const J_TAU    = 10
const PHI_MAX  = 0.5
const TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID   = (1, 10)
const N_EVAL   = parse(Int, get(ENV, "N_EVAL", "100"))
const N_INIT   = parse(Int, get(ENV, "N_INIT", "10"))
const SEED     = parse(Int, get(ENV, "SEED", "42"))

const TWO_PI = 2π
const LO5 = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9,   TWO_PI * 1.0e9 ]
const HI5 = [12.0e9,   0.4e9,   5.0e6,   5.0e9,   TWO_PI * 12.0e9 ]
const SCALE5 = HI5 .- LO5

println(@sprintf("Config: PHI_MAX=%.2f  K_EPOCHS=%d  K_PHI=%d  N_EVAL=%d  N_INIT=%d  SEED=%d",
                 PHI_MAX, K_EPOCHS, K_PHI, N_EVAL, N_INIT, SEED))
@printf("Box:\n")
@printf("  f_q_max  ∈ [%.2f, %.2f] GHz\n", LO5[1]/1e9, HI5[1]/1e9)
@printf("  E_C/h    ∈ [%.2f, %.2f] GHz\n", LO5[2]/1e9, HI5[2]/1e9)
@printf("  κ        ∈ [%.2f, %.2f] MHz\n", LO5[3]/1e6, HI5[3]/1e6)
@printf("  Δ_qr     ∈ [%.2f, %.2f] GHz\n", LO5[4]/1e9, HI5[4]/1e9)
@printf("  ω_d/(2π) ∈ [%.2f, %.2f] GHz   (free)\n",
        LO5[5]/TWO_PI/1e9, HI5[5]/TWO_PI/1e9)
flush(stdout)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)

const HIST = (V = Float64[], c_vec = Vector{Vector{Float64}}(),
              omega_d = Float64[], elapsed = Float64[],
              z = Vector{Vector{Float64}}())

function objective(z)
    t0 = time()
    x = LO5 .+ SCALE5 .* z
    v = [x[1], x[2], x[3], x[4],
         PAPER_BASELINE.temperature, PAPER_BASELINE.A_phi, PAPER_BASELINE.A_Ic]
    c = vec_as_c(v)
    ω_d = x[5]
    (V, _memo, _st) = solve_bellman_threaded_full(grid, K_EPOCHS, c, ω_d; terminal=:mse)
    push!(HIST.V, V); push!(HIST.c_vec, copy(v)); push!(HIST.omega_d, ω_d)
    push!(HIST.z, copy(z)); push!(HIST.elapsed, time() - t0)
    n = length(HIST.V)
    @printf("[joint_widephi %3d] V=%+.6e  c=(%.4f, %.4f, %.4f MHz, %.4f)  ω_d/(2π)=%.4f GHz  %.1fs\n",
            n, V, x[1]/1e9, x[2]/1e9, x[3]/1e6, x[4]/1e9, ω_d/TWO_PI/1e9,
            HIST.elapsed[end])
    flush(stdout)
    return V
end

Random.seed!(SEED)

model = ElasticGPE(5,
    mean = MeanConst(0.0),
    kernel = SEArd([0.0 for _ in 1:5], 5.0),
    logNoise = -3.0,
    capacity = 1000)

modeloptimizer = MAPGPOptimizer(
    every = 10,
    noisebounds = [-6.0, 1.0],
    kernbounds = [[-5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
                  [ 5.0,  5.0,  5.0,  5.0,  5.0,  5.0]],
    maxeval = 40)

opt = BOpt(objective, model,
    ExpectedImprovement(),
    modeloptimizer,
    zeros(5), ones(5);
    repetitions = 1,
    maxiterations = N_EVAL,
    sense = Max,
    initializer_iterations = N_INIT,
    verbosity = Progress)

println("\nLaunching joint-DP 5-D BOpt at PHI_MAX=$PHI_MAX with $N_EVAL evals ($N_INIT random init)..."); flush(stdout)
t_run0 = time()
result = boptimize!(opt)
elapsed = time() - t_run0
println("\nbopt total: $(round(elapsed/60; digits=1)) min"); flush(stdout)

ibest = argmax(HIST.V)
v_best = HIST.c_vec[ibest]
ω_best = HIST.omega_d[ibest]
V_best = HIST.V[ibest]
@printf("\nBest V = %.6e at eval %d\n", V_best, ibest)
@printf("  c_best:  f_q=%.4f GHz  E_C=%.4f GHz  κ=%.4f MHz  Δ=%.4f GHz\n",
        v_best[1]/1e9, v_best[2]/1e9, v_best[3]/1e6, v_best[4]/1e9)
@printf("  ω_d/(2π) = %.4f GHz   ω_d = %.4e rad/s\n", ω_best/TWO_PI/1e9, ω_best)
flush(stdout)

phi_star_fn = make_phi_star_fn()
omega_d_fn  = make_omega_d_fn(; phi_star_fn=phi_star_fn)
ω_fn = omega_d_fn(vec_as_c(v_best))
@printf("  ω_d_fn(c_best)/(2π) = %.4f GHz  (Δ from BOpt's ω_d = %+.4f GHz, %+.1f%%)\n",
        ω_fn/TWO_PI/1e9, (ω_best-ω_fn)/TWO_PI/1e9, 100*(ω_best-ω_fn)/ω_fn)
flush(stdout)

outdir = joinpath(@__DIR__, "results", "bayesopt_joint_widephi_$(K_TAG)")
isdir(outdir) || mkpath(outdir)
open(joinpath(outdir, "result.jls"), "w") do io
    serialize(io, (; hist=HIST, ibest, V_best, v_best, omega_d_best=ω_best,
                     omega_d_fn_at_best=ω_fn,
                     N_EVAL, N_INIT, SEED, elapsed,
                     K_EPOCHS, K_PHI, PHI_MAX, TAU_GRID, N_GRID,
                     LO5, HI5, terminal=:mse,
                     optimizer=:BayesOpt_EI_ARDMatern_5D_widephi,
                     timestamp=now()))
end
println("Saved $(joinpath(outdir, "result.jls"))")
