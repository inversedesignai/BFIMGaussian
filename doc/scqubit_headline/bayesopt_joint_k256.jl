#=
bayesopt_joint_k256.jl — Bayesian optimization of joint-DP V_adaptive(c)
at K_PHI=256, gradient-free, globally aware.

Uses BayesianOptimization.jl's GP surrogate with ARD-Matérn kernel (per-dim
length scales) and Expected-Improvement acquisition. Each evaluation is one
Bellman re-solve at K_PHI=256 (≈30s) — no gradient. The GP surrogate uses
all past evaluations to decide where to probe next, so 100 evals globally
explore the 4D box rather than locally descending in one basin.

Search space: 4D normalized box [0,1]^4 (avoids unit-scale anisotropy).

Output: results/bayesopt_joint_k256/result.jls
Wall-clock: ~50 min on 64 threads (100 evals × ~30s).
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

println("bayesopt_joint_k256.jl"); flush(stdout)
println("Threads: $(Threads.nthreads())"); flush(stdout)

const K_EPOCHS = 4
const K_PHI    = parse(Int, get(ENV, "K_PHI", "256"))
const J_TAU    = 10
const PHI_MAX  = 0.1
const TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID   = (1, 10)
const N_EVAL   = parse(Int, get(ENV, "N_EVAL", "100"))
const N_INIT   = parse(Int, get(ENV, "N_INIT", "10"))
const SEED     = parse(Int, get(ENV, "SEED", "42"))

# Box on the 4 free dims (T, A_phi, A_Ic pinned at PAPER_BASELINE)
const LO = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9 ]
const HI = [12.0e9,   0.4e9,   5.0e6,   5.0e9 ]
const SCALE = HI .- LO

println(@sprintf("Config: K_PHI=%d  N_EVAL=%d  N_INIT=%d  SEED=%d",
                 K_PHI, N_EVAL, N_INIT, SEED))
flush(stdout)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)
phi_star_fn = make_phi_star_fn()
omega_d_fn  = make_omega_d_fn(; phi_star_fn=phi_star_fn)

# ---------- Black-box objective in normalized [0,1]^4 ----------
const HIST = (V = Float64[], c_vec = Vector{Vector{Float64}}(),
              elapsed = Float64[], z = Vector{Vector{Float64}}())

function objective(z)
    t0 = time()
    x = LO .+ SCALE .* z
    v = [x[1], x[2], x[3], x[4],
         PAPER_BASELINE.temperature, PAPER_BASELINE.A_phi, PAPER_BASELINE.A_Ic]
    c = vec_as_c(v)
    ω_d = omega_d_fn(c)
    (V, _memo, _st) = solve_bellman_threaded_full(grid, K_EPOCHS, c, ω_d; terminal=:mse)
    push!(HIST.V, V); push!(HIST.c_vec, copy(v))
    push!(HIST.z, copy(z)); push!(HIST.elapsed, time() - t0)
    n = length(HIST.V)
    @printf("[joint bopt %3d] V=%+.6e  c=(%.4f, %.4f, %.4f MHz, %.4f) %.1fs\n",
            n, V, x[1]/1e9, x[2]/1e9, x[3]/1e6, x[4]/1e9, HIST.elapsed[end])
    flush(stdout)
    return V
end

# ---------- BayesOpt setup ----------
Random.seed!(SEED)

# ARD Matérn-5/2 GP with per-dim log-length-scales initialised at log(0.5),
# constant mean, log-noise σ_n initialised at -3 (since V is on order 1e-5).
model = ElasticGPE(4,
    mean = MeanConst(0.0),
    kernel = SEArd([0.0 for _ in 1:4], 5.0),
    logNoise = -3.0,
    capacity = 1000)

# Periodic GP hyperparameter optimisation (every 10 evals).
modeloptimizer = MAPGPOptimizer(
    every = 10,
    noisebounds = [-6.0, 1.0],
    kernbounds = [[-5.0, -5.0, -5.0, -5.0, -5.0],
                  [ 5.0,  5.0,  5.0,  5.0,  5.0]],
    maxeval = 40)

opt = BOpt(objective, model,
    ExpectedImprovement(),
    modeloptimizer,
    zeros(4), ones(4);
    repetitions = 1,
    maxiterations = N_EVAL,
    sense = Max,
    initializer_iterations = N_INIT,
    verbosity = Progress)

# ---------- Run ----------
println("\nLaunching BOpt with $N_EVAL evals ($N_INIT random init points)..."); flush(stdout)
t_run0 = time()
result = boptimize!(opt)
elapsed = time() - t_run0
println("\nbopt total: $(round(elapsed/60; digits=1)) min")
flush(stdout)

# ---------- Pick best V_adaptive (max in our convention) ----------
ibest = argmax(HIST.V)
v_best = HIST.c_vec[ibest]
V_best = HIST.V[ibest]
@printf("\nBOpt best V = %.6e at eval %d\n", V_best, ibest)
@printf("  c_best: f_q=%.4f GHz  E_C=%.4f GHz  κ=%.4f MHz  Δ=%.4f GHz\n",
        v_best[1]/1e9, v_best[2]/1e9, v_best[3]/1e6, v_best[4]/1e9)
flush(stdout)

# ---------- Save ----------
outdir = joinpath(@__DIR__, "results", "bayesopt_joint_k256")
isdir(outdir) || mkpath(outdir)
open(joinpath(outdir, "result.jls"), "w") do io
    serialize(io, (; hist=HIST, ibest, V_best, v_best,
                     N_EVAL, N_INIT, SEED, elapsed,
                     K_EPOCHS, K_PHI, PHI_MAX, TAU_GRID, N_GRID,
                     terminal=:mse, optimizer=:BayesOpt_EI_ARDMatern,
                     timestamp=now()))
end
println("Saved $(joinpath(outdir, "result.jls"))")
