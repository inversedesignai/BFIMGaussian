#=
bayesopt_pcrb_k256.jl — Bayesian optimization of PCRB log_J_P(c, schedule)
at K_PHI=256, gradient-free, globally aware.

Schedule is fixed to (j=10, ℓ=2)^K = (320 ns, n=10)^4, which is consistently
the optimal schedule across all PCRB optimizations we've tried — both at
PAPER_BASELINE and from random inits. We verify this assumption at the
BayesOpt winner by re-enumerating all 20^4 = 160k schedules and confirming.

Search space: 4D normalized box [0,1]^4.
Objective: log J_P(c, schedule_fixed) under the prior-averaged BIM. BayesOpt
maximises this (just like for joint-DP, max V_adaptive).

Output: results/bayesopt_pcrb_k256/result.jls
Wall-clock: ~30-45 min (each log_J_P eval is essentially free; the cost is
the per-eval bim_prior_averaged at K_PHI=256).
=#
using Printf, Serialization, Dates, Random
using BayesianOptimization, GaussianProcesses, Distributions

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt, .PCRB

println("bayesopt_pcrb_k256.jl"); flush(stdout)
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

const LO = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9 ]
const HI = [12.0e9,   0.4e9,   5.0e6,   5.0e9 ]
const SCALE = HI .- LO

# Fixed schedule: (j=J_TAU, ℓ=L)^K = (320 ns, n=10)^4 — always optimal in our experiments.
const FIXED_SCHED = [(J_TAU, 2) for _ in 1:K_EPOCHS]

println(@sprintf("Config: K_PHI=%d  N_EVAL=%d  N_INIT=%d  SEED=%d  fixed_schedule=%s",
                 K_PHI, N_EVAL, N_INIT, SEED, string(FIXED_SCHED)))
flush(stdout)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)
omega_d_fn = make_omega_d_fn()

# ---------- Black-box objective in normalized [0,1]^4 ----------
const HIST = (logJP = Float64[], c_vec = Vector{Vector{Float64}}(),
              elapsed = Float64[], z = Vector{Vector{Float64}}())

function objective(z)
    t0 = time()
    x = LO .+ SCALE .* z
    v = [x[1], x[2], x[3], x[4],
         PAPER_BASELINE.temperature, PAPER_BASELINE.A_phi, PAPER_BASELINE.A_Ic]
    c = vec_as_c(v)
    ω_d = omega_d_fn(c)
    lJP = log_JP_of_schedule(FIXED_SCHED, grid, c, ω_d; J_0=1e-4)
    push!(HIST.logJP, lJP); push!(HIST.c_vec, copy(v))
    push!(HIST.z, copy(z)); push!(HIST.elapsed, time() - t0)
    n = length(HIST.logJP)
    @printf("[pcrb bopt %3d] log_JP=%+.6f  c=(%.4f, %.4f, %.4f MHz, %.4f) %.1fs\n",
            n, lJP, x[1]/1e9, x[2]/1e9, x[3]/1e6, x[4]/1e9, HIST.elapsed[end])
    flush(stdout)
    return lJP
end

Random.seed!(SEED)

model = ElasticGPE(4,
    mean = MeanConst(0.0),
    kernel = SEArd([0.0 for _ in 1:4], 5.0),
    logNoise = -4.0,
    capacity = 1000)

modeloptimizer = MAPGPOptimizer(
    every = 10,
    noisebounds = [-7.0, 0.0],
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

println("\nLaunching BOpt with $N_EVAL evals ($N_INIT random init)..."); flush(stdout)
t_run0 = time()
result = boptimize!(opt)
elapsed = time() - t_run0
println("\nbopt total: $(round(elapsed/60; digits=1)) min"); flush(stdout)

ibest = argmax(HIST.logJP)
v_best = HIST.c_vec[ibest]
logJP_best = HIST.logJP[ibest]
@printf("\nBOpt best log_JP = %.6f at eval %d\n", logJP_best, ibest)
@printf("  c_best: f_q=%.4f GHz  E_C=%.4f GHz  κ=%.4f MHz  Δ=%.4f GHz\n",
        v_best[1]/1e9, v_best[2]/1e9, v_best[3]/1e6, v_best[4]/1e9)
flush(stdout)

# ---------- Sanity check: enumerate schedules at the BayesOpt winner ----------
# (cheap: 160k schedules × ~few µs each at K_PHI=256 ≈ a few seconds)
println("\nVerifying assumed schedule by enumeration at c_best...")
c_best = vec_as_c(v_best)
ω_best = omega_d_fn(c_best)
t_enum = time()
(sched_argmax, logJP_argmax) = argmax_schedule_enumerate(grid, c_best, ω_best, K_EPOCHS; J_0=1e-4)
@printf("  argmax schedule: %s   log_JP=%.6f   (%.1fs)\n",
        string(sched_argmax), logJP_argmax, time()-t_enum)
@printf("  fixed assumed:   %s   log_JP=%.6f\n", string(FIXED_SCHED), logJP_best)
if sched_argmax == FIXED_SCHED
    println("  ✓ assumption confirmed: (320 ns, n=10)^4 is optimal at BayesOpt c_best")
else
    @printf("  ⚠  optimal schedule differs.  log_JP gap = %.6f\n",
            logJP_argmax - logJP_best)
end
flush(stdout)

outdir = joinpath(@__DIR__, "results", "bayesopt_pcrb_k256")
isdir(outdir) || mkpath(outdir)
open(joinpath(outdir, "result.jls"), "w") do io
    serialize(io, (; hist=HIST, ibest, logJP_best, v_best, c_best,
                     fixed_schedule=FIXED_SCHED,
                     sched_argmax_at_best=sched_argmax,
                     logJP_argmax_at_best=logJP_argmax,
                     N_EVAL, N_INIT, SEED, elapsed,
                     K_EPOCHS, K_PHI, PHI_MAX, TAU_GRID, N_GRID,
                     optimizer=:BayesOpt_EI_ARDMatern,
                     timestamp=now()))
end
println("Saved $(joinpath(outdir, "result.jls"))")
