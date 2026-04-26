#=
bayesopt_pcrb_k256_full.jl — strictly fair Bayesian optimization of PCRB,
where the schedule is **re-enumerated at every BayesOpt probe** instead of
held fixed. The objective at each c is

    f(c) = max over schedules s of log_JP(c, s)

evaluated by argmax_schedule_enumerate (160k schedules at K=4, J=10, L=2).

This is the "true" BayesOpt of PCRB: at every c probed by the GP, the
inner schedule problem is solved to optimum, so we never assume any
schedule. Slower than the fixed-schedule version (each eval ≈ 40-80s
for the enumeration at K_PHI=256) but rigorous.

Inits / acquisition / GP setup are identical to bayesopt_joint_k256.jl
and bayesopt_pcrb_k256.jl: ARD-Matern GP, Expected Improvement, 4D
normalized [0,1]^4 box, 10 random init + 90 EI evals = 100 total.

Output: results/bayesopt_pcrb_k256_full/result.jls
Wall-clock: ~60-90 min on 32 threads.
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

println("bayesopt_pcrb_k256_full.jl"); flush(stdout)
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

println(@sprintf("Config: K_PHI=%d  N_EVAL=%d  N_INIT=%d  SEED=%d  schedule=ARGMAX (re-enumerated per eval)",
                 K_PHI, N_EVAL, N_INIT, SEED))
flush(stdout)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)
omega_d_fn = make_omega_d_fn()

const HIST = (logJP = Float64[], c_vec = Vector{Vector{Float64}}(),
              sched = Vector{Vector{Tuple{Int,Int}}}(),
              elapsed = Float64[], z = Vector{Vector{Float64}}())

function objective(z)
    t0 = time()
    x = LO .+ SCALE .* z
    v = [x[1], x[2], x[3], x[4],
         PAPER_BASELINE.temperature, PAPER_BASELINE.A_phi, PAPER_BASELINE.A_Ic]
    c = vec_as_c(v)
    ω_d = omega_d_fn(c)
    # Re-enumerate the optimal schedule at THIS c (this is what makes it rigorous)
    (sched_argmax, lJP) = argmax_schedule_enumerate(grid, c, ω_d, K_EPOCHS; J_0=1e-4)
    push!(HIST.logJP, lJP); push!(HIST.c_vec, copy(v))
    push!(HIST.sched, copy(sched_argmax)); push!(HIST.z, copy(z))
    push!(HIST.elapsed, time() - t0)
    n = length(HIST.logJP)
    @printf("[pcrb-full bopt %3d] log_JP=%+.6f  c=(%.4f, %.4f, %.4f MHz, %.4f)  sched=%s  %.1fs\n",
            n, lJP, x[1]/1e9, x[2]/1e9, x[3]/1e6, x[4]/1e9, string(sched_argmax),
            HIST.elapsed[end])
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

println("\nLaunching BOpt with $N_EVAL evals ($N_INIT random init), schedule re-enumerated per eval"); flush(stdout)
t_run0 = time()
result = boptimize!(opt)
elapsed = time() - t_run0
println("\nbopt total: $(round(elapsed/60; digits=1)) min"); flush(stdout)

ibest = argmax(HIST.logJP)
v_best = HIST.c_vec[ibest]
sched_best = HIST.sched[ibest]
logJP_best = HIST.logJP[ibest]
@printf("\nBOpt best log_JP = %.6f at eval %d\n", logJP_best, ibest)
@printf("  c_best: f_q=%.4f GHz  E_C=%.4f GHz  κ=%.4f MHz  Δ=%.4f GHz\n",
        v_best[1]/1e9, v_best[2]/1e9, v_best[3]/1e6, v_best[4]/1e9)
@printf("  sched_best: %s\n", string(sched_best))
flush(stdout)

# Distribution of schedules across the 100 evals (do they all collapse to (10,2)^4?)
sched_counts = Dict{Vector{Tuple{Int,Int}}, Int}()
for s in HIST.sched
    sched_counts[s] = get(sched_counts, s, 0) + 1
end
println("\nSchedule distribution across $(length(HIST.sched)) evaluations:")
for (s, cnt) in sort(collect(sched_counts), by=x->-x[2])
    @printf("  %4d  %s\n", cnt, string(s))
end
flush(stdout)

outdir = joinpath(@__DIR__, "results", "bayesopt_pcrb_k256_full")
isdir(outdir) || mkpath(outdir)
open(joinpath(outdir, "result.jls"), "w") do io
    serialize(io, (; hist=HIST, ibest, logJP_best, v_best, sched_best,
                     sched_counts,
                     N_EVAL, N_INIT, SEED, elapsed,
                     K_EPOCHS, K_PHI, PHI_MAX, TAU_GRID, N_GRID,
                     optimizer=:BayesOpt_EI_ARDMatern_with_argmax_sched,
                     timestamp=now()))
end
println("Saved $(joinpath(outdir, "result.jls"))")
