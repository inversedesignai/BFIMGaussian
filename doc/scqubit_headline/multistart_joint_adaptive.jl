#=
multistart_joint_adaptive.jl — joint-DP multistart with REOPT_EVERY=1
(every iteration re-solves the Bellman policy, so the gradient is always
the true envelope-theorem gradient — no stale-policy bias) and adaptive
lr decay (halve when V plateaus for N iters).

Each restart explores from a different c_0; the global joint-DP optimum
is the best across all restarts.

Inits:
  paper   - PAPER_BASELINE
  naive   - mid-box
  rand_1  - uniform-random sample, MersenneTwister(1)
  rand_2  - uniform-random sample, MersenneTwister(2)

Output: results/joint_multistart/<INIT_ID>.jls per restart.
Wall-clock: ~30-40 min per restart on 64 threads (Bellman re-solve dominates).
=#
using Printf, Serialization, Dates, LinearAlgebra, Random

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "BellmanThreaded.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "GradientThreaded.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
using .ScqubitModel, .Belief, .Bellman, .BellmanThreaded, .Gradient, .GradientThreaded, .JointOpt

println("multistart_joint_adaptive.jl"); flush(stdout)
println("Threads: $(Threads.nthreads())"); flush(stdout)

const K_EPOCHS = 4
const K_PHI    = 128
const J_TAU    = 10
const PHI_MAX  = 0.1
const TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID   = (1, 10)
const PARALLEL_DEPTH = 1

# Adam + adaptive-lr config
const MAX_ITERS  = parse(Int, get(ENV, "MAX_ITERS", "60"))
const LR_INIT    = parse(Float64, get(ENV, "LR_INIT", "5e-3"))
const LR_DECAY   = parse(Float64, get(ENV, "LR_DECAY", "0.5"))
const LR_MIN     = parse(Float64, get(ENV, "LR_MIN", "1e-6"))
const PATIENCE   = parse(Int, get(ENV, "PATIENCE", "8"))
const TOL        = parse(Float64, get(ENV, "TOL", "1e-8"))   # min V improvement to count as progress

# Box (T, A_phi, A_Ic pinned)
function realistic_box(bl::ScqubitParams)
    lo = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    hi = [12.0e9,   0.4e9,   5.0e6,   5.0e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    CBox(lo, hi)
end
const LO4 = [3.0e9, 0.15e9, 0.1e6, 0.8e9]
const HI4 = [12.0e9, 0.4e9, 5.0e6, 5.0e9]

# Init set
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
        x = LO4 .+ (HI4 .- LO4) .* rand(rng, 4)
        return ScqubitParams(
            f_q_max=x[1], E_C_over_h=x[2], kappa=x[3], Delta_qr=x[4],
            temperature=PAPER_BASELINE.temperature,
            A_phi=PAPER_BASELINE.A_phi, A_Ic=PAPER_BASELINE.A_Ic,
        )
    else
        error("unknown INIT_ID=$id")
    end
end

const INIT_LIST = let raw = get(ENV, "INITS", "paper,naive,rand_1,rand_2")
    isempty(raw) ? String[] : split(raw, ",")
end

println("Inits: ", INIT_LIST)
println("Config: max_iters=$MAX_ITERS  lr_init=$LR_INIT  patience=$PATIENCE  decay=$LR_DECAY  lr_min=$LR_MIN")
flush(stdout)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)
phi_star_fn = make_phi_star_fn()
omega_d_fn  = make_omega_d_fn(; phi_star_fn=phi_star_fn)
box = realistic_box(PAPER_BASELINE)

outdir = joinpath(@__DIR__, "results", "joint_multistart")
isdir(outdir) || mkpath(outdir)

# ---------- Per-restart Adam with REOPT_EVERY=1 + adaptive lr ----------
function run_restart(init_c, init_id::AbstractString)
    println("\n" * "="^72)
    println("RESTART: $init_id")
    println("="^72)
    @printf("init c:  f_q=%.4f GHz  E_C=%.4f GHz  κ=%.4f MHz  Δ=%.4f GHz\n",
            init_c.f_q_max/1e9, init_c.E_C_over_h/1e9,
            init_c.kappa/1e6, init_c.Delta_qr/1e9)
    flush(stdout)

    v = c_as_vec(init_c); project_c!(v, box)
    scale = max.(box.hi .- box.lo, 0.0)
    state = AdamState(length(v); lr=LR_INIT, scale=scale)

    hist = (V_adaptive = Float64[], grad_norm = Float64[],
            c_vec = Vector{Vector{Float64}}(),
            omega_d = Float64[],
            elapsed = Float64[],
            lr = Float64[])

    V_best = -Inf
    iters_since_improve = 0

    t_run0 = time()
    for iter in 1:MAX_ITERS
        t_iter = time()
        c_cur = vec_as_c(v)
        ω_d   = omega_d_fn(c_cur)
        (V, memo, st) = solve_bellman_threaded_full(grid, K_EPOCHS, c_cur, ω_d; terminal=:mse)
        g = grad_c_exact_fd_threaded(v, memo, grid, ω_d, K_EPOCHS;
                                     terminal=:mse, parallel_depth=PARALLEL_DEPTH)
        gn = norm(g)
        adam_update!(v, g, state); project_c!(v, box)

        push!(hist.V_adaptive, V); push!(hist.grad_norm, gn)
        push!(hist.c_vec, copy(v)); push!(hist.omega_d, ω_d)
        push!(hist.elapsed, time() - t_iter); push!(hist.lr, state.lr)

        # Adaptive lr: check for improvement
        improved = V > V_best + TOL * abs(V_best)
        if improved
            V_best = V
            iters_since_improve = 0
        else
            iters_since_improve += 1
        end

        @printf("[%-6s] iter %3d  V=%+.6e  |g|=%.3e  lr=%.2e  Δt=%.1fs%s\n",
                init_id, iter, V, gn, state.lr, hist.elapsed[end],
                improved ? "  <best>" : "")
        flush(stdout)

        if iters_since_improve >= PATIENCE
            state.lr *= LR_DECAY
            iters_since_improve = 0
            @printf("       [%s] lr decay -> %.2e\n", init_id, state.lr); flush(stdout)
            if state.lr < LR_MIN
                println("       [$init_id] lr below LR_MIN, terminating restart"); flush(stdout)
                break
            end
        end
    end
    elapsed = time() - t_run0
    @printf("[%s] total: %.1f min  (%d iters)\n", init_id, elapsed/60, length(hist.V_adaptive)); flush(stdout)

    nj = length(hist.V_adaptive)
    ibest = argmax(hist.V_adaptive[1:nj])
    v_best = hist.c_vec[ibest]
    V_best_final = hist.V_adaptive[ibest]
    @printf("[%s] V_best = %.6e at iter %d  c=[%.4f, %.4f, %.4f MHz, %.4f]\n",
            init_id, V_best_final, ibest,
            v_best[1]/1e9, v_best[2]/1e9, v_best[3]/1e6, v_best[4]/1e9)
    flush(stdout)

    out_path = joinpath(outdir, "$(init_id).jls")
    open(out_path, "w") do io
        serialize(io, (; init_id, init=init_c,
                         hist, ibest, v_best, V_best=V_best_final,
                         K_EPOCHS, K_PHI, PHI_MAX, TAU_GRID, N_GRID,
                         elapsed, timestamp=now(),
                         max_iters=MAX_ITERS, lr_init=LR_INIT,
                         lr_decay=LR_DECAY, patience=PATIENCE, lr_min=LR_MIN,
                         terminal=:mse,
                         fixed_components=(:temperature, :A_phi, :A_Ic)))
    end
    println("Saved $out_path"); flush(stdout)
    (V_best_final, v_best, ibest)
end

# ---------- Run all restarts ----------
all_results = []
for init_id in INIT_LIST
    init_c = init_c_from_id(init_id)
    (V_best, v_best, ibest) = run_restart(init_c, init_id)
    push!(all_results, (init_id, V_best, v_best, ibest))
end

println("\n" * "="^72)
println("MULTISTART SUMMARY (joint-DP)")
println("="^72)
@printf("%-10s %-15s %-10s %s\n", "init", "V_best", "iter", "c (f_q, E_C, κ MHz, Δ)")
for (id, V, v, i) in all_results
    @printf("%-10s %+.6e  %-10d (%.4f, %.4f, %.4f, %.4f)\n",
            id, V, i, v[1]/1e9, v[2]/1e9, v[3]/1e6, v[4]/1e9)
end
println("-"^72)
i_global = argmax(i -> all_results[i][2], 1:length(all_results))
(id_g, V_g, v_g, _) = all_results[i_global]
@printf("GLOBAL BEST: init=%s  V=%.6e  c=(%.4f, %.4f, %.4f MHz, %.4f)\n",
        id_g, V_g, v_g[1]/1e9, v_g[2]/1e9, v_g[3]/1e6, v_g[4]/1e9)
println("="^72)
flush(stdout)

# Save summary
open(joinpath(outdir, "_summary.jls"), "w") do io
    serialize(io, (; all_results, i_global,
                     INIT_LIST, MAX_ITERS, LR_INIT, LR_DECAY, PATIENCE, LR_MIN,
                     timestamp=now()))
end
println("Saved $(joinpath(outdir, "_summary.jls"))")
