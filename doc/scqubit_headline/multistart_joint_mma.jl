#=
multistart_joint_mma.jl — joint-DP multistart with MMA (Method of Moving
Asymptotes, NLopt :LD_MMA) at K_PHI=256 throughout (training matches
deployment, so no coarse-grid artifact).

MMA is applied in **normalized [0,1]^4 parameter space** (z = (x-LO)/(HI-LO))
so the per-dim asymptote logic is uniform across the four free `c` dims
(otherwise f_q's huge physical scale dominates the descent direction
just like it did for L-BFGS-B).

At each MMA evaluation:
  - solve_bellman_threaded_full at K_PHI=256 (~140 s)  ← dominant cost
  - grad_c_exact_fd_threaded at K_PHI=256 (~1-3 s)

Wall-clock per restart: ~70-120 min  (~30-50 MMA iters × ~140s/eval).

Inits: paper, naive, rand_1, rand_2 (same set as Adam multistart).
Output: results/joint_mma_k256/<INIT_ID>.jls per restart.
=#
using Printf, Serialization, Dates, LinearAlgebra, Random
using NLopt

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "BellmanThreaded.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "GradientThreaded.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
using .ScqubitModel, .Belief, .Bellman, .BellmanThreaded, .Gradient, .GradientThreaded, .JointOpt

println("multistart_joint_mma.jl"); flush(stdout)
println("Threads: $(Threads.nthreads())"); flush(stdout)

const K_EPOCHS = 4
const K_PHI    = parse(Int, get(ENV, "K_PHI", "256"))
const J_TAU    = 10
const PHI_MAX  = 0.1
const TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID   = (1, 10)
const PARALLEL_DEPTH = 1
const MAX_EVALS  = parse(Int, get(ENV, "MAX_EVALS", "60"))
const XTOL_REL   = parse(Float64, get(ENV, "XTOL_REL", "1e-4"))
const FTOL_REL   = parse(Float64, get(ENV, "FTOL_REL", "1e-5"))

# Free dims: f_q_max, E_C/h, kappa, Delta_qr  (T, A_phi, A_Ic pinned)
const LO = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9 ]
const HI = [12.0e9,   0.4e9,   5.0e6,   5.0e9 ]
const SCALE = HI .- LO   # for normalization

function init_c_from_id(id::AbstractString)
    if id == "paper"
        return PAPER_BASELINE
    elseif id == "naive"
        return ScqubitParams(f_q_max=7.5e9, E_C_over_h=0.275e9, kappa=2.55e6, Delta_qr=2.9e9,
            temperature=PAPER_BASELINE.temperature, A_phi=PAPER_BASELINE.A_phi, A_Ic=PAPER_BASELINE.A_Ic)
    elseif startswith(id, "rand_")
        seed = parse(Int, replace(id, "rand_" => ""))
        rng = MersenneTwister(seed)
        x = LO .+ SCALE .* rand(rng, 4)
        return ScqubitParams(f_q_max=x[1], E_C_over_h=x[2], kappa=x[3], Delta_qr=x[4],
            temperature=PAPER_BASELINE.temperature, A_phi=PAPER_BASELINE.A_phi, A_Ic=PAPER_BASELINE.A_Ic)
    else
        error("unknown INIT_ID=$id")
    end
end

const INIT_LIST = let raw = get(ENV, "INITS", "paper,naive,rand_1,rand_2")
    isempty(raw) ? String[] : split(raw, ",")
end

println("Inits: ", INIT_LIST)
println("Config: K_PHI=$K_PHI  max_evals=$MAX_EVALS  xtol_rel=$XTOL_REL  ftol_rel=$FTOL_REL")
flush(stdout)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)
phi_star_fn = make_phi_star_fn()
omega_d_fn  = make_omega_d_fn(; phi_star_fn=phi_star_fn)

outdir = joinpath(@__DIR__, "results", "joint_mma_k256")
isdir(outdir) || mkpath(outdir)

# ---------- Per-restart MMA ----------
function run_restart(init_c, init_id::AbstractString)
    println("\n" * "="^72)
    println("RESTART: $init_id  (MMA at K_PHI=$K_PHI, normalized [0,1]^4)")
    println("="^72)
    @printf("init c:  f_q=%.4f GHz  E_C=%.4f GHz  κ=%.4f MHz  Δ=%.4f GHz\n",
            init_c.f_q_max/1e9, init_c.E_C_over_h/1e9,
            init_c.kappa/1e6, init_c.Delta_qr/1e9)
    flush(stdout)

    hist = (V_adaptive = Float64[], grad_norm = Float64[],
            c_vec = Vector{Vector{Float64}}(),
            omega_d = Float64[],
            elapsed = Float64[])

    n_eval = Ref(0)

    # Objective in normalized coordinates z ∈ [0,1]^4
    function obj(z, grad)
        n_eval[] += 1
        t0 = time()
        x = LO .+ SCALE .* z
        v = [x[1], x[2], x[3], x[4],
             init_c.temperature, init_c.A_phi, init_c.A_Ic]
        c = vec_as_c(v)
        ω_d = omega_d_fn(c)
        (V, memo, _) = solve_bellman_threaded_full(grid, K_EPOCHS, c, ω_d; terminal=:mse)
        if length(grad) > 0
            g_full = grad_c_exact_fd_threaded(v, memo, grid, ω_d, K_EPOCHS;
                                              terminal=:mse, parallel_depth=PARALLEL_DEPTH)
            # Chain rule: dV/dz_i = SCALE_i * dV/dx_i
            for i in 1:4
                grad[i] = SCALE[i] * g_full[i]
            end
            push!(hist.grad_norm, norm(grad))
        else
            push!(hist.grad_norm, NaN)
        end
        push!(hist.V_adaptive, V); push!(hist.c_vec, copy(v))
        push!(hist.omega_d, ω_d); push!(hist.elapsed, time() - t0)
        @printf("[%-6s eval %3d] V=%+.6e  |g_z|=%.3e  c=(%.4f, %.4f, %.4f MHz, %.4f) %.1fs\n",
                init_id, n_eval[], V,
                isnan(hist.grad_norm[end]) ? 0.0 : hist.grad_norm[end],
                x[1]/1e9, x[2]/1e9, x[3]/1e6, x[4]/1e9, hist.elapsed[end])
        flush(stdout)
        return V
    end

    z0_phys = clamp.([init_c.f_q_max, init_c.E_C_over_h, init_c.kappa, init_c.Delta_qr],
                     LO, HI)
    z0 = (z0_phys .- LO) ./ SCALE

    opt = NLopt.Opt(:LD_MMA, 4)
    NLopt.lower_bounds!(opt, zeros(4))
    NLopt.upper_bounds!(opt, ones(4))
    NLopt.xtol_rel!(opt, XTOL_REL)
    NLopt.ftol_rel!(opt, FTOL_REL)
    NLopt.maxeval!(opt, MAX_EVALS)
    NLopt.max_objective!(opt, obj)

    t_run0 = time()
    (Vmax, zmax, ret) = NLopt.optimize(opt, z0)
    elapsed = time() - t_run0
    @printf("\n[%s] MMA done: %s  (%.1f min, %d evals)\n",
            init_id, string(ret), elapsed/60, n_eval[]); flush(stdout)

    nj = length(hist.V_adaptive)
    ibest = argmax(hist.V_adaptive[1:nj])
    v_best = hist.c_vec[ibest]
    V_best = hist.V_adaptive[ibest]
    @printf("[%s] V_best = %.6e  iter %d  c=[%.4f, %.4f, %.4f MHz, %.4f]\n",
            init_id, V_best, ibest,
            v_best[1]/1e9, v_best[2]/1e9, v_best[3]/1e6, v_best[4]/1e9)
    flush(stdout)

    open(joinpath(outdir, "$(init_id).jls"), "w") do io
        serialize(io, (; init_id, init=init_c,
                         hist, ibest, v_best, V_best, Vmax, zmax,
                         optimizer = :LD_MMA, ret,
                         K_EPOCHS, K_PHI, PHI_MAX, TAU_GRID, N_GRID,
                         elapsed, n_eval=n_eval[], timestamp=now(),
                         max_evals=MAX_EVALS, xtol_rel=XTOL_REL, ftol_rel=FTOL_REL,
                         terminal=:mse,
                         fixed_components=(:temperature, :A_phi, :A_Ic)))
    end
    println("Saved $(joinpath(outdir, "$(init_id).jls"))")
    flush(stdout)
    (V_best, v_best, ibest)
end

# ---------- Run all restarts sequentially ----------
all_results = NamedTuple[]
for init_id in INIT_LIST
    init_c = init_c_from_id(init_id)
    (V_best, v_best, ibest) = run_restart(init_c, init_id)
    push!(all_results, (init_id=init_id, V_best=V_best, v_best=v_best, ibest=ibest))
end

println("\n" * "="^72)
println("MMA MULTISTART SUMMARY (joint-DP, K_PHI=$K_PHI)")
println("="^72)
@printf("%-10s %-15s %-6s %s\n", "init", "V_best", "iter", "c (f_q, E_C, κ MHz, Δ_qr)")
for r in all_results
    @printf("%-10s %+.6e   %-4d  (%.4f, %.4f, %.4f, %.4f)\n",
            r.init_id, r.V_best, r.ibest,
            r.v_best[1]/1e9, r.v_best[2]/1e9, r.v_best[3]/1e6, r.v_best[4]/1e9)
end
println("-"^72)
i_g = argmax(i -> all_results[i].V_best, 1:length(all_results))
g = all_results[i_g]
@printf("GLOBAL BEST: init=%s  V=%.6e  c=(%.4f, %.4f, %.4f MHz, %.4f)\n",
        g.init_id, g.V_best,
        g.v_best[1]/1e9, g.v_best[2]/1e9, g.v_best[3]/1e6, g.v_best[4]/1e9)
println("="^72); flush(stdout)

open(joinpath(outdir, "_summary.jls"), "w") do io
    serialize(io, (; all_results, i_global=i_g,
                     INIT_LIST, MAX_EVALS, XTOL_REL, FTOL_REL,
                     K_PHI, optimizer=:LD_MMA, timestamp=now()))
end
println("Saved $(joinpath(outdir, "_summary.jls"))")
