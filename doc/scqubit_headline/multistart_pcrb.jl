#=
multistart_pcrb.jl — PCRB multistart from same init set as joint-DP
multistart. PCRB's V (= log J_P) landscape is much smoother than joint-DP's,
so we don't need adaptive lr — fixed Adam at lr=5e-3 with 150 iters from
each init suffices. Schedule re-enumerated every 2 iters.

Inits: paper, naive, rand_1, rand_2 (same as joint-DP multistart).
Output: results/pcrb_multistart/<INIT_ID>.jls + _summary.jls
Wall-clock: ~20-45 min per restart (PCRB enumeration is single-threaded).
=#
using Printf, Serialization, Dates, Random

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt, .PCRB

println("multistart_pcrb.jl"); flush(stdout)
println("Threads: $(Threads.nthreads())"); flush(stdout)

const K_EPOCHS = 4
const K_PHI    = 128
const J_TAU    = 10
const PHI_MAX  = 0.1
const TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID   = (1, 10)
const OUTER_ITERS = parse(Int, get(ENV, "PCRB_ITERS", "150"))
const OUTER_LR    = parse(Float64, get(ENV, "PCRB_LR", "5e-3"))
const REOPT_EVERY = parse(Int, get(ENV, "PCRB_REOPT", "2"))

function realistic_box(bl::ScqubitParams)
    lo = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    hi = [12.0e9,   0.4e9,   5.0e6,   5.0e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    CBox(lo, hi)
end
const LO4 = [3.0e9, 0.15e9, 0.1e6, 0.8e9]
const HI4 = [12.0e9, 0.4e9, 5.0e6, 5.0e9]

function init_c_from_id(id::AbstractString)
    if id == "paper"
        return PAPER_BASELINE
    elseif id == "naive"
        return ScqubitParams(f_q_max=7.5e9, E_C_over_h=0.275e9, kappa=2.55e6, Delta_qr=2.9e9,
            temperature=PAPER_BASELINE.temperature, A_phi=PAPER_BASELINE.A_phi, A_Ic=PAPER_BASELINE.A_Ic)
    elseif startswith(id, "rand_")
        seed = parse(Int, replace(id, "rand_" => ""))
        rng = MersenneTwister(seed)
        x = LO4 .+ (HI4 .- LO4) .* rand(rng, 4)
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
println("Config: iters=$OUTER_ITERS  lr=$OUTER_LR  reopt=$REOPT_EVERY")
flush(stdout)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)
box  = realistic_box(PAPER_BASELINE)
omega_d_fn = make_omega_d_fn()

outdir = joinpath(@__DIR__, "results", "pcrb_multistart")
isdir(outdir) || mkpath(outdir)

all_results = NamedTuple[]
for init_id in INIT_LIST
    init_c = init_c_from_id(init_id)
    println("\n" * "="^72)
    println("RESTART: $init_id")
    println("="^72)
    @printf("init c:  f_q=%.4f GHz  E_C=%.4f GHz  κ=%.4f MHz  Δ=%.4f GHz\n",
            init_c.f_q_max/1e9, init_c.E_C_over_h/1e9,
            init_c.kappa/1e6, init_c.Delta_qr/1e9)
    flush(stdout)

    t0 = time()
    (c_final, sched_final, hist) = pcrb_baseline(init_c;
        grid=grid, K_epochs=K_EPOCHS,
        outer_iters=OUTER_ITERS, outer_lr=OUTER_LR,
        schedule_reopt_every=REOPT_EVERY,
        omega_d_fn=omega_d_fn, cbox=box, verbose=true)
    elapsed = time() - t0
    @printf("[%s] elapsed: %.1f min\n", init_id, elapsed/60); flush(stdout)

    imax = argmax(hist.log_JP)
    v_best = hist.c_vec[imax]
    sched_best = hist.sched[imax]
    @printf("[%s] log_JP best @ iter %d = %.6f   sched=%s\n",
            init_id, imax, hist.log_JP[imax], string(sched_best))
    @printf("[%s]   c=[%.4f, %.4f, %.4f MHz, %.4f]\n",
            init_id, v_best[1]/1e9, v_best[2]/1e9, v_best[3]/1e6, v_best[4]/1e9)

    open(joinpath(outdir, "$(init_id).jls"), "w") do io
        serialize(io, (; init_id, init=init_c,
                         c_final, v_final=c_as_vec(c_final),
                         sched_final, history=hist,
                         imax, v_best, sched_best,
                         logJP_best=hist.log_JP[imax],
                         K_EPOCHS, K_PHI, PHI_MAX, TAU_GRID, N_GRID,
                         elapsed, timestamp=now()))
    end
    println("Saved.")
    flush(stdout)

    push!(all_results, (init_id=init_id, logJP_best=hist.log_JP[imax],
                        v_best=v_best, sched_best=sched_best, imax=imax))
end

println("\n" * "="^72)
println("MULTISTART SUMMARY (PCRB)")
println("="^72)
@printf("%-10s %-12s %-6s %s\n", "init", "log_JP_best", "iter", "c (f_q, E_C, κ MHz, Δ_qr)")
for r in all_results
    @printf("%-10s %+.6f    %-4d  (%.4f, %.4f, %.4f, %.4f)\n",
            r.init_id, r.logJP_best, r.imax,
            r.v_best[1]/1e9, r.v_best[2]/1e9, r.v_best[3]/1e6, r.v_best[4]/1e9)
end
println("-"^72)
i_g = argmax(i -> all_results[i].logJP_best, 1:length(all_results))
g = all_results[i_g]
@printf("GLOBAL BEST (PCRB): init=%s  log_JP=%.6f  c=(%.4f, %.4f, %.4f MHz, %.4f)\n",
        g.init_id, g.logJP_best,
        g.v_best[1]/1e9, g.v_best[2]/1e9, g.v_best[3]/1e6, g.v_best[4]/1e9)
println("="^72); flush(stdout)

open(joinpath(outdir, "_summary.jls"), "w") do io
    serialize(io, (; all_results, i_global=i_g, timestamp=now()))
end
println("Saved $(joinpath(outdir, "_summary.jls"))")
