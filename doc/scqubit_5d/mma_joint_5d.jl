#=
mma_joint_5d.jl — 5-D MMA optimization of joint-DP V_adaptive(c, ω_d) at
K_PHI=256, with ω_d treated as a free 5th parameter (no longer derived
from c via single-shot Fisher logic).

Used as a validation/refinement step on top of the 5-D BayesOpt result:
initialize MMA at (or near) the BayesOpt winner and check whether MMA's
local descent confirms / improves the basin V.

If MMA converges to V comparable to BayesOpt's, we've established that the
basin is genuine and that BayesOpt found a true local minimum. If MMA
finds a deeper V with small displacement, we get a tighter polish on the
BayesOpt winner.

Inits supported via INIT_ID env var:
  bayesopt   - initialize at the saved BayesOpt 5-D winner
  custom     - read c0 from BAYESOPT_5D_C env (5 floats: f_q, E_C, κ, Δ, ω_d)

Output: results/mma_joint_5d/<INIT_ID>.jls
Wall-clock: ~10-25 min (≤ 60 MMA evals × ~30s each).
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

println("mma_joint_5d.jl"); flush(stdout)
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
const FTOL_REL   = parse(Float64, get(ENV, "FTOL_REL", "1e-6"))
const INIT_ID    = get(ENV, "INIT_ID", "bayesopt")

const TWO_PI = 2π
const LO5 = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9,   TWO_PI * 1.0e9 ]
const HI5 = [12.0e9,   0.4e9,   5.0e6,   5.0e9,   TWO_PI * 12.0e9 ]
const SCALE5 = HI5 .- LO5

# ---- Load init c+ω_d ----
function load_init()
    if INIT_ID == "bayesopt"
        path = joinpath(@__DIR__, "results", "bayesopt_joint_5d", "result.jls")
        isfile(path) || error("Need $path; run bayesopt_joint_5d.jl first.")
        j = deserialize(path)
        v = j.v_best          # 7-vec; we use [1..4] for hardware, ω_d separately
        x0 = [v[1], v[2], v[3], v[4], j.omega_d_best]
        @printf("INIT_ID=bayesopt  loaded V=%.4e  c=(%.4f, %.4f, %.4f MHz, %.4f)  ω_d/(2π)=%.4f GHz\n",
                j.V_best, x0[1]/1e9, x0[2]/1e9, x0[3]/1e6, x0[4]/1e9, x0[5]/TWO_PI/1e9)
        return x0
    elseif INIT_ID == "custom"
        raw = get(ENV, "BAYESOPT_5D_C", "")
        isempty(raw) && error("INIT_ID=custom requires BAYESOPT_5D_C env var (5 numbers)")
        x0 = parse.(Float64, split(raw, ","))
        length(x0) == 5 || error("BAYESOPT_5D_C must have 5 entries")
        @printf("INIT_ID=custom  c=(%.4f, %.4f, %.4f MHz, %.4f)  ω_d/(2π)=%.4f GHz\n",
                x0[1]/1e9, x0[2]/1e9, x0[3]/1e6, x0[4]/1e9, x0[5]/TWO_PI/1e9)
        return x0
    else
        error("unknown INIT_ID=$INIT_ID")
    end
end

x0 = load_init()
println(@sprintf("Config: K_PHI=%d  max_evals=%d  xtol_rel=%.0e  ftol_rel=%.0e",
                 K_PHI, MAX_EVALS, XTOL_REL, FTOL_REL))
flush(stdout)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)

const HIST = (V = Float64[], grad_norm = Float64[],
              c_vec = Vector{Vector{Float64}}(), omega_d = Float64[],
              elapsed = Float64[], z = Vector{Vector{Float64}}())

n_eval = Ref(0)
function objective(z, grad)
    n_eval[] += 1
    t0 = time()
    x = LO5 .+ SCALE5 .* z
    v = [x[1], x[2], x[3], x[4],
         PAPER_BASELINE.temperature, PAPER_BASELINE.A_phi, PAPER_BASELINE.A_Ic]
    c = vec_as_c(v)
    ω_d = x[5]
    (V, memo, _st) = solve_bellman_threaded_full(grid, K_EPOCHS, c, ω_d; terminal=:mse)
    if length(grad) > 0
        # Gradient w.r.t. c only (ω_d gradient via finite differences below)
        g4 = grad_c_exact_fd_threaded(v, memo, grid, ω_d, K_EPOCHS;
                                      terminal=:mse, parallel_depth=PARALLEL_DEPTH)
        # Gradient w.r.t. ω_d via central FD (cheap: 2 Bellman evals).  We use
        # a relative step so it scales with the magnitude of ω_d.
        h = max(abs(ω_d) * 1e-5, 1e3)
        (Vp, _, _) = solve_bellman_threaded_full(grid, K_EPOCHS, c, ω_d + h; terminal=:mse)
        (Vm, _, _) = solve_bellman_threaded_full(grid, K_EPOCHS, c, ω_d - h; terminal=:mse)
        gω = (Vp - Vm) / (2*h)
        # Box-scaled gradient in normalized [0,1]^5 space
        for i in 1:4
            grad[i] = SCALE5[i] * g4[i]
        end
        grad[5] = SCALE5[5] * gω
        push!(HIST.grad_norm, norm(grad))
    else
        push!(HIST.grad_norm, NaN)
    end
    push!(HIST.V, V); push!(HIST.c_vec, copy(v))
    push!(HIST.omega_d, ω_d); push!(HIST.z, copy(z))
    push!(HIST.elapsed, time() - t0)
    @printf("[mma5d %3d] V=%+.6e  |g_z|=%.3e  c=(%.4f, %.4f, %.4f MHz, %.4f)  ω_d/(2π)=%.4f GHz  %.1fs\n",
            n_eval[], V,
            isnan(HIST.grad_norm[end]) ? 0.0 : HIST.grad_norm[end],
            x[1]/1e9, x[2]/1e9, x[3]/1e6, x[4]/1e9, ω_d/TWO_PI/1e9,
            HIST.elapsed[end])
    flush(stdout)
    return V
end

# Convert physical x0 → normalized z0 in [0,1]^5
z0 = clamp.((x0 .- LO5) ./ SCALE5, 0.0, 1.0)
@printf("Init z0 = %s   (physical x0 in box: %s)\n", string(round.(z0; digits=4)),
        all(0 .≤ z0 .≤ 1) ? "yes" : "NO — clamped")
flush(stdout)

opt = NLopt.Opt(:LD_MMA, 5)
NLopt.lower_bounds!(opt, zeros(5))
NLopt.upper_bounds!(opt, ones(5))
NLopt.xtol_rel!(opt, XTOL_REL)
NLopt.ftol_rel!(opt, FTOL_REL)
NLopt.maxeval!(opt, MAX_EVALS)
NLopt.max_objective!(opt, objective)

println("\nLaunching 5-D MMA from $INIT_ID init..."); flush(stdout)
t_run0 = time()
(Vmax, zmax, ret) = NLopt.optimize(opt, z0)
elapsed = time() - t_run0
@printf("\n5-D MMA done: %s  (%.1f min, %d evals)\n",
        string(ret), elapsed/60, n_eval[]); flush(stdout)

ibest = argmax(HIST.V)
v_best = HIST.c_vec[ibest]
ω_best = HIST.omega_d[ibest]
V_best = HIST.V[ibest]
@printf("MMA best V = %.6e at eval %d\n", V_best, ibest)
@printf("  c_best:  f_q=%.4f GHz  E_C=%.4f GHz  κ=%.4f MHz  Δ=%.4f GHz\n",
        v_best[1]/1e9, v_best[2]/1e9, v_best[3]/1e6, v_best[4]/1e9)
@printf("  ω_d/(2π) = %.4f GHz   ω_d = %.4e rad/s\n", ω_best/TWO_PI/1e9, ω_best)

# Comparison to init
@printf("\nComparison to init:\n")
@printf("  V    init=%.6e   final=%.6e   improvement=%+.2f%%\n",
        HIST.V[1], V_best, 100*(V_best - HIST.V[1])/abs(HIST.V[1]))
@printf("  drift (best vs init):\n")
@printf("    f_q:   %+.4f GHz   E_C: %+.4f GHz   κ: %+.4f MHz   Δ: %+.4f GHz   ω_d/(2π): %+.4f GHz\n",
        (v_best[1]-x0[1])/1e9, (v_best[2]-x0[2])/1e9,
        (v_best[3]-x0[3])/1e6, (v_best[4]-x0[4])/1e9,
        (ω_best-x0[5])/TWO_PI/1e9)
flush(stdout)

outdir = joinpath(@__DIR__, "results", "mma_joint_5d")
isdir(outdir) || mkpath(outdir)
open(joinpath(outdir, "$(INIT_ID).jls"), "w") do io
    serialize(io, (; init_id=INIT_ID, x0,
                     hist=HIST, ibest, V_best, v_best, omega_d_best=ω_best,
                     ret, elapsed, n_eval=n_eval[],
                     K_EPOCHS, K_PHI, PHI_MAX, TAU_GRID, N_GRID,
                     LO5, HI5, max_evals=MAX_EVALS, xtol_rel=XTOL_REL, ftol_rel=FTOL_REL,
                     terminal=:mse, optimizer=:LD_MMA_5D, timestamp=now()))
end
println("Saved $(joinpath(outdir, "$(INIT_ID).jls"))")
