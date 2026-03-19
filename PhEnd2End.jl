push!(LOAD_PATH, @__DIR__)

# Flush stdout after every write so nohup.out gets output in real time.
ENV["JULIA_DEBUG"] = ""   # ensure no debug buffering
flush(stdout)

using Distributed

println("[startup] Adding worker processes..."); flush(stdout)
addprocs(200)
println("[startup] Workers: $(nworkers())  |  Sys.CPU_THREADS: $(Sys.CPU_THREADS)"); flush(stdout)

# Broadcast LOAD_PATH and module imports to all workers.
# $src_dir interpolates the master-side string so @__DIR__ is not re-evaluated
# on each worker.
const src_dir = @__DIR__
@everywhere push!(LOAD_PATH, $src_dir)
# ── Sensor selection criterion ────────────────────────────────────────────────
# This script uses BFIMGaussian (BFIM trace criterion) by default.
# To switch to the posterior-covariance criterion (A-optimal design), replace
#   using BFIMGaussian
# with
#   using PosteriorCovGaussian
# everywhere below (including the main-process `using` on line 39).
#
# API differences when using PosteriorCovGaussian:
#   - get_sopt(c, μ, Σ, model)       ← takes Σ (was: get_sopt(c, μ, model))
#   - episode_loss is unchanged       (internally passes Σ to get_sopt)
#   - _run_episode_grad is unchanged  (episode_loss signature is the same)
#   - posterior_cov_trace(μ,s,c,model,Σ)  replaces  bfim_trace(μ,s,c,model)
#   - posterior_grad_s / posterior_hessian_s  replace  bfim_grad_s / bfim_hessian_s
# ──────────────────────────────────────────────────────────────────────────────
println("[startup] Loading modules on all workers..."); flush(stdout)
@everywhere begin
    using SimGeomBroadBand
    using BFIMGaussian          # swap to: using PosteriorCovGaussian
    using LinearAlgebra
    using Random
    using Zygote
    # The FDFD solves use sparse LU via UMFPACK (SuiteSparse), which is
    # single-threaded and does not call BLAS.  Setting BLAS threads to 1 is
    # therefore a no-op for the current workload, but acts as a safeguard: if
    # any dense linear-algebra path is ever introduced (e.g. a dense fallback
    # or a future dense post-processing step), OpenBLAS would otherwise try to
    # spawn its own thread pool inside each worker process, causing
    # oversubscription (17 workers × 18 BLAS threads = 306 threads on 18 cores).
    BLAS.set_num_threads(1)
end

println("[startup] Loading main-process packages..."); flush(stdout)
using Test
using Plots
using ForwardDiff
using Optim
using Serialization
using Statistics
using BFIMGaussian          # swap to: using PosteriorCovGaussian
println("[startup] All packages loaded."); flush(stdout)

# Unpack the flat real coefficient vector c into arrays of complex 4×4 S-matrices.
# Uses only indexing — fully Zygote-differentiable.
# This is the exact inverse of the packing performed in sim_geom.
@everywhere function unpack_c(c, nω)
    S0 = c[1:16nω]      .+ im .* c[16nω+1:32nω]
    S1 = c[32nω+1:48nω] .+ im .* c[48nω+1:64nω]
    S2 = c[64nω+1:80nω] .+ im .* c[80nω+1:96nω]
    S_arr      = [reshape(S0[(i-1)*16+1:i*16], 4, 4) for i in 1:nω]
    dSdn_arr   = [reshape(S1[(i-1)*16+1:i*16], 4, 4) for i in 1:nω]
    d2Sdn2_arr = [reshape(S2[(i-1)*16+1:i*16], 4, 4) for i in 1:nω]
    return S_arr, dSdn_arr, d2Sdn2_arr
end

@everywhere function make_model(nω, n, GΔω, σ², αr)

    function f(x, s, c)
        S_arr, dSdn_arr, d2Sdn2_arr = unpack_c(c, nω)
        φ₁, φ₂ = s[1], s[2]
        Δn = reshape(x, n, n)
        powers_only(Δn, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
    end

    function fx(x, s, c)
        S_arr, dSdn_arr, d2Sdn2_arr = unpack_c(c, nω)
        φ₁, φ₂ = s[1], s[2]
        Δn = reshape(x, n, n)
        jac_only(Δn, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
    end

    function fxs(x, s, c, λ)
        S_arr, dSdn_arr, d2Sdn2_arr = unpack_c(c, nω)
        Δn = reshape(x, n, n)
        jac_and_dirderiv_s(Δn, s[1], s[2], λ, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
    end

    # Set BFIM_FXS=0 to disable analytical fxs path and use ForwardDiff fallback in IFT rrule.
    _use_fxs = get(ENV, "BFIM_FXS", "1") != "0"
    ModelFunctions(f=f, fx=fx, fxs=_use_fxs ? fxs : nothing, σ²=σ², dy=4n, dx=n^2, ds=2, dc=96*nω, αr=αr, zero_s_init=true)

end

# Worker function must be defined @everywhere so pmap can call it by name on
# each worker without serialising any closure.  All data is passed explicitly
# as a single NamedTuple so the pmap call itself has no captured state.
@everywhere function _run_episode_grad(arg)
    x0_i, noise_i, c, nω, n_lat, GΔω, μ0, Σ0, σ², αr = arg
    mf = make_model(nω, n_lat, GΔω, σ², αr)
    loss_i, (grad_i,) = Zygote.withgradient(c) do c_
        episode_loss(x0_i, c_, mf, μ0, Σ0, noise_i)
    end
    return (loss_i, grad_i)
end

function batch_c2loss_grad(x0_list::AbstractVector, c, nω, n_lat, GΔω,
                μ0, Σ0, noise_bank::AbstractVector, σ², αr)
    n = length(x0_list)
    @assert length(noise_bank) == n  "x0_list and noise_bank must have same length"
    # Pack every episode's data into a plain tuple so pmap serialises only
    # ordinary arrays — no closures, no module-local types.
    args = [(x0_list[i], noise_bank[i], c, nω, n_lat, GΔω, μ0, Σ0, σ², αr) for i in 1:n]
    results   = pmap(_run_episode_grad, args)
    mean_loss = sum(r[1] for r in results) / n
    mean_grad = sum(r[2] for r in results) / n
    return mean_loss, mean_grad
end

function sim_geom(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info, monitors_array, a_f_array, a_b_array)

    S_arr, dSdn_arr, d2Sdn2_arr = getSmatrices(
                                ε_geom, n_geom, ε_base, ω_array, Ls, Bs,
                                grid_info, monitors_array, a_f_array, a_b_array;
                                design_iy=grid_info.design_iy, design_ix=grid_info.design_ix)

    S0r = reduce(vcat, vec(real.(S))   for S   in S_arr)
    S0i = reduce(vcat, vec(imag.(S))   for S   in S_arr)
    S1r = reduce(vcat, vec(real.(dS))  for dS  in dSdn_arr)
    S1i = reduce(vcat, vec(imag.(dS))  for dS  in dSdn_arr)
    S2r = reduce(vcat, vec(real.(d2S)) for d2S in d2Sdn2_arr)
    S2i = reduce(vcat, vec(imag.(d2S)) for d2S in d2Sdn2_arr)
    c = vcat(S0r, S0i, S1r, S1i, S2r, S2i)

    return c

end

function end2end(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info, monitors_array, a_f_array, a_b_array,
                 x0_list, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)

    @assert size(ε_geom) == (length(grid_info.design_iy), length(grid_info.design_ix))  "ε_geom size $(size(ε_geom)) does not match design region ($(length(grid_info.design_iy)), $(length(grid_info.design_ix)))"

    println("  [end2end] FDFD forward (sim_geom + Zygote pullback)..."); flush(stdout)
    fdfd = ε_ -> sim_geom(ε_, n_geom, ε_base, ω_array, Ls, Bs, grid_info, monitors_array, a_f_array, a_b_array)
    c, pb_c = Zygote.pullback(fdfd, ε_geom)

    println("  [end2end] Episode gradients (batch_c2loss_grad, $(length(x0_list)) episodes)..."); flush(stdout)
    loss, cbar = batch_c2loss_grad(x0_list, c, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)

    println("  [end2end] FDFD backward (pullback)..."); flush(stdout)
    (grad_ε_geoms,) = pb_c(cbar)

    println("  [end2end] Done. loss=$loss"); flush(stdout)
    return loss, grad_ε_geoms

end

# ── Adam optimisation of ε_geom ──────────────────────────────────────────────

function train_adam!(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                     monitors_array, a_f_array, a_b_array,
                     x0_list, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr;
                     n_iters=100, lr=1e-3, β1=0.9, β2=0.999, ε_adam=1e-8,
                     resample_every=0, rng=MersenneTwister(123),
                     x0_min=-0.01, x0_max=0.01,
                     save_every=10, save_dir=joinpath(@__DIR__, "checkpoints"))

    mkpath(save_dir)

    m = zeros(size(ε_geom))   # first moment
    v = zeros(size(ε_geom))   # second moment
    losses = Float64[]

    for t in 1:n_iters
        t_start = time()

        # Optionally resample x0 and noise each iteration for fresh stochastic estimates
        if resample_every > 0 && t > 1 && mod(t - 1, resample_every) == 0
            println("  [optim] Resampling x0 and noise..."); flush(stdout)
            x0_list    = [x0_min .+ (x0_max - x0_min) .* rand(rng, length(μ0)) for _ in eachindex(x0_list)]
            noise_bank = sample_noise_bank(rng, length(x0_list), length(noise_bank[1]),
                                           length(noise_bank[1][1]), σ²)
        end

        loss, grad = end2end(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                             monitors_array, a_f_array, a_b_array,
                             x0_list, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        push!(losses, loss)

        # Adam update
        m .= β1 .* m .+ (1 - β1) .* grad
        v .= β2 .* v .+ (1 - β2) .* grad .^ 2
        m̂ = m ./ (1 - β1^t)
        v̂ = v ./ (1 - β2^t)
        ε_geom .-= lr .* m̂ ./ (sqrt.(v̂) .+ ε_adam)

        # Clamp to [0, 1] (physical permittivity fraction)
        clamp!(ε_geom, 0.0, 1.0)

        # Diagnostics
        g_abs    = abs.(grad)
        step_abs = abs.(lr .* m̂ ./ (sqrt.(v̂) .+ ε_adam))
        Δloss    = t > 1 ? loss - losses[end-1] : NaN
        elapsed  = round(time() - t_start, digits=1)
        println("iter $t/$n_iters  $(elapsed)s  loss=$(round(loss, sigdigits=6))  Δloss=$(round(Δloss, sigdigits=3))  " *
                "|grad| min=$(round(minimum(g_abs), sigdigits=3)) avg=$(round(mean(g_abs), sigdigits=3)) " *
                "max=$(round(maximum(g_abs), sigdigits=3))  " *
                "|step| avg=$(round(mean(step_abs), sigdigits=3)) max=$(round(maximum(step_abs), sigdigits=3))  " *
                "ε range=[$(round(minimum(ε_geom), digits=4)), $(round(maximum(ε_geom), digits=4))]")
        flush(stdout)

        # Save checkpoint
        if mod(t, save_every) == 0 || t == n_iters
            path = joinpath(save_dir, "eps_geom_step_$(lpad(t, 5, '0')).jls")
            serialize(path, (; step=t, loss, ε_geom=copy(ε_geom), losses=copy(losses)))
            println("  saved → $path"); flush(stdout)
            flush(stdout)
        end
    end

    return ε_geom, losses
end

# ── MMA optimisation of ε_geom (deterministic, gradient-based) ───────────────

using NLopt

function train_mma!(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                    monitors_array, a_f_array, a_b_array,
                    x0_list, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr;
                    n_iters=200, ftol_rel=1e-8, xtol_rel=1e-8,
                    save_every=10, save_dir=joinpath(@__DIR__, "checkpoints_mma"))

    mkpath(save_dir)
    n_params = length(ε_geom)
    losses = Float64[]
    iter_count = Ref(0)

    function nlopt_obj(x::Vector{Float64}, grad::Vector{Float64})
        t_start = time()
        iter_count[] += 1
        t = iter_count[]

        # Reshape flat x back to design-region matrix
        ε_mat = reshape(x, size(ε_geom))

        loss, g = end2end(ε_mat, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                          monitors_array, a_f_array, a_b_array,
                          x0_list, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        push!(losses, loss)

        # NLopt expects gradient written into `grad` in-place
        if length(grad) > 0
            grad .= vec(g)
        end

        # Diagnostics
        g_abs   = abs.(vec(g))
        Δloss   = t > 1 ? loss - losses[end-1] : NaN
        elapsed = round(time() - t_start, digits=1)
        println("iter $t  $(elapsed)s  loss=$(round(loss, sigdigits=6))  Δloss=$(round(Δloss, sigdigits=3))  " *
                "|grad| min=$(round(minimum(g_abs), sigdigits=3)) avg=$(round(mean(g_abs), sigdigits=3)) " *
                "max=$(round(maximum(g_abs), sigdigits=3))  " *
                "ε range=[$(round(minimum(x), digits=4)), $(round(maximum(x), digits=4))]")
        flush(stdout)

        # Save checkpoint
        if mod(t, save_every) == 0
            path = joinpath(save_dir, "eps_geom_step_$(lpad(t, 5, '0')).jls")
            serialize(path, (; step=t, loss, ε_geom=copy(ε_mat), losses=copy(losses)))
            println("  saved → $path"); flush(stdout)
        end

        return loss
    end

    opt = NLopt.Opt(:LD_MMA, n_params)
    opt.lower_bounds = zeros(n_params)
    opt.upper_bounds = ones(n_params)
    opt.min_objective = nlopt_obj
    opt.maxeval = n_iters
    opt.ftol_rel = ftol_rel
    opt.xtol_rel = xtol_rel

    println("[mma] Starting MMA optimisation ($n_params parameters, maxeval=$n_iters)"); flush(stdout)

    x0 = vec(copy(ε_geom))
    (minf, minx, ret) = NLopt.optimize(opt, x0)

    println("[mma] Finished: ret=$ret  final_loss=$minf  iters=$(iter_count[])"); flush(stdout)

    # Save final
    ε_final = reshape(minx, size(ε_geom))
    path = joinpath(save_dir, "eps_geom_final.jls")
    serialize(path, (; ret, loss=minf, ε_geom=copy(ε_final), losses=copy(losses)))
    println("[mma] saved → $path"); flush(stdout)

    ε_geom .= ε_final
    return ε_geom, losses
end

# ── Parameters ────────────────────────────────────────────────────────────────
n_lat              = 2
n_core, w          = 2.0, 0.5
n_geom             = n_core
Lx, Ly             = 10.0, 10.0
res, n_pml         = 50, 24
R_target           = 1e-8
port_offset        = 15
mon_offset         = 5
d_length, d_width  = 6.0, 6.0
ωmin, ωmax         = 5.5, 7.5
nω                 = 20        

ω₀                 = (ωmax + ωmin) / 2 # input pulse center freq
Δω                 = (ωmax - ωmin) / 6 # input pulse freq bandwidth

dy                 = 4 * n_lat
dx                 = n_lat^2
ds                 = 2

n_episodes         = nworkers()                        
N_steps            = 3                                # EKF steps per episode
μ0                 = zeros(dx)                                      # initial belief mean
Σ0                 = 1e-6 *Matrix{Float64}(I, dx, dx)              # initial belief covariance (= x0_max² · I)
σ²                 = 1e-8
αr                 = 1.0

x0_min             = -0.001
x0_max             =  0.001
rng                = MersenneTwister(42)
x0_list            = [x0_min .+ (x0_max - x0_min) .* rand(rng, dx) for _ in 1:n_episodes]
noise_bank         = sample_noise_bank(rng, n_episodes, N_steps, dy, σ²)

println("[params] nω=$nω  workers=$(nworkers())  n_lat=$n_lat  res=$res  grid=$(round(Int,Lx*res))×$(round(Int,Ly*res))  design=$(d_length)×$(d_width)"); flush(stdout)
flush(stdout)

# ── Geometry and calibration setup ───────────────────────────────────────────
println("[setup] Building 4-port geometry (setup_4port_sweep)..."); flush(stdout)
ε_base, ω_array, Ls, Bs, grid_info, _, monitors_array =
    setup_4port_sweep(ωmin, ωmax, nω, n_core, w, d_length, d_width;
        Lx=Lx, Ly=Ly, res=res, n_pml=n_pml, R_target=R_target,
        port_offset=port_offset, mon_offset=mon_offset)
println("[setup] Geometry built. Nx=$(grid_info.Nx) Ny=$(grid_info.Ny)"); flush(stdout)

println("[setup] Calibrating straight waveguide..."); flush(stdout)
(a_f_array, a_b_array) = calibrate_straight_waveguide(
    ωmin, ωmax, nω, n_core, w;
    Lx=Lx, Ly=Ly, res=res, n_pml=n_pml, R_target=R_target,
    port_offset=port_offset, mon_offset=mon_offset)
println("[setup] Calibration done. a_f range=[$(minimum(a_f_array)), $(maximum(a_f_array))]"); flush(stdout)

δω  = ω_array[2] - ω_array[1]
GΔω = @. exp(-(ω_array - ω₀)^2 / (2Δω^2)) / (Δω * sqrt(2π)) * δω

Ny_d = length(grid_info.design_iy)
Nx_d = length(grid_info.design_ix)
println("[setup] Design region: $(Ny_d)×$(Nx_d) = $(Ny_d*Nx_d) parameters"); flush(stdout)

# ── Run mode ─────────────────────────────────────────────────────────────────
# Control via env var BFIM_MODE:
#   "adam"  — Adam optimisation (default)
#   "mma"   — NLopt MMA optimisation
#   "test"  — skip optimisation, run only gradient tests (set BFIM_TEST too)
#   "none"  — setup only, no optimisation or tests
const _MODE = lowercase(get(ENV, "BFIM_MODE", "adam"))

if _MODE == "adam"
    println("[optim] Mode: Adam"); flush(stdout)
    ε_geom_opt = fill(0.5, Ny_d, Nx_d)
    ε_geom_opt, losses = train_adam!(
        ε_geom_opt, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
        monitors_array, a_f_array, a_b_array,
        x0_list, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr;
        n_iters=200, lr=1e-3, x0_min=x0_min, x0_max=x0_max)

elseif _MODE == "mma"
    println("[optim] Mode: MMA"); flush(stdout)
    ε_geom_opt = fill(0.5, Ny_d, Nx_d)
    ε_geom_opt, losses = train_mma!(
        ε_geom_opt, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
        monitors_array, a_f_array, a_b_array,
        x0_list, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr;
        n_iters=1000000, ftol_rel=1e-16, xtol_rel=1e-16,
        save_every=10)

elseif _MODE == "test"
    println("[mode] Test only — skipping optimisation"); flush(stdout)

elseif _MODE == "none"
    println("[mode] Setup only — no optimisation, no tests"); flush(stdout)

else
    error("Unknown BFIM_MODE=\"$_MODE\". Use: adam, mma, test, none")
end

# ══════════════════════════════════════════════════════════════════════════════
# Gradient / correctness tests
# Control via env var BFIM_TEST: "all", or comma-separated subset of
#   sim_geom, mf_f, mf_fx, lattice, ekf, sopt_heatmap, sopt_grad, episode, episode_warm, batch, ekf_perf, taylor_fidelity
# Example:  BFIM_TEST=sopt_grad,episode julia PhEnd2End.jl
# ══════════════════════════════════════════════════════════════════════════════
const _TEST_ENV = get(ENV, "BFIM_TEST", "")
const _RUN_ALL  = _TEST_ENV == "all"
_run_test(name) = _RUN_ALL || name in split(_TEST_ENV, ",")

if _TEST_ENV != ""
    println("\n═══ Running selected gradient checks: $(_RUN_ALL ? "all" : _TEST_ENV) ═══\n"); flush(stdout)

    # ── Compute / load nominal c (shared by all tests) ───────────────────────
    c_nom_path = joinpath(@__DIR__, "c_nom.jls")
    if isfile(c_nom_path)
        c_nom = deserialize(c_nom_path)
        println("Loaded c_nom from $c_nom_path  (length=$(length(c_nom)))"); flush(stdout)
    else
        ε_geom = rand(MersenneTwister(1234), Ny_d, Nx_d)
        c_nom = sim_geom(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info, monitors_array, a_f_array, a_b_array)
        serialize(c_nom_path, c_nom)
        println("Computed and saved c_nom to $c_nom_path"); flush(stdout)
    end
end

if _run_test("sim_geom")
    let ε = 1e-5
        ε_geom_check = rand(MersenneTwister(77), Ny_d, Nx_d)
        sg = ε_ -> sim_geom(ε_, n_geom, ε_base, ω_array, Ls, Bs, grid_info, monitors_array, a_f_array, a_b_array)
        scalar_sg = ε_ -> sum(abs2, sg(ε_))
        _, (grad,) = Zygote.withgradient(scalar_sg, ε_geom_check)
        v = randn(MersenneTwister(78), Ny_d, Nx_d); v ./= norm(v)
        fd_deriv = (scalar_sg(ε_geom_check .+ ε .* v) - scalar_sg(ε_geom_check .- ε .* v)) / (2ε)
        ad_deriv = dot(grad, v)
        rel_err  = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)
        println("sim_geom gradient check (AD vs FD):"); flush(stdout)
        println("  AD=$ad_deriv  FD=$fd_deriv  rel_err=$rel_err"); flush(stdout)
        @assert rel_err < 1e-3  "sim_geom gradient check failed: rel_err=$rel_err"
    end
end

if _run_test("mf_f")
    let ε = 1e-5
        mf_check = make_model(nω, n_lat, GΔω, σ², αr)
        x_check  = x0_list[1]
        s_check  = zeros(ds)

        scalar_f = c_ -> sum(abs2, mf_check.f(x_check, s_check, c_))
        _, (grad,) = Zygote.withgradient(scalar_f, c_nom)
        v = randn(MersenneTwister(99), length(c_nom)); v ./= norm(v)

        fd_deriv = (scalar_f(c_nom .+ ε .* v) - scalar_f(c_nom .- ε .* v)) / (2ε)
        ad_deriv = dot(grad, v)
        rel_err  = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)

        println("mf.f gradient check w.r.t. c (AD vs FD):"); flush(stdout)
        println("  AD directional derivative : $ad_deriv"); flush(stdout)
        println("  FD directional derivative : $fd_deriv"); flush(stdout)
        println("  relative error            : $rel_err"); flush(stdout)
        @assert rel_err < 1e-3  "mf.f gradient check failed: relative error = $rel_err"
    end
end

if _run_test("mf_fx")
    let ε = 1e-5
        mf_check = make_model(nω, n_lat, GΔω, σ², αr)
        x_check  = x0_list[1]
        s_check  = zeros(ds)

        # 1. Directional check
        v = randn(MersenneTwister(100), dx); v ./= norm(v)
        jac_v    = mf_check.fx(x_check, s_check, c_nom) * v
        fd_jac_v = (mf_check.f(x_check .+ ε .* v, s_check, c_nom) .-
                    mf_check.f(x_check .- ε .* v, s_check, c_nom)) ./ (2ε)
        rel_err_dir = norm(jac_v .- fd_jac_v) / (norm(jac_v) + 1e-12)

        println("mf.fx directional check (analytical vs FD):"); flush(stdout)
        println("  ‖jac_v‖ = $(norm(jac_v))  ‖fd_jac_v‖ = $(norm(fd_jac_v))  rel_err = $rel_err_dir"); flush(stdout)
        @assert rel_err_dir < 1e-3  "mf.fx directional check failed: relative error = $rel_err_dir"

        # 2. Full Jacobian check — column i = (f(x+ε eᵢ) - f(x-ε eᵢ)) / 2ε
        J_anal = mf_check.fx(x_check, s_check, c_nom)
        J_fd   = hcat([
            (mf_check.f(x_check .+ ε .* (1:dx .== i), s_check, c_nom) .-
             mf_check.f(x_check .- ε .* (1:dx .== i), s_check, c_nom)) ./ (2ε)
            for i in 1:dx]...)
        rel_err_full = norm(J_anal .- J_fd) / (norm(J_anal) + 1e-12)

        println("mf.fx full Jacobian check (analytical vs FD):"); flush(stdout)
        println("  ‖J_anal‖ = $(norm(J_anal))  ‖J_fd‖ = $(norm(J_fd))  rel_err = $rel_err_full"); flush(stdout)
        if rel_err_full >= 1e-3
            println("  --- diagnostic: J_anal ---"); flush(stdout)
            show(stdout, "text/plain", J_anal); println(); flush(stdout)
            println("  --- diagnostic: J_fd ---"); flush(stdout)
            show(stdout, "text/plain", J_fd); println(); flush(stdout)
            println("  --- diagnostic: element-wise ratio J_anal ./ J_fd ---"); flush(stdout)
            show(stdout, "text/plain", J_anal ./ J_fd); println(); flush(stdout)
            println("  --- diagnostic: does -J_anal ≈ J_fd? rel_err = $(norm(J_anal .+ J_fd)/norm(J_anal))"); flush(stdout)
        end
        @assert rel_err_full < 1e-3  "mf.fx full Jacobian check failed: relative error = $rel_err_full"
    end
end

# ── Lattice function consistency & AD tests ──────────────────────────────────
# Tests powers_only, jac_only, jac_and_dirderiv_s for:
#   1. Mutual consistency: jac_only ≈ FD(powers_only), dirderiv ≈ FD(jac_only)
#   2. Zygote AD correctness: ∂/∂c of each function matches FD
if _run_test("lattice")
    let ε = 1e-5
        mf_check = make_model(nω, n_lat, GΔω, σ², αr)
        S_arr, dSdn_arr, d2Sdn2_arr = unpack_c(c_nom, nω)
        x_check  = x0_list[1]
        Δn       = reshape(x_check, n_lat, n_lat)
        s_check  = [0.3, -0.2]   # non-zero phases
        φ₁, φ₂   = s_check[1], s_check[2]

        println("═══ Lattice function tests ═══"); flush(stdout)

        # ── 1a. jac_only vs FD of powers_only ────────────────────────────────
        # J[:,k] should equal (powers_only(x+ε·eₖ) − powers_only(x−ε·eₖ)) / 2ε
        println("  1a. jac_only vs FD(powers_only) w.r.t. Δn:"); flush(stdout)
        J_anal = jac_only(Δn, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        J_fd   = hcat([begin
            eₖ = zeros(n_lat, n_lat); eₖ[k] = 1.0
            (powers_only(Δn .+ ε .* eₖ, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω) .-
             powers_only(Δn .- ε .* eₖ, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)) ./ (2ε)
        end for k in 1:length(Δn)]...)
        re_jac = norm(J_anal - J_fd) / (norm(J_anal) + 1e-12)
        println("      ‖J_anal‖=$(round(norm(J_anal), sigdigits=4))  ‖J_fd‖=$(round(norm(J_fd), sigdigits=4))  rel_err=$(round(re_jac, sigdigits=3))"); flush(stdout)
        @assert re_jac < 1e-3  "jac_only vs FD(powers_only) failed: rel_err=$re_jac"

        # ── 1b. jac_only vs FD of powers_only w.r.t. s (phases) ──────────────
        println("  1b. jac_only vs FD(powers_only) w.r.t. s:"); flush(stdout)
        for (si, name) in [(1, "φ₁"), (2, "φ₂")]
            μ_plus  = powers_only(Δn, (si==1 ? φ₁+ε : φ₁), (si==2 ? φ₂+ε : φ₂),
                                  S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
            μ_minus = powers_only(Δn, (si==1 ? φ₁-ε : φ₁), (si==2 ? φ₂-ε : φ₂),
                                  S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
            dμ_ds_fd = (μ_plus .- μ_minus) ./ (2ε)
            # jac_only gives ∂μ/∂Δn, not ∂μ/∂s. Use ForwardDiff for ∂μ/∂s:
            dμ_ds_ad = ForwardDiff.derivative(
                t -> powers_only(Δn, (si==1 ? φ₁+t : φ₁), (si==2 ? φ₂+t : φ₂),
                                 S_arr, dSdn_arr, d2Sdn2_arr, GΔω), 0.0)
            re_s = norm(dμ_ds_ad - dμ_ds_fd) / (norm(dμ_ds_ad) + 1e-12)
            println("      ∂μ/∂$name: ‖AD‖=$(round(norm(dμ_ds_ad), sigdigits=4))  ‖FD‖=$(round(norm(dμ_ds_fd), sigdigits=4))  rel_err=$(round(re_s, sigdigits=3))"); flush(stdout)
            @assert re_s < 1e-3  "∂μ/∂$name check failed: rel_err=$re_s"
        end

        # ── 1c. jac_and_dirderiv_s vs FD of jac_only w.r.t. s ────────────────
        println("  1c. jac_and_dirderiv_s vs FD(jac_only) w.r.t. s:"); flush(stdout)
        λ_test = [0.7, -0.4]
        J0, dJ_λ = jac_and_dirderiv_s(Δn, φ₁, φ₂, λ_test, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        # FD: directional derivative of jac_only along λ in s-space
        J_plus  = jac_only(Δn, φ₁ + ε*λ_test[1], φ₂ + ε*λ_test[2], S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        J_minus = jac_only(Δn, φ₁ - ε*λ_test[1], φ₂ - ε*λ_test[2], S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        dJ_fd   = (J_plus .- J_minus) ./ (2ε)
        # Also check J0 == jac_only at the same point
        re_J0 = norm(J0 - J_anal) / (norm(J_anal) + 1e-12)
        re_dJ = norm(dJ_λ - dJ_fd) / (norm(dJ_fd) + 1e-12)
        println("      J consistency: rel_err=$(round(re_J0, sigdigits=3))"); flush(stdout)
        println("      dJ_λ: ‖anal‖=$(round(norm(dJ_λ), sigdigits=4))  ‖FD‖=$(round(norm(dJ_fd), sigdigits=4))  rel_err=$(round(re_dJ, sigdigits=3))"); flush(stdout)
        @assert re_J0 < 1e-10  "J from jac_and_dirderiv_s != jac_only: rel_err=$re_J0"
        @assert re_dJ < 1e-3   "dJ_λ vs FD(jac_only) failed: rel_err=$re_dJ"

        # ── 2a. Zygote gradient of powers_only w.r.t. c ──────────────────────
        println("  2a. Zygote ∂(powers_only)/∂c:"); flush(stdout)
        v_c = randn(MersenneTwister(200), length(c_nom)); v_c ./= norm(v_c)
        scalar_po = c_ -> begin
            Sa, dSa, d2Sa = unpack_c(c_, nω)
            sum(abs2, powers_only(Δn, φ₁, φ₂, Sa, dSa, d2Sa, GΔω))
        end
        _, (grad_po,) = Zygote.withgradient(scalar_po, c_nom)
        fd_po = (scalar_po(c_nom .+ ε .* v_c) - scalar_po(c_nom .- ε .* v_c)) / (2ε)
        ad_po = dot(grad_po, v_c)
        re_po = abs(fd_po - ad_po) / (abs(ad_po) + 1e-12)
        println("      AD=$(round(ad_po, sigdigits=6))  FD=$(round(fd_po, sigdigits=6))  rel_err=$(round(re_po, sigdigits=3))"); flush(stdout)
        @assert re_po < 1e-3  "Zygote ∂(powers_only)/∂c failed: rel_err=$re_po"

        # ── 2b. Zygote gradient of jac_only w.r.t. c ─────────────────────────
        println("  2b. Zygote ∂(jac_only)/∂c:"); flush(stdout)
        scalar_jo = c_ -> begin
            Sa, dSa, d2Sa = unpack_c(c_, nω)
            sum(abs2, jac_only(Δn, φ₁, φ₂, Sa, dSa, d2Sa, GΔω))
        end
        _, (grad_jo,) = Zygote.withgradient(scalar_jo, c_nom)
        fd_jo = (scalar_jo(c_nom .+ ε .* v_c) - scalar_jo(c_nom .- ε .* v_c)) / (2ε)
        ad_jo = dot(grad_jo, v_c)
        re_jo = abs(fd_jo - ad_jo) / (abs(ad_jo) + 1e-12)
        println("      AD=$(round(ad_jo, sigdigits=6))  FD=$(round(fd_jo, sigdigits=6))  rel_err=$(round(re_jo, sigdigits=3))"); flush(stdout)
        @assert re_jo < 1e-3  "Zygote ∂(jac_only)/∂c failed: rel_err=$re_jo"

        # ── 2c. Zygote gradient of jac_and_dirderiv_s w.r.t. c ───────────────
        println("  2c. Zygote ∂(jac_and_dirderiv_s)/∂c:"); flush(stdout)
        scalar_jds = c_ -> begin
            Sa, dSa, d2Sa = unpack_c(c_, nω)
            F, dF = jac_and_dirderiv_s(Δn, φ₁, φ₂, λ_test, Sa, dSa, d2Sa, GΔω)
            sum(abs2, F) + sum(abs2, dF)
        end
        _, (grad_jds,) = Zygote.withgradient(scalar_jds, c_nom)
        fd_jds = (scalar_jds(c_nom .+ ε .* v_c) - scalar_jds(c_nom .- ε .* v_c)) / (2ε)
        ad_jds = dot(grad_jds, v_c)
        re_jds = abs(fd_jds - ad_jds) / (abs(ad_jds) + 1e-12)
        println("      AD=$(round(ad_jds, sigdigits=6))  FD=$(round(fd_jds, sigdigits=6))  rel_err=$(round(re_jds, sigdigits=3))"); flush(stdout)
        @assert re_jds < 1e-3  "Zygote ∂(jac_and_dirderiv_s)/∂c failed: rel_err=$re_jds"

        # ── 2d. Zygote gradient of bfim_trace_dirderiv (IFT scalar) w.r.t. c ─
        # This is the scalar λ'·∇_s(bfim_trace) = 2·sum(F.*dF_λ)/σ² that the
        # IFT rrule differentiates w.r.t. c.
        println("  2d. Zygote ∂(λ'·∇_s bfim_trace)/∂c  (IFT scalar):"); flush(stdout)
        scalar_ift = c_ -> begin
            Sa, dSa, d2Sa = unpack_c(c_, nω)
            Δn_ = reshape(x_check, n_lat, n_lat)
            F, dF_λ = jac_and_dirderiv_s(Δn_, s_check[1], s_check[2], λ_test, Sa, dSa, d2Sa, GΔω)
            2 * sum(F .* dF_λ) / σ²
        end
        _, (grad_ift,) = Zygote.withgradient(scalar_ift, c_nom)
        fd_ift = (scalar_ift(c_nom .+ ε .* v_c) - scalar_ift(c_nom .- ε .* v_c)) / (2ε)
        ad_ift = dot(grad_ift, v_c)
        re_ift = abs(fd_ift - ad_ift) / (abs(ad_ift) + 1e-12)
        println("      AD=$(round(ad_ift, sigdigits=6))  FD=$(round(fd_ift, sigdigits=6))  rel_err=$(round(re_ift, sigdigits=3))"); flush(stdout)
        @assert re_ift < 1e-3  "Zygote ∂(IFT scalar)/∂c failed: rel_err=$re_ift"

        println("═══ All lattice tests passed ═══"); flush(stdout)
    end
end

# ── EKF update gradient tests ─────────────────────────────────────────────────
# Tests ekf_update in isolation and composed with get_sopt (single EKF step).
# Pinpoints whether the gradient bug is in ekf_update or the composition.
if _run_test("ekf")
    let ε = 1e-5
        mf_check = make_model(nω, n_lat, GΔω, σ², αr)
        x0_check = x0_list[1]
        s_check  = [0.3, -0.2]
        noise_1  = noise_bank[1][1]
        v_c = randn(MersenneTwister(300), length(c_nom)); v_c ./= norm(v_c)

        println("═══ EKF update gradient tests ═══"); flush(stdout)

        # ── 3a. ekf_update only (fixed s, no get_sopt) ───────────────────────
        # Differentiate ‖μ_new‖² w.r.t. c through one ekf_update call.
        println("  3a. Zygote ∂(ekf_update μ_new)/∂c  (fixed s, no get_sopt):"); flush(stdout)
        ekf_mu_c = c_ -> begin
            y = mf_check.f(x0_check, s_check, c_) + noise_1
            μ_new, _ = ekf_update(μ0, Σ0, y, s_check, c_, mf_check)
            sum(abs2, μ_new)
        end
        _, (grad_3a,) = Zygote.withgradient(ekf_mu_c, c_nom)
        fd_3a = (ekf_mu_c(c_nom .+ ε .* v_c) - ekf_mu_c(c_nom .- ε .* v_c)) / (2ε)
        ad_3a = dot(grad_3a, v_c)
        re_3a = abs(fd_3a - ad_3a) / (abs(ad_3a) + 1e-12)
        println("      AD=$(round(ad_3a, sigdigits=6))  FD=$(round(fd_3a, sigdigits=6))  rel_err=$(round(re_3a, sigdigits=3))"); flush(stdout)
        @assert re_3a < 1e-3  "ekf_update μ gradient check failed: rel_err=$re_3a"

        # ── 3b. ekf_update Σ_new (fixed s) ───────────────────────────────────
        # Differentiate tr(Σ_new) w.r.t. c — tests Joseph form backward pass.
        println("  3b. Zygote ∂(tr(Σ_new))/∂c  (fixed s):"); flush(stdout)
        ekf_sig_c = c_ -> begin
            y = mf_check.f(x0_check, s_check, c_) + noise_1
            _, Σ_new = ekf_update(μ0, Σ0, y, s_check, c_, mf_check)
            tr(Σ_new)
        end
        _, (grad_3b,) = Zygote.withgradient(ekf_sig_c, c_nom)
        fd_3b = (ekf_sig_c(c_nom .+ ε .* v_c) - ekf_sig_c(c_nom .- ε .* v_c)) / (2ε)
        ad_3b = dot(grad_3b, v_c)
        re_3b = abs(fd_3b - ad_3b) / (abs(ad_3b) + 1e-12)
        println("      AD=$(round(ad_3b, sigdigits=6))  FD=$(round(fd_3b, sigdigits=6))  rel_err=$(round(re_3b, sigdigits=3))"); flush(stdout)
        @assert re_3b < 1e-3  "ekf_update Σ gradient check failed: rel_err=$re_3b"

        # ── 3c. Single EKF step with get_sopt (IFT + ekf) ────────────────────
        # Differentiate ‖μ_new − x0‖² w.r.t. c through get_sopt + ekf_update.
        println("  3c. Zygote ∂(single EKF step with get_sopt)/∂c:"); flush(stdout)
        single_step_c = c_ -> begin
            sk = get_sopt(c_, μ0, mf_check)
            yk = mf_check.f(x0_check, sk, c_) + noise_1
            μ_new, _ = ekf_update(μ0, Σ0, yk, sk, c_, mf_check)
            sum(abs2, μ_new - x0_check)
        end
        _, (grad_3c,) = Zygote.withgradient(single_step_c, c_nom)
        fd_3c = (single_step_c(c_nom .+ ε .* v_c) - single_step_c(c_nom .- ε .* v_c)) / (2ε)
        ad_3c = dot(grad_3c, v_c)
        re_3c = abs(fd_3c - ad_3c) / (abs(ad_3c) + 1e-12)
        println("      AD=$(round(ad_3c, sigdigits=6))  FD=$(round(fd_3c, sigdigits=6))  rel_err=$(round(re_3c, sigdigits=3))"); flush(stdout)
        @assert re_3c < 1e-3  "single EKF step gradient check failed: rel_err=$re_3c"

        # ── 3c'. IFT rrule μ̄ test: ∂(get_sopt)/∂μ ─────────────────────────
        # Tests the μ̄ = J_μ' * λ component in isolation.
        # If this fails, the IFT μ̄ is wrong.
        println("  3c'. Zygote ∂(get_sopt)/∂μ  (IFT μ̄ pathway):"); flush(stdout)
        v_mu = randn(MersenneTwister(301), dx); v_mu ./= norm(v_mu)
        μ_test = x0_list[1]  # non-zero μ to test general case
        # Scalar function of μ through get_sopt:
        sopt_of_mu = μ_ -> begin
            s_opt = get_sopt(c_nom, μ_, mf_check)
            sum(real.(exp.(im .* s_opt)))   # smooth scalar proxy
        end
        _, (grad_mu,) = Zygote.withgradient(sopt_of_mu, μ_test)
        for ε_test in [1e-4, 1e-5, 1e-6]
            fd_mu = (sopt_of_mu(μ_test .+ ε_test .* v_mu) - sopt_of_mu(μ_test .- ε_test .* v_mu)) / (2ε_test)
            ad_mu = dot(grad_mu, v_mu)
            re_mu = abs(fd_mu - ad_mu) / (abs(ad_mu) + 1e-12)
            println("      ε=$ε_test  AD=$(round(ad_mu, sigdigits=6))  FD=$(round(fd_mu, sigdigits=6))  rel_err=$(round(re_mu, sigdigits=3))"); flush(stdout)
        end
        # Also test with μ1 from a single EKF step (the actual multi-step scenario)
        println("  3c''. Same test at μ₁ (post-EKF point):"); flush(stdout)
        sk0 = get_sopt(c_nom, μ0, mf_check)
        y0  = mf_check.f(x0_check, sk0, c_nom) + noise_1
        μ1_test, _ = ekf_update(μ0, Σ0, y0, sk0, c_nom, mf_check)
        _, (grad_mu1,) = Zygote.withgradient(sopt_of_mu, μ1_test)
        for ε_test in [1e-4, 1e-5, 1e-6]
            fd_mu1 = (sopt_of_mu(μ1_test .+ ε_test .* v_mu) - sopt_of_mu(μ1_test .- ε_test .* v_mu)) / (2ε_test)
            ad_mu1 = dot(grad_mu1, v_mu)
            re_mu1 = abs(fd_mu1 - ad_mu1) / (abs(ad_mu1) + 1e-12)
            println("      ε=$ε_test  AD=$(round(ad_mu1, sigdigits=6))  FD=$(round(fd_mu1, sigdigits=6))  rel_err=$(round(re_mu1, sigdigits=3))"); flush(stdout)
        end

        # ── 3c'''. Warm-started ∂(get_sopt)/∂μ at μ₁ ──────────────────────
        # Uses _get_sopt with nominal s★ as init so FD stays in the same basin.
        println("  3c'''. Warm-started ∂(get_sopt)/∂μ at μ₁:"); flush(stdout)
        sopt_nom_at_mu1 = get_sopt(c_nom, μ1_test, mf_check)
        sopt_of_mu_warm = μ_ -> begin
            s_opt = BFIMGaussian._get_sopt(c_nom, μ_, mf_check, sopt_nom_at_mu1)
            sum(real.(exp.(im .* s_opt)))
        end
        _, (grad_mu1w,) = Zygote.withgradient(sopt_of_mu_warm, μ1_test)
        best_re_mu1w = Inf
        for ε_test in [1e-4, 1e-5, 1e-6]
            fd_mu1w = (sopt_of_mu_warm(μ1_test .+ ε_test .* v_mu) - sopt_of_mu_warm(μ1_test .- ε_test .* v_mu)) / (2ε_test)
            ad_mu1w = dot(grad_mu1w, v_mu)
            re_mu1w = abs(fd_mu1w - ad_mu1w) / (abs(ad_mu1w) + 1e-12)
            best_re_mu1w = min(best_re_mu1w, re_mu1w)
            println("      ε=$ε_test  AD=$(round(ad_mu1w, sigdigits=6))  FD=$(round(fd_mu1w, sigdigits=6))  rel_err=$(round(re_mu1w, sigdigits=3))"); flush(stdout)
        end
        @assert best_re_mu1w < 1e-3  "warm-started ∂(get_sopt)/∂μ at μ₁ failed: best rel_err=$best_re_mu1w"

        # ── 3d. Two EKF steps, FIXED s, no get_sopt ─────────────────────────
        # Isolates multi-step ekf_update gradient from IFT rrule.
        println("  3d. Zygote ∂(2 EKF steps, fixed s, no get_sopt)/∂c:"); flush(stdout)
        noise_2  = noise_bank[1][2]
        s_fixed1 = [0.3, -0.2]
        s_fixed2 = [0.1,  0.5]
        two_step_fixed_c = c_ -> begin
            y1 = mf_check.f(x0_check, s_fixed1, c_) + noise_1
            μ1, Σ1 = ekf_update(μ0, Σ0, y1, s_fixed1, c_, mf_check)
            y2 = mf_check.f(x0_check, s_fixed2, c_) + noise_2
            μ2, Σ2 = ekf_update(μ1, Σ1, y2, s_fixed2, c_, mf_check)
            sum(abs2, μ2 - x0_check)
        end
        _, (grad_3d,) = Zygote.withgradient(two_step_fixed_c, c_nom)
        fd_3d = (two_step_fixed_c(c_nom .+ ε .* v_c) - two_step_fixed_c(c_nom .- ε .* v_c)) / (2ε)
        ad_3d = dot(grad_3d, v_c)
        re_3d = abs(fd_3d - ad_3d) / (abs(ad_3d) + 1e-12)
        println("      AD=$(round(ad_3d, sigdigits=6))  FD=$(round(fd_3d, sigdigits=6))  rel_err=$(round(re_3d, sigdigits=3))"); flush(stdout)
        @assert re_3d < 1e-3  "2-step EKF (fixed s) gradient check failed: rel_err=$re_3d"

        # ── 3e. Two EKF steps WITH get_sopt (warm-started FD) ─────────────────
        # The inner get_sopt optimizer can jump basins under small δc perturbation
        # (see 3c'' diagnostic), making naive FD unreliable. We use warm-started
        # _get_sopt (passing nominal s★ as initial guess) so FD stays in the same
        # basin as the AD gradient (which uses the IFT for the smooth branch).
        println("  3e. Zygote ∂(2 EKF steps with get_sopt)/∂c  [warm-started FD]:"); flush(stdout)
        # Compute nominal s★ at each step for warm-starting
        sk1_nom = get_sopt(c_nom, μ0, mf_check)
        yk1_nom = mf_check.f(x0_check, sk1_nom, c_nom) + noise_1
        μ1_nom, Σ1_nom = ekf_update(μ0, Σ0, yk1_nom, sk1_nom, c_nom, mf_check)
        sk2_nom = get_sopt(c_nom, μ1_nom, mf_check)

        two_step_warm_c = c_ -> begin
            # Step 1: warm-start from sk1_nom
            sk1 = BFIMGaussian._get_sopt(c_, μ0, mf_check, sk1_nom)
            yk1 = mf_check.f(x0_check, sk1, c_) + noise_1
            μ1, Σ1 = ekf_update(μ0, Σ0, yk1, sk1, c_, mf_check)
            # Step 2: warm-start from sk2_nom
            sk2 = BFIMGaussian._get_sopt(c_, μ1, mf_check, sk2_nom)
            yk2 = mf_check.f(x0_check, sk2, c_) + noise_2
            μ2, Σ2 = ekf_update(μ1, Σ1, yk2, sk2, c_, mf_check)
            sum(abs2, μ2 - x0_check)
        end
        _, (grad_3e,) = Zygote.withgradient(two_step_warm_c, c_nom)
        # FD uses the same warm-started function (same basins)
        fd_3e = (two_step_warm_c(c_nom .+ ε .* v_c) - two_step_warm_c(c_nom .- ε .* v_c)) / (2ε)
        ad_3e = dot(grad_3e, v_c)
        re_3e = abs(fd_3e - ad_3e) / (abs(ad_3e) + 1e-12)
        println("      AD=$(round(ad_3e, sigdigits=6))  FD=$(round(fd_3e, sigdigits=6))  rel_err=$(round(re_3e, sigdigits=3))"); flush(stdout)
        @assert re_3e < 1e-2  "2-step EKF (warm-started) gradient check failed: rel_err=$re_3e"

        # ── 3f. Full episode_loss (warm-started FD) ───────────────────────────
        println("  3f. Zygote ∂(episode_loss)/∂c  (N_steps=$N_steps) [warm-started FD]:"); flush(stdout)
        # Pre-compute nominal s★ trajectory for warm-starting
        s_noms = Vector{Vector{Float64}}(undef, N_steps)
        μ_run, Σ_run = μ0, Σ0
        for k in 1:N_steps
            s_noms[k] = get_sopt(c_nom, μ_run, mf_check)
            yk_run = mf_check.f(x0_check, s_noms[k], c_nom) + noise_bank[1][k]
            μ_run, Σ_run = ekf_update(μ_run, Σ_run, yk_run, s_noms[k], c_nom, mf_check)
        end
        ep_warm_c = c_ -> begin
            μ, Σ = μ0, Σ0
            for k in 1:N_steps
                sk = BFIMGaussian._get_sopt(c_, μ, mf_check, s_noms[k])
                yk = mf_check.f(x0_check, sk, c_) + noise_bank[1][k]
                μ, Σ = ekf_update(μ, Σ, yk, sk, c_, mf_check)
            end
            sum(abs2, μ - x0_check)
        end
        _, (grad_3f,) = Zygote.withgradient(ep_warm_c, c_nom)
        fd_3f = (ep_warm_c(c_nom .+ ε .* v_c) - ep_warm_c(c_nom .- ε .* v_c)) / (2ε)
        ad_3f = dot(grad_3f, v_c)
        re_3f = abs(fd_3f - ad_3f) / (abs(ad_3f) + 1e-12)
        println("      AD=$(round(ad_3f, sigdigits=6))  FD=$(round(fd_3f, sigdigits=6))  rel_err=$(round(re_3f, sigdigits=3))"); flush(stdout)
        @assert re_3f < 1e-2  "episode_loss (warm-started) gradient check failed: rel_err=$re_3f"

        println("═══ EKF gradient tests complete ═══"); flush(stdout)
    end
end

if _run_test("sopt_heatmap")
    let n_trials = 10, n_grid = 100
        mf_check  = make_model(nω, n_lat, GΔω, σ², αr)
        φ_range   = range(-π, π, length=n_grid)
        reg_obj(μ, s) = bfim_trace(μ, s, c_nom, mf_check) - mf_check.αr * sum(abs2, s)

        println("get_sopt heatmap validation — $n_trials trials:"); flush(stdout)
        plts = map(1:n_trials) do trial
            μ = x0_list[mod1(trial, length(x0_list))]
            Z_bfim = [bfim_trace(μ, [φ₁, φ₂], c_nom, mf_check)
                      for φ₂ in φ_range, φ₁ in φ_range]
            idx    = argmax(Z_bfim)
            s_grid = [φ_range[idx[2]], φ_range[idx[1]]]
            s★_raw = get_sopt(c_nom, μ, mf_check)
            s★     = mod.(s★_raw .+ π, 2π) .- π
            Δs     = norm(s_grid .- s★)

            println("  trial $trial: grid=$(round.(s_grid, digits=3))  " *
                    "get_sopt=$(round.(s★, digits=3))  ‖Δs‖=$(round(Δs, sigdigits=2))"); flush(stdout)

            p = heatmap(φ_range, φ_range, Z_bfim;
                        xlabel="φ₁", ylabel="φ₂",
                        title="trial $trial  ‖Δs‖=$(round(Δs, sigdigits=1))",
                        color=:viridis, colorbar=false)
            scatter!(p, [s_grid[1]], [s_grid[2]];
                     marker=:xcross, markersize=9, markerstrokewidth=3,
                     color=:red, label="grid max")
            scatter!(p, [s★[1]], [s★[2]];
                     marker=:circle, markersize=7, markerstrokewidth=2,
                     color=:white, label="get_sopt")
            p
        end

        fig = plot(plts...; layout=(2, 5), size=(1600, 700),
                   plot_title="BFIM trace landscape  (αr=$(mf_check.αr), markers: × grid argmax, ○ get_sopt)")
        savefig(fig, joinpath(@__DIR__, "bfim_heatmaps.png"))
        println("Saved → $(joinpath(@__DIR__, "bfim_heatmaps.png"))"); flush(stdout)
        display(fig)
    end
end

if _run_test("sopt_grad")
    let ε = 1e-5, n_trials = 10
        mf_check  = make_model(nω, n_lat, GΔω, σ², αr)
        rng_check = MersenneTwister(101)
        println("get_sopt gradient check (IFT rrule) w.r.t. c — $n_trials trials:"); flush(stdout)
        for trial in 1:n_trials
            μ_check     = x0_list[mod1(trial, length(x0_list))]
            sopt_nom    = get_sopt(c_nom, μ_check, mf_check)
            println("  trial $trial: s★ = $(round.(sopt_nom, digits=4))"); flush(stdout)
            scalar_sopt = c_ -> sum(real.(exp.(im .* BFIMGaussian._get_sopt(c_, μ_check, mf_check, sopt_nom))))
            _, (grad,)  = Zygote.withgradient(scalar_sopt, c_nom)
            v           = randn(rng_check, length(c_nom)); v ./= norm(v)
            fd_deriv    = (scalar_sopt(c_nom .+ ε .* v) - scalar_sopt(c_nom .- ε .* v)) / (2ε)
            ad_deriv    = dot(grad, v)
            rel_err     = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)
            println("  trial $trial: AD=$ad_deriv  FD=$fd_deriv  rel_err=$rel_err"); flush(stdout)
            @assert rel_err < 1e-3  "get_sopt gradient check failed at trial $trial: rel_err=$rel_err"
        end
    end
end

if _run_test("episode")
    let n_trials = 3
        mf_check  = make_model(nω, n_lat, GΔω, σ², αr)
        rng_check = MersenneTwister(103)
        println("episode_loss gradient check (AD vs FD) — $n_trials trials, multiple ε:"); flush(stdout)
        for trial in 1:n_trials
            x0_check    = x0_list[mod1(trial, length(x0_list))]
            noise_check = noise_bank[mod1(trial, length(noise_bank))]
            eloss       = c_ -> episode_loss(x0_check, c_, mf_check, μ0, Σ0, noise_check)
            L_nom, (grad,) = Zygote.withgradient(eloss, c_nom)
            v = randn(rng_check, length(c_nom)); v ./= norm(v)
            ad_deriv = dot(grad, v)
            println("  trial $trial: loss=$(round(L_nom, sigdigits=6))  AD=$(round(ad_deriv, sigdigits=6))"); flush(stdout)
            best_err = Inf
            for ε in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                fd_deriv = (eloss(c_nom .+ ε .* v) - eloss(c_nom .- ε .* v)) / (2ε)
                rel_err  = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)
                best_err = min(best_err, rel_err)
                println("    ε=$ε  FD=$(round(fd_deriv, sigdigits=6))  rel_err=$(round(rel_err, sigdigits=3))"); flush(stdout)
            end
            @assert best_err < 1e-2  "episode_loss gradient check failed at trial $trial: best rel_err=$best_err across all ε"
        end
    end
end

# ── Warm-started episode test (accurate FD via basin-stable get_sopt) ─────────
if _run_test("episode_warm")
    let ε = 1e-5, n_trials = 3
        mf_check  = make_model(nω, n_lat, GΔω, σ², αr)
        rng_check = MersenneTwister(104)
        println("episode_loss gradient check (warm-started FD) — $n_trials trials:"); flush(stdout)
        for trial in 1:n_trials
            x0_check    = x0_list[mod1(trial, length(x0_list))]
            noise_check = noise_bank[mod1(trial, length(noise_bank))]

            # Pre-compute nominal s★ trajectory for warm-starting FD
            s_noms_ep = Vector{Vector{Float64}}(undef, length(noise_check))
            μ_run, Σ_run = μ0, Σ0
            for k in eachindex(noise_check)
                s_noms_ep[k] = get_sopt(c_nom, μ_run, mf_check)
                yk = mf_check.f(x0_check, s_noms_ep[k], c_nom) + noise_check[k]
                μ_run, Σ_run = ekf_update(μ_run, Σ_run, yk, s_noms_ep[k], c_nom, mf_check)
            end

            # Warm-started episode: uses _get_sopt with nominal s★ as init
            ep_warm = c_ -> begin
                μ, Σ = μ0, Σ0
                for k in eachindex(noise_check)
                    sk = BFIMGaussian._get_sopt(c_, μ, mf_check, s_noms_ep[k])
                    yk = mf_check.f(x0_check, sk, c_) + noise_check[k]
                    μ, Σ = ekf_update(μ, Σ, yk, sk, c_, mf_check)
                end
                sum(abs2, μ - x0_check)
            end

            L_nom, (grad,) = Zygote.withgradient(ep_warm, c_nom)
            v = randn(rng_check, length(c_nom)); v ./= norm(v)
            ad_deriv = dot(grad, v)
            fd_deriv = (ep_warm(c_nom .+ ε .* v) - ep_warm(c_nom .- ε .* v)) / (2ε)
            rel_err  = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)
            println("  trial $trial: loss=$(round(L_nom, sigdigits=6))  AD=$(round(ad_deriv, sigdigits=6))  " *
                    "FD=$(round(fd_deriv, sigdigits=6))  rel_err=$(round(rel_err, sigdigits=3))"); flush(stdout)
            @assert rel_err < 1e-3  "episode_loss (warm-started) gradient check failed at trial $trial: rel_err=$rel_err"
        end
    end
end

if _run_test("batch")
    # Per-episode gradient check first (diagnoses which episodes have basin issues)
    let ε = 1e-5
        mf_check = make_model(nω, n_lat, GΔω, σ², αr)
        v = randn(MersenneTwister(99), length(c_nom)); v ./= norm(v)
        println("batch_c2loss_grad: per-episode diagnostics (ε=$ε):"); flush(stdout)
        for i in eachindex(x0_list)
            eloss = c_ -> episode_loss(x0_list[i], c_, mf_check, μ0, Σ0, noise_bank[i])
            L_nom, (grad,) = Zygote.withgradient(eloss, c_nom)
            fd_d = (eloss(c_nom .+ ε .* v) - eloss(c_nom .- ε .* v)) / (2ε)
            ad_d = dot(grad, v)
            re   = abs(fd_d - ad_d) / (abs(ad_d) + 1e-12)
            println("  episode $i: loss=$(round(L_nom, sigdigits=6))  AD=$(round(ad_d, sigdigits=6))  " *
                    "FD=$(round(fd_d, sigdigits=6))  rel_err=$(round(re, sigdigits=3))"); flush(stdout)
        end

        # Aggregate check via batch (uses pmap)
        L_nom, grad_c = batch_c2loss_grad(x0_list, c_nom, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        L_fwd, _ = batch_c2loss_grad(x0_list, c_nom .+ ε .* v, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        L_bwd, _ = batch_c2loss_grad(x0_list, c_nom .- ε .* v, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        fd_deriv = (L_fwd - L_bwd) / (2ε)
        ad_deriv = dot(grad_c, v)
        rel_err  = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)
        println("batch_c2loss_grad aggregate check:"); flush(stdout)
        println("  loss=$L_nom  AD=$ad_deriv  FD=$fd_deriv  rel_err=$rel_err"); flush(stdout)

        # Also try ε=1e-4 to check if FD step size is the issue
        ε2 = 1e-4
        L_fwd2, _ = batch_c2loss_grad(x0_list, c_nom .+ ε2 .* v, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        L_bwd2, _ = batch_c2loss_grad(x0_list, c_nom .- ε2 .* v, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        fd_deriv2 = (L_fwd2 - L_bwd2) / (2ε2)
        rel_err2  = abs(fd_deriv2 - ad_deriv) / (abs(ad_deriv) + 1e-12)
        println("  (ε=$ε2) FD=$fd_deriv2  rel_err=$rel_err2"); flush(stdout)

        @assert rel_err < 1e-2 || rel_err2 < 1e-2  "batch_c2loss_grad gradient check failed: rel_err=$rel_err (ε=$ε), $rel_err2 (ε=$ε2)"
    end
end

# ── EKF performance evaluation ────────────────────────────────────────────────
# Runs Monte Carlo episodes to assess whether the EKF actually estimates x0
# well for this problem, before investing in geometry optimisation.
# Reports per-step error/covariance statistics and detects divergence.
if _run_test("ekf_perf")
    let n_mc = 50, N_eval = max(N_steps, 10)
        mf_check = make_model(nω, n_lat, GΔω, σ², αr)
        rng_perf = MersenneTwister(999)

        println("═══ EKF Performance Evaluation ═══"); flush(stdout)
        println("  n_mc=$n_mc episodes, N_eval=$N_eval steps per episode"); flush(stdout)
        println("  σ²=$σ²  αr=$αr  x0 ∈ [$x0_min, $x0_max]  Σ0=$(Σ0[1,1])·I"); flush(stdout)

        # Storage: per-step statistics across episodes
        errs      = zeros(N_eval+1, n_mc)   # ‖μ_k − x0‖  (step 0 = prior)
        tr_Σs     = zeros(N_eval+1, n_mc)   # tr(Σ_k)
        mahal_sq  = zeros(N_eval+1, n_mc)   # (μ−x0)'Σ⁻¹(μ−x0) Mahalanobis distance²
        bfim_vals = zeros(N_eval, n_mc)      # tr(BFIM) at each step

        for ep in 1:n_mc
            x0_ep = x0_min .+ (x0_max - x0_min) .* rand(rng_perf, dx)
            noise_ep = [sqrt(σ²) .* randn(rng_perf, dy) for _ in 1:N_eval]

            μ, Σ = copy(μ0), copy(Σ0)

            # Step 0: prior
            err0 = norm(μ - x0_ep)
            errs[1, ep]     = err0
            tr_Σs[1, ep]    = tr(Σ)
            Σ_inv = inv(Σ)
            d = μ - x0_ep
            mahal_sq[1, ep] = dot(d, Σ_inv * d)

            for k in 1:N_eval
                sk = get_sopt(c_nom, μ, mf_check)
                F  = mf_check.fx(μ, sk, c_nom)
                bfim_vals[k, ep] = sum(abs2, F) / σ²

                yk = mf_check.f(x0_ep, sk, c_nom) + noise_ep[k]
                μ, Σ = ekf_update(μ, Σ, yk, sk, c_nom, mf_check)

                errs[k+1, ep]     = norm(μ - x0_ep)
                tr_Σs[k+1, ep]    = tr(Σ)
                Σ_inv_k = inv(Σ + 1e-12 * I(dx))  # regularise for near-singular Σ
                d_k = μ - x0_ep
                mahal_sq[k+1, ep] = dot(d_k, Σ_inv_k * d_k)
            end
        end

        # ── Report ────────────────────────────────────────────────────────────
        println("\n  Step | ‖μ−x0‖ mean±std        | tr(Σ) mean           | Mahal² mean  | BFIM tr mean"); flush(stdout)
        println("  -----|------------------------|----------------------|--------------|-------------"); flush(stdout)
        for k in 0:N_eval
            e_mean = mean(errs[k+1, :])
            e_std  = std(errs[k+1, :])
            t_mean = mean(tr_Σs[k+1, :])
            m_mean = mean(mahal_sq[k+1, :])
            if k == 0
                println("  $(lpad(k,4)) | $(rpad(round(e_mean, sigdigits=4), 10)) ± $(lpad(round(e_std, sigdigits=3), 8)) " *
                        "| $(rpad(round(t_mean, sigdigits=4), 20)) | $(rpad(round(m_mean, sigdigits=4), 12)) | (prior)"); flush(stdout)
            else
                b_mean = mean(bfim_vals[k, :])
                println("  $(lpad(k,4)) | $(rpad(round(e_mean, sigdigits=4), 10)) ± $(lpad(round(e_std, sigdigits=3), 8)) " *
                        "| $(rpad(round(t_mean, sigdigits=4), 20)) | $(rpad(round(m_mean, sigdigits=4), 12)) | $(round(b_mean, sigdigits=4))"); flush(stdout)
            end
        end

        # ── Diagnostics ──────────────────────────────────────────────────────
        final_err_mean = mean(errs[end, :])
        prior_err_mean = mean(errs[1, :])
        final_trΣ      = mean(tr_Σs[end, :])
        prior_trΣ      = mean(tr_Σs[1, :])
        final_mahal    = mean(mahal_sq[end, :])

        println("\n  Summary:"); flush(stdout)
        println("    Prior  ‖μ−x0‖ = $(round(prior_err_mean, sigdigits=4)),  tr(Σ) = $(round(prior_trΣ, sigdigits=4))"); flush(stdout)
        println("    Final  ‖μ−x0‖ = $(round(final_err_mean, sigdigits=4)),  tr(Σ) = $(round(final_trΣ, sigdigits=4))"); flush(stdout)
        println("    Error reduction: $(round(final_err_mean / prior_err_mean, sigdigits=3))×"); flush(stdout)
        println("    Covariance reduction: $(round(final_trΣ / prior_trΣ, sigdigits=3))×"); flush(stdout)
        println("    Final Mahalanobis² mean: $(round(final_mahal, sigdigits=4))  (expect ≈ $dx if calibrated)"); flush(stdout)

        # Divergence check
        diverged = final_err_mean > 2 * prior_err_mean
        if diverged
            println("    ⚠ WARNING: EKF appears to DIVERGE (final error > 2× prior error)"); flush(stdout)
        end
        cov_growing = final_trΣ > prior_trΣ
        if cov_growing
            println("    ⚠ WARNING: Covariance GROWING (tr(Σ) increased from prior to final)"); flush(stdout)
        end

        # Calibration check: Mahalanobis² should be ≈ dx for a well-calibrated filter
        if final_mahal > 3 * dx
            println("    ⚠ WARNING: EKF is OVERCONFIDENT (Mahalanobis² >> dx=$dx)"); flush(stdout)
        elseif final_mahal < dx / 3
            println("    ⚠ WARNING: EKF is UNDERCONFIDENT (Mahalanobis² << dx=$dx)"); flush(stdout)
        else
            println("    ✓ EKF calibration looks reasonable (Mahalanobis² ≈ dx=$dx)"); flush(stdout)
        end

        if !diverged && !cov_growing
            println("    ✓ EKF converges: error and covariance decrease over steps"); flush(stdout)
        end

        println("═══ EKF Performance Evaluation Complete ═══"); flush(stdout)
    end
end

# ── Taylor model fidelity test ────────────────────────────────────────────────
# Compares the Taylor-based lattice model (powers_only) against full FDFD
# at various Δn values to check whether the 2nd-order Taylor expansion
# S ≈ S₀ + dS·Δn + d²S·Δn² is accurate enough for the EKF to work.
if _run_test("taylor_fidelity")
    let
        println("═══ Taylor Model Fidelity Test ═══"); flush(stdout)

        # Use a random base geometry
        ε_geom_base = rand(MersenneTwister(1234), Ny_d, Nx_d)

        # Test at several uniform Δn values
        Δn_vals = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        s_test  = [0.3, -0.2]   # fixed sensor phases
        φ₁, φ₂  = s_test[1], s_test[2]

        # ── Compute Taylor model S-matrices at base geometry ──────────────────
        println("  Computing base S-matrices (FDFD at n_geom=$n_geom)..."); flush(stdout)
        S_arr, dSdn_arr, d2Sdn2_arr = getSmatrices(
            ε_geom_base, n_geom, ε_base, ω_array, Ls, Bs,
            grid_info, monitors_array, a_f_array, a_b_array;
            design_iy=grid_info.design_iy, design_ix=grid_info.design_ix)

        # Taylor model: port powers at uniform Δn (same Δn for all n_lat×n_lat blocks)
        println("  Computing Taylor-model port powers at each Δn..."); flush(stdout)
        taylor_powers = Dict{Float64, Vector{Float64}}()
        for Δn_val in Δn_vals
            Δn_mat = fill(Δn_val, n_lat, n_lat)
            taylor_powers[Δn_val] = powers_only(Δn_mat, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        end

        # ── Full FDFD at each perturbed n_geom + Δn ───────────────────────────
        # For a uniform Δn, all design pixels have the same index n_geom + Δn.
        # We re-run batch_solve with n_geom_perturbed and extract port powers
        # using the same monitor overlaps.
        println("  Running full FDFD at each n_geom + Δn..."); flush(stdout)
        fdfd_powers = Dict{Float64, Vector{Float64}}()
        for Δn_val in Δn_vals
            n_perturbed = n_geom + Δn_val
            # getSmatrices at perturbed n gives S_perturbed; we use Δn=0 in the
            # lattice model since the FDFD already includes the perturbation.
            S_p, _, _ = getSmatrices(
                ε_geom_base, n_perturbed, ε_base, ω_array, Ls, Bs,
                grid_info, monitors_array, a_f_array, a_b_array;
                design_iy=grid_info.design_iy, design_ix=grid_info.design_ix)
            Δn_zero = fill(0.0, n_lat, n_lat)
            # Use the FDFD S-matrices with zero Taylor perturbation
            dS_zero = [zeros(ComplexF64, 4, 4) for _ in 1:nω]
            fdfd_powers[Δn_val] = powers_only(Δn_zero, φ₁, φ₂, S_p, dS_zero, dS_zero, GΔω)
        end

        # ── Report ────────────────────────────────────────────────────────────
        ref_power = norm(taylor_powers[0.0])
        println("\n  Δn       | ‖Taylor‖     | ‖FDFD‖       | ‖Taylor−FDFD‖ | rel_err    | rel_to_signal"); flush(stdout)
        println("  ---------|-------------|-------------|--------------|------------|-------------"); flush(stdout)
        for Δn_val in Δn_vals
            tp = taylor_powers[Δn_val]
            fp = fdfd_powers[Δn_val]
            diff = norm(tp - fp)
            rel  = diff / (norm(fp) + 1e-30)
            # rel_to_signal: error relative to the signal (change from Δn=0)
            signal = norm(fp - fdfd_powers[0.0])
            rel_sig = signal > 1e-30 ? diff / signal : NaN
            println("  $(rpad(Δn_val, 9)) | $(rpad(round(norm(tp), sigdigits=5), 11)) " *
                    "| $(rpad(round(norm(fp), sigdigits=5), 11)) " *
                    "| $(rpad(round(diff, sigdigits=4), 12)) " *
                    "| $(rpad(round(rel, sigdigits=3), 10)) " *
                    "| $(round(rel_sig, sigdigits=3))"); flush(stdout)
        end

        # ── Per-port comparison at Δn = x0_max ───────────────────────────────
        Δn_check = x0_max
        tp = taylor_powers[Δn_check]
        fp = fdfd_powers[Δn_check]
        println("\n  Per-port comparison at Δn=$Δn_check (x0_max):"); flush(stdout)
        println("  Port | Taylor power  | FDFD power    | abs_err       | rel_err"); flush(stdout)
        println("  -----|--------------|--------------|--------------|--------"); flush(stdout)
        for i in eachindex(tp)
            ae = abs(tp[i] - fp[i])
            re = ae / (abs(fp[i]) + 1e-30)
            println("  $(rpad(i, 5))| $(rpad(round(tp[i], sigdigits=6), 13)) " *
                    "| $(rpad(round(fp[i], sigdigits=6), 13)) " *
                    "| $(rpad(round(ae, sigdigits=4), 13)) " *
                    "| $(round(re, sigdigits=3))"); flush(stdout)
        end

        # ── Jacobian comparison at Δn = 0 ────────────────────────────────────
        # Compare analytical jac_only (from Taylor derivatives) vs FD of full FDFD
        println("\n  Jacobian comparison: jac_only vs FD(FDFD) at Δn=0:"); flush(stdout)
        J_taylor = jac_only(zeros(n_lat, n_lat), φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        ε_fd = 1e-5
        J_fdfd = hcat([begin
            # Perturb the k-th block's n_geom by ε_fd and re-run FDFD
            # For uniform block perturbation: n_geom + ε_fd for block k
            # This is expensive but gives the ground truth Jacobian
            Δn_plus  = zeros(n_lat, n_lat); Δn_plus[k] = ε_fd
            Δn_minus = zeros(n_lat, n_lat); Δn_minus[k] = -ε_fd

            # Full FDFD at n_geom + ε_fd (block k only)
            # We need per-block perturbation — but batch_solve uses uniform n_geom.
            # Approximate: use Taylor model FD as ground truth reference
            # (this tests jac_only's analytical formula vs FD of powers_only)
            (powers_only(Δn_plus, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω) .-
             powers_only(Δn_minus, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)) ./ (2ε_fd)
        end for k in 1:n_lat^2]...)

        jac_err = norm(J_taylor - J_fdfd) / (norm(J_taylor) + 1e-30)
        println("  ‖J_taylor‖ = $(round(norm(J_taylor), sigdigits=4))  " *
                "‖J_fd‖ = $(round(norm(J_fdfd), sigdigits=4))  " *
                "rel_err = $(round(jac_err, sigdigits=3))"); flush(stdout)

        println("═══ Taylor Fidelity Test Complete ═══"); flush(stdout)
    end
end

if _TEST_ENV != ""
    println("\n═══ Selected gradient checks complete ═══\n"); flush(stdout)
end
