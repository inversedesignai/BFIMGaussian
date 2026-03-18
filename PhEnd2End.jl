push!(LOAD_PATH, @__DIR__)

# Flush stdout after every write so nohup.out gets output in real time.
ENV["JULIA_DEBUG"] = ""   # ensure no debug buffering
flush(stdout)

using Distributed

println("[startup] Adding worker processes..."); flush(stdout)
addprocs(4)
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

    ModelFunctions(f=f, fx=fx, fxs=fxs, σ²=σ², dy=4n, dx=n^2, ds=2, dc=96*nω, αr=αr, zero_s_init=true)

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
            println("  saved → $path")
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
Σ0                 = 0.01 *Matrix{Float64}(I, dx, dx)              # initial belief covariance
σ²                 = 0.01
αr                 = 10.0

x0_min             = -0.01
x0_max             =  0.01
rng                = MersenneTwister(42)
x0_list            = [x0_min .+ (x0_max - x0_min) .* rand(rng, dx) for _ in 1:n_episodes]
noise_bank         = sample_noise_bank(rng, n_episodes, N_steps, dy, σ²)

println("[params] nω=$nω  workers=$(nworkers())  n_lat=$n_lat  res=$res  grid=$(round(Int,Lx*res))×$(round(Int,Ly*res))  design=$(d_length)×$(d_width)")
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
        n_iters=1000000, ftol_rel=1e-8, xtol_rel=1e-8,
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
#   sim_geom, mf_f, mf_fx, lattice, ekf, sopt_heatmap, sopt_grad, episode, batch
# Example:  BFIM_TEST=sopt_grad,episode julia PhEnd2End.jl
# ══════════════════════════════════════════════════════════════════════════════
const _TEST_ENV = get(ENV, "BFIM_TEST", "")
const _RUN_ALL  = _TEST_ENV == "all"
_run_test(name) = _RUN_ALL || name in split(_TEST_ENV, ",")

if _TEST_ENV != ""
    println("\n═══ Running selected gradient checks: $(_RUN_ALL ? "all" : _TEST_ENV) ═══\n")

    # ── Compute / load nominal c (shared by all tests) ───────────────────────
    c_nom_path = joinpath(@__DIR__, "c_nom.jls")
    if isfile(c_nom_path)
        c_nom = deserialize(c_nom_path)
        println("Loaded c_nom from $c_nom_path  (length=$(length(c_nom)))")
    else
        ε_geom = rand(MersenneTwister(1234), Ny_d, Nx_d)
        c_nom = sim_geom(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info, monitors_array, a_f_array, a_b_array)
        serialize(c_nom_path, c_nom)
        println("Computed and saved c_nom to $c_nom_path")
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
        println("sim_geom gradient check (AD vs FD):")
        println("  AD=$ad_deriv  FD=$fd_deriv  rel_err=$rel_err")
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

        println("mf.f gradient check w.r.t. c (AD vs FD):")
        println("  AD directional derivative : $ad_deriv")
        println("  FD directional derivative : $fd_deriv")
        println("  relative error            : $rel_err")
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

        println("mf.fx directional check (analytical vs FD):")
        println("  ‖jac_v‖ = $(norm(jac_v))  ‖fd_jac_v‖ = $(norm(fd_jac_v))  rel_err = $rel_err_dir")
        @assert rel_err_dir < 1e-3  "mf.fx directional check failed: relative error = $rel_err_dir"

        # 2. Full Jacobian check — column i = (f(x+ε eᵢ) - f(x-ε eᵢ)) / 2ε
        J_anal = mf_check.fx(x_check, s_check, c_nom)
        J_fd   = hcat([
            (mf_check.f(x_check .+ ε .* (1:dx .== i), s_check, c_nom) .-
             mf_check.f(x_check .- ε .* (1:dx .== i), s_check, c_nom)) ./ (2ε)
            for i in 1:dx]...)
        rel_err_full = norm(J_anal .- J_fd) / (norm(J_anal) + 1e-12)

        println("mf.fx full Jacobian check (analytical vs FD):")
        println("  ‖J_anal‖ = $(norm(J_anal))  ‖J_fd‖ = $(norm(J_fd))  rel_err = $rel_err_full")
        if rel_err_full >= 1e-3
            println("  --- diagnostic: J_anal ---")
            show(stdout, "text/plain", J_anal); println()
            println("  --- diagnostic: J_fd ---")
            show(stdout, "text/plain", J_fd); println()
            println("  --- diagnostic: element-wise ratio J_anal ./ J_fd ---")
            show(stdout, "text/plain", J_anal ./ J_fd); println()
            println("  --- diagnostic: does -J_anal ≈ J_fd? rel_err = $(norm(J_anal .+ J_fd)/norm(J_anal))")
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
        println("  1a. jac_only vs FD(powers_only) w.r.t. Δn:")
        J_anal = jac_only(Δn, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        J_fd   = hcat([begin
            eₖ = zeros(n_lat, n_lat); eₖ[k] = 1.0
            (powers_only(Δn .+ ε .* eₖ, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω) .-
             powers_only(Δn .- ε .* eₖ, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)) ./ (2ε)
        end for k in 1:length(Δn)]...)
        re_jac = norm(J_anal - J_fd) / (norm(J_anal) + 1e-12)
        println("      ‖J_anal‖=$(round(norm(J_anal), sigdigits=4))  ‖J_fd‖=$(round(norm(J_fd), sigdigits=4))  rel_err=$(round(re_jac, sigdigits=3))")
        @assert re_jac < 1e-3  "jac_only vs FD(powers_only) failed: rel_err=$re_jac"

        # ── 1b. jac_only vs FD of powers_only w.r.t. s (phases) ──────────────
        println("  1b. jac_only vs FD(powers_only) w.r.t. s:")
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
            println("      ∂μ/∂$name: ‖AD‖=$(round(norm(dμ_ds_ad), sigdigits=4))  ‖FD‖=$(round(norm(dμ_ds_fd), sigdigits=4))  rel_err=$(round(re_s, sigdigits=3))")
            @assert re_s < 1e-3  "∂μ/∂$name check failed: rel_err=$re_s"
        end

        # ── 1c. jac_and_dirderiv_s vs FD of jac_only w.r.t. s ────────────────
        println("  1c. jac_and_dirderiv_s vs FD(jac_only) w.r.t. s:")
        λ_test = [0.7, -0.4]
        J0, dJ_λ = jac_and_dirderiv_s(Δn, φ₁, φ₂, λ_test, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        # FD: directional derivative of jac_only along λ in s-space
        J_plus  = jac_only(Δn, φ₁ + ε*λ_test[1], φ₂ + ε*λ_test[2], S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        J_minus = jac_only(Δn, φ₁ - ε*λ_test[1], φ₂ - ε*λ_test[2], S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        dJ_fd   = (J_plus .- J_minus) ./ (2ε)
        # Also check J0 == jac_only at the same point
        re_J0 = norm(J0 - J_anal) / (norm(J_anal) + 1e-12)
        re_dJ = norm(dJ_λ - dJ_fd) / (norm(dJ_fd) + 1e-12)
        println("      J consistency: rel_err=$(round(re_J0, sigdigits=3))")
        println("      dJ_λ: ‖anal‖=$(round(norm(dJ_λ), sigdigits=4))  ‖FD‖=$(round(norm(dJ_fd), sigdigits=4))  rel_err=$(round(re_dJ, sigdigits=3))")
        @assert re_J0 < 1e-10  "J from jac_and_dirderiv_s != jac_only: rel_err=$re_J0"
        @assert re_dJ < 1e-3   "dJ_λ vs FD(jac_only) failed: rel_err=$re_dJ"

        # ── 2a. Zygote gradient of powers_only w.r.t. c ──────────────────────
        println("  2a. Zygote ∂(powers_only)/∂c:")
        v_c = randn(MersenneTwister(200), length(c_nom)); v_c ./= norm(v_c)
        scalar_po = c_ -> begin
            Sa, dSa, d2Sa = unpack_c(c_, nω)
            sum(abs2, powers_only(Δn, φ₁, φ₂, Sa, dSa, d2Sa, GΔω))
        end
        _, (grad_po,) = Zygote.withgradient(scalar_po, c_nom)
        fd_po = (scalar_po(c_nom .+ ε .* v_c) - scalar_po(c_nom .- ε .* v_c)) / (2ε)
        ad_po = dot(grad_po, v_c)
        re_po = abs(fd_po - ad_po) / (abs(ad_po) + 1e-12)
        println("      AD=$(round(ad_po, sigdigits=6))  FD=$(round(fd_po, sigdigits=6))  rel_err=$(round(re_po, sigdigits=3))")
        @assert re_po < 1e-3  "Zygote ∂(powers_only)/∂c failed: rel_err=$re_po"

        # ── 2b. Zygote gradient of jac_only w.r.t. c ─────────────────────────
        println("  2b. Zygote ∂(jac_only)/∂c:")
        scalar_jo = c_ -> begin
            Sa, dSa, d2Sa = unpack_c(c_, nω)
            sum(abs2, jac_only(Δn, φ₁, φ₂, Sa, dSa, d2Sa, GΔω))
        end
        _, (grad_jo,) = Zygote.withgradient(scalar_jo, c_nom)
        fd_jo = (scalar_jo(c_nom .+ ε .* v_c) - scalar_jo(c_nom .- ε .* v_c)) / (2ε)
        ad_jo = dot(grad_jo, v_c)
        re_jo = abs(fd_jo - ad_jo) / (abs(ad_jo) + 1e-12)
        println("      AD=$(round(ad_jo, sigdigits=6))  FD=$(round(fd_jo, sigdigits=6))  rel_err=$(round(re_jo, sigdigits=3))")
        @assert re_jo < 1e-3  "Zygote ∂(jac_only)/∂c failed: rel_err=$re_jo"

        # ── 2c. Zygote gradient of jac_and_dirderiv_s w.r.t. c ───────────────
        println("  2c. Zygote ∂(jac_and_dirderiv_s)/∂c:")
        scalar_jds = c_ -> begin
            Sa, dSa, d2Sa = unpack_c(c_, nω)
            F, dF = jac_and_dirderiv_s(Δn, φ₁, φ₂, λ_test, Sa, dSa, d2Sa, GΔω)
            sum(abs2, F) + sum(abs2, dF)
        end
        _, (grad_jds,) = Zygote.withgradient(scalar_jds, c_nom)
        fd_jds = (scalar_jds(c_nom .+ ε .* v_c) - scalar_jds(c_nom .- ε .* v_c)) / (2ε)
        ad_jds = dot(grad_jds, v_c)
        re_jds = abs(fd_jds - ad_jds) / (abs(ad_jds) + 1e-12)
        println("      AD=$(round(ad_jds, sigdigits=6))  FD=$(round(fd_jds, sigdigits=6))  rel_err=$(round(re_jds, sigdigits=3))")
        @assert re_jds < 1e-3  "Zygote ∂(jac_and_dirderiv_s)/∂c failed: rel_err=$re_jds"

        # ── 2d. Zygote gradient of bfim_trace_dirderiv (IFT scalar) w.r.t. c ─
        # This is the scalar λ'·∇_s(bfim_trace) = 2·sum(F.*dF_λ)/σ² that the
        # IFT rrule differentiates w.r.t. c.
        println("  2d. Zygote ∂(λ'·∇_s bfim_trace)/∂c  (IFT scalar):")
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
        println("      AD=$(round(ad_ift, sigdigits=6))  FD=$(round(fd_ift, sigdigits=6))  rel_err=$(round(re_ift, sigdigits=3))")
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
        println("  3a. Zygote ∂(ekf_update μ_new)/∂c  (fixed s, no get_sopt):")
        ekf_mu_c = c_ -> begin
            y = mf_check.f(x0_check, s_check, c_) + noise_1
            μ_new, _ = ekf_update(μ0, Σ0, y, s_check, c_, mf_check)
            sum(abs2, μ_new)
        end
        _, (grad_3a,) = Zygote.withgradient(ekf_mu_c, c_nom)
        fd_3a = (ekf_mu_c(c_nom .+ ε .* v_c) - ekf_mu_c(c_nom .- ε .* v_c)) / (2ε)
        ad_3a = dot(grad_3a, v_c)
        re_3a = abs(fd_3a - ad_3a) / (abs(ad_3a) + 1e-12)
        println("      AD=$(round(ad_3a, sigdigits=6))  FD=$(round(fd_3a, sigdigits=6))  rel_err=$(round(re_3a, sigdigits=3))")
        @assert re_3a < 1e-3  "ekf_update μ gradient check failed: rel_err=$re_3a"

        # ── 3b. ekf_update Σ_new (fixed s) ───────────────────────────────────
        # Differentiate tr(Σ_new) w.r.t. c — tests Joseph form backward pass.
        println("  3b. Zygote ∂(tr(Σ_new))/∂c  (fixed s):")
        ekf_sig_c = c_ -> begin
            y = mf_check.f(x0_check, s_check, c_) + noise_1
            _, Σ_new = ekf_update(μ0, Σ0, y, s_check, c_, mf_check)
            tr(Σ_new)
        end
        _, (grad_3b,) = Zygote.withgradient(ekf_sig_c, c_nom)
        fd_3b = (ekf_sig_c(c_nom .+ ε .* v_c) - ekf_sig_c(c_nom .- ε .* v_c)) / (2ε)
        ad_3b = dot(grad_3b, v_c)
        re_3b = abs(fd_3b - ad_3b) / (abs(ad_3b) + 1e-12)
        println("      AD=$(round(ad_3b, sigdigits=6))  FD=$(round(fd_3b, sigdigits=6))  rel_err=$(round(re_3b, sigdigits=3))")
        @assert re_3b < 1e-3  "ekf_update Σ gradient check failed: rel_err=$re_3b"

        # ── 3c. Single EKF step with get_sopt (IFT + ekf) ────────────────────
        # Differentiate ‖μ_new − x0‖² w.r.t. c through get_sopt + ekf_update.
        println("  3c. Zygote ∂(single EKF step with get_sopt)/∂c:")
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
        println("      AD=$(round(ad_3c, sigdigits=6))  FD=$(round(fd_3c, sigdigits=6))  rel_err=$(round(re_3c, sigdigits=3))")
        @assert re_3c < 1e-3  "single EKF step gradient check failed: rel_err=$re_3c"

        # ── 3d. Two EKF steps, FIXED s, no get_sopt ─────────────────────────
        # Isolates multi-step ekf_update gradient from IFT rrule.
        println("  3d. Zygote ∂(2 EKF steps, fixed s, no get_sopt)/∂c:")
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
        println("      AD=$(round(ad_3d, sigdigits=6))  FD=$(round(fd_3d, sigdigits=6))  rel_err=$(round(re_3d, sigdigits=3))")
        @assert re_3d < 1e-3  "2-step EKF (fixed s) gradient check failed: rel_err=$re_3d"

        # ── 3e. Two EKF steps WITH get_sopt ──────────────────────────────────
        println("  3e. Zygote ∂(2 EKF steps with get_sopt)/∂c:")
        two_step_sopt_c = c_ -> begin
            μ, Σ = μ0, Σ0
            for noise_k in [noise_1, noise_2]
                sk = get_sopt(c_, μ, mf_check)
                yk = mf_check.f(x0_check, sk, c_) + noise_k
                μ, Σ = ekf_update(μ, Σ, yk, sk, c_, mf_check)
            end
            sum(abs2, μ - x0_check)
        end
        _, (grad_3e,) = Zygote.withgradient(two_step_sopt_c, c_nom)
        fd_3e = (two_step_sopt_c(c_nom .+ ε .* v_c) - two_step_sopt_c(c_nom .- ε .* v_c)) / (2ε)
        ad_3e = dot(grad_3e, v_c)
        re_3e = abs(fd_3e - ad_3e) / (abs(ad_3e) + 1e-12)
        println("      AD=$(round(ad_3e, sigdigits=6))  FD=$(round(fd_3e, sigdigits=6))  rel_err=$(round(re_3e, sigdigits=3))")
        @assert re_3e < 1e-2  "2-step EKF (with get_sopt) gradient check failed: rel_err=$re_3e"

        # ── 3f. Full episode_loss (N_steps EKF steps) ─────────────────────────
        println("  3f. Zygote ∂(episode_loss)/∂c  (N_steps=$N_steps):")
        ep_c = c_ -> episode_loss(x0_check, c_, mf_check, μ0, Σ0, noise_bank[1])
        _, (grad_3f,) = Zygote.withgradient(ep_c, c_nom)
        fd_3f = (ep_c(c_nom .+ ε .* v_c) - ep_c(c_nom .- ε .* v_c)) / (2ε)
        ad_3f = dot(grad_3f, v_c)
        re_3f = abs(fd_3f - ad_3f) / (abs(ad_3f) + 1e-12)
        println("      AD=$(round(ad_3f, sigdigits=6))  FD=$(round(fd_3f, sigdigits=6))  rel_err=$(round(re_3f, sigdigits=3))")
        @assert re_3f < 1e-2  "episode_loss gradient check failed: rel_err=$re_3f"

        println("═══ EKF gradient tests complete ═══"); flush(stdout)
    end
end

if _run_test("sopt_heatmap")
    let n_trials = 10, n_grid = 100
        mf_check  = make_model(nω, n_lat, GΔω, σ², αr)
        φ_range   = range(-π, π, length=n_grid)
        reg_obj(μ, s) = bfim_trace(μ, s, c_nom, mf_check) - mf_check.αr * sum(abs2, s)

        println("get_sopt heatmap validation — $n_trials trials:")
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
                    "get_sopt=$(round.(s★, digits=3))  ‖Δs‖=$(round(Δs, sigdigits=2))")

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
        println("Saved → $(joinpath(@__DIR__, "bfim_heatmaps.png"))")
        display(fig)
    end
end

if _run_test("sopt_grad")
    let ε = 1e-5, n_trials = 10
        mf_check  = make_model(nω, n_lat, GΔω, σ², αr)
        rng_check = MersenneTwister(101)
        println("get_sopt gradient check (IFT rrule) w.r.t. c — $n_trials trials:")
        for trial in 1:n_trials
            μ_check     = x0_list[mod1(trial, length(x0_list))]
            sopt_nom    = get_sopt(c_nom, μ_check, mf_check)
            println("  trial $trial: s★ = $(round.(sopt_nom, digits=4))")
            scalar_sopt = c_ -> sum(real.(exp.(im .* BFIMGaussian._get_sopt(c_, μ_check, mf_check, sopt_nom))))
            _, (grad,)  = Zygote.withgradient(scalar_sopt, c_nom)
            v           = randn(rng_check, length(c_nom)); v ./= norm(v)
            fd_deriv    = (scalar_sopt(c_nom .+ ε .* v) - scalar_sopt(c_nom .- ε .* v)) / (2ε)
            ad_deriv    = dot(grad, v)
            rel_err     = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)
            println("  trial $trial: AD=$ad_deriv  FD=$fd_deriv  rel_err=$rel_err")
            @assert rel_err < 1e-3  "get_sopt gradient check failed at trial $trial: rel_err=$rel_err"
        end
    end
end

if _run_test("episode")
    let n_trials = 3
        mf_check  = make_model(nω, n_lat, GΔω, σ², αr)
        rng_check = MersenneTwister(103)
        println("episode_loss gradient check (AD vs FD) — $n_trials trials, multiple ε:")
        for trial in 1:n_trials
            x0_check    = x0_list[mod1(trial, length(x0_list))]
            noise_check = noise_bank[mod1(trial, length(noise_bank))]
            eloss       = c_ -> episode_loss(x0_check, c_, mf_check, μ0, Σ0, noise_check)
            L_nom, (grad,) = Zygote.withgradient(eloss, c_nom)
            v = randn(rng_check, length(c_nom)); v ./= norm(v)
            ad_deriv = dot(grad, v)
            println("  trial $trial: loss=$(round(L_nom, sigdigits=6))  AD=$(round(ad_deriv, sigdigits=6))")
            best_err = Inf
            for ε in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                fd_deriv = (eloss(c_nom .+ ε .* v) - eloss(c_nom .- ε .* v)) / (2ε)
                rel_err  = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)
                best_err = min(best_err, rel_err)
                println("    ε=$ε  FD=$(round(fd_deriv, sigdigits=6))  rel_err=$(round(rel_err, sigdigits=3))")
            end
            @assert best_err < 1e-2  "episode_loss gradient check failed at trial $trial: best rel_err=$best_err across all ε"
        end
    end
end

if _run_test("batch")
    # Per-episode gradient check first (diagnoses which episodes have basin issues)
    let ε = 1e-5
        mf_check = make_model(nω, n_lat, GΔω, σ², αr)
        v = randn(MersenneTwister(99), length(c_nom)); v ./= norm(v)
        println("batch_c2loss_grad: per-episode diagnostics (ε=$ε):")
        for i in eachindex(x0_list)
            eloss = c_ -> episode_loss(x0_list[i], c_, mf_check, μ0, Σ0, noise_bank[i])
            L_nom, (grad,) = Zygote.withgradient(eloss, c_nom)
            fd_d = (eloss(c_nom .+ ε .* v) - eloss(c_nom .- ε .* v)) / (2ε)
            ad_d = dot(grad, v)
            re   = abs(fd_d - ad_d) / (abs(ad_d) + 1e-12)
            println("  episode $i: loss=$(round(L_nom, sigdigits=6))  AD=$(round(ad_d, sigdigits=6))  " *
                    "FD=$(round(fd_d, sigdigits=6))  rel_err=$(round(re, sigdigits=3))")
        end

        # Aggregate check via batch (uses pmap)
        L_nom, grad_c = batch_c2loss_grad(x0_list, c_nom, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        L_fwd, _ = batch_c2loss_grad(x0_list, c_nom .+ ε .* v, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        L_bwd, _ = batch_c2loss_grad(x0_list, c_nom .- ε .* v, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        fd_deriv = (L_fwd - L_bwd) / (2ε)
        ad_deriv = dot(grad_c, v)
        rel_err  = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)
        println("batch_c2loss_grad aggregate check:")
        println("  loss=$L_nom  AD=$ad_deriv  FD=$fd_deriv  rel_err=$rel_err")

        # Also try ε=1e-4 to check if FD step size is the issue
        ε2 = 1e-4
        L_fwd2, _ = batch_c2loss_grad(x0_list, c_nom .+ ε2 .* v, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        L_bwd2, _ = batch_c2loss_grad(x0_list, c_nom .- ε2 .* v, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        fd_deriv2 = (L_fwd2 - L_bwd2) / (2ε2)
        rel_err2  = abs(fd_deriv2 - ad_deriv) / (abs(ad_deriv) + 1e-12)
        println("  (ε=$ε2) FD=$fd_deriv2  rel_err=$rel_err2")

        @assert rel_err < 1e-2 || rel_err2 < 1e-2  "batch_c2loss_grad gradient check failed: rel_err=$rel_err (ε=$ε), $rel_err2 (ε=$ε2)"
    end
end

if _TEST_ENV != ""
    println("\n═══ Selected gradient checks complete ═══\n")
end
