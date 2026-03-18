push!(LOAD_PATH, @__DIR__)

# Flush stdout after every write so nohup.out gets output in real time.
ENV["JULIA_DEBUG"] = ""   # ensure no debug buffering
flush(stdout)

using Distributed

addprocs(4)
println("Workers: $(nworkers())  |  Sys.CPU_THREADS: $(Sys.CPU_THREADS)")

# Broadcast LOAD_PATH and module imports to all workers.
# $src_dir interpolates the master-side string so @__DIR__ is not re-evaluated
# on each worker.
const src_dir = @__DIR__
@everywhere push!(LOAD_PATH, $src_dir)
@everywhere begin
    using SimGeomBroadBand
    using BFIMGaussian
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

using Test
using Plots
using ForwardDiff
using Optim
using Serialization
using Statistics
using BFIMGaussian

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
    fdfd = ε_ -> sim_geom(ε_, n_geom, ε_base, ω_array, Ls, Bs, grid_info, monitors_array, a_f_array, a_b_array)
    c, pb_c = Zygote.pullback(fdfd, ε_geom)

    loss, cbar = batch_c2loss_grad(x0_list, c, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)

    (grad_ε_geoms,) = pb_c(cbar) 
        
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
        # Optionally resample x0 and noise each iteration for fresh stochastic estimates
        if resample_every > 0 && t > 1 && mod(t - 1, resample_every) == 0
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
        println("iter $t/$n_iters  loss=$(round(loss, sigdigits=6))  Δloss=$(round(Δloss, sigdigits=3))  " *
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

println("number of frequencies = $nω  |  workers = $(nworkers())")

# ── Geometry and calibration setup ───────────────────────────────────────────
ε_base, ω_array, Ls, Bs, grid_info, _, monitors_array =
    setup_4port_sweep(ωmin, ωmax, nω, n_core, w, d_length, d_width;
        Lx=Lx, Ly=Ly, res=res, n_pml=n_pml, R_target=R_target,
        port_offset=port_offset, mon_offset=mon_offset)


(a_f_array, a_b_array) = calibrate_straight_waveguide(
    ωmin, ωmax, nω, n_core, w;
    Lx=Lx, Ly=Ly, res=res, n_pml=n_pml, R_target=R_target,
    port_offset=port_offset, mon_offset=mon_offset)

δω  = ω_array[2] - ω_array[1]
GΔω = @. exp(-(ω_array - ω₀)^2 / (2Δω^2)) / (Δω * sqrt(2π)) * δω

Ny_d = length(grid_info.design_iy)
Nx_d = length(grid_info.design_ix)

# ── Run optimisation ─────────────────────────────────────────────────────────

ε_geom_opt = fill(0.5, Ny_d, Nx_d)

ε_geom_opt, losses = train_adam!(
    ε_geom_opt, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
    monitors_array, a_f_array, a_b_array,
    x0_list, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr;
    n_iters=200, lr=1e-3, x0_min=x0_min, x0_max=x0_max)

# ══════════════════════════════════════════════════════════════════════════════
# Gradient / correctness tests
# Control via env var BFIM_TEST: "all", or comma-separated subset of
#   sim_geom, mf_f, mf_fx, sopt_heatmap, sopt_grad, episode, batch
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
    let ε = 1e-5, n_trials = 5
        mf_check  = make_model(nω, n_lat, GΔω, σ², αr)
        rng_check = MersenneTwister(103)
        println("episode_loss gradient check (AD vs FD) — $n_trials trials:")
        for trial in 1:n_trials
            x0_check    = x0_list[mod1(trial, length(x0_list))]
            noise_check = noise_bank[mod1(trial, length(noise_bank))]
            eloss       = c_ -> episode_loss(x0_check, c_, mf_check, μ0, Σ0, noise_check)
            L_nom, (grad,) = Zygote.withgradient(eloss, c_nom)
            v = randn(rng_check, length(c_nom)); v ./= norm(v)
            fd_deriv = (eloss(c_nom .+ ε .* v) - eloss(c_nom .- ε .* v)) / (2ε)
            ad_deriv = dot(grad, v)
            rel_err  = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)
            println("  trial $trial: loss=$L_nom  AD=$ad_deriv  FD=$fd_deriv  rel_err=$rel_err")
            @assert rel_err < 1e-3  "episode_loss gradient check failed at trial $trial: rel_err=$rel_err"
        end
    end
end

if _run_test("batch")
    let ε = 1e-5
        v = randn(MersenneTwister(99), length(c_nom)); v ./= norm(v)
        L_nom, grad_c = batch_c2loss_grad(x0_list, c_nom, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        L_fwd, _ = batch_c2loss_grad(x0_list, c_nom .+ ε .* v, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        L_bwd, _ = batch_c2loss_grad(x0_list, c_nom .- ε .* v, nω, n_lat, GΔω, μ0, Σ0, noise_bank, σ², αr)
        fd_deriv = (L_fwd - L_bwd) / (2ε)
        ad_deriv = dot(grad_c, v)
        rel_err  = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-12)
        println("batch_c2loss_grad gradient check (AD vs FD):")
        println("  loss=$L_nom  AD=$ad_deriv  FD=$fd_deriv  rel_err=$rel_err")
        @assert rel_err < 1e-3  "batch_c2loss_grad gradient check failed: rel_err=$rel_err"
    end
end

if _TEST_ENV != ""
    println("\n═══ Selected gradient checks complete ═══\n")
end
