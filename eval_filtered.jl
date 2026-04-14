#!/usr/bin/env julia
"""
Evaluate loss for raw and filtered optimized geometries from both
opt_result1 (MMA) and opt_result2 (Adam).

Sets up the full FDFD pipeline once, then evaluates each geometry variant.
"""

push!(LOAD_PATH, @__DIR__)
flush(stdout)

using Distributed
println("[eval] Adding workers..."); flush(stdout)
addprocs(20)
println("[eval] Workers: $(nworkers())"); flush(stdout)

const src_dir = @__DIR__
@everywhere push!(LOAD_PATH, $src_dir)
@everywhere begin
    using SimGeomBroadBand
    using BFIMGaussian
    using LinearAlgebra
    using Random
    using Zygote
    BLAS.set_num_threads(1)
end

using Serialization
using Statistics

# ── Spatial filters (same as filter_and_eval.jl) ─────────────────────────────

function gaussian_kernel(r::Int, σ::Float64)
    k = [exp(-((i-r-1)^2 + (j-r-1)^2) / (2σ^2)) for i in 1:2r+1, j in 1:2r+1]
    return k ./ sum(k)
end

function conv2d(img::Matrix{Float64}, kernel::Matrix{Float64})
    Ny, Nx = size(img)
    ky, kx = size(kernel)
    ry, rx = ky ÷ 2, kx ÷ 2
    out = zeros(Ny, Nx)
    for ix in 1:Nx, iy in 1:Ny
        s = 0.0; w = 0.0
        for dx in -rx:rx, dy in -ry:ry
            jx, jy = ix + dx, iy + dy
            if 1 <= jx <= Nx && 1 <= jy <= Ny
                wk = kernel[dy + ry + 1, dx + rx + 1]
                s += wk * img[jy, jx]; w += wk
            end
        end
        out[iy, ix] = s / w
    end
    return out
end

threshold(img, η=0.5) = Float64.(img .>= η)

function apply_gauss_filter(ε, r)
    σ = r / 2.0
    k = gaussian_kernel(r, σ)
    return threshold(conv2d(ε, k), 0.5)
end

# ── Reuse PhEnd2End setup/model code ─────────────────────────────────────────

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

@everywhere function _run_episode_fwd(arg)
    x0_i, noise_i, c, nω, n_lat, GΔω, μ0, Σ0, σ², αr = arg
    mf = make_model(nω, n_lat, GΔω, σ², αr)
    return episode_loss(x0_i, c, mf, μ0, Σ0, noise_i)
end

function sim_geom(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info, monitors_array, a_f_array, a_b_array)
    S_arr, dSdn_arr, d2Sdn2_arr = getSmatrices(
        ε_geom, n_geom, ε_base, ω_array, Ls, Bs,
        grid_info, monitors_array, a_f_array, a_b_array;
        design_iy=grid_info.design_iy, design_ix=grid_info.design_ix)
    S0r = reduce(vcat, vec(real.(S)) for S in S_arr)
    S0i = reduce(vcat, vec(imag.(S)) for S in S_arr)
    S1r = reduce(vcat, vec(real.(dS)) for dS in dSdn_arr)
    S1i = reduce(vcat, vec(imag.(dS)) for dS in dSdn_arr)
    S2r = reduce(vcat, vec(real.(d2S)) for d2S in d2Sdn2_arr)
    S2i = reduce(vcat, vec(imag.(d2S)) for d2S in d2Sdn2_arr)
    return vcat(S0r, S0i, S1r, S1i, S2r, S2i)
end

# ── Parameters (must match PhEnd2End.jl) ─────────────────────────────────────

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

ω₀                 = (ωmax + ωmin) / 2
Δω                 = (ωmax - ωmin) / 6

dx                 = n_lat^2
dy                 = 4 * n_lat
N_steps            = 3
μ0                 = fill((1e-5 + 1e-4)/2, dx)
Σ0                 = 7e-10 * Matrix{Float64}(I, dx, dx)
σ²                 = 1e-10
αr                 = 0.0
x0_min             = 1e-5
x0_max             = 1e-4

# Episodes for evaluation (parallelized via pmap)
n_eval_episodes    = 50
rng_eval           = MersenneTwister(12345)
x0_list            = [x0_min .+ (x0_max - x0_min) .* rand(rng_eval, dx) for _ in 1:n_eval_episodes]
noise_bank         = sample_noise_bank(rng_eval, n_eval_episodes, N_steps, dy, σ²)

# ── FDFD setup ───────────────────────────────────────────────────────────────

println("[setup] Building 4-port geometry..."); flush(stdout)
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
println("[setup] Calibration done."); flush(stdout)

δω  = ω_array[2] - ω_array[1]
GΔω = @. exp(-(ω_array - ω₀)^2 / (2Δω^2)) / (Δω * sqrt(2π)) * δω

Ny_d = length(grid_info.design_iy)
Nx_d = length(grid_info.design_ix)
println("[setup] Design region: $(Ny_d)×$(Nx_d)"); flush(stdout)

# ── Evaluate one geometry ────────────────────────────────────────────────────

function eval_geometry(ε_geom, label)
    println("\n  [$label] Running FDFD..."); flush(stdout)
    t0 = time()
    c = sim_geom(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                 monitors_array, a_f_array, a_b_array)
    t_fdfd = round(time() - t0, digits=1)
    println("  [$label] FDFD done ($(t_fdfd)s). Evaluating $(n_eval_episodes) episodes via pmap..."); flush(stdout)

    t1 = time()
    args = [(x0_list[i], noise_bank[i], c, nω, n_lat, GΔω, μ0, Σ0, σ², αr) for i in 1:n_eval_episodes]
    losses = pmap(_run_episode_fwd, args)
    t_ep = round(time() - t1, digits=1)

    mean_loss = mean(losses)
    std_loss  = std(losses)
    med_loss  = median(losses)
    rms_err   = sqrt(mean_loss)

    println("  [$label] episodes done ($(t_ep)s). loss: mean=$(round(mean_loss, sigdigits=4)) ± $(round(std_loss, sigdigits=3))  " *
            "median=$(round(med_loss, sigdigits=4))  RMS_rel_err=$(round(100*rms_err, digits=2))%")
    flush(stdout)
    return (; label, mean_loss, std_loss, med_loss, rms_err)
end

# ── Evaluate all geometries ──────────────────────────────────────────────────

# Each checkpoint evaluated with the αr used during its optimization
checkpoints = [
    ("MMA step 310",   joinpath(@__DIR__, "opt_result1", "checkpoints_mma", "eps_geom_step_00310.jls"),  1.0),
    ("Adam step 4750", joinpath(@__DIR__, "opt_result2", "checkpoints_adam", "eps_geom_step_04750.jls"), 0.0),
]

filter_radii = [0, 3, 5, 8, 12]  # 0 = no filter (raw)

println("\n" * "="^80)
println("LOSS EVALUATION: raw vs filtered geometries")
println("  Episodes: $n_eval_episodes  |  N_steps: $N_steps  |  σ²=$σ²  |  Δn ∈ [$x0_min, $x0_max]")
println("="^80)

all_results = []

for (ckpt_name, ckpt_path, αr_run) in checkpoints
    println("\n── $ckpt_name (αr=$αr_run) ──"); flush(stdout)
    d = deserialize(ckpt_path)
    ε_raw = d.ε_geom

    # Override αr to match the run that produced this checkpoint
    global αr = αr_run

    for r in filter_radii
        if r == 0
            label = "$ckpt_name / raw"
            ε_eval = ε_raw
        else
            label = "$ckpt_name / gauss r=$r"
            ε_eval = apply_gauss_filter(ε_raw, r)
        end
        res = eval_geometry(ε_eval, label)
        push!(all_results, res)
    end
end

# Also evaluate random init baseline
println("\n── Baseline (random init) ──"); flush(stdout)
ε_random = rand(MersenneTwister(1234), Ny_d, Nx_d)
res = eval_geometry(ε_random, "Random init")
push!(all_results, res)

# ── Summary table ────────────────────────────────────────────────────────────

println("\n" * "="^80)
println("SUMMARY")
println("="^80)
println(rpad("Geometry", 40) * rpad("Mean Loss", 14) * rpad("Median", 14) * "RMS Rel Err")
println("-"^80)
for r in all_results
    println(rpad(r.label, 40) *
            rpad(string(round(r.mean_loss, sigdigits=4)), 14) *
            rpad(string(round(r.med_loss, sigdigits=4)), 14) *
            "$(round(100*r.rms_err, digits=2))%")
end
println("="^80)
