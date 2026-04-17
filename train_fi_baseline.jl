#!/usr/bin/env julia
#
# FI baseline: optimize geometry to maximize E_x[tr(FI(x, s0, c))] at fixed s0.
#
# This is the simplest geometry optimization baseline — no EKF loop, no adaptive
# sensor selection.  Just maximize average Fisher information at a random fixed
# sensor setting.  Uses the same density filter + β continuation as the autotune
# run for a fair comparison.
#
# Usage:  julia -p 20 train_fi_baseline.jl

push!(LOAD_PATH, @__DIR__)

ENV["JULIA_DEBUG"] = ""
flush(stdout)

using Distributed

println("[startup] Adding worker processes..."); flush(stdout)
addprocs(20)
println("[startup] Workers: $(nworkers())"); flush(stdout)

const src_dir = @__DIR__
@everywhere push!(LOAD_PATH, $src_dir)

println("[startup] Loading modules on all workers..."); flush(stdout)
@everywhere begin
    using SimGeomBroadBand
    using BFIMGaussian
    using LinearAlgebra
    using Random
    using Zygote
    BLAS.set_num_threads(1)
end

println("[startup] Loading main-process packages..."); flush(stdout)
using Serialization
using Statistics
using SparseArrays
println("[startup] All packages loaded."); flush(stdout)

# ── Density filter + projection (from PhEnd2End.jl) ─────────────────────────

function build_density_filter(Ny::Int, Nx::Int, R::Float64)
    R_int = ceil(Int, R)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for ix in 1:Nx, iy in 1:Ny
        k = (ix - 1) * Ny + iy
        wsum = 0.0
        local_entries = Tuple{Int, Float64}[]
        for jx in max(1, ix - R_int):min(Nx, ix + R_int)
            for jy in max(1, iy - R_int):min(Ny, iy + R_int)
                d = sqrt(Float64((ix - jx)^2 + (iy - jy)^2))
                if d <= R
                    w = R - d
                    l = (jx - 1) * Ny + jy
                    push!(local_entries, (l, w))
                    wsum += w
                end
            end
        end
        for (l, w) in local_entries
            push!(rows, k)
            push!(cols, l)
            push!(vals, w / wsum)
        end
    end
    return sparse(rows, cols, vals, Ny * Nx, Ny * Nx)
end

function project_density(ρ, β, η=0.5)
    t_eta  = tanh(β * η)
    t_one  = tanh(β * (1 - η))
    return (t_eta .+ tanh.(β .* (ρ .- η))) ./ (t_eta + t_one)
end

# ── Unpack c + model (from PhEnd2End.jl) ─────────────────────────────────────

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

function sim_geom(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                  monitors_array, a_f_array, a_b_array)
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
    return vcat(S0r, S0i, S1r, S1i, S2r, S2i)
end

# ── FI objective: -E_x[tr(F^T F / σ²)] at fixed s0 ─────────────────────────

@everywhere function _fi_episode_grad(arg)
    x0_i, s0, c, nω, n_lat, GΔω, σ² = arg
    mf = make_model(nω, n_lat, GΔω, σ², 0.0)
    # Minimize negative FI (= maximize FI)
    loss_i, (grad_i,) = Zygote.withgradient(c) do c_
        F = mf.fx(x0_i, s0, c_)
        -sum(abs2, F) / σ²
    end
    return (loss_i, grad_i)
end

function batch_fi_grad(x0_list, s0, c, nω, n_lat, GΔω, σ²)
    n = length(x0_list)
    args = [(x0_list[i], s0, c, nω, n_lat, GΔω, σ²) for i in 1:n]
    results = pmap(_fi_episode_grad, args)
    mean_loss = sum(r[1] for r in results) / n
    mean_grad = sum(r[2] for r in results) / n
    return mean_loss, mean_grad
end

function end2end_fi(ε_raw, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                    monitors_array, a_f_array, a_b_array,
                    x0_list, s0, nω, n_lat, GΔω, σ²;
                    W_filter=nothing, β_proj=nothing, η_proj=0.5)
    Ny_d = length(grid_info.design_iy)
    Nx_d = length(grid_info.design_ix)

    println("  [end2end_fi] FDFD forward..."); flush(stdout)
    fdfd = ε_ -> begin
        ε_filt = W_filter !== nothing ? reshape(W_filter * vec(ε_), Ny_d, Nx_d) : ε_
        ε_proj = β_proj !== nothing ? project_density(ε_filt, β_proj, η_proj) : ε_filt
        sim_geom(ε_proj, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                 monitors_array, a_f_array, a_b_array)
    end
    c, pb_c = Zygote.pullback(fdfd, ε_raw)

    println("  [end2end_fi] FI gradients ($(length(x0_list)) episodes)..."); flush(stdout)
    loss, cbar = batch_fi_grad(x0_list, s0, c, nω, n_lat, GΔω, σ²)

    println("  [end2end_fi] FDFD backward..."); flush(stdout)
    (grad_ε_raw,) = pb_c(cbar)

    println("  [end2end_fi] Done. neg_FI=$loss"); flush(stdout)
    return loss, grad_ε_raw
end

# ── Noise bank helper (needed for compatibility) ──��─────────────────────────

@everywhere function sample_noise_bank(rng, n_ep, N_steps, dy, σ²)
    [[ sqrt(σ²) .* randn(rng, dy) for _ in 1:N_steps ] for _ in 1:n_ep]
end

# ══════════════════════════════════════════════════════════════════════════════
# Parameters (must match PhEnd2End.jl)
# ══════════════════════════════════════════════════════════════════════════════

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

ω₀  = (ωmax + ωmin) / 2
Δω  = (ωmax - ωmin) / 6

dy  = 4 * n_lat
dx  = n_lat^2
ds  = 2

N_steps    = 3
μ0         = fill((1e-5 + 1e-4)/2, dx)
Σ0         = 7e-10 * Matrix{Float64}(I, dx, dx)
σ²         = 1e-10
αr         = 0.0

x0_min     = 1e-5
x0_max     = 1e-4
n_episodes = 20

# Fixed random sensor params
s0 = let rng_s = MersenneTwister(7777)
    2π .* rand(rng_s, ds) .- π   # s0 ∈ [-π, π]^2
end
println("[params] Fixed s0 = $s0"); flush(stdout)

# ─�� Geometry and calibration setup ───────────────────────────────────────────

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
println("[setup] Design region: $(Ny_d)×$(Nx_d) = $(Ny_d*Nx_d) parameters"); flush(stdout)

# ── Adam optimization ────────────────────────────────────────────────────────

filter_radius  = 5.0
β_proj_init    = 16.0
β_proj_max     = 256.0
# β schedule matching autotune run timing: double every ~75 iters
β_proj_schedule = [75, 75, 75, 75]
lr             = 1e-3
n_iters        = 600
save_every     = 10
save_dir       = joinpath(@__DIR__, "checkpoints_fi")
mkpath(save_dir)

rng = MersenneTwister(42)
ε_geom = rand(MersenneTwister(1234), Ny_d, Nx_d)   # same init as autotune run

# Build density filter
println("[optim] Building density filter (R=$filter_radius)..."); flush(stdout)
W_filter = build_density_filter(Ny_d, Nx_d, filter_radius)
println("[optim] Filter: $(nnz(W_filter)) nonzeros"); flush(stdout)

# β schedule
β_proj = β_proj_init
β_milestones = Int[]
let cum = 0, β_val = β_proj_init, idx = 1
    while β_val < β_proj_max
        interval = β_proj_schedule[min(idx, length(β_proj_schedule))]
        cum += interval
        β_val *= 2
        push!(β_milestones, cum)
        idx += 1
    end
end
println("[optim] β schedule: double at iters $β_milestones (β: $β_proj_init → $β_proj_max)"); flush(stdout)
println("[optim] n_iters=$n_iters  lr=$lr  n_episodes=$n_episodes  s0=$s0"); flush(stdout)

# Adam state
m_adam = zeros(Ny_d, Nx_d)
v_adam = zeros(Ny_d, Nx_d)
β1, β2, ε_adam_val = 0.9, 0.999, 1e-8
losses = Float64[]

for t in 1:n_iters
    global β_proj, ε_geom, m_adam, v_adam
    t_start = time()

    # β continuation
    if !isempty(β_milestones) && t == β_milestones[1] && β_proj < β_proj_max
        β_proj = min(2 * β_proj, β_proj_max)
        popfirst!(β_milestones)
        println("  [optim] β_proj → $β_proj"); flush(stdout)
    end

    # Resample x0 each iteration (stochastic, same as autotune run)
    x0_list = [x0_min .+ (x0_max - x0_min) .* rand(rng, dx) for _ in 1:n_episodes]

    loss, grad = end2end_fi(ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                            monitors_array, a_f_array, a_b_array,
                            x0_list, s0, nω, n_lat, GΔω, σ²;
                            W_filter=W_filter, β_proj=β_proj)
    push!(losses, loss)

    # Adam update
    m_adam .= β1 .* m_adam .+ (1 - β1) .* grad
    v_adam .= β2 .* v_adam .+ (1 - β2) .* grad .^ 2
    m̂ = m_adam ./ (1 - β1^t)
    v̂ = v_adam ./ (1 - β2^t)
    ε_geom .-= lr .* m̂ ./ (sqrt.(v̂) .+ ε_adam_val)
    clamp!(ε_geom, 0.0, 1.0)

    # Diagnostics
    g_abs    = abs.(grad)
    step_abs = abs.(lr .* m̂ ./ (sqrt.(v̂) .+ ε_adam_val))
    Δloss    = t > 1 ? loss - losses[end-1] : NaN
    elapsed  = round(time() - t_start, digits=1)
    println("iter $t/$n_iters  $(elapsed)s  neg_FI=$(round(loss, sigdigits=6))  Δ=$(round(Δloss, sigdigits=3))  " *
            "|grad| avg=$(round(mean(g_abs), sigdigits=3)) max=$(round(maximum(g_abs), sigdigits=3))  " *
            "|step| avg=$(round(mean(step_abs), sigdigits=3))  " *
            "ε=[$(round(minimum(ε_geom), digits=4)), $(round(maximum(ε_geom), digits=4))]  β=$β_proj")
    flush(stdout)

    # Save checkpoint
    if mod(t, save_every) == 0 || t == n_iters
        path = joinpath(save_dir, "eps_geom_step_$(lpad(t, 5, '0')).jls")
        serialize(path, (; step=t, loss, ε_geom=copy(ε_geom), losses=copy(losses),
                          β_proj=β_proj, filter_radius=filter_radius))
        println("  saved → $path"); flush(stdout)
    end
end

println("\n═══ FI Baseline Training Complete ═══")
println("Final neg_FI = $(round(losses[end], sigdigits=6))")
println("Checkpoint dir: $save_dir")
flush(stdout)
