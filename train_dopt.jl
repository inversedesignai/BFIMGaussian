#!/usr/bin/env julia
#
# D-optimal joint optimization of geometry c and sensor sequence s_{1:N}.
#
# Objective: min_{ε_geom, s_{1:N}} E_x[-ln det J_N(x, s_{1:N}, c(ε_geom))]
#   where J_N = Σ₀⁻¹ + (1/σ²) Σ_k F_k(x, s_k, c)ᵀ F_k(x, s_k, c)
#
# No adaptive policy, no EKF in the loop, no inner optimization.
# Single-level optimization over d_c + N·d_s variables.
#
# The forward pass and custom rrule are fully analytical except for the
# c-VJP through jac_only (which uses Zygote.pullback on the lattice model).
# The s-gradient uses jac_and_dirderiv_s (analytical, no AD).
#
# Usage:  julia -p 20 train_dopt.jl

push!(LOAD_PATH, @__DIR__)

ENV["JULIA_DEBUG"] = ""
flush(stdout)

using Distributed

println("[startup] Adding worker processes..."); flush(stdout)
if nworkers() == 1
    addprocs(20)
end
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
    using ChainRulesCore
    BLAS.set_num_threads(1)
end

println("[startup] Loading main-process packages..."); flush(stdout)
using Serialization
using Statistics
using SparseArrays
println("[startup] All packages loaded."); flush(stdout)

# ── Density filter + projection ──────────────────────────────────────────────

function build_density_filter(Ny::Int, Nx::Int, R::Float64)
    R_int = ceil(Int, R)
    rows = Int[]; cols = Int[]; vals = Float64[]
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
            push!(rows, k); push!(cols, l); push!(vals, w / wsum)
        end
    end
    return sparse(rows, cols, vals, Ny * Nx, Ny * Nx)
end

function project_density(ρ, β, η=0.5)
    t_eta = tanh(β * η); t_one = tanh(β * (1 - η))
    return (t_eta .+ tanh.(β .* (ρ .- η))) ./ (t_eta + t_one)
end

# ── Unpack c + model ─────────────────────────────────────────────────────────

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
        Δn = reshape(x, n, n)
        powers_only(Δn, s[1], s[2], S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
    end
    function fx(x, s, c)
        S_arr, dSdn_arr, d2Sdn2_arr = unpack_c(c, nω)
        Δn = reshape(x, n, n)
        jac_only(Δn, s[1], s[2], S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
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

# ══════════════════════════════════════════════════════════════════════════════
# D-optimal loss and analytical rrule
# ══════════════════════════════════════════════════════════════════════════════

# Worker function: forward pass for one episode.
# Returns (loss_i, F_list, J_N, J_N_inv) — cached for the backward pass.
@everywhere function _dopt_forward(x0, s_list, c, Σ0_inv, nω, n_lat, GΔω, σ²)
    mf = make_model(nω, n_lat, GΔω, σ², 0.0)
    N = length(s_list)
    dx = n_lat^2

    # Compute F_k for each step
    F_list = [mf.fx(x0, s_list[k], c) for k in 1:N]

    # Accumulate J_N = Σ₀⁻¹ + Σ_k F_kᵀ F_k / σ²
    J_N = copy(Σ0_inv)
    for k in 1:N
        J_N .+= (F_list[k]' * F_list[k]) ./ σ²
    end

    loss_i = -logdet(J_N)
    J_N_inv = inv(J_N)

    return (; loss=loss_i, F_list, J_N, J_N_inv)
end

# Worker function: backward pass for one episode.
# Given cached forward quantities, computes gradients w.r.t. c and s_list.
# Uses:
#   - Zygote.pullback through model.fx for c-gradient
#   - Analytical jac_and_dirderiv_s for s-gradient (no AD)
@everywhere function _dopt_backward(x0, s_list, c, fwd, nω, n_lat, GΔω, σ²)
    mf = make_model(nω, n_lat, GΔω, σ², 0.0)
    N = length(s_list)
    J_N_inv = fwd.J_N_inv
    F_list = fwd.F_list

    grad_c = zeros(length(c))
    grad_s_list = [zeros(2) for _ in 1:N]

    # Unpack once for all steps (used by jac_and_dirderiv_s)
    S_arr, dSdn_arr, d2Sdn2_arr = unpack_c(c, nω)
    Δn = reshape(x0, n_lat, n_lat)

    for k in 1:N
        F_k = F_list[k]
        s_k = s_list[k]

        # Cotangent of F_k from -logdet:
        #   ∂(-logdet J_N)/���F_k = -(2/σ²) F_k J_N⁻¹
        # (from ∂logdet(A)/∂A = A⁻¹, chain rule through F_kᵀ F_k)
        F_bar = (-2.0 / σ²) .* (F_k * J_N_inv)  # dy × dx, same shape as F_k

        # ── s-gradient (analytical via jac_and_dirderiv_s) ────────────────
        # ∂F/∂φ₁ and ∂F/∂φ₂ via canonical basis vectors
        _, dF_dφ1 = jac_and_dirderiv_s(Δn, s_k[1], s_k[2], [1.0, 0.0],
                                        S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        _, dF_dφ2 = jac_and_dirderiv_s(Δn, s_k[1], s_k[2], [0.0, 1.0],
                                        S_arr, dSdn_arr, d2Sdn2_arr, GΔω)

        # s̄_k[j] = Σ_{i,l} F̄_k[i,l] · ∂F_k[i,l]/∂s_k[j]
        grad_s_list[k][1] = sum(F_bar .* dF_dφ1)
        grad_s_list[k][2] = sum(F_bar .* dF_dφ2)

        # ── c-gradient (Zygote pullback through model.fx) ────────────────
        _, pb_fx = Zygote.pullback(c_ -> mf.fx(x0, s_k, c_), c)
        (dc_k,) = pb_fx(F_bar)
        if dc_k !== nothing
            grad_c .+= dc_k
        end
    end

    return (; grad_c, grad_s_list)
end

# Combined forward+backward for one episode (for pmap).
@everywhere function _dopt_episode(arg)
    x0, s_list, c, Σ0_inv, nω, n_lat, GΔω, σ² = arg
    fwd = _dopt_forward(x0, s_list, c, Σ0_inv, nω, n_lat, GΔω, σ²)
    bwd = _dopt_backward(x0, s_list, c, fwd, nω, n_lat, GΔω, σ²)
    return (; loss=fwd.loss, grad_c=bwd.grad_c, grad_s_list=bwd.grad_s_list)
end

# Batched D-optimal loss + gradient over episodes (pmap).
function batch_dopt_grad(x0_list, s_list, c, Σ0_inv, nω, n_lat, GΔω, σ²)
    n = length(x0_list)
    args = [(x0_list[i], s_list, c, Σ0_inv, nω, n_lat, GΔω, σ²) for i in 1:n]
    results = pmap(_dopt_episode, args)

    mean_loss = sum(r.loss for r in results) / n
    mean_grad_c = sum(r.grad_c for r in results) / n
    N = length(s_list)
    mean_grad_s = [sum(r.grad_s_list[k] for r in results) / n for k in 1:N]

    return mean_loss, mean_grad_c, mean_grad_s
end

# End-to-end: ε_geom → filter → project → sim_geom → c → D-optimal loss.
function end2end_dopt(ε_raw, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                      monitors_array, a_f_array, a_b_array,
                      x0_list, s_list, Σ0_inv, nω, n_lat, GΔω, σ²;
                      W_filter=nothing, β_proj=nothing, η_proj=0.5)
    Ny_d = length(grid_info.design_iy)
    Nx_d = length(grid_info.design_ix)

    println("  [dopt] FDFD forward..."); flush(stdout)
    fdfd = ε_ -> begin
        ε_filt = W_filter !== nothing ? reshape(W_filter * vec(ε_), Ny_d, Nx_d) : ε_
        ε_proj = β_proj !== nothing ? project_density(ε_filt, β_proj, η_proj) : ε_filt
        sim_geom(ε_proj, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                 monitors_array, a_f_array, a_b_array)
    end
    c, pb_c = Zygote.pullback(fdfd, ε_raw)

    println("  [dopt] D-optimal gradients ($(length(x0_list)) episodes, $(length(s_list)) steps)..."); flush(stdout)
    loss, grad_c, grad_s = batch_dopt_grad(x0_list, s_list, c, Σ0_inv, nω, n_lat, GΔω, σ²)

    println("  [dopt] FDFD backward..."); flush(stdout)
    (grad_ε_raw,) = pb_c(grad_c)

    println("  [dopt] Done. -logdet(J_N)=$(round(loss, sigdigits=6))"); flush(stdout)
    return loss, grad_ε_raw, grad_s
end

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
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
Σ0_inv     = inv(Σ0)
σ²         = 1e-10

x0_min     = 1e-5
x0_max     = 1e-4
n_episodes = 20

# ── Geometry and calibration setup ───────────────────────────────────────────

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

# ── Optimization setup ───────────────────────────────────────────────────────

filter_radius  = 5.0
β_proj_init    = 16.0
β_proj_max     = 256.0
β_proj_schedule = [75, 75, 75, 75]

lr_geom        = 1e-3     # Adam lr for ε_geom
lr_s           = 1e-2     # Adam lr for sensor params (fewer params, smoother landscape)
n_iters        = 600
save_every     = 10
save_dir       = joinpath(@__DIR__, "checkpoints_dopt")
mkpath(save_dir)

rng = MersenneTwister(42)
ε_geom = rand(MersenneTwister(1234), Ny_d, Nx_d)   # same init as other runs

# Initialize sensor params: random in [-π, π]
rng_s = MersenneTwister(7777)
s_list = [2π .* rand(rng_s, ds) .- π for _ in 1:N_steps]
println("[init] s₁=$(round.(s_list[1], digits=3)), s₂=$(round.(s_list[2], digits=3)), s₃=$(round.(s_list[3], digits=3))")
flush(stdout)

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
println("[optim] n_iters=$n_iters  lr_geom=$lr_geom  lr_s=$lr_s  N_steps=$N_steps  n_episodes=$n_episodes")
flush(stdout)

# Adam states: geometry
m_geom = zeros(Ny_d, Nx_d)
v_geom = zeros(Ny_d, Nx_d)

# Adam states: sensor params (one per step)
m_s = [zeros(ds) for _ in 1:N_steps]
v_s = [zeros(ds) for _ in 1:N_steps]

β1, β2, ε_adam = 0.9, 0.999, 1e-8
losses = Float64[]

# ── Training loop ────────────────────────────────────────────────────────────

for t in 1:n_iters
    global β_proj, ε_geom, m_geom, v_geom, s_list, m_s, v_s
    t_start = time()

    # β continuation
    if !isempty(β_milestones) && t == β_milestones[1] && β_proj < β_proj_max
        β_proj = min(2 * β_proj, β_proj_max)
        popfirst!(β_milestones)
        println("  [optim] β_proj → $β_proj"); flush(stdout)
    end

    # Resample x0 each iteration
    x0_list = [x0_min .+ (x0_max - x0_min) .* rand(rng, dx) for _ in 1:n_episodes]

    loss, grad_ε, grad_s = end2end_dopt(
        ε_geom, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
        monitors_array, a_f_array, a_b_array,
        x0_list, s_list, Σ0_inv, nω, n_lat, GΔω, σ²;
        W_filter=W_filter, β_proj=β_proj)
    push!(losses, loss)

    # ── Adam update: geometry ─────────────────────────────────────────────
    m_geom .= β1 .* m_geom .+ (1 - β1) .* grad_ε
    v_geom .= β2 .* v_geom .+ (1 - β2) .* grad_ε .^ 2
    m̂_g = m_geom ./ (1 - β1^t)
    v̂_g = v_geom ./ (1 - β2^t)
    ε_geom .-= lr_geom .* m̂_g ./ (sqrt.(v̂_g) .+ ε_adam)
    clamp!(ε_geom, 0.0, 1.0)

    # ── Adam update: sensor params ────────────────────────────────────────
    for k in 1:N_steps
        m_s[k] .= β1 .* m_s[k] .+ (1 - β1) .* grad_s[k]
        v_s[k] .= β2 .* v_s[k] .+ (1 - β2) .* grad_s[k] .^ 2
        m̂_s = m_s[k] ./ (1 - β1^t)
        v̂_s = v_s[k] ./ (1 - β2^t)
        s_list[k] .-= lr_s .* m̂_s ./ (sqrt.(v̂_s) .+ ε_adam)
        # Wrap to [-π, π] (BFIM is 2π-periodic)
        s_list[k] .= mod.(s_list[k] .+ π, 2π) .- π
    end

    # ── Diagnostics ───────────────────────────────────────────────────────
    g_abs    = abs.(grad_ε)
    elapsed  = round(time() - t_start, digits=1)
    s_str    = join(["($(round(s[1],digits=3)),$(round(s[2],digits=3)))" for s in s_list], " ")
    println("iter $t/$n_iters  $(elapsed)s  -logdet=$(round(loss, sigdigits=6))  " *
            "|grad_ε| avg=$(round(mean(g_abs), sigdigits=3)) max=$(round(maximum(g_abs), sigdigits=3))  " *
            "|grad_s| $(round.(norm.(grad_s), sigdigits=3))  " *
            "s=$s_str  β=$β_proj")
    flush(stdout)

    # ── Save checkpoint ───────────────────────────────────────────────────
    if mod(t, save_every) == 0 || t == n_iters
        path = joinpath(save_dir, "dopt_step_$(lpad(t, 5, '0')).jls")
        serialize(path, (; step=t, loss, ε_geom=copy(ε_geom), s_list=deepcopy(s_list),
                          losses=copy(losses), β_proj=β_proj, filter_radius=filter_radius))
        println("  saved → $path"); flush(stdout)
    end
end

println("\n═══ D-Optimal Training Complete ═══")
println("Final -logdet(J_N) = $(round(losses[end], sigdigits=6))")
println("Final s: $(s_list)")
println("Checkpoint dir: $save_dir")
flush(stdout)
