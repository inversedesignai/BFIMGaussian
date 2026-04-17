#!/usr/bin/env julia
#
# Gradient check for the D-optimal loss and its analytical rrule.
# Tests:
#   1. Forward: -logdet(J_N) matches manual computation
#   2. s-gradient vs finite differences
#   3. c-gradient (directional) vs finite differences
#
# Usage:  julia test_dopt_grad.jl

push!(LOAD_PATH, @__DIR__)

using SimGeomBroadBand
using BFIMGaussian
using LinearAlgebra
using Random
using Zygote
using Serialization
using SparseArrays

BLAS.set_num_threads(1)

# ── Helpers from train_dopt.jl ───────────────────────────────────────────────

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

function unpack_c(c, nω)
    S0 = c[1:16nω]      .+ im .* c[16nω+1:32nω]
    S1 = c[32nω+1:48nω] .+ im .* c[48nω+1:64nω]
    S2 = c[64nω+1:80nω] .+ im .* c[80nω+1:96nω]
    S_arr      = [reshape(S0[(i-1)*16+1:i*16], 4, 4) for i in 1:nω]
    dSdn_arr   = [reshape(S1[(i-1)*16+1:i*16], 4, 4) for i in 1:nω]
    d2Sdn2_arr = [reshape(S2[(i-1)*16+1:i*16], 4, 4) for i in 1:nω]
    return S_arr, dSdn_arr, d2Sdn2_arr
end

function make_model(nω, n, GΔω, σ², αr)
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

# ── D-optimal forward (single x0) ───────────────────────────────────────────

function dopt_loss(x0, s_list, c, Σ0_inv, nω, n_lat, GΔω, σ²)
    mf = make_model(nω, n_lat, GΔω, σ², 0.0)
    N = length(s_list)
    F_list = [mf.fx(x0, s_list[k], c) for k in 1:N]
    J_N = copy(Σ0_inv)
    for k in 1:N
        J_N .+= (F_list[k]' * F_list[k]) ./ σ²
    end
    return -logdet(J_N)
end

# ── Analytical backward (single x0) ─────────────────────────────────────────

function dopt_grad(x0, s_list, c, Σ0_inv, nω, n_lat, GΔω, σ²)
    mf = make_model(nω, n_lat, GΔω, σ², 0.0)
    N = length(s_list)

    # Forward
    F_list = [mf.fx(x0, s_list[k], c) for k in 1:N]
    J_N = copy(Σ0_inv)
    for k in 1:N
        J_N .+= (F_list[k]' * F_list[k]) ./ σ²
    end
    J_N_inv = inv(J_N)

    # Backward
    S_arr, dSdn_arr, d2Sdn2_arr = unpack_c(c, nω)
    Δn = reshape(x0, n_lat, n_lat)

    grad_c = zeros(length(c))
    grad_s_list = [zeros(2) for _ in 1:N]

    for k in 1:N
        F_k = F_list[k]
        s_k = s_list[k]

        # F̄_k = -(2/σ²) F_k J_N⁻¹
        F_bar = (-2.0 / σ²) .* (F_k * J_N_inv)

        # s-gradient
        _, dF_dφ1 = jac_and_dirderiv_s(Δn, s_k[1], s_k[2], [1.0, 0.0],
                                        S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        _, dF_dφ2 = jac_and_dirderiv_s(Δn, s_k[1], s_k[2], [0.0, 1.0],
                                        S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
        grad_s_list[k][1] = sum(F_bar .* dF_dφ1)
        grad_s_list[k][2] = sum(F_bar .* dF_dφ2)

        # c-gradient via Zygote pullback
        _, pb_fx = Zygote.pullback(c_ -> mf.fx(x0, s_k, c_), c)
        (dc_k,) = pb_fx(F_bar)
        if dc_k !== nothing
            grad_c .+= dc_k
        end
    end

    return grad_c, grad_s_list
end

# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

n_lat = 2; n_core = 2.0; w = 0.5; n_geom = n_core
Lx = 10.0; Ly = 10.0; res = 50; n_pml = 24; R_target = 1e-8
port_offset = 15; mon_offset = 5; d_length = 6.0; d_width = 6.0
ωmin = 5.5; ωmax = 7.5; nω = 20
ω₀ = (ωmax + ωmin) / 2; Δω = (ωmax - ωmin) / 6
dx = n_lat^2; dy = 4 * n_lat; ds = 2; N_steps = 3
σ² = 1e-10
Σ0 = 7e-10 * Matrix{Float64}(I, dx, dx)
Σ0_inv = inv(Σ0)

println("Setting up geometry..."); flush(stdout)
ε_base, ω_array, Ls, Bs, grid_info, _, monitors_array =
    setup_4port_sweep(ωmin, ωmax, nω, n_core, w, d_length, d_width;
        Lx=Lx, Ly=Ly, res=res, n_pml=n_pml, R_target=R_target,
        port_offset=port_offset, mon_offset=mon_offset)

(a_f_array, a_b_array) = calibrate_straight_waveguide(
    ωmin, ωmax, nω, n_core, w;
    Lx=Lx, Ly=Ly, res=res, n_pml=n_pml, R_target=R_target,
    port_offset=port_offset, mon_offset=mon_offset)

δω = ω_array[2] - ω_array[1]
GΔω = @. exp(-(ω_array - ω₀)^2 / (2Δω^2)) / (Δω * sqrt(2π)) * δω
Ny_d = length(grid_info.design_iy); Nx_d = length(grid_info.design_ix)

# Compute c from a random geometry
ε_geom_test = rand(MersenneTwister(77), Ny_d, Nx_d)
println("Computing c from test geometry (FDFD forward)..."); flush(stdout)
c = sim_geom(ε_geom_test, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
             monitors_array, a_f_array, a_b_array)
println("Done. |c|=$(length(c))"); flush(stdout)

# Test point
x0 = [3e-5, 5e-5, 7e-5, 4e-5]
s_list = [[1.0, -0.5], [0.3, 2.1], [-1.5, 0.8]]

# ══════════════════════════════════════════════════════════════════════════════
# Test 1: Forward value sanity check
# ══════════════════════════════════════════════════════════════════════════════

println("\n═══ Test 1: Forward value ═══")
loss = dopt_loss(x0, s_list, c, Σ0_inv, nω, n_lat, GΔω, σ²)
println("  -logdet(J_N) = $loss")

# Manual check
mf = make_model(nω, n_lat, GΔω, σ², 0.0)
F1 = mf.fx(x0, s_list[1], c)
F2 = mf.fx(x0, s_list[2], c)
F3 = mf.fx(x0, s_list[3], c)
J_N = Σ0_inv + (F1'*F1 + F2'*F2 + F3'*F3) / σ²
loss_manual = -logdet(J_N)
println("  Manual:       $loss_manual")
println("  Match: $(abs(loss - loss_manual) < 1e-10 ? "PASS" : "FAIL ($(abs(loss - loss_manual)))")")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# Test 2: s-gradient vs finite differences
# ══════════════════════════════════════════════════════════════════════════════

println("\n═══ Test 2: s-gradient vs FD ═══")
grad_c, grad_s = dopt_grad(x0, s_list, c, Σ0_inv, nω, n_lat, GΔω, σ²)

ε_fd = 1e-6
for k in 1:N_steps
    for j in 1:ds
        s_plus = deepcopy(s_list); s_plus[k][j] += ε_fd
        s_minus = deepcopy(s_list); s_minus[k][j] -= ε_fd
        fd = (dopt_loss(x0, s_plus, c, Σ0_inv, nω, n_lat, GΔω, σ²) -
              dopt_loss(x0, s_minus, c, Σ0_inv, nω, n_lat, GΔω, σ²)) / (2ε_fd)
        ad = grad_s[k][j]
        rel_err = abs(fd - ad) / (abs(ad) + 1e-12)
        status = rel_err < 1e-4 ? "PASS" : "FAIL"
        println("  s[$k][$j]:  AD=$(round(ad, sigdigits=6))  FD=$(round(fd, sigdigits=6))  rel_err=$(round(rel_err, sigdigits=3))  $status")
    end
end
flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# Test 3: c-gradient (directional) vs finite differences
# ══════════════════════════════════════════════════════════════════════════════

println("\n═══ Test 3: c-gradient (directional) vs FD ═══")
v = randn(MersenneTwister(99), length(c)); v ./= norm(v)
ad_dir = dot(grad_c, v)

ε_fd_c = 1e-5
loss_p = dopt_loss(x0, s_list, c .+ ε_fd_c .* v, Σ0_inv, nω, n_lat, GΔω, σ²)
loss_m = dopt_loss(x0, s_list, c .- ε_fd_c .* v, Σ0_inv, nω, n_lat, GΔω, σ²)
fd_dir = (loss_p - loss_m) / (2ε_fd_c)
rel_err_c = abs(fd_dir - ad_dir) / (abs(ad_dir) + 1e-12)
status_c = rel_err_c < 1e-4 ? "PASS" : "FAIL"
println("  AD directional = $(round(ad_dir, sigdigits=6))")
println("  FD directional = $(round(fd_dir, sigdigits=6))")
println("  rel_err = $(round(rel_err_c, sigdigits=3))  $status_c")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# Test 4: Full end-to-end ε_geom gradient (directional) vs FD
# ══════════════════════════════════════════════════════════════════════════════

println("\n═══ Test 4: ε_geom gradient (directional) vs FD ═══")

# Full forward+backward through FDFD + D-optimal
W_filter = build_density_filter(Ny_d, Nx_d, 5.0)
β_proj = 16.0

fdfd_fn = ε_ -> begin
    ε_filt = reshape(W_filter * vec(ε_), Ny_d, Nx_d)
    ε_proj = project_density(ε_filt, β_proj)
    sim_geom(ε_proj, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
             monitors_array, a_f_array, a_b_array)
end

println("  Computing AD gradient (FDFD forward + pullback)..."); flush(stdout)
c_val, pb_c = Zygote.pullback(fdfd_fn, ε_geom_test)
# Recompute grad_c at this c (should match since we used same ε_geom_test)
grad_c2, _ = dopt_grad(x0, s_list, c_val, Σ0_inv, nω, n_lat, GΔω, σ²)
(grad_ε,) = pb_c(grad_c2)

v_ε = randn(MersenneTwister(101), Ny_d, Nx_d); v_ε ./= norm(v_ε)
ad_dir_ε = dot(grad_ε, v_ε)

println("  Computing FD (2 FDFD solves)..."); flush(stdout)
ε_fd_e = 1e-5
c_p = fdfd_fn(ε_geom_test .+ ε_fd_e .* v_ε)
c_m = fdfd_fn(ε_geom_test .- ε_fd_e .* v_ε)
loss_p_e = dopt_loss(x0, s_list, c_p, Σ0_inv, nω, n_lat, GΔω, σ²)
loss_m_e = dopt_loss(x0, s_list, c_m, Σ0_inv, nω, n_lat, GΔω, σ²)
fd_dir_ε = (loss_p_e - loss_m_e) / (2ε_fd_e)

rel_err_ε = abs(fd_dir_ε - ad_dir_ε) / (abs(ad_dir_ε) + 1e-12)
status_ε = rel_err_ε < 1e-3 ? "PASS" : "FAIL"
println("  AD directional = $(round(ad_dir_ε, sigdigits=6))")
println("  FD directional = $(round(fd_dir_ε, sigdigits=6))")
println("  rel_err = $(round(rel_err_ε, sigdigits=3))  $status_ε")
flush(stdout)

println("\n═══ All tests complete ═══"); flush(stdout)
