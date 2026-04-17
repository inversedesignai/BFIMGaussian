#!/usr/bin/env julia
#
# EKF performance evaluation: optimized vs unoptimized geometry
#
# Loads the final checkpoint from the autotune run, applies density filter +
# projection, computes S-matrix coefficients via FDFD, and runs Monte Carlo
# EKF episodes.  Repeats with a random (unoptimized) geometry for comparison.
#
# Usage:  julia eval_ekf_perf.jl
#         julia eval_ekf_perf.jl path/to/checkpoint.jls

push!(LOAD_PATH, @__DIR__)

ENV["JULIA_DEBUG"] = ""
flush(stdout)

using SimGeomBroadBand
using BFIMGaussian
using LinearAlgebra
using Random
using Serialization
using Statistics
using SparseArrays
using Optim
using ForwardDiff
using ADTypes: AutoForwardDiff
BLAS.set_num_threads(1)

# ── Density filter + projection (copied from PhEnd2End.jl) ──────────────────

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

# ── sim_geom: FDFD forward pass → packed coefficient vector c ────────────────

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

# ── Unpack c (same as PhEnd2End.jl) ─────────────────────────────────────────

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

# ── EKF performance evaluation ───────────────────────────────────────────────

function eval_ekf_perf(c, label, mf, μ0, Σ0, σ², dx, dy, x0_min, x0_max;
                       n_mc=200, N_eval=10, seed=999)
    rng_perf = MersenneTwister(seed)

    println("\n═══ EKF Performance: $label ═══")
    println("  n_mc=$n_mc episodes, N_eval=$N_eval steps")
    println("  σ²=$σ²  x0 ∈ [$x0_min, $x0_max]  Σ0=$(Σ0[1,1])·I")
    flush(stdout)

    errs_ekf  = zeros(N_eval+1, n_mc)
    errs_mle  = zeros(N_eval+1, n_mc)
    tr_Σs     = zeros(N_eval+1, n_mc)
    mahal_sq  = zeros(N_eval+1, n_mc)
    bfim_vals = zeros(N_eval, n_mc)
    loss_vals = zeros(n_mc)  # episode loss (sum relative squared error)

    for ep in 1:n_mc
        x0_ep = x0_min .+ (x0_max - x0_min) .* rand(rng_perf, dx)
        noise_ep = [sqrt(σ²) .* randn(rng_perf, dy) for _ in 1:N_eval]

        μ, Σ = copy(μ0), copy(Σ0)

        ys_so_far = Vector{Float64}[]
        ss_so_far = Vector{Float64}[]

        err0 = norm((μ .- x0_ep) ./ x0_ep)
        errs_ekf[1, ep] = err0
        errs_mle[1, ep] = err0
        tr_Σs[1, ep]    = tr(Σ)
        Σ_inv = inv(Σ)
        d = μ - x0_ep
        mahal_sq[1, ep] = dot(d, Σ_inv * d)

        for k in 1:N_eval
            sk = get_sopt(c, μ, mf)
            F  = mf.fx(μ, sk, c)
            bfim_vals[k, ep] = sum(abs2, F) / σ²

            yk = mf.f(x0_ep, sk, c) + noise_ep[k]
            μ, Σ = ekf_update(μ, Σ, yk, sk, c, mf)

            errs_ekf[k+1, ep] = norm((μ .- x0_ep) ./ x0_ep)
            tr_Σs[k+1, ep]    = tr(Σ)
            Σ_inv_k = inv(Σ + 1e-12 * Matrix{Float64}(I, dx, dx))
            d_k = μ - x0_ep
            mahal_sq[k+1, ep] = dot(d_k, Σ_inv_k * d_k)

            # MLE
            push!(ys_so_far, yk)
            push!(ss_so_far, sk)
            local ys_k = copy(ys_so_far)
            local ss_k = copy(ss_so_far)
            mle_obj = x -> begin
                total = zero(eltype(x))
                for j in eachindex(ys_k)
                    pred = mf.f(x, ss_k[j], c)
                    total += sum(abs2, ys_k[j] .- pred)
                end
                total / σ²
            end
            mle_result = optimize(mle_obj, copy(μ0), LBFGS(),
                                  Optim.Options(g_tol=1e-12, iterations=1000),
                                  inplace=false, autodiff=AutoForwardDiff())
            x_mle = Optim.minimizer(mle_result)
            errs_mle[k+1, ep] = norm((x_mle .- x0_ep) ./ x0_ep)
        end

        # Episode loss (same as training: sum of relative squared errors at each step)
        loss_vals[ep] = sum(((μ .- x0_ep) ./ x0_ep).^2)

        if mod(ep, 50) == 0
            println("  ... completed $ep/$n_mc episodes"); flush(stdout)
        end
    end

    # ── Report ────────────────────────────────────────────────────────────────
    println("\n  Step | EKF rel_err mean±std   | MLE rel_err mean±std   | tr(Σ) mean           | Mahal² | BFIM tr")
    println("  -----|------------------------|------------------------|----------------------|--------|--------")
    flush(stdout)
    for k in 0:N_eval
        ekf_m = mean(errs_ekf[k+1, :]);  ekf_s = std(errs_ekf[k+1, :])
        mle_m = mean(errs_mle[k+1, :]);  mle_s = std(errs_mle[k+1, :])
        t_m   = mean(tr_Σs[k+1, :])
        mh_m  = mean(mahal_sq[k+1, :])
        bfim_str = k == 0 ? "(prior)" : "$(round(mean(bfim_vals[k, :]), sigdigits=4))"
        println("  $(lpad(k,4)) | $(rpad(round(ekf_m, sigdigits=4), 10)) ± $(lpad(round(ekf_s, sigdigits=3), 8)) " *
                "| $(rpad(round(mle_m, sigdigits=4), 10)) ± $(lpad(round(mle_s, sigdigits=3), 8)) " *
                "| $(rpad(round(t_m, sigdigits=4), 20)) " *
                "| $(rpad(round(mh_m, sigdigits=4), 6)) " *
                "| $bfim_str")
        flush(stdout)
    end

    prior_err = mean(errs_ekf[1, :])
    final_ekf = mean(errs_ekf[end, :])
    final_mle = mean(errs_mle[end, :])
    final_trΣ = mean(tr_Σs[end, :])
    prior_trΣ = mean(tr_Σs[1, :])
    final_mahal = mean(mahal_sq[end, :])
    mean_loss = mean(loss_vals)

    println("\n  Summary ($label):")
    println("    Prior     rel_err = $(round(prior_err, sigdigits=4))")
    println("    EKF final rel_err = $(round(final_ekf, sigdigits=4))  ($(round(final_ekf/prior_err, sigdigits=3))× prior)")
    println("    MLE final rel_err = $(round(final_mle, sigdigits=4))  ($(round(final_mle/prior_err, sigdigits=3))× prior)")
    println("    EKF vs MLE: EKF is $(round(final_ekf/final_mle, sigdigits=3))× MLE")
    println("    tr(Σ) reduction: $(round(final_trΣ/prior_trΣ, sigdigits=3))×")
    println("    Final Mahalanobis² mean: $(round(final_mahal, sigdigits=4))  (expect ≈ $dx if calibrated)")
    println("    Episode loss (mean): $(round(mean_loss, sigdigits=4))")
    flush(stdout)

    # Calibration check
    if final_mahal > 3 * dx
        println("    WARNING: EKF is OVERCONFIDENT (Mahalanobis² >> dx=$dx)")
    elseif final_mahal < dx / 3
        println("    WARNING: EKF is UNDERCONFIDENT (Mahalanobis² << dx=$dx)")
    else
        println("    EKF calibration looks reasonable (Mahalanobis² ≈ dx=$dx)")
    end
    flush(stdout)

    return (; prior_err, final_ekf, final_mle, final_trΣ, prior_trΣ, final_mahal, mean_loss,
              errs_ekf, errs_mle, tr_Σs, mahal_sq, bfim_vals)
end

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

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

ω₀  = (ωmax + ωmin) / 2
Δω  = (ωmax - ωmin) / 6

dy  = 4 * n_lat
dx  = n_lat^2
ds  = 2

N_steps = 3
μ0      = fill((1e-5 + 1e-4)/2, dx)
Σ0      = 7e-10 * Matrix{Float64}(I, dx, dx)
σ²      = 1e-10
αr      = 0.0

x0_min  = 1e-5
x0_max  = 1e-4

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

mf = make_model(nω, n_lat, GΔω, σ², αr)

# ── Load optimized geometry ──────────────────────────────────────────────────
ckpt_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "checkpoints", "eps_geom_step_00580.jls")
println("\n[load] Loading checkpoint: $ckpt_path"); flush(stdout)
ckpt = deserialize(ckpt_path)
println("[load] Step=$(ckpt.step), loss=$(round(ckpt.loss, sigdigits=4)), β=$(ckpt.β_proj), R=$(ckpt.filter_radius)")
flush(stdout)

# Apply density filter + projection (same pipeline as training)
R_filter = ckpt.filter_radius
β_proj   = ckpt.β_proj
ε_raw    = ckpt.ε_geom

println("[load] Building density filter (R=$R_filter)..."); flush(stdout)
W_filter = build_density_filter(Ny_d, Nx_d, R_filter)
ε_filt   = reshape(W_filter * vec(ε_raw), Ny_d, Nx_d)
ε_proj   = project_density(ε_filt, β_proj)
println("[load] Projected geometry: binary_pct=$(round(100*mean((ε_proj .< 0.01) .| (ε_proj .> 0.99)), sigdigits=4))%")
flush(stdout)

println("\n[fdfd] Computing S-matrix coefficients for OPTIMIZED geometry..."); flush(stdout)
t0 = time()
c_opt = sim_geom(ε_proj, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                 monitors_array, a_f_array, a_b_array)
println("[fdfd] Done in $(round(time()-t0, digits=1))s. |c|=$(length(c_opt))"); flush(stdout)

# ── Random (unoptimized) geometry ────────────────────────────────────────────
println("\n[fdfd] Computing S-matrix coefficients for RANDOM geometry..."); flush(stdout)
ε_rand = rand(MersenneTwister(1234), Ny_d, Nx_d)
# Apply same filter + projection so comparison is fair (same pipeline)
ε_rand_filt = reshape(W_filter * vec(ε_rand), Ny_d, Nx_d)
ε_rand_proj = project_density(ε_rand_filt, β_proj)
println("[fdfd] Random projected binary_pct=$(round(100*mean((ε_rand_proj .< 0.01) .| (ε_rand_proj .> 0.99)), sigdigits=4))%")
flush(stdout)

t0 = time()
c_rand = sim_geom(ε_rand_proj, n_geom, ε_base, ω_array, Ls, Bs, grid_info,
                  monitors_array, a_f_array, a_b_array)
println("[fdfd] Done in $(round(time()-t0, digits=1))s."); flush(stdout)

# ── Evaluate both ────────────────────────────────────────────────────────────
n_mc   = 200
N_eval = 10

println("\n" * "="^80)
println("Running $n_mc Monte Carlo episodes, $N_eval EKF steps each")
println("="^80)

res_opt  = eval_ekf_perf(c_opt,  "OPTIMIZED (step=$(ckpt.step), β=$(ckpt.β_proj))",
                         mf, μ0, Σ0, σ², dx, dy, x0_min, x0_max;
                         n_mc=n_mc, N_eval=N_eval)

res_rand = eval_ekf_perf(c_rand, "RANDOM (unoptimized)",
                         mf, μ0, Σ0, σ², dx, dy, x0_min, x0_max;
                         n_mc=n_mc, N_eval=N_eval)

# ── Comparison ───────────────────────────────────────────────────────────────
println("\n" * "="^80)
println("COMPARISON: OPTIMIZED vs RANDOM")
println("="^80)

println("\n  Metric                    | Optimized        | Random           | Ratio (Rand/Opt)")
println("  --------------------------|------------------|------------------|------------------")

function fmt(x); return rpad(round(x, sigdigits=4), 16); end

println("  EKF final rel_err (mean)  | $(fmt(res_opt.final_ekf)) | $(fmt(res_rand.final_ekf)) | $(round(res_rand.final_ekf/res_opt.final_ekf, sigdigits=3))×")
println("  MLE final rel_err (mean)  | $(fmt(res_opt.final_mle)) | $(fmt(res_rand.final_mle)) | $(round(res_rand.final_mle/res_opt.final_mle, sigdigits=3))×")
println("  Episode loss (mean)       | $(fmt(res_opt.mean_loss)) | $(fmt(res_rand.mean_loss)) | $(round(res_rand.mean_loss/res_opt.mean_loss, sigdigits=3))×")
println("  tr(Σ) final (mean)        | $(fmt(res_opt.final_trΣ)) | $(fmt(res_rand.final_trΣ)) | $(round(res_rand.final_trΣ/res_opt.final_trΣ, sigdigits=3))×")
println("  tr(Σ) reduction           | $(fmt(res_opt.final_trΣ/res_opt.prior_trΣ)) | $(fmt(res_rand.final_trΣ/res_rand.prior_trΣ)) |")
println("  Mahalanobis² final        | $(fmt(res_opt.final_mahal)) | $(fmt(res_rand.final_mahal)) |")
println("  BFIM tr step 1 (mean)     | $(fmt(mean(res_opt.bfim_vals[1,:]))) | $(fmt(mean(res_rand.bfim_vals[1,:]))) | $(round(mean(res_opt.bfim_vals[1,:])/mean(res_rand.bfim_vals[1,:]), sigdigits=3))×")

# Per-step comparison
println("\n  Per-step EKF relative error comparison:")
println("  Step | Optimized          | Random             | Improvement")
println("  -----|--------------------|--------------------|------------")
for k in 0:N_eval
    ek_o = mean(res_opt.errs_ekf[k+1, :])
    ek_r = mean(res_rand.errs_ekf[k+1, :])
    imp = round(ek_r / ek_o, sigdigits=3)
    println("  $(lpad(k,4)) | $(rpad(round(ek_o, sigdigits=4), 18)) | $(rpad(round(ek_r, sigdigits=4), 18)) | $(imp)×")
end
flush(stdout)

println("\n═══ Evaluation complete ═══"); flush(stdout)
