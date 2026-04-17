#!/usr/bin/env julia
# autotune_dopt_inspect.jl — diagnostic snapshot for the D-optimal cron autotune agent.
#
# Usage:
#   julia --project=. autotune_dopt_inspect.jl [checkpoint.jls] [log_path]
#
# Defaults:
#   checkpoint = latest checkpoints_dopt/dopt_step_*.jls
#   log        = nohup_dopt.out
#
# Outputs:
#   - key=value stats to stdout (one per line)
#   - PNGs to autotune_dopt_snapshot/:
#       geometry.png        (raw | filtered | projected ε)
#       gray_zone.png       (|ε_proj - 0.5| heatmap)
#       loss_trajectory.png (-logdet vs iter, β doublings marked; linear y)
#       grad_hist.png       (log10 avg|grad_ε| per iter, last 300 iters)
#       sensor_trajectory.png (φ₁, φ₂ for each of N_steps over time)

using Serialization
using SparseArrays
using Plots
using Statistics
using Dates
using FFTW

# ── Filter & projection (identical to train_dopt_interactive.jl) ─────────────

function build_density_filter(Ny::Int, Nx::Int, R::Float64)
    R_int = ceil(Int, R)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for ix in 1:Nx, iy in 1:Ny
        k = (ix - 1) * Ny + iy
        wsum = 0.0
        local_entries = Tuple{Int, Float64}[]
        for jx in max(1, ix-R_int):min(Nx, ix+R_int)
            for jy in max(1, iy-R_int):min(Ny, iy+R_int)
                d = sqrt(Float64((ix-jx)^2 + (iy-jy)^2))
                if d <= R
                    w = R - d
                    push!(local_entries, ((jx-1)*Ny+jy, w))
                    wsum += w
                end
            end
        end
        for (l, w) in local_entries
            push!(rows, k); push!(cols, l); push!(vals, w/wsum)
        end
    end
    sparse(rows, cols, vals, Ny*Nx, Ny*Nx)
end

function project_density(ρ, β, η=0.5)
    t_eta = tanh(β * η)
    t_one = tanh(β * (1 - η))
    return (t_eta .+ tanh.(β .* (ρ .- η))) ./ (t_eta + t_one)
end

# ── Log parsing ──────────────────────────────────────────────────────────────

struct IterRow
    iter::Int
    loss::Float64        # -logdet(J_N); more-negative is better
    β::Float64
    grad_avg::Float64    # avg |grad_ε|
    grad_max::Float64    # max |grad_ε|
    grad_s1::Float64     # |grad_s[1]|
    grad_s2::Float64     # |grad_s[2]|
    grad_s3::Float64     # |grad_s[3]| (assumed N_steps=3)
    s_params::Vector{Float64}  # [φ₁¹,φ₂¹,φ₁²,φ₂²,φ₁³,φ₂³]
end

# Matches a train_dopt_interactive.jl iter line, e.g.:
# iter 12/2000  62.5s  -logdet=-99.077  |grad_ε| avg=0.0122 max=0.159  |grad_s| [0.0787, 0.209, 0.0376]  s=(1.982,2.779) (1.542,2.523) (2.34,2.633)  β=16.0
const ITER_RE = r"iter (\d+)/\d+\s+[\d.]+s\s+-logdet=([-\d.eE+]+)\s+\|grad_ε\|\s+avg=([\d.eE+\-]+)\s+max=([\d.eE+\-]+)\s+\|grad_s\|\s+\[([\d.eE+\-,\s]+)\]\s+s=\(([-\d.eE+]+),([-\d.eE+]+)\)\s+\(([-\d.eE+]+),([-\d.eE+]+)\)\s+\(([-\d.eE+]+),([-\d.eE+]+)\)\s+β=([\d.eE+\-]+)"

function parse_log(path::String; last_n_iters::Int=300)
    lines = isfile(path) ? readlines(path) : String[]
    rows = IterRow[]
    for ln in lines
        m = match(ITER_RE, ln)
        if m !== nothing
            try
                grad_s_parts = [parse(Float64, strip(p)) for p in split(m.captures[5], ',')]
                # Pad or truncate to 3 entries
                while length(grad_s_parts) < 3; push!(grad_s_parts, NaN); end
                s_params = [parse(Float64, m.captures[i]) for i in 6:11]
                push!(rows, IterRow(
                    parse(Int, m.captures[1]),
                    parse(Float64, m.captures[2]),
                    parse(Float64, m.captures[12]),
                    parse(Float64, m.captures[3]),
                    parse(Float64, m.captures[4]),
                    grad_s_parts[1], grad_s_parts[2], grad_s_parts[3],
                    s_params,
                ))
            catch e
                # skip malformed line
            end
        end
    end
    length(rows) > last_n_iters ? rows[end-last_n_iters+1:end] : rows
end

# ── Min-feature violations (isolated 1-pixel components, 4-connectivity) ─────

function count_isolated_pixels(bin::BitMatrix)
    Ny, Nx = size(bin)
    iso_black = 0
    iso_white = 0
    for i in 1:Ny, j in 1:Nx
        v = bin[i, j]
        alone = true
        if i > 1  && bin[i-1, j] == v; alone = false; end
        if alone && i < Ny && bin[i+1, j] == v; alone = false; end
        if alone && j > 1  && bin[i, j-1] == v; alone = false; end
        if alone && j < Nx && bin[i, j+1] == v; alone = false; end
        if alone
            v ? (iso_white += 1) : (iso_black += 1)
        end
    end
    iso_black, iso_white
end

# ── Main ─────────────────────────────────────────────────────────────────────

function find_latest_checkpoint(dir)
    isdir(dir) || return nothing
    files = filter(f -> startswith(f, "dopt_step_") && endswith(f, ".jls"), readdir(dir))
    isempty(files) && return nothing
    joinpath(dir, sort(files)[end])
end

function main()
    root = @__DIR__
    ckpt_path = length(ARGS) >= 1 ? ARGS[1] : find_latest_checkpoint(joinpath(root, "checkpoints_dopt"))
    log_path  = length(ARGS) >= 2 ? ARGS[2] : joinpath(root, "nohup_dopt.out")
    snap_dir  = joinpath(root, "autotune_dopt_snapshot")
    mkpath(snap_dir)

    println("# autotune_dopt_inspect @ ", Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
    println("checkpoint=$ckpt_path")
    println("log=$log_path")
    println("snapshot_dir=$snap_dir")

    if ckpt_path === nothing || !isfile(ckpt_path)
        println("error=no_checkpoint")
        return
    end

    ckpt = deserialize(ckpt_path)
    ε_raw = ckpt.ε_geom
    step  = ckpt.step
    loss_ckpt = ckpt.loss
    β_ckpt = hasproperty(ckpt, :β_proj) ? ckpt.β_proj : nothing
    R     = hasproperty(ckpt, :filter_radius) ? ckpt.filter_radius : 0.0
    s_list_ckpt = hasproperty(ckpt, :s_list) ? ckpt.s_list : nothing
    Ny, Nx = size(ε_raw)

    println("step=$step")
    println("loss_ckpt=$loss_ckpt")
    println("beta_ckpt=$(something(β_ckpt, -1.0))")
    println("filter_radius=$R")
    println("grid=$(Ny)x$(Nx)")

    if s_list_ckpt !== nothing
        s_str = join(["($(round(s[1],digits=3)),$(round(s[2],digits=3)))" for s in s_list_ckpt], " ")
        println("s_ckpt=$s_str")
    end

    # Apply filter + projection
    if R > 0 && β_ckpt !== nothing
        W = build_density_filter(Ny, Nx, Float64(R))
        ε_filt = reshape(W * vec(ε_raw), Ny, Nx)
        ε_proj = project_density(ε_filt, Float64(β_ckpt))

        n_bin = count(x -> x < 0.01 || x > 0.99, ε_proj)
        binary_pct = 100 * n_bin / length(ε_proj)
        println("binary_pct=", round(binary_pct, digits=3))

        gray_center = count(x -> 0.4 <= x <= 0.6, ε_proj)
        gray_any    = count(x -> 0.1 <= x <= 0.9, ε_proj)
        println("gray_center_pct=", round(100 * gray_center / length(ε_proj), digits=3))
        println("gray_any_pct=",    round(100 * gray_any    / length(ε_proj), digits=3))

        ε_bin = ε_proj .>= 0.5
        iso_black, iso_white = count_isolated_pixels(ε_bin)
        println("iso_black=$iso_black")
        println("iso_white=$iso_white")
        println("iso_total_pct=", round(100*(iso_black+iso_white)/length(ε_proj), digits=3))

        raw_sat = count(x -> x < 1e-3 || x > 1 - 1e-3, ε_raw)
        println("raw_saturated_pct=", round(100 * raw_sat / length(ε_raw), digits=3))

        vf_proj = mean(ε_proj)
        println("vol_frac_proj=", round(vf_proj, digits=4))

        # ── Dominant feature wavelength (Fourier radial peak) ────────────
        # Diagnostic for pixel-speckle pathology: λ near 1 px = checkerboard.
        F  = abs2.(fftshift(fft(ε_proj .- mean(ε_proj))))
        kyv = fftshift(collect(0:Ny-1)) .- div(Ny,2)
        kxv = fftshift(collect(0:Nx-1)) .- div(Nx,2)
        kmax = sqrt(Float64(div(Ny,2))^2 + Float64(div(Nx,2))^2)
        nb = 50
        hist = zeros(nb); cnt = zeros(Int, nb)
        for i in 1:Ny, j in 1:Nx
            kr = sqrt(Float64(kyv[i])^2 + Float64(kxv[j])^2)
            b = clamp(ceil(Int, kr/kmax*nb), 1, nb)
            hist[b] += F[i,j]; cnt[b] += 1
        end
        rad = hist ./ max.(cnt, 1)
        peak_bin = argmax(rad)
        peak_k = peak_bin/nb * kmax
        λ_px = peak_k > 0 ? Ny/peak_k : 0.0
        println("peak_feature_wavelength_px=", round(λ_px, digits=2))

        p1 = heatmap(ε_raw; aspect_ratio=:equal, color=:grays, clims=(0,1),
                     title="Raw ε", colorbar=false,
                     xlabel="x", ylabel="y")
        p2 = heatmap(ε_filt; aspect_ratio=:equal, color=:grays, clims=(0,1),
                     title="Filtered (R=$R)", colorbar=false,
                     xlabel="x", ylabel="y")
        p3 = heatmap(ε_proj; aspect_ratio=:equal, color=:grays, clims=(0,1),
                     title="Projected (β=$(β_ckpt))  $(round(binary_pct,digits=1))% binary",
                     colorbar=true, xlabel="x", ylabel="y")
        fig_geom = plot(p1, p2, p3; layout=(1,3), size=(1500, 500),
                        plot_title="step $step   -logdet=$(round(loss_ckpt,sigdigits=4))   λ_feat≈$(round(λ_px,digits=1))px")
        savefig(fig_geom, joinpath(snap_dir, "geometry.png"))

        gray_map = 1 .- 2 .* abs.(ε_proj .- 0.5)
        fig_gray = heatmap(gray_map; aspect_ratio=:equal, color=:hot, clims=(0, 1),
                           title="Gray zone  (bright = undecided)",
                           colorbar=true, xlabel="x", ylabel="y", size=(700, 650))
        savefig(fig_gray, joinpath(snap_dir, "gray_zone.png"))
    else
        println("binary_pct=NaN")
        println("note=no_projection_in_use")
    end

    # ── Log-derived trajectory stats ─────────────────────────────────────────
    rows = parse_log(log_path; last_n_iters=300)
    println("n_iters_parsed=$(length(rows))")

    if !isempty(rows)
        iters  = [r.iter for r in rows]
        losses = [r.loss for r in rows]
        βs     = [r.β    for r in rows]
        grads  = [r.grad_avg for r in rows]

        last_β = βs[end]
        println("current_beta_from_log=$last_β")
        iters_at_β = count(==(last_β), βs)
        println("iters_at_current_beta=$iters_at_β")

        ma20 = length(losses) >= 20 ? mean(losses[end-19:end]) : mean(losses)
        ma40 = length(losses) >= 40 ? mean(losses[end-39:end]) : mean(losses)
        prev_ma20 = length(losses) >= 40 ? mean(losses[end-39:end-20]) : NaN
        std20 = length(losses) >= 20 ? std(losses[end-19:end]) : NaN
        println("ma_loss_20=",      round(ma20, sigdigits=6))
        println("ma_loss_40=",      round(ma40, sigdigits=6))
        println("ma_loss_prev20=",  isnan(prev_ma20) ? "NaN" : round(prev_ma20, sigdigits=6))
        # For -logdet: lower (more negative) is better. Ratio < 1 means improving.
        println("ma_ratio_20_over_prev20=", isnan(prev_ma20) || prev_ma20 == 0 ? "NaN" : round(ma20/prev_ma20, sigdigits=4))
        println("loss_std_20=",     isnan(std20) ? "NaN" : round(std20, sigdigits=4))
        # "min" = most-negative = best for d-opt
        println("loss_min_recent=", round(minimum(losses[max(1,end-99):end]), sigdigits=4))
        println("loss_max_recent=", round(maximum(losses[max(1,end-99):end]), sigdigits=4))
        println("loss_best_ever=",  round(minimum(losses), sigdigits=4))

        last_doubling_iter = -1
        for i in length(βs):-1:2
            if βs[i] > βs[i-1]
                last_doubling_iter = iters[i]
                break
            end
        end
        println("last_doubling_iter=$last_doubling_iter")
        iters_since_doubling = last_doubling_iter > 0 ? iters[end] - last_doubling_iter : -1
        println("iters_since_doubling=$iters_since_doubling")

        grad_sat = count(g -> g < 1e-8, grads)
        println("grad_sat_pct_last_$(length(grads))=", round(100*grad_sat/length(grads), digits=2))
        println("grad_avg_recent=", round(mean(grads[max(1,end-19):end]), sigdigits=4))

        # Sensor gradient magnitudes
        gs1 = [r.grad_s1 for r in rows]
        gs2 = [r.grad_s2 for r in rows]
        gs3 = [r.grad_s3 for r in rows]
        println("grad_s_avg_recent=[",
                round(mean(gs1[max(1,end-19):end]), sigdigits=3), ",",
                round(mean(gs2[max(1,end-19):end]), sigdigits=3), ",",
                round(mean(gs3[max(1,end-19):end]), sigdigits=3), "]")

        # ── Loss trajectory PNG (linear y; -logdet can be negative) ──────
        β_change_idx = Int[]
        for i in 2:length(βs)
            if βs[i] != βs[i-1]
                push!(β_change_idx, i)
            end
        end
        fig_loss = plot(iters, losses; lw=1, legend=:topright, label="-logdet(J_N)",
                        xlabel="iter", ylabel="-logdet(J_N)  (lower=better)",
                        title="loss trajectory  (β doublings marked)",
                        size=(1000, 500))
        for idx in β_change_idx
            vline!(fig_loss, [iters[idx]]; ls=:dash, color=:red, alpha=0.5, label="")
            annotate!(fig_loss, iters[idx], maximum(losses), text("β=$(βs[idx])", 8, :red, :left))
        end
        if length(losses) >= 20
            ma_series = [mean(losses[max(1,i-19):i]) for i in 1:length(losses)]
            plot!(fig_loss, iters, ma_series; lw=2, color=:blue, label="MA-20")
        end
        savefig(fig_loss, joinpath(snap_dir, "loss_trajectory.png"))

        log_grads = log10.(max.(grads, 1e-16))
        fig_grad = histogram(log_grads; bins=40, legend=false,
                             xlabel="log10(avg |grad_ε|)", ylabel="count",
                             title="geom grad magnitude (last $(length(grads)) iters)",
                             size=(700, 400))
        savefig(fig_grad, joinpath(snap_dir, "grad_hist.png"))

        # ── Sensor trajectory ────────────────────────────────────────────
        s_mat = hcat([r.s_params for r in rows]...)   # 6 × n_rows
        fig_s = plot(xlabel="iter", ylabel="φ (rad)", title="sensor params φ₁,φ₂ per step (N=3)",
                     size=(1000, 500), legend=:outerright)
        labels = ["s1.φ1","s1.φ2","s2.φ1","s2.φ2","s3.φ1","s3.φ2"]
        for k in 1:6
            plot!(fig_s, iters, s_mat[k, :]; lw=1.5, label=labels[k])
        end
        for idx in β_change_idx
            vline!(fig_s, [iters[idx]]; ls=:dash, color=:red, alpha=0.4, label="")
        end
        savefig(fig_s, joinpath(snap_dir, "sensor_trajectory.png"))
    else
        println("note=no_parseable_iter_lines")
    end

    # Pending control file?
    control = joinpath(root, "checkpoints_dopt", "control.toml")
    println("control_pending=", isfile(control))

    applied = filter(f -> occursin(r"^control\.toml\.applied\.", f),
                     isdir(joinpath(root,"checkpoints_dopt")) ? readdir(joinpath(root,"checkpoints_dopt")) : String[])
    println("control_applied_count=", length(applied))
    if !isempty(applied)
        println("last_applied=", sort(applied)[end])
    end

    # Process alive?
    out_str = try
        read(`pgrep -af julia`, String)
    catch
        ""
    end
    any_dopt = occursin("train_dopt", out_str)
    interactive = occursin("train_dopt_interactive", out_str)
    non_interactive = any_dopt && !interactive
    println("julia_alive=$any_dopt")
    println("julia_controllable=$interactive")
    println("julia_running_non_interactive=$non_interactive")

    println("done=true")
end

main()
