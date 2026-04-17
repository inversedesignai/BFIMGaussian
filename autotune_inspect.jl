#!/usr/bin/env julia
# autotune_inspect.jl — diagnostic snapshot for the cron-driven autotune agent.
#
# Usage:
#   julia --project=. autotune_inspect.jl [checkpoint.jls] [log_path]
#
# Defaults:
#   checkpoint = latest checkpoints/eps_geom_step_*.jls
#   log        = nohup.out
#
# Outputs:
#   - key=value stats to stdout (one per line)
#   - PNGs to autotune_snapshot/:
#       geometry.png        (raw | filtered | projected ε)
#       gray_zone.png       (|ε_proj - 0.5| heatmap — where the fight still is)
#       loss_trajectory.png (log-scale loss vs iter, β doublings marked)
#       grad_hist.png       (log10 avg|grad| per iter, last 300 iters)

using Serialization
using SparseArrays
using Plots
using Statistics
using Dates

# ── Filter & projection (identical to PhEnd2End_interactive.jl) ──────────────

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
    loss::Float64
    β::Float64
    grad_avg::Float64
    grad_max::Float64
    step_avg::Float64
end

# Matches:
# iter 676/1000000  71.6s  loss=0.000186998  Δloss=3.4e-5  |grad| min=1.53e-10 avg=6.14e-6 max=6.05e-5  |step| avg=0.000199 max=0.000479  ε range=[0.0, 1.0]  β_proj=2.0
const ITER_RE = r"iter (\d+)/\d+.*?loss=([^\s]+).*?\|grad\| min=[^\s]+ avg=([^\s]+) max=([^\s]+).*?\|step\| avg=([^\s]+).*?β_proj=([^\s]+)"

function parse_log(path::String; last_n_iters::Int=300)
    lines = isfile(path) ? readlines(path) : String[]
    rows = IterRow[]
    for ln in lines
        m = match(ITER_RE, ln)
        if m !== nothing
            try
                push!(rows, IterRow(
                    parse(Int, m.captures[1]),
                    parse(Float64, m.captures[2]),
                    parse(Float64, m.captures[6]),
                    parse(Float64, m.captures[3]),
                    parse(Float64, m.captures[4]),
                    parse(Float64, m.captures[5]),
                ))
            catch
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
        # neighbors (4-connectivity)
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
    files = filter(f -> startswith(f, "eps_geom_step_") && endswith(f, ".jls"), readdir(dir))
    isempty(files) && return nothing
    joinpath(dir, sort(files)[end])
end

function main()
    root = @__DIR__
    ckpt_path = length(ARGS) >= 1 ? ARGS[1] : find_latest_checkpoint(joinpath(root, "checkpoints"))
    log_path  = length(ARGS) >= 2 ? ARGS[2] : joinpath(root, "nohup.out")
    snap_dir  = joinpath(root, "autotune_snapshot")
    mkpath(snap_dir)

    println("# autotune_inspect @ ", Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
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
    Ny, Nx = size(ε_raw)

    println("step=$step")
    println("loss_ckpt=$loss_ckpt")
    println("beta_ckpt=$(something(β_ckpt, -1.0))")
    println("filter_radius=$R")
    println("grid=$(Ny)x$(Nx)")

    # Apply filter + projection
    if R > 0 && β_ckpt !== nothing
        W = build_density_filter(Ny, Nx, Float64(R))
        ε_filt = reshape(W * vec(ε_raw), Ny, Nx)
        ε_proj = project_density(ε_filt, Float64(β_ckpt))

        n_bin = count(x -> x < 0.01 || x > 0.99, ε_proj)
        binary_pct = 100 * n_bin / length(ε_proj)
        println("binary_pct=", round(binary_pct, digits=3))

        # Gray-zone breakdown
        gray_center = count(x -> 0.4 <= x <= 0.6, ε_proj)
        gray_any    = count(x -> 0.1 <= x <= 0.9, ε_proj)
        println("gray_center_pct=", round(100 * gray_center / length(ε_proj), digits=3))
        println("gray_any_pct=",    round(100 * gray_any    / length(ε_proj), digits=3))

        # Min-feature violations
        ε_bin = ε_proj .>= 0.5
        iso_black, iso_white = count_isolated_pixels(ε_bin)
        println("iso_black=$iso_black")
        println("iso_white=$iso_white")
        println("iso_total_pct=", round(100*(iso_black+iso_white)/length(ε_proj), digits=3))

        # Raw saturation (how "committed" is raw ε)
        raw_sat = count(x -> x < 1e-3 || x > 1 - 1e-3, ε_raw)
        println("raw_saturated_pct=", round(100 * raw_sat / length(ε_raw), digits=3))

        # ── Geometry panel ───────────────────────────────────────────────
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
                        plot_title="step $step   loss=$(round(loss_ckpt,sigdigits=4))")
        savefig(fig_geom, joinpath(snap_dir, "geometry.png"))

        # ── Gray-zone map ────────────────────────────────────────────────
        gray_map = 1 .- 2 .* abs.(ε_proj .- 0.5)   # 1 at ε=0.5, 0 at ε∈{0,1}
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

        # Moving averages
        ma20 = length(losses) >= 20 ? mean(losses[end-19:end]) : mean(losses)
        ma40 = length(losses) >= 40 ? mean(losses[end-39:end]) : mean(losses)
        prev_ma20 = length(losses) >= 40 ? mean(losses[end-39:end-20]) : NaN
        std20 = length(losses) >= 20 ? std(losses[end-19:end]) : NaN
        println("ma_loss_20=",      round(ma20, sigdigits=6))
        println("ma_loss_40=",      round(ma40, sigdigits=6))
        println("ma_loss_prev20=",  isnan(prev_ma20) ? "NaN" : round(prev_ma20, sigdigits=6))
        println("ma_ratio_20_over_prev20=", isnan(prev_ma20) || prev_ma20 == 0 ? "NaN" : round(ma20/prev_ma20, sigdigits=4))
        println("loss_std_20=",     isnan(std20) ? "NaN" : round(std20, sigdigits=4))
        println("loss_min_recent=", round(minimum(losses[max(1,end-99):end]), sigdigits=4))
        println("loss_max_recent=", round(maximum(losses[max(1,end-99):end]), sigdigits=4))

        # Last β doubling
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

        # Gradient saturation
        grad_sat = count(g -> g < 1e-8, grads)
        println("grad_sat_pct_last_$(length(grads))=", round(100*grad_sat/length(grads), digits=2))
        println("grad_avg_recent=", round(mean(grads[max(1,end-19):end]), sigdigits=4))

        # ── Loss trajectory PNG ──────────────────────────────────────────
        β_change_idx = Int[]
        for i in 2:length(βs)
            if βs[i] != βs[i-1]
                push!(β_change_idx, i)
            end
        end
        fig_loss = plot(iters, losses; yscale=:log10, lw=1, legend=false,
                        xlabel="iter", ylabel="loss (log)",
                        title="loss trajectory  (β shown as vertical lines)",
                        size=(1000, 500))
        for idx in β_change_idx
            vline!(fig_loss, [iters[idx]]; ls=:dash, color=:red, alpha=0.5)
            annotate!(fig_loss, iters[idx], maximum(losses), text("β=$(βs[idx])", 8, :red, :left))
        end
        # also plot MA-20
        if length(losses) >= 20
            ma_series = [mean(losses[max(1,i-19):i]) for i in 1:length(losses)]
            plot!(fig_loss, iters, ma_series; lw=2, color=:blue, label="MA-20")
        end
        savefig(fig_loss, joinpath(snap_dir, "loss_trajectory.png"))

        # ── Gradient histogram ───────────────────────────────────────────
        log_grads = log10.(max.(grads, 1e-16))
        fig_grad = histogram(log_grads; bins=40, legend=false,
                             xlabel="log10(avg |grad|)", ylabel="count",
                             title="grad magnitude (last $(length(grads)) iters)",
                             size=(700, 400))
        savefig(fig_grad, joinpath(snap_dir, "grad_hist.png"))
    else
        println("note=no_parseable_iter_lines")
    end

    # Pending control file?
    control = joinpath(root, "checkpoints", "control.toml")
    println("control_pending=", isfile(control))

    # Recent control applications
    applied = filter(f -> occursin(r"^control\.toml\.applied\.", f),
                     isdir(joinpath(root,"checkpoints")) ? readdir(joinpath(root,"checkpoints")) : String[])
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
    any_phend = occursin(r"PhEnd2End", out_str)
    interactive = occursin("PhEnd2End_interactive", out_str)
    non_interactive = any_phend && !interactive
    println("julia_alive=$any_phend")
    println("julia_controllable=$interactive")
    println("julia_running_non_interactive=$non_interactive")

    println("done=true")
end

main()
