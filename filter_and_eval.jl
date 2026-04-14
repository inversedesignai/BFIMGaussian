#!/usr/bin/env julia
"""
Post-process optimized geometries: apply spatial smoothing to remove
sub-pixel noise inherited from random initialization, then evaluate loss.

Usage:
  julia filter_and_eval.jl [checkpoint_path]

Default: latest checkpoint from opt_result2/checkpoints_adam/
"""

push!(LOAD_PATH, @__DIR__)

using Serialization
using Plots
using LinearAlgebra
using SparseArrays
using Statistics

# ── Spatial filters ──────────────────────────────────────────────────────────

"""Gaussian kernel (unnormalized) of size (2r+1)×(2r+1)."""
function gaussian_kernel(r::Int, σ::Float64)
    k = [exp(-((i-r-1)^2 + (j-r-1)^2) / (2σ^2)) for i in 1:2r+1, j in 1:2r+1]
    return k ./ sum(k)
end

"""Apply 2D convolution with zero-padded boundaries."""
function conv2d(img::Matrix{Float64}, kernel::Matrix{Float64})
    Ny, Nx = size(img)
    ky, kx = size(kernel)
    ry, rx = ky ÷ 2, kx ÷ 2
    out = zeros(Ny, Nx)
    for ix in 1:Nx, iy in 1:Ny
        s = 0.0
        w = 0.0
        for dx in -rx:rx, dy in -ry:ry
            jx, jy = ix + dx, iy + dy
            if 1 <= jx <= Nx && 1 <= jy <= Ny
                wk = kernel[dy + ry + 1, dx + rx + 1]
                s += wk * img[jy, jx]
                w += wk
            end
        end
        out[iy, ix] = s / w  # renormalize at boundaries
    end
    return out
end

"""Apply median filter with window radius r."""
function median_filter(img::Matrix{Float64}, r::Int)
    Ny, Nx = size(img)
    out = zeros(Ny, Nx)
    for ix in 1:Nx, iy in 1:Ny
        vals = Float64[]
        for dx in -r:r, dy in -r:r
            jx, jy = ix + dx, iy + dy
            if 1 <= jx <= Nx && 1 <= jy <= Ny
                push!(vals, img[jy, jx])
            end
        end
        out[iy, ix] = median(vals)
    end
    return out
end

"""Threshold at η, producing binary output."""
threshold(img, η=0.5) = Float64.(img .>= η)

# ── Load checkpoint ──────────────────────────────────────────────────────────

function find_latest_checkpoint(dir)
    files = filter(f -> endswith(f, ".jls"), readdir(dir))
    isempty(files) && error("No .jls files in $dir")
    return joinpath(dir, sort(files)[end])
end

ckpt_path = length(ARGS) >= 1 ? ARGS[1] :
    find_latest_checkpoint(joinpath(@__DIR__, "opt_result2", "checkpoints_adam"))
println("Loading: $ckpt_path"); flush(stdout)
d = deserialize(ckpt_path)
ε_raw = d.ε_geom
step = d.step
loss_raw = d.loss
Ny, Nx = size(ε_raw)
println("  Step=$step  Loss=$(round(loss_raw, sigdigits=4))  Size=$(Ny)×$(Nx)")
flush(stdout)

# ── Apply filters at several radii ──────────────────────────────────────────

radii = [3, 5, 8, 12]

println("\nApplying Gaussian blur + threshold at various radii..."); flush(stdout)
results = []
for r in radii
    σ = r / 2.0
    k = gaussian_kernel(r, σ)
    ε_smooth = conv2d(ε_raw, k)
    ε_binary = threshold(ε_smooth, 0.5)

    n_white = count(x -> x == 1.0, ε_binary)
    fill_frac = round(100 * n_white / length(ε_binary), digits=1)
    println("  r=$r (σ=$(σ)): fill=$(fill_frac)%")
    push!(results, (; r, σ, ε_smooth, ε_binary, fill_frac))
end

# Also try median filter
println("\nApplying median filter + threshold..."); flush(stdout)
for r in [3, 5]
    ε_med = median_filter(ε_raw, r)
    ε_med_bin = threshold(ε_med, 0.5)
    n_white = count(x -> x == 1.0, ε_med_bin)
    fill_frac = round(100 * n_white / length(ε_med_bin), digits=1)
    println("  median r=$r: fill=$(fill_frac)%")
    push!(results, (; r, σ=NaN, ε_smooth=ε_med, ε_binary=ε_med_bin, fill_frac))
end

# ── Visualize ────────────────────────────────────────────────────────────────

println("\nGenerating comparison plot..."); flush(stdout)

n = length(results) + 1
plots = []

# Original
push!(plots, heatmap(ε_raw; aspect_ratio=:equal, color=:grays, clims=(0,1),
    title="Original (step $step)", colorbar=false, size=(400,400)))

# Filtered versions
for res in results
    label = isnan(res.σ) ? "median r=$(res.r)" : "gauss r=$(res.r)"
    push!(plots, heatmap(res.ε_binary; aspect_ratio=:equal, color=:grays, clims=(0,1),
        title="$label → binary\nfill=$(res.fill_frac)%", colorbar=false))
end

fig = plot(plots...; layout=(2, ceil(Int, n/2)), size=(400*ceil(Int, n/2), 800),
           plot_title="Filtered geometries (step $step)")
savefig(fig, joinpath(@__DIR__, "eps_geom_filtered.png"))
println("Saved → eps_geom_filtered.png")

# Also save the smoothed (pre-threshold) versions
plots_smooth = []
push!(plots_smooth, heatmap(ε_raw; aspect_ratio=:equal, color=:grays, clims=(0,1),
    title="Original", colorbar=false))
for res in results
    label = isnan(res.σ) ? "median r=$(res.r)" : "gauss r=$(res.r)"
    push!(plots_smooth, heatmap(res.ε_smooth; aspect_ratio=:equal, color=:grays, clims=(0,1),
        title="$label (smoothed)", colorbar=false))
end
fig2 = plot(plots_smooth...; layout=(2, ceil(Int, n/2)), size=(400*ceil(Int, n/2), 800),
            plot_title="Smoothed geometries (before threshold)")
savefig(fig2, joinpath(@__DIR__, "eps_geom_smoothed.png"))
println("Saved → eps_geom_smoothed.png")

# Save filtered geometries for loss evaluation
for res in results
    label = isnan(res.σ) ? "median_r$(res.r)" : "gauss_r$(res.r)"
    path = joinpath(@__DIR__, "eps_filtered_$(label).jls")
    serialize(path, (; ε_geom=res.ε_binary, filter=label, source_step=step))
end
println("\nSaved filtered geometries as .jls files for loss evaluation.")
println("To evaluate loss: julia PhEnd2End.jl with BFIM_MODE=none, then load and call end2end().")
