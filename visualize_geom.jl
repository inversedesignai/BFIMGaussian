#!/usr/bin/env julia
"""
Visualize an optimized ε_geom checkpoint as a PNG heatmap.

Usage:
  julia visualize_geom.jl [checkpoint_path] [output_path]

Defaults:
  checkpoint_path = latest checkpoint in opt_result2/checkpoints_adam/
  output_path     = eps_geom_latest.png
"""

using Serialization
using Plots

function find_latest_checkpoint(dir)
    files = filter(f -> endswith(f, ".jls"), readdir(dir))
    isempty(files) && error("No .jls files found in $dir")
    return joinpath(dir, sort(files)[end])
end

function visualize_geom(checkpoint_path::String, output_path::String)
    println("Loading: $checkpoint_path")
    d = deserialize(checkpoint_path)

    ε = d.ε_geom
    step = d.step
    loss = d.loss
    Ny, Nx = size(ε)

    println("  Step: $step  Loss: $(round(loss, sigdigits=4))  Size: $(Ny)×$(Nx)")
    println("  ε range: [$(minimum(ε)), $(maximum(ε))]")

    # Count binary pixels
    n_zero = count(x -> x == 0.0, ε)
    n_one  = count(x -> x == 1.0, ε)
    pct_binary = round(100 * (n_zero + n_one) / length(ε), digits=1)
    println("  Binary pixels: $(pct_binary)% ($(n_zero) zeros, $(n_one) ones)")

    p = heatmap(ε;
        aspect_ratio=:equal,
        color=:grays,
        clims=(0, 1),
        xlabel="x (pixels)",
        ylabel="y (pixels)",
        title="ε_geom — step $step, loss=$(round(loss, sigdigits=4))",
        size=(800, 800),
        colorbar_title="ε_geom")

    savefig(p, output_path)
    println("Saved → $output_path")
end

# Parse arguments
checkpoint_dir = joinpath(@__DIR__, "opt_result2", "checkpoints_adam")
checkpoint_path = length(ARGS) >= 1 ? ARGS[1] : find_latest_checkpoint(checkpoint_dir)
output_path     = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "eps_geom_latest.png")

visualize_geom(checkpoint_path, output_path)
