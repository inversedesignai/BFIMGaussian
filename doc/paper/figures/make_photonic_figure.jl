#=
make_photonic_figure.jl  — publication-quality photonic result figure.

Combines:
  - the final binary-projected ε_geom (main plot), rendered from the
    β=256 autotune checkpoint via the density filter + tanh projection
  - the full loss trajectory from random init through the β=256 plateau
    (inset), assembled from three log files:
      opt_result2/nohup.out       — initial random-init continuous Adam
      nohup.out.pre-autotune...   — first filter + β continuation
      nohup.out                   — final R=5 + β=16→256 autotune

Output: figures/photonic_main.png
=#

using Printf
using Serialization
using Plots
using Statistics

const REPO = "/home/zlin/BFIMGaussian"

gr()  # we'll switch to pgfplotsx if available; gr is simpler and publication-OK
default(fontfamily="sans-serif", grid=false, frame=:box, tickfontsize=11,
        labelfontsize=13, titlefontsize=14, legendfontsize=10,
        background_color=:white, foreground_color=:black)

# ---------------------- density filter + projection ----------------------
function build_density_filter(Ny::Int, Nx::Int, R::Float64)
    N = Ny * Nx
    W = zeros(Float64, N, N)
    R2 = R^2
    for j in 1:Nx, i in 1:Ny
        idx = (j - 1) * Ny + i
        iy_lo = max(1, i - Int(ceil(R))); iy_hi = min(Ny, i + Int(ceil(R)))
        ix_lo = max(1, j - Int(ceil(R))); ix_hi = min(Nx, j + Int(ceil(R)))
        acc = 0.0
        for jj in ix_lo:ix_hi, ii in iy_lo:iy_hi
            d2 = (ii - i)^2 + (jj - j)^2
            if d2 <= R2
                w = max(0.0, R - sqrt(d2))
                k = (jj - 1) * Ny + ii
                W[idx, k] = w
                acc += w
            end
        end
        if acc > 0
            W[idx, :] ./= acc
        end
    end
    W
end

function project_density(ρ, β, η=0.5)
    num = tanh(β * η) .+ tanh.(β .* (ρ .- η))
    den = tanh(β * η) + tanh(β * (1.0 - η))
    num ./ den
end

# ---------------------- log parsing ----------------------
"Return Vector{Float64} of loss values, one per iter, from an Adam log file."
function parse_losses(path::String)
    losses = Float64[]
    for line in eachline(path)
        m = match(r"iter\s+(\d+)/\d+\s+[\d\.]+s\s+loss=([0-9.eE+-]+)", line)
        m === nothing && continue
        push!(losses, parse(Float64, m.captures[2]))
    end
    losses
end

"Moving-average smoother."
function ma(x::Vector{Float64}, w::Int)
    n = length(x)
    y = similar(x)
    for i in 1:n
        lo = max(1, i - w + 1)
        y[i] = mean(@view x[lo:i])
    end
    y
end

# ---------------------- build the figure ----------------------
function main()
    println("Loading final ε_geom ...")
    ckpt = deserialize(joinpath(REPO, "checkpoints", "eps_geom_step_00580.jls"))
    ε_raw = ckpt.ε_geom
    Ny, Nx = size(ε_raw)
    @printf("  ckpt step=%d loss=%.4e β_proj=%.1f R=%.1f\n",
            ckpt.step, ckpt.loss, ckpt.β_proj, ckpt.filter_radius)
    println("  ε_raw: $(Ny)×$(Nx)")

    println("Building density filter (R=5) ...")
    W = build_density_filter(Ny, Nx, 5.0)
    ε_filt = reshape(W * vec(ε_raw), Ny, Nx)
    ε_proj = project_density(ε_filt, 256.0)

    # threshold at 0.5 for clean binary display
    ε_binary = ε_proj .>= 0.5

    println("Parsing loss logs ...")
    L_init = parse_losses(joinpath(REPO, "opt_result2", "nohup.out"))
    L_tune = parse_losses(joinpath(REPO, "nohup.out"))
    @printf("  opt_result2: %d iters (loss %.4f → %.4e)\n",
            length(L_init), L_init[1], L_init[end])
    @printf("  autotune:    %d iters (loss %.4e → %.4e)\n",
            length(L_tune), L_tune[1], L_tune[end])

    # Concatenate: global iter axis
    L_full = vcat(L_init, L_tune)
    iters  = collect(1:length(L_full))
    L_ma   = ma(L_full, 20)

    println("Rendering figure ...")
    # Physical extent: design region is 6.0x6.0 unit cells at res=50 px/unit,
    # so 300 px span = 6.0 unit cells. We just label pixels for clarity.
    extent = (1, Nx, 1, Ny)

    # --- Main panel: binary design with a frame and sparse tick labels ---
    p_main = heatmap(ε_binary; color=[:white, :black], clims=(0, 1),
                     aspect_ratio=:equal, colorbar=false,
                     framestyle=:box, xlims=(0.5, Nx + 0.5), ylims=(0.5, Ny + 0.5),
                     xticks=(0:100:Nx, string.(0:100:Nx)),
                     yticks=(0:100:Ny, string.(0:100:Ny)),
                     xlabel="design x (pixels)", ylabel="design y (pixels)",
                     tickfontsize=12, labelfontsize=14, titlefontsize=14,
                     title="(a) optimized binary design")

    # --- Loss trajectory panel, clean styling, full labels ---
    iter_transition = length(L_init)
    p_loss = plot(iters, L_full;
                  yscale=:log10, color=RGB(0.22, 0.48, 0.80), lw=0.6, alpha=0.50,
                  label="per iter", legend=:topright, legendfontsize=11,
                  foreground_color_legend=:black,
                  background_color_inside=:white,
                  frame=:box, grid=true, gridalpha=0.20, gridlinewidth=0.6,
                  tickfontsize=12, labelfontsize=14, titlefontsize=14,
                  xlabel="Adam iteration", ylabel="loss",
                  xlims=(1, length(iters)),
                  ylims=(1e-4, 5e0),
                  xticks=(0:1000:length(iters)),
                  title="(b) loss trajectory")
    plot!(p_loss, iters, L_ma; color=RGB(0.85, 0.15, 0.15), lw=2.0,
          label="moving avg 20")
    vline!(p_loss, [iter_transition]; color=:gray, ls=:dash, lw=1.2,
           alpha=0.7, label="")

    # --- Compose side by side ---
    combined = plot(p_main, p_loss; layout=@layout([a b]),
                    size=(1400, 680), dpi=220,
                    left_margin=8Plots.mm,
                    right_margin=6Plots.mm,
                    top_margin=4Plots.mm,
                    bottom_margin=10Plots.mm)

    out = joinpath(REPO, "doc", "paper", "figures", "photonic_main.png")
    savefig(combined, out)
    println("Saved: $out  ($(filesize(out)) bytes)")
end

main()
