#=
make_photonic_3d_figure.jl — 3D illustration for Case Study C (photonic
metasensor).

Scene:
  * The 90,000-pixel binary permittivity landscape of the β=256 autotune
    checkpoint, extruded into a 3D slab (silicon blocks against a lighter
    cladding).
  * Four waveguide ports entering/exiting the four sides of the slab.
  * A translucent E-field intensity overlay on top of the slab (mock
    amplitude from a Gaussian envelope, for illustration; swap in actual
    FDFD output if available).
  * Color scheme tuned to resemble silicon-on-insulator photonic renders.

Dependencies: GLMakie + Serialization (to load the checkpointed ε_geom).

Output: doc/paper/figures/photonic_3d_scene.png
=#

using GLMakie
using LinearAlgebra
using Serialization
using Statistics

GLMakie.activate!()
set_theme!(theme_light(); fontsize=14)

const REPO = get(ENV, "BFIM_REPO", dirname(dirname(dirname(@__DIR__))))

# ---------- density filter + projection (mirrors make_photonic_figure.jl) ----------
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

project_density(ρ, β, η=0.5) =
    (tanh(β * η) .+ tanh.(β .* (ρ .- η))) ./ (tanh(β * η) + tanh(β * (1.0 - η)))

# Fallback synthetic permittivity pattern in case the checkpoint is unavailable
function synthetic_epsilon(Ny=300, Nx=300)
    ε = zeros(Float64, Ny, Nx)
    # seed with a few Gaussian blobs + one central cross
    for (cy, cx, σ) in [(80, 80, 20.0), (220, 80, 22.0), (80, 220, 18.0),
                        (220, 220, 25.0), (150, 150, 30.0)]
        for j in 1:Nx, i in 1:Ny
            ε[i, j] += exp(-((i - cy)^2 + (j - cx)^2) / (2σ^2))
        end
    end
    # central cross
    for j in 140:160, i in 80:220
        ε[i, j] += 0.5
    end
    for i in 140:160, j in 80:220
        ε[i, j] += 0.5
    end
    ε ./= maximum(ε)
    # binarize
    ε .>= 0.30
end

function load_epsilon()
    ckpt_path = joinpath(REPO, "checkpoints", "eps_geom_step_00580.jls")
    if isfile(ckpt_path)
        println("Loading ε_geom from $ckpt_path ...")
        ckpt = deserialize(ckpt_path)
        ε_raw = ckpt.ε_geom
        Ny, Nx = size(ε_raw)
        W = build_density_filter(Ny, Nx, 5.0)
        ε_filt = reshape(W * vec(ε_raw), Ny, Nx)
        ε_proj = project_density(ε_filt, 256.0)
        return ε_proj .>= 0.5
    else
        println("Checkpoint not found at $ckpt_path; using synthetic fallback.")
        return synthetic_epsilon()
    end
end

function render_photonic_scene()
    ε_binary = load_epsilon()

    # downsample for 3D rendering (block-average to keep the pattern but reduce voxel count)
    function blockavg(A, k)
        Ny_, Nx_ = size(A)
        ny = div(Ny_, k); nx = div(Nx_, k)
        B = zeros(Float64, ny, nx)
        for j in 1:nx, i in 1:ny
            B[i, j] = mean(A[(i-1)*k+1:i*k, (j-1)*k+1:j*k])
        end
        B .>= 0.5
    end
    ε_ds = blockavg(ε_binary, 4)  # 75x75 from 300x300
    Ny_ds, Nx_ds = size(ε_ds)

    # physical extent (design region is 6.0 x 6.0 unit cells)
    extent_x = range(-3.0, 3.0; length=Nx_ds + 1)
    extent_y = range(-3.0, 3.0; length=Ny_ds + 1)
    dz_slab = 0.25

    fig = Figure(; size=(1500, 1000), backgroundcolor=:white)
    ax = Axis3(fig[1, 1];
        aspect=(1.4, 1.4, 0.8),
        xlabel="x (unit cells)", ylabel="y (unit cells)", zlabel="z",
        title="Case C — Photonic metasensor (~9×10⁴ pixel design)",
        azimuth=0.28π, elevation=0.28π, perspectiveness=0.35)

    # ---------- substrate slab (BOX) ----------
    # bottom cladding layer (thin, light)
    surface!(ax,
        [extent_x[1], extent_x[end]], [extent_y[1], extent_y[end]],
        fill(-0.05, 2, 2);
        colormap=[RGBAf(0.85, 0.88, 0.92, 1.0), RGBAf(0.85, 0.88, 0.92, 1.0)],
        shading=FastShading)

    # ---------- silicon pillars (each "true" pixel = a box) ----------
    # use voxels!: Makie has voxels for exactly this use case
    voxel_values = Float32.(ε_ds) .|> x -> x > 0.5f0 ? 1.0f0 : 0.0f0
    # voxels! expects a 3D array; extrude to thickness n_z = 1 voxel layer
    n_zv = max(1, round(Int, dz_slab / (6.0 / Nx_ds)))
    voxel_3d = zeros(Float32, Nx_ds, Ny_ds, n_zv)
    for k in 1:n_zv, j in 1:Ny_ds, i in 1:Nx_ds
        voxel_3d[i, j, k] = voxel_values[j, i]
    end
    x_range = (extent_x[1], extent_x[end])
    y_range = (extent_y[1], extent_y[end])
    z_range = (0.0, dz_slab)
    voxels!(ax, x_range, y_range, z_range, voxel_3d;
        colormap=[RGBAf(0, 0, 0, 0), RGBAf(0.22, 0.25, 0.38, 1.0)],
        shading=FastShading)

    # ---------- four waveguide ports (rectangular stubs) ----------
    port_w = 0.25      # waveguide width
    port_len = 1.8
    port_z0 = 0.02
    port_z1 = dz_slab - 0.02
    port_color = RGBAf(0.30, 0.35, 0.50, 1.0)
    # +x port (east)
    surface!(ax,
        [3.0, 3.0 + port_len], [-port_w/2, port_w/2], fill(port_z1, 2, 2);
        colormap=[port_color, port_color], shading=FastShading)
    surface!(ax,
        [3.0, 3.0 + port_len], [-port_w/2, port_w/2], fill(port_z0, 2, 2);
        colormap=[port_color, port_color], shading=FastShading)
    # -x port (west)
    surface!(ax,
        [-3.0 - port_len, -3.0], [-port_w/2, port_w/2], fill(port_z1, 2, 2);
        colormap=[port_color, port_color], shading=FastShading)
    surface!(ax,
        [-3.0 - port_len, -3.0], [-port_w/2, port_w/2], fill(port_z0, 2, 2);
        colormap=[port_color, port_color], shading=FastShading)
    # +y port (north)
    surface!(ax,
        [-port_w/2, port_w/2], [3.0, 3.0 + port_len], fill(port_z1, 2, 2);
        colormap=[port_color, port_color], shading=FastShading)
    surface!(ax,
        [-port_w/2, port_w/2], [3.0, 3.0 + port_len], fill(port_z0, 2, 2);
        colormap=[port_color, port_color], shading=FastShading)
    # -y port (south)
    surface!(ax,
        [-port_w/2, port_w/2], [-3.0 - port_len, -3.0], fill(port_z1, 2, 2);
        colormap=[port_color, port_color], shading=FastShading)
    surface!(ax,
        [-port_w/2, port_w/2], [-3.0 - port_len, -3.0], fill(port_z0, 2, 2);
        colormap=[port_color, port_color], shading=FastShading)

    # ---------- E-field intensity overlay (mock Gaussian multi-lobe) ----------
    Nxo, Nyo = 200, 200
    xs = range(extent_x[1], extent_x[end]; length=Nxo)
    ys = range(extent_y[1], extent_y[end]; length=Nyo)
    E = [exp(-((x)^2 + (y)^2) / (2 * 0.9^2)) +
         0.55 * exp(-((x + 1.8)^2 + (y - 1.2)^2) / (2 * 0.7^2)) +
         0.45 * exp(-((x - 1.5)^2 + (y + 1.6)^2) / (2 * 0.8^2))
         for y in ys, x in xs]
    E ./= maximum(E)
    # floating transparent surface just above the slab
    surface!(ax, xs, ys, fill(dz_slab + 0.12, Nyo, Nxo);
        color=E, colormap=:plasma, transparency=true,
        shading=FastShading, alpha=0.55)

    # ---------- port labels ----------
    text!(ax, 3.0 + port_len + 0.2, 0.0, port_z1; text="port 1 (out)",
        color=:black, align=(:left, :center), fontsize=13)
    text!(ax, -3.0 - port_len - 0.2, 0.0, port_z1; text="port 2 (in)",
        color=:black, align=(:right, :center), fontsize=13)
    text!(ax, 0.0, 3.0 + port_len + 0.2, port_z1; text="port 3 (out)",
        color=:black, align=(:center, :bottom), fontsize=13)
    text!(ax, 0.0, -3.0 - port_len - 0.2, port_z1; text="port 4 (in)",
        color=:black, align=(:center, :top), fontsize=13)

    # incident arrow on port 2 and port 4
    arrows!(ax, [-3.0 - port_len - 0.3], [0.0], [port_z1 + 0.05],
        [0.6], [0.0], [0.0];
        arrowsize=Vec3f(0.08, 0.08, 0.12), color=RGBAf(0.85, 0.25, 0.25, 1.0),
        linewidth=0.02)
    arrows!(ax, [0.0], [-3.0 - port_len - 0.3], [port_z1 + 0.05],
        [0.0], [0.6], [0.0];
        arrowsize=Vec3f(0.08, 0.08, 0.12), color=RGBAf(0.85, 0.25, 0.25, 1.0),
        linewidth=0.02)

    hidespines!(ax)
    ax.xticklabelsvisible = false
    ax.yticklabelsvisible = false
    ax.zticklabelsvisible = false

    out = joinpath(@__DIR__, "photonic_3d_scene.png")
    save(out, fig; px_per_unit=2)
    println("Saved: $out")
    return fig
end

render_photonic_scene()
