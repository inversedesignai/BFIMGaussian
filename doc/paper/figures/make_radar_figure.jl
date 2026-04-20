#=
make_radar_figure.jl — 3D illustration for Case Study A (radar beam search).

Scene:
  * A linear phased-array antenna of 16 elements on a metallic baseplate.
  * A ring of K=16 target cells arrayed on a circular arc above the antenna;
    one cell highlighted as the true target, one as the current belief mode.
  * A translucent volumetric beam lobe emerging from the array, shaped to
    resemble the optimized gain pattern G_c(alpha) of an inverse-designed
    antenna (narrow main lobe, small sidelobes).
  * A ghost overlay of the next-step beam showing the adaptive policy's
    binary-search descent.

Dependencies: GLMakie (requires a working OpenGL/GPU). Swap to CairoMakie
if running headless — the 3D rendering degrades gracefully but stays usable.

Output: doc/paper/figures/radar_scene.png
=#

using GLMakie
using LinearAlgebra

GLMakie.activate!()
set_theme!(theme_light(); fontsize=14)

# ---------- geometric parameters ----------
const K = 16                     # number of target cells on the ring
const N_elements = 16            # antenna elements
const lambda = 1.0               # nominal wavelength (dimensionless)
const d_elem = lambda / 2        # element spacing (half-wavelength array)
const array_y = 0.0              # array sits along x-axis, y = 0
const ring_R = 8.0 * lambda      # radius of target ring
const ring_z = 6.0 * lambda      # target ring elevation above array
const target_idx = 5             # true target cell (0-indexed)
const belief_idx = 7             # current policy points here

# ---------- helper: beam-lobe mesh ----------
"Parametric beam-lobe surface.  A narrow cone with a Gaussian falloff in
angular coordinates, pointing toward `target`."
function beam_lobe_mesh(origin, target; n_theta=48, n_phi=24, lobe_width=0.10)
    direction = normalize(target - origin)
    # orthonormal frame around `direction`
    tmp = abs(direction[3]) < 0.9 ? [0.0, 0.0, 1.0] : [1.0, 0.0, 0.0]
    u = normalize(cross(direction, tmp))
    v = normalize(cross(direction, u))

    phis = range(0, 2pi; length=n_phi)
    ts = range(0, norm(target - origin); length=n_theta)
    verts = Point3f[]
    for t in ts, phi in phis
        # Gaussian lobe: radial extent grows linearly in t then tapers near tip
        r = lobe_width * t * (1.0 - 0.2 * (t / ts[end])^2)
        p = origin .+ t .* direction .+ (r * cos(phi)) .* u .+ (r * sin(phi)) .* v
        push!(verts, Point3f(p...))
    end
    # build triangular faces
    faces = GLTriangleFace[]
    for i in 1:(n_theta-1), j in 1:(n_phi-1)
        a = (i-1)*n_phi + j
        b = (i-1)*n_phi + j + 1
        c = i*n_phi + j + 1
        d = i*n_phi + j
        push!(faces, GLTriangleFace(a, b, c))
        push!(faces, GLTriangleFace(a, c, d))
    end
    verts, faces
end

function render_radar_scene()
    fig = Figure(; size=(1400, 900), backgroundcolor=:white)
    ax = Axis3(fig[1, 1];
        aspect=(1.5, 1.2, 1.0),
        xlabel="x / λ", ylabel="y / λ", zlabel="z / λ",
        title="Case A — Radar beam-search geometry",
        xgridcolor=(:grey, 0.2), ygridcolor=(:grey, 0.2), zgridcolor=(:grey, 0.2),
        xlabeloffset=30, ylabeloffset=30, zlabeloffset=40,
        azimuth=0.35π, elevation=0.12π, perspectiveness=0.45)

    # ---------- baseplate ----------
    baseplate_x = range(-(N_elements*d_elem)/2 - 1.0, (N_elements*d_elem)/2 + 1.0; length=2)
    baseplate_y = range(-1.5, 1.5; length=2)
    baseplate_z = fill(-0.05, 2, 2)
    surface!(ax, baseplate_x, baseplate_y, baseplate_z;
        colormap=[:grey25, :grey25], shading=FastShading, transparency=false)

    # ---------- phased-array elements ----------
    for k in 1:N_elements
        x0 = (k - (N_elements+1)/2) * d_elem
        # each element is a small box on the baseplate
        box_x = [x0 - 0.12, x0 + 0.12]
        box_y = [-0.25, 0.25]
        box_z = [0.0, 0.15]
        # draw top face
        for z in box_z
            surface!(ax, box_x, box_y, fill(z, 2, 2); colormap=[:gold, :gold],
                shading=FastShading)
        end
        # side faces (simplified: two rectangles)
        xs = [box_x[1], box_x[2], box_x[2], box_x[1], box_x[1]]
        ys_front = fill(box_y[1], 5); ys_back = fill(box_y[2], 5)
        zs = [box_z[1], box_z[1], box_z[2], box_z[2], box_z[1]]
        lines!(ax, xs, ys_front, zs; color=:gold, linewidth=1.2)
        lines!(ax, xs, ys_back,  zs; color=:gold, linewidth=1.2)
    end

    # ---------- ring of K target cells ----------
    ring_phis = [2π * (k - 1) / K for k in 1:K]
    targets = [Point3f(ring_R * cos(ϕ), ring_R * sin(ϕ), ring_z) for ϕ in ring_phis]

    # scatter the cells as small spheres
    cell_colors = [k == target_idx + 1 ? RGBAf(0.85, 0.15, 0.15, 1.0) :
                   k == belief_idx + 1 ? RGBAf(0.15, 0.55, 0.85, 1.0) :
                   RGBAf(0.70, 0.70, 0.70, 0.8)
                   for k in 1:K]
    meshscatter!(ax, targets;
        markersize=0.35, color=cell_colors, shading=FastShading)

    # thin translucent disc connecting the ring (to visually anchor the ring)
    disc_phis = range(0, 2π; length=128)
    disc_pts = [Point3f(ring_R * cos(ϕ), ring_R * sin(ϕ), ring_z) for ϕ in disc_phis]
    lines!(ax, disc_pts; color=RGBAf(0.3, 0.3, 0.3, 0.4), linewidth=1.0)

    # ---------- main beam lobe toward the CURRENT belief cell ----------
    array_center = Point3f(0.0, 0.0, 0.08)
    belief_tgt = targets[belief_idx + 1]
    verts, faces = beam_lobe_mesh(array_center, belief_tgt; lobe_width=0.08)
    mesh!(ax, verts, faces;
        color=RGBAf(0.15, 0.55, 0.85, 0.22), shading=FastShading, transparency=true)

    # ---------- ghost next-step lobe toward a different cell (adaptive policy) ----------
    next_idx = mod(belief_idx + 3, K)
    next_tgt = targets[next_idx + 1]
    verts2, faces2 = beam_lobe_mesh(array_center, next_tgt; lobe_width=0.065)
    mesh!(ax, verts2, faces2;
        color=RGBAf(0.50, 0.20, 0.70, 0.15), shading=FastShading, transparency=true)

    # ---------- arrows showing beam steering directions ----------
    arrows!(ax,
        [array_center[1]], [array_center[2]], [array_center[3]],
        [belief_tgt[1] - array_center[1]], [belief_tgt[2] - array_center[2]], [belief_tgt[3] - array_center[3]];
        arrowsize=Vec3f(0.5, 0.5, 0.8), color=RGBAf(0.10, 0.40, 0.75, 0.9),
        linewidth=0.03)

    # ---------- legend / annotation text ----------
    text!(ax, belief_tgt[1], belief_tgt[2], belief_tgt[3] + 0.7;
        text="belief mode", color=:dodgerblue, align=(:center, :bottom), fontsize=14)
    text!(ax, targets[target_idx + 1][1], targets[target_idx + 1][2], targets[target_idx + 1][3] + 0.7;
        text="true target", color=:crimson, align=(:center, :bottom), fontsize=14)
    text!(ax, 0.0, 0.0, -0.7;
        text="phased array (16 elements, λ/2 spacing)", color=:goldenrod3,
        align=(:center, :top), fontsize=12)

    hidespines!(ax)
    ax.xticklabelsvisible = false
    ax.yticklabelsvisible = false
    ax.zticklabelsvisible = false

    out = joinpath(@__DIR__, "radar_scene.png")
    save(out, fig; px_per_unit=2)
    println("Saved: $out")
    return fig
end

render_radar_scene()
