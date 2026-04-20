#=
make_scqubit_figure.jl — 3D illustration for Case Study B (superconducting-
qubit flux sensor).

Scene:
  * X-mon transmon cross electrode on a sapphire substrate (two perpendicular
    rectangular pads).
  * Small SQUID loop at the end of one arm, with two Josephson-junction
    gaps visibly drawn.
  * Flux-bias line (current-carrying wire) routed parallel to the arm,
    with a magnetic field line threading the SQUID loop.
  * A Bloch sphere inset to the right, showing the Ramsey phase-accumulation
    arc (|0> + |1>)/sqrt(2) -> |0> + e^{i phi} |1>.
  * Readout resonator shown as a meander line on the opposite arm.

Dependencies: GLMakie (3D) + Colors for the Bloch sphere.

Output: doc/paper/figures/scqubit_scene.png
=#

using GLMakie
using LinearAlgebra

GLMakie.activate!()
set_theme!(theme_light(); fontsize=14)

# ---------- dimensions (arbitrary units, tuned for clean layout) ----------
const arm_L = 4.0           # arm half-length
const arm_w = 0.6           # arm width
const arm_h = 0.06          # arm thickness (metal film)
const subs_L = 6.0          # substrate extent
const subs_h = 0.02         # substrate thickness
const squid_side = 0.8
const squid_gap = 0.12      # junction-gap spacing
const bias_L = 4.0
const bias_offset_y = -1.2

# ---------- helper: axis-aligned box as two surfaces (top + bottom) + wireframe ----------
function draw_box!(ax, x_range, y_range, z_range; top_color=:orange,
                   edge_color=:black, alpha=1.0)
    # top and bottom faces
    surface!(ax, x_range, y_range, fill(z_range[2], 2, 2);
        colormap=[top_color, top_color], shading=FastShading, transparency=alpha<1)
    surface!(ax, x_range, y_range, fill(z_range[1], 2, 2);
        colormap=[top_color, top_color], shading=FastShading, transparency=alpha<1)
    # edges
    x0, x1 = x_range; y0, y1 = y_range; z0, z1 = z_range
    pts = Point3f[
        (x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(x0,y0,z0),
        (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1),(x0,y0,z1)]
    lines!(ax, pts; color=edge_color, linewidth=1.0)
    for (p1, p2) in [((x1,y0,z0),(x1,y0,z1)),((x1,y1,z0),(x1,y1,z1)),((x0,y1,z0),(x0,y1,z1))]
        lines!(ax, [Point3f(p1...), Point3f(p2...)]; color=edge_color, linewidth=1.0)
    end
end

function render_scqubit_scene()
    fig = Figure(; size=(1500, 900), backgroundcolor=:white)
    ax = Axis3(fig[1, 1:2];
        aspect=(1.6, 1.4, 0.6),
        xlabel="x", ylabel="y", zlabel="z",
        title="Case B — Superconducting x-mon flux sensor",
        xlabeloffset=20, ylabeloffset=20, zlabeloffset=30,
        azimuth=0.30π, elevation=0.38π, perspectiveness=0.3)

    # ---------- substrate ----------
    draw_box!(ax, (-subs_L/2, subs_L/2), (-subs_L/2, subs_L/2), (-subs_h, 0.0);
        top_color=RGBAf(0.65, 0.75, 0.85, 1.0), edge_color=:grey40)

    # ---------- x-mon cross (two overlapping rectangles) ----------
    # horizontal arm
    draw_box!(ax, (-arm_L, arm_L), (-arm_w/2, arm_w/2), (0.0, arm_h);
        top_color=RGBAf(0.95, 0.55, 0.15, 1.0), edge_color=:black)
    # vertical arm
    draw_box!(ax, (-arm_w/2, arm_w/2), (-arm_L, arm_L), (0.0, arm_h);
        top_color=RGBAf(0.95, 0.55, 0.15, 1.0), edge_color=:black)

    # ---------- SQUID loop at the end of the +x arm ----------
    loop_cx = arm_L + 0.6
    loop_cy = 0.0
    # loop outer square
    outer = [Point3f(loop_cx - squid_side/2, loop_cy - squid_side/2, arm_h),
             Point3f(loop_cx + squid_side/2, loop_cy - squid_side/2, arm_h),
             Point3f(loop_cx + squid_side/2, loop_cy + squid_side/2, arm_h),
             Point3f(loop_cx - squid_side/2, loop_cy + squid_side/2, arm_h),
             Point3f(loop_cx - squid_side/2, loop_cy - squid_side/2, arm_h)]
    lines!(ax, outer; color=:black, linewidth=3.0)
    # junction gaps: small breaks in two sides (top and bottom)
    # mark with red dots where the JJs sit
    meshscatter!(ax,
        [Point3f(loop_cx, loop_cy - squid_side/2, arm_h),
         Point3f(loop_cx, loop_cy + squid_side/2, arm_h)];
        markersize=0.10, color=RGBAf(0.85, 0.1, 0.1, 1.0))
    # connector to the +x arm
    lines!(ax, [Point3f(arm_L, 0.0, arm_h), Point3f(loop_cx - squid_side/2, 0.0, arm_h)];
        color=:black, linewidth=3.0)

    text!(ax, loop_cx, loop_cy + squid_side/2 + 0.4, arm_h;
        text="SQUID loop (2 JJs)", color=:crimson, fontsize=13, align=(:center, :bottom))

    # ---------- flux-bias line: wire alongside the +x arm ----------
    # represented as a cylinder (line segment + surface) for current-carrying
    bias_y = loop_cy - 1.2
    bias_start = Point3f(loop_cx - 1.2, bias_y, 0.15)
    bias_end   = Point3f(loop_cx + 1.2, bias_y, 0.15)
    lines!(ax, [bias_start, bias_end]; color=RGBAf(0.85, 0.3, 0.7, 1.0), linewidth=6.0)
    # current arrow
    arrows!(ax, [bias_start[1]], [bias_start[2]], [bias_start[3]],
        [bias_end[1] - bias_start[1]], [bias_end[2] - bias_start[2]], [bias_end[3] - bias_start[3]];
        arrowsize=Vec3f(0.2, 0.2, 0.3), color=RGBAf(0.55, 0.05, 0.50, 1.0),
        linewidth=0.02)
    text!(ax, bias_start[1] - 0.1, bias_y, 0.4;
        text="I_bias", color=:purple, fontsize=13, align=(:right, :center))

    # ---------- magnetic field line threading the SQUID ----------
    # a helix-like curve from near bias line up through SQUID, out the other side
    tt = range(-1.2, 1.2; length=80)
    field_x = [loop_cx + t * 0.05 for t in tt]
    field_y = [bias_y + 0.8 * sin(2π * 0.8 * t + π/2) for t in tt]
    field_z = [0.4 + 0.4 * cos(2π * 0.8 * t + π/2) for t in tt]
    # manual adjustment: make a single loop
    phis = range(0, π; length=80)
    field_x = fill(loop_cx, 80)
    field_y = bias_y .+ (loop_cy - bias_y) .* (1.0 .- cos.(phis)) ./ 2.0
    field_z = 0.4 .+ 0.5 .* sin.(phis)
    lines!(ax, collect(zip(field_x, field_y, field_z)) .|> Point3f;
        color=RGBAf(0.2, 0.3, 0.85, 0.9), linewidth=2.5)
    text!(ax, loop_cx + 0.1, (loop_cy + bias_y)/2, 0.9;
        text="Φ_ext", color=:navy, fontsize=15, align=(:left, :bottom))

    # ---------- readout resonator (meander) at the -x arm tip ----------
    res_cx = -arm_L - 0.4
    meander_x = Float64[]; meander_y = Float64[]; meander_z = Float64[]
    for i in 0:6
        push!(meander_x, res_cx - 0.15 * i)
        push!(meander_y, (-1)^i * 0.6)
        push!(meander_z, arm_h)
    end
    lines!(ax, collect(zip(meander_x, meander_y, meander_z)) .|> Point3f;
        color=RGBAf(0.15, 0.60, 0.15, 1.0), linewidth=4.0)
    text!(ax, res_cx - 0.6, 0.0, 0.4;
        text="readout\nresonator", color=:darkgreen, fontsize=12, align=(:right, :center))

    # ---------- xy-control line, entering from +y side ----------
    xy_line = [Point3f(0.0, arm_L + 0.8, 0.15), Point3f(0.0, arm_L + 0.1, 0.15)]
    lines!(ax, xy_line; color=RGBAf(0.1, 0.1, 0.9, 1.0), linewidth=4.0)
    text!(ax, 0.0, arm_L + 1.1, 0.3;
        text="xy-control", color=:blue, fontsize=12, align=(:center, :bottom))

    hidespines!(ax)
    ax.xticklabelsvisible = false
    ax.yticklabelsvisible = false
    ax.zticklabelsvisible = false

    # ---------- Bloch sphere inset ----------
    ax_bloch = Axis3(fig[1, 3];
        aspect=(1, 1, 1), title="Ramsey phase accumulation",
        elevation=0.25π, azimuth=0.4π, perspectiveness=0.4)
    hidespines!(ax_bloch)
    ax_bloch.xticklabelsvisible = false
    ax_bloch.yticklabelsvisible = false
    ax_bloch.zticklabelsvisible = false

    # sphere wireframe
    theta = range(0, π; length=24)
    phi = range(0, 2π; length=48)
    sx = [sin(t) * cos(p) for t in theta, p in phi]
    sy = [sin(t) * sin(p) for t in theta, p in phi]
    sz = [cos(t) for _ in theta, _ in phi]
    wireframe!(ax_bloch, sx, sy, sz; color=RGBAf(0.7, 0.7, 0.7, 0.45), linewidth=0.4)

    # Axes
    arrows!(ax_bloch, [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0],
        [1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2];
        arrowsize=Vec3f(0.05, 0.05, 0.1), color=:black, linewidth=0.015)
    text!(ax_bloch, 1.35, 0.0, 0.0; text="x", align=(:center, :center), fontsize=12)
    text!(ax_bloch, 0.0, 1.35, 0.0; text="y", align=(:center, :center), fontsize=12)
    text!(ax_bloch, 0.0, 0.0, 1.35; text="|0⟩", align=(:center, :bottom), fontsize=13)
    text!(ax_bloch, 0.0, 0.0, -1.35; text="|1⟩", align=(:center, :top), fontsize=13)

    # equatorial arc showing phase accumulation
    phi_arc = range(0, 1.7π; length=60)
    arc_pts = [Point3f(cos(p), sin(p), 0.0) for p in phi_arc]
    lines!(ax_bloch, arc_pts; color=RGBAf(0.85, 0.25, 0.25, 1.0), linewidth=3.0)
    # starting state
    meshscatter!(ax_bloch, [Point3f(1.0, 0.0, 0.0)]; markersize=0.10, color=:dodgerblue)
    # current state (after accumulated phase phi = Δω τ)
    p_end = Point3f(cos(1.7π), sin(1.7π), 0.0)
    meshscatter!(ax_bloch, [p_end]; markersize=0.10, color=:crimson)
    arrows!(ax_bloch, [0.0], [0.0], [0.0],
        [p_end[1]], [p_end[2]], [p_end[3]];
        arrowsize=Vec3f(0.08, 0.08, 0.10), color=:crimson, linewidth=0.02)

    out = joinpath(@__DIR__, "scqubit_scene.png")
    save(out, fig; px_per_unit=2)
    println("Saved: $out")
    return fig
end

render_scqubit_scene()
