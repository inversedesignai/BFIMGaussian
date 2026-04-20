#=
composite_scqubit.jl — Case Study B hero figure (v6 — stripped down).

Two-panel layout with no header banner, no Bloch inset, no explanatory
caption (the paper caption will describe everything).

    LEFT:  headline MSE-ratio number + two comparison bars.
    RIGHT: 3D Blender scene with large labels placed outside the image.
=#

using CairoMakie
using Printf
using FileIO
using Colors
using FixedPointNumbers

CairoMakie.activate!()

# -----------------------------------------------------------------------------
const BG       = RGBf(1.00, 1.00, 1.00)
const PANEL    = RGBf(0.965, 0.970, 0.980)
const FRAME    = RGBf(0.20, 0.24, 0.34)
const BAR_HI   = RGBf(0.10, 0.16, 0.38)
const BAR_LO   = RGBf(0.62, 0.70, 0.86)
const ACCENT   = RGBf(0.82, 0.25, 0.20)
const YELLOW   = RGBf(1.00, 0.84, 0.32)
const TEXT     = RGBf(0.08, 0.10, 0.16)
const GRAY     = RGBf(0.42, 0.44, 0.52)
const NAVY     = RGBf(0.12, 0.18, 0.44)

function flatten_white(img)
    out = similar(img, RGB{N0f8})
    for I in eachindex(img)
        c = img[I]
        α = Float64(alpha(c))
        r = Float64(red(c))   * α + (1 - α)
        g = Float64(green(c)) * α + (1 - α)
        b = Float64(blue(c))  * α + (1 - α)
        out[I] = RGB{N0f8}(clamp(r,0,1), clamp(g,0,1), clamp(b,0,1))
    end
    out
end

function rounded_rect!(ax, x0, y0, x1, y1; r=0.010, color=BG,
                       strokecolor=FRAME, strokewidth=1.0)
    n = 14
    pts = Point2f[]
    for θ in range(π,    1.5π; length=n); push!(pts, Point2f(x0+r + r*cos(θ), y0+r + r*sin(θ))); end
    for θ in range(1.5π, 2π;   length=n); push!(pts, Point2f(x1-r + r*cos(θ), y0+r + r*sin(θ))); end
    for θ in range(0,    0.5π; length=n); push!(pts, Point2f(x1-r + r*cos(θ), y1-r + r*sin(θ))); end
    for θ in range(0.5π, π;    length=n); push!(pts, Point2f(x0+r + r*cos(θ), y1-r + r*sin(θ))); end
    poly!(ax, pts; color=color, strokecolor=strokecolor, strokewidth=strokewidth)
end

# =============================================================================
# CANVAS
# =============================================================================
fig = Figure(size=(1500, 820), backgroundcolor=BG, figure_padding=14)
body = fig[1, 1] = GridLayout()

ax_l_holder = Axis(body[1, 1])
ax_r_holder = Axis(body[1, 2])
delete!(ax_l_holder); delete!(ax_r_holder)

colsize!(body, 1, Relative(0.36))
colsize!(body, 2, Relative(0.64))
colgap!(body, 14)

# =============================================================================
# LEFT COLUMN — Performance
# =============================================================================
ax_l = Axis(body[1, 1]; backgroundcolor=PANEL,
            leftspinecolor=FRAME, rightspinecolor=FRAME,
            topspinecolor=FRAME, bottomspinecolor=FRAME, spinewidth=1.6,
            xticksvisible=false, yticksvisible=false,
            xticklabelsvisible=false, yticklabelsvisible=false,
            limits=(0,1,0,1))
hidedecorations!(ax_l)

# Panel title strip
rounded_rect!(ax_l, 0.03, 0.93, 0.97, 0.99; r=0.010,
              color=FRAME, strokewidth=0)
text!(ax_l, 0.50, 0.96; text="Performance",
      align=(:center, :center), fontsize=30, color=YELLOW, font=:bold)

# ----- Huge headline number -----------------------------------------------
text!(ax_l, 0.50, 0.80; text="11.3×",
      align=(:center, :center), fontsize=130, color=ACCENT, font=:bold)
text!(ax_l, 0.50, 0.67; text="lower MSE than PCRB",
      align=(:center, :center), fontsize=26, color=TEXT, font=:bold)
text!(ax_l, 0.50, 0.61;
      text="z = +132 σ   (paired MC, n = 20 000)",
      align=(:center, :center), fontsize=18, color=GRAY, font=:italic)

# Divider
lines!(ax_l, [0.07, 0.93], [0.54, 0.54]; color=GRAY, linewidth=1.0)

# ----- Comparison bars -----------------------------------------------------
function bar_row!(ax, y, label, value_str, frac; highlight=false)
    clr = highlight ? BAR_HI : BAR_LO
    text!(ax, 0.06, y+0.065; text=label,
          align=(:left, :baseline), fontsize=21, color=TEXT,
          font=highlight ? :bold : :regular)
    text!(ax, 0.94, y+0.065; text=value_str,
          align=(:right, :baseline), fontsize=26,
          color=highlight ? BAR_HI : TEXT, font=:bold)
    x0, x1 = 0.06, 0.94
    by0, by1 = y-0.025, y+0.025
    rounded_rect!(ax, x0, by0, x1, by1; r=0.006,
                  color=RGBf(0.88, 0.89, 0.92), strokewidth=0)
    bend = x0 + frac * (x1 - x0)
    if bend > x0 + 0.01
        rounded_rect!(ax, x0, by0, bend, by1; r=0.006,
                      color=clr, strokewidth=0)
    end
end

MSE_SCALE = 10.0   # 10 × 10⁻⁴ Φ₀²  full bar
bar_row!(ax_l, 0.395, "PCRB co-designed",
         "MSE̅ = 8.43", 8.43/MSE_SCALE)
bar_row!(ax_l, 0.230, "Joint-DP co-designed",
         "MSE̅ = 0.75", 0.75/MSE_SCALE; highlight=true)
text!(ax_l, 0.50, 0.10;
      text="units:  × 10⁻⁴ Φ₀²",
      align=(:center, :center), fontsize=18, color=GRAY, font=:italic)

# =============================================================================
# RIGHT COLUMN — 3D scene with labels only (no Bloch, no caption)
# =============================================================================
ax_r = Axis(body[1, 2]; backgroundcolor=BG,
            leftspinevisible=false, rightspinevisible=false,
            topspinevisible=false, bottomspinevisible=false,
            xticksvisible=false, yticksvisible=false,
            xticklabelsvisible=false, yticklabelsvisible=false,
            limits=(0,1,0,1))
hidedecorations!(ax_r)

# 3D scene fills the central band.  Width × height chosen so the image's
# native 4:3 landscape aspect is preserved given the right-panel pixel
# aspect (≈ 935 × 792 → aspect 1.18), avoiding horizontal stretch.
#   want  w_rel × 935 / (h_rel × 792) = 1.333
#   pick  w_rel = 0.90, h_rel = 0.80
scqubit_img = flatten_white(load(joinpath(@__DIR__, "blender_scqubit.png")))
img_x0, img_x1 = 0.05, 0.95
img_y0, img_y1 = 0.10, 0.82
image!(ax_r, img_x0..img_x1, img_y0..img_y1, rotr90(scqubit_img))

# ---------------------------------------------------------------------------
# Labels placed ENTIRELY OUTSIDE the image footprint (top strip y > 0.82,
# bottom strip y < 0.18) so there is zero overlap with 3D content.
# ---------------------------------------------------------------------------
function annotate_out!(ax, tip_xy::Tuple{<:Real,<:Real},
                       lbl_xy::Tuple{<:Real,<:Real}, lbl::AbstractString;
                       color=TEXT, fontsize=26, bold=true,
                       align=(:center, :center))
    lines!(ax, [lbl_xy[1], tip_xy[1]], [lbl_xy[2], tip_xy[2]];
           color=GRAY, linewidth=1.4, linestyle=:dash)
    scatter!(ax, [tip_xy[1]], [tip_xy[2]]; color=color, markersize=12)
    text!(ax, lbl_xy[1], lbl_xy[2]; text=lbl,
          align=align, fontsize=fontsize, color=color,
          font=bold ? :bold : :regular)
end

# ----- Two labels ABOVE the image (top strip, y ≈ 0.92) ----------------
# "External flux Φ_ext" → orange up-arrows above the SQUID
annotate_out!(ax_r, (0.62, 0.56), (0.46, 0.92),
              "External flux  Φ_ext"; color=ACCENT, fontsize=26)
# "Flux-bias line" → purple curving trace on the +x side of the chip
annotate_out!(ax_r, (0.81, 0.40), (0.80, 0.92),
              "Flux-bias line"; color=RGBf(0.55, 0.22, 0.62),
              fontsize=26)

# ----- Two labels BELOW the image (bottom strip, y ≈ 0.07) -------------
# "Transmon" → gold cross (x-mon) in the left half of the chip
annotate_out!(ax_r, (0.38, 0.43), (0.22, 0.07),
              "Transmon"; color=RGBf(0.65, 0.48, 0.12), fontsize=28)
# "SQUID + Josephson jcts" → the rectangular ring with blue JJ stacks
annotate_out!(ax_r, (0.62, 0.43), (0.62, 0.07),
              "SQUID + Josephson jcts"; color=NAVY, fontsize=26)

# =============================================================================
out = joinpath(@__DIR__, "scqubit_hero.png")
save(out, fig; px_per_unit=2.0)
println("Saved: $out")
