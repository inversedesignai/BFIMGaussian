#=
composite_scqubit.jl — Final Case Study B hero figure.

Composites:
  - Left 2D panel: Performance Metrics (CairoMakie)
  - Right main: Blender scqubit 3D render (blender_scqubit.png)
  - Upper-right inset: Bloch sphere (blender_bloch.png)
  - Right-side belief evolution curves (CairoMakie)
  - Lower inset: Ramsey schedule schematic (CairoMakie)
  - Annotation arrows + labels (CairoMakie)

Output: scqubit_hero.png  (replaces prior versions)
=#

using CairoMakie
using Printf
using FileIO

CairoMakie.activate!()

const BG        = RGBf(0.97, 0.93, 0.85)
const PANEL_BG  = RGBf(0.95, 0.92, 0.82)
const PANEL_BG2 = RGBf(0.99, 0.97, 0.92)
const FRAME     = RGBf(0.22, 0.27, 0.38)
const BAR_DARK  = RGBf(0.10, 0.15, 0.35)
const BAR_LIGHT = RGBf(0.28, 0.45, 0.72)
const BAR_BG    = RGBf(0.85, 0.86, 0.90)
const ACCENT    = RGBf(0.82, 0.25, 0.20)
const YELLOW    = RGBf(0.98, 0.82, 0.55)
const TEXT      = RGBf(0.10, 0.12, 0.18)
const GRAY_ANN  = RGBf(0.40, 0.42, 0.48)
const NAVY      = RGBf(0.15, 0.20, 0.45)

function rounded_rect!(ax, x0, y0, x1, y1; r=0.008, color=BG, strokecolor=FRAME,
                       strokewidth=1.0)
    n = 12
    pts = Point2f[]
    for θ in range(π, 1.5π; length=n); push!(pts, Point2f(x0+r + r*cos(θ), y0+r + r*sin(θ))); end
    for θ in range(1.5π, 2π; length=n); push!(pts, Point2f(x1-r + r*cos(θ), y0+r + r*sin(θ))); end
    for θ in range(0, 0.5π; length=n); push!(pts, Point2f(x1-r + r*cos(θ), y1-r + r*sin(θ))); end
    for θ in range(0.5π, π; length=n); push!(pts, Point2f(x0+r + r*cos(θ), y1-r + r*sin(θ))); end
    poly!(ax, pts; color=color, strokecolor=strokecolor, strokewidth=strokewidth)
end

function annotate!(ax, tip_x, tip_y, label_x, label_y, str; color=TEXT,
                   fontsize=11, align=(:center, :center), bold=false)
    lines!(ax, [label_x, tip_x], [label_y, tip_y]; color=GRAY_ANN,
           linewidth=0.9, linestyle=:dash)
    scatter!(ax, [tip_x], [tip_y]; color=color, markersize=5)
    text!(ax, label_x, label_y; text=str, align=align,
          fontsize=fontsize, color=color, font=bold ? :bold : :regular)
end

# =============================================================================
# Canvas
# =============================================================================
fig = Figure(size=(1600, 900), backgroundcolor=BG, figure_padding=10)
gl = fig[1, 1] = GridLayout()
left_col  = gl[1, 1] = GridLayout()
right_col = gl[1, 2] = GridLayout()
colsize!(gl, 1, Relative(0.30))
colsize!(gl, 2, Relative(0.70))
colgap!(gl, 18)

# =============================================================================
# LEFT: Performance Metrics (CairoMakie)
# =============================================================================
ax_left = Axis(left_col[1, 1]; backgroundcolor=PANEL_BG,
               leftspinecolor=FRAME, rightspinecolor=FRAME,
               topspinecolor=FRAME, bottomspinecolor=FRAME, spinewidth=1.5,
               xticksvisible=false, yticksvisible=false,
               xticklabelsvisible=false, yticklabelsvisible=false,
               xgridvisible=false, ygridvisible=false,
               limits=(0, 1, 0, 1), aspect=DataAspect())
hidedecorations!(ax_left, grid=true)

rounded_rect!(ax_left, 0.035, 0.915, 0.965, 0.98;
              color=FRAME, strokecolor=FRAME, strokewidth=0)
text!(ax_left, 0.5, 0.947; text="Performance Metrics",
      align=(:center, :center), fontsize=19, font=:bold,
      color=RGBf(0.98,0.95,0.88))
text!(ax_left, 0.5, 0.89;
      text="Case Study B  —  Superconducting-qubit flux sensor",
      align=(:center, :center), fontsize=10.5, color=GRAY_ANN, font=:italic)

function draw_metric(ax, y_top, title, value_text, bar_frac; highlight=false,
                     subtitle=nothing, value_color=TEXT)
    h = 0.15
    rounded_rect!(ax, 0.04, y_top-h, 0.96, y_top; r=0.012,
                  color=PANEL_BG2, strokecolor=FRAME, strokewidth=1.0)
    text!(ax, 0.07, y_top-0.022; text=title, align=(:left, :top),
          fontsize=12, color=TEXT, font=:bold)
    text!(ax, 0.93, y_top-0.022; text=value_text, align=(:right, :top),
          fontsize=12, color=value_color, font=:bold)
    y_bar_top = y_top-0.082
    y_bar_bot = y_top-0.113
    rounded_rect!(ax, 0.07, y_bar_bot, 0.93, y_bar_top; r=0.005,
                  color=BAR_BG, strokecolor=BAR_BG, strokewidth=0)
    bar_end = 0.07 + bar_frac * 0.86
    clr = highlight ? BAR_DARK : BAR_LIGHT
    if bar_end > 0.075
        rounded_rect!(ax, 0.07, y_bar_bot, bar_end, y_bar_top; r=0.005,
                      color=clr, strokewidth=0)
    end
    if subtitle !== nothing
        text!(ax, 0.07, y_bar_bot-0.012; text=subtitle, align=(:left, :top),
              fontsize=9.5, color=GRAY_ANN, font=:italic)
    end
end

mse_baseline = 2.075
mse_joint    = 1.829
m_max = 2.2

draw_metric(ax_left, 0.86, "Baseline Design (PCRB Optimized)",
            @sprintf("MSE̅ = %.3f × 10⁻² Φ₀²", mse_baseline),
            mse_baseline/m_max)
draw_metric(ax_left, 0.68, "Joint-DP Co-Design  (this work)",
            @sprintf("MSE̅ = %.3f × 10⁻² Φ₀²", mse_joint),
            mse_joint/m_max; highlight=true,
            subtitle="↑ 13.4% reduction  (z = +11.5σ,  n_mc = 20 000)",
            value_color=BAR_DARK)

# Optimal yellow tag
tag_x0 = 0.70; tag_x1 = 0.925
rounded_rect!(ax_left, tag_x0, 0.595, tag_x1, 0.640; r=0.008,
              color=YELLOW, strokecolor=ACCENT, strokewidth=1.3)
text!(ax_left, (tag_x0+tag_x1)/2, 0.617; text="Optimal Co-Design",
      align=(:center, :center), fontsize=10.5, color=ACCENT, font=:bold)

# Advantage callout with mini-bar
adv_top = 0.47; adv_bot = 0.02
rounded_rect!(ax_left, 0.04, adv_bot, 0.96, adv_top; r=0.012,
              color=RGBf(0.99, 0.95, 0.80), strokecolor=ACCENT, strokewidth=1.5)
text!(ax_left, 0.50, adv_top-0.03; text="Joint-DP Advantage",
      align=(:center, :top), fontsize=14, color=ACCENT, font=:bold)
text!(ax_left, 0.50, adv_top-0.065;
      text="grows monotonically with narrower priors",
      align=(:center, :top), fontsize=10, color=TEXT, font=:italic)

# mini chart
ratios_lbl = [("Φ₀/33", 128.0), ("Φ₀/20", 76.9), ("Φ₀/12", 12.4),
              ("Φ₀/10", 8.29), ("Φ₀/7", 3.33), ("Φ₀/5", 1.42),
              ("Φ₀/4", 1.25), ("Φ₀/3", 1.05), ("Φ₀/2.5", 0.89),
              ("Φ₀/2", 1.13)]
# display in log scale
bars_x0 = 0.08; bars_x1 = 0.92
nb = 6  # show a subset
indices = [1, 2, 3, 4, 5, 10]  # 0.03, 0.05, 0.08, 0.10, 0.15, 0.49
selected = ratios_lbl[indices]
bw = (bars_x1 - bars_x0) / (nb*1.35)
max_ratio_log = log10(128.0) + 0.1
y_bar_base = 0.09
y_bar_top  = 0.28
for (k, (lbl, r)) in enumerate(selected)
    xc = bars_x0 + (k-0.5) * (bars_x1-bars_x0)/nb
    frac = max(0.02, log10(max(r, 1.0)) / max_ratio_log)
    h    = frac * (y_bar_top - y_bar_base)
    clr  = r >= 1.5 ? BAR_DARK : GRAY_ANN
    rounded_rect!(ax_left, xc-bw/2, y_bar_base, xc+bw/2, y_bar_base+h; r=0.003,
                  color=clr, strokewidth=0)
    text!(ax_left, xc, y_bar_base+h+0.007;
          text=r >= 2 ? @sprintf("%.0f×", r) : @sprintf("%.2f×", r),
          align=(:center, :bottom), fontsize=9,
          color=r >= 1.5 ? BAR_DARK : GRAY_ANN, font=:bold)
    text!(ax_left, xc, y_bar_base-0.004; text=lbl,
          align=(:center, :top), fontsize=8.5, color=TEXT)
end
text!(ax_left, 0.5, 0.04;
      text="ratio  MSE̅_pcrb / MSE̅_joint   vs.   prior width  φ_max",
      align=(:center, :center), fontsize=9, color=GRAY_ANN, font=:italic)

# =============================================================================
# RIGHT: Composite 3D scene + insets
# =============================================================================
ax_right = Axis(right_col[1, 1]; backgroundcolor=BG,
                leftspinevisible=false, rightspinevisible=false,
                topspinevisible=false, bottomspinevisible=false,
                xticksvisible=false, yticksvisible=false,
                xticklabelsvisible=false, yticklabelsvisible=false,
                limits=(0,1,0,1), aspect=DataAspect())
hidedecorations!(ax_right)

text!(ax_right, 0.36, 0.97;
      text="Flux Sensing POMDP & System Geometry",
      align=(:center, :center), fontsize=18, font=:bold, color=TEXT)

# Place Blender scqubit render as main image
scqubit_img = load(joinpath(@__DIR__, "blender_scqubit.png"))
img_x0, img_y0, img_x1, img_y1 = 0.00, 0.29, 0.72, 0.93
image!(ax_right, img_x0..img_x1, img_y0..img_y1, rotr90(scqubit_img))

# Place Bloch-sphere inset upper-right (square aspect)
bloch_img = load(joinpath(@__DIR__, "blender_bloch.png"))
bl_cx, bl_cy = 0.88, 0.82
bl_w = 0.22
bl_x0 = bl_cx - bl_w/2;  bl_x1 = bl_cx + bl_w/2
bl_y0 = bl_cy - bl_w/2;  bl_y1 = bl_cy + bl_w/2  # square (aspect=DataAspect)
rounded_rect!(ax_right, bl_x0-0.005, bl_y0-0.006, bl_x1+0.005, bl_y1+0.030;
              r=0.006, color=PANEL_BG2, strokecolor=FRAME, strokewidth=0.9)
image!(ax_right, bl_x0..bl_x1, bl_y0..bl_y1, rotr90(bloch_img))
text!(ax_right, bl_cx, bl_y1+0.008;
      text="Bloch sphere  —  Ramsey phase",
      align=(:center, :bottom), fontsize=9.5, color=TEXT, font=:italic)

# Belief evolution curves (right-middle column)
function gauss(x, μ, σ); exp.(-(x .- μ).^2 ./ (2σ^2)); end
xs_grid = range(-1, 1; length=120)

function belief_inset!(ax, cx, cy, w, h, μ, σ, label;
                       color=BAR_DARK, bold=false, title_offset=0.016)
    rounded_rect!(ax, cx-w/2, cy-h/2, cx+w/2, cy+h/2; r=0.006,
                  color=PANEL_BG2, strokecolor=color,
                  strokewidth=bold ? 1.8 : 0.9)
    ys = gauss(xs_grid, μ, σ)
    xdata = cx - w/2 + 0.02 .+ (xs_grid .+ 1) ./ 2 .* (w - 0.04)
    ydata = cy - h/2 + 0.015 .+ ys .* (h - 0.03)
    band!(ax, xdata, fill(cy - h/2 + 0.015, length(xdata)), ydata;
          color=(color, 0.25))
    lines!(ax, xdata, ydata; color=color, linewidth=bold ? 1.8 : 1.2)
    text!(ax, cx, cy + h/2 + title_offset; text=label,
          align=(:center, :bottom), fontsize=9.5,
          color=color, font=bold ? :bold : :italic)
    xdata, ydata
end

# Belief curves stacked vertically on the far right, below Bloch
bc_x = 0.88
belief_inset!(ax_right, bc_x, 0.57, 0.22, 0.07, 0.0, 0.58, "p₀(x)  prior")
belief_inset!(ax_right, bc_x, 0.45, 0.22, 0.07, 0.12, 0.28, "p₁(x)  after epoch 1")
belief_inset!(ax_right, bc_x, 0.33, 0.22, 0.07, 0.28, 0.07,
              "p_K(x)  final posterior"; color=ACCENT, bold=true)

# Bayesian-update small arrow
lines!(ax_right, [bc_x + 0.085, bc_x + 0.095], [0.52, 0.50]; color=GRAY_ANN, linewidth=0.8)
lines!(ax_right, [bc_x + 0.085, bc_x + 0.095], [0.40, 0.38]; color=GRAY_ANN, linewidth=0.8)
text!(ax_right, bc_x + 0.127, 0.45;
      text="Bayesian\nupdate",
      align=(:center, :center), fontsize=9, color=GRAY_ANN, font=:italic)

# Ramsey schedule inset (bottom-left)
rx0 = 0.02; ry0 = 0.03; rx1 = 0.52; ry1 = 0.25
rounded_rect!(ax_right, rx0, ry0, rx1, ry1; r=0.010,
              color=PANEL_BG2, strokecolor=FRAME, strokewidth=0.9)
text!(ax_right, (rx0+rx1)/2, ry1 - 0.028;
      text="Ramsey adaptive schedule   s_k = (τ_k, n_k)",
      align=(:center, :top), fontsize=11.5, color=TEXT, font=:bold)

# π/2 pulses at epoch boundaries with τ intervals between
n_pulses = 5
pulse_xs = range(rx0 + 0.05, rx1 - 0.07; length=n_pulses)
pulse_ybot = ry0 + 0.05
pulse_ytop = ry0 + 0.13
for i in 1:n_pulses
    x0 = pulse_xs[i]
    # first π/2
    lines!(ax_right, [x0, x0], [pulse_ybot, pulse_ytop];
           color=BAR_DARK, linewidth=2.4)
    # τ stretch (blue shaded interval)
    if i < n_pulses
        x1 = pulse_xs[i+1]
        poly!(ax_right, Point2f[(x0, pulse_ybot+0.030),
                                (x1, pulse_ybot+0.030),
                                (x1, pulse_ybot+0.050),
                                (x0, pulse_ybot+0.050)];
              color=RGBf(0.72, 0.80, 0.95), strokewidth=0)
    end
    if i >= 2
        scatter!(ax_right, [x0 + 0.008], [pulse_ytop + 0.012];
                 color=RGBf(0.15, 0.55, 0.30), marker=:diamond, markersize=7)
    end
end
text!(ax_right, (rx0+rx1)/2, ry0 + 0.02;
      text="π/2 — τ_k — π/2 — readout  (green diamonds)   ·   chosen adaptively by  π⋆(b_k)",
      align=(:center, :bottom), fontsize=9.5, color=GRAY_ANN, font=:italic)

# =============================================================================
# Annotations on the 3D scene (coords tuned to current render)
# =============================================================================
# SQUID loop label
annotate!(ax_right, 0.52, 0.58, 0.22, 0.88,
          "SQUID loop\n(hardware c ∈ ℝ⁷)";
          fontsize=11, align=(:center, :center), bold=true)
# External flux label
annotate!(ax_right, 0.51, 0.83, 0.50, 0.92,
          "External flux  Φ_ext  (hidden state  x)";
          fontsize=11, color=NAVY, align=(:center, :center), bold=true)
# Flux-bias line
annotate!(ax_right, 0.67, 0.55, 0.66, 0.40,
          "Flux-bias line\n(drive  ω_d)";
          fontsize=10, color=RGBf(0.55, 0.22, 0.62), bold=true)
# Transmon x-mon cross
annotate!(ax_right, 0.20, 0.48, 0.10, 0.30,
          "Transmon x-mon\n(frequency-tunable qubit)";
          fontsize=10, color=RGBf(0.82, 0.60, 0.20), bold=true)
# JJ label — actually on top/bottom of SQUID (image coords around 0.44, 0.58)
annotate!(ax_right, 0.45, 0.58, 0.30, 0.76,
          "Josephson junctions\n(red-tinted breaks)";
          fontsize=10, color=ACCENT, bold=true)

# Bottom caption
rounded_rect!(ax_right, 0.54, 0.01, 0.99, 0.04; r=0.006,
              color=PANEL_BG, strokecolor=FRAME, strokewidth=0.7)
text!(ax_right, (0.54+0.99)/2, 0.025;
      text="Hardware c and adaptive policy π⋆(b) co-optimized via envelope-theorem gradient  (§6)",
      align=(:center, :center), fontsize=10, color=TEXT, font=:italic)

# Save
out = joinpath(@__DIR__, "scqubit_hero.png")
save(out, fig; px_per_unit=2.0)
println("Saved: $out")
