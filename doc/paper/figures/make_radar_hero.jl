#=
make_radar_hero.jl — Case Study A hero figure in Gemini-editorial aesthetic.

Replicates the warm beige/navy palette of `gemini_radar.png`, with the exact
enumeration values from Case Study A (K=16, N=4, narrow-beam m=1 / wide m=8
regimes, in-family optimum m*=9):

  Oracle     V_oracle(m=9)   = 1.879 nats    (header comparison baseline)
  Adaptive   V_adaptive(m=9) = 1.704 nats    (Optimal Adaptive Value)
  Fixed      V_fixed(m=9)    = 1.401 nats
  EVPI       E[IG](m=9)      = 0.478 nats

  Headline: EVPI-argmax (m=1 or m=16) loses ~2.8× in attainable
            V_adaptive to the joint-DP optimum at m*=9.

Output: doc/paper/figures/radar_hero.png
=#

using CairoMakie
using Printf

CairoMakie.activate!()

# --- palette ---
const BG        = RGBf(0.98, 0.95, 0.88)
const PANEL_BG  = RGBf(0.95, 0.92, 0.82)
const PANEL_BG2 = RGBf(0.99, 0.97, 0.92)
const FRAME     = RGBf(0.22, 0.27, 0.38)
const BAR_DARK  = RGBf(0.10, 0.15, 0.35)
const BAR_LIGHT = RGBf(0.28, 0.45, 0.72)
const BAR_BG    = RGBf(0.85, 0.86, 0.90)
const ACCENT    = RGBf(0.82, 0.25, 0.20)
const GOLD      = RGBf(0.82, 0.60, 0.18)
const YELLOW    = RGBf(0.98, 0.82, 0.55)
const TEXT      = RGBf(0.10, 0.12, 0.18)
const GRAY_ANN  = RGBf(0.40, 0.42, 0.48)
const GREEN     = RGBf(0.15, 0.55, 0.30)
const METAL     = RGBf(0.60, 0.62, 0.68)  # antenna baseplate

function rounded_rect!(ax, x0, y0, x1, y1; r=0.008, color=BG, strokecolor=FRAME,
                       strokewidth=1.0)
    n = 12
    pts = Point2f[]
    for θ in range(π, 1.5π; length=n)
        push!(pts, Point2f(x0+r + r*cos(θ), y0+r + r*sin(θ)))
    end
    for θ in range(1.5π, 2π; length=n)
        push!(pts, Point2f(x1-r + r*cos(θ), y0+r + r*sin(θ)))
    end
    for θ in range(0, 0.5π; length=n)
        push!(pts, Point2f(x1-r + r*cos(θ), y1-r + r*sin(θ)))
    end
    for θ in range(0.5π, π; length=n)
        push!(pts, Point2f(x0+r + r*cos(θ), y1-r + r*sin(θ)))
    end
    poly!(ax, pts; color=color, strokecolor=strokecolor, strokewidth=strokewidth)
end

function annotate!(ax, tip_x, tip_y, label_x, label_y, str; color=TEXT,
                   fontsize=11, align=(:center, :center), bold=false, tip_color=nothing)
    lines!(ax, [label_x, tip_x], [label_y, tip_y]; color=GRAY_ANN,
           linewidth=0.8, linestyle=:dash)
    scatter!(ax, [tip_x], [tip_y]; color=isnothing(tip_color) ? color : tip_color,
             markersize=5)
    text!(ax, label_x, label_y; text=str, align=align,
          fontsize=fontsize, color=color, font=bold ? :bold : :regular)
end

fig = Figure(size=(1450, 820), backgroundcolor=BG, figure_padding=10)
gl = fig[1, 1] = GridLayout()
left_col  = gl[1, 1] = GridLayout()
right_col = gl[1, 2] = GridLayout()
colsize!(gl, 1, Relative(0.34))
colsize!(gl, 2, Relative(0.66))
colgap!(gl, 20)

# =============================================================================
# LEFT: Metrics
# =============================================================================
ax_left = Axis(left_col[1, 1]; backgroundcolor=PANEL_BG,
               leftspinecolor=FRAME, rightspinecolor=FRAME,
               topspinecolor=FRAME, bottomspinecolor=FRAME, spinewidth=1.5,
               xticksvisible=false, yticksvisible=false,
               xticklabelsvisible=false, yticklabelsvisible=false,
               xgridvisible=false, ygridvisible=false,
               limits=(0,1,0,1), aspect=DataAspect())
hidedecorations!(ax_left, grid=true)

rounded_rect!(ax_left, 0.035, 0.915, 0.965, 0.98;
              color=FRAME, strokecolor=FRAME, strokewidth=0)
text!(ax_left, 0.5, 0.947; text="Information Metrics",
      align=(:center, :center), fontsize=19, font=:bold,
      color=RGBf(0.98,0.95,0.88))
text!(ax_left, 0.5, 0.89; text="Case Study A  —  Radar beam-search POMDP (K=16, N=4)",
      align=(:center, :center), fontsize=10.5, color=GRAY_ANN, font=:italic)

function draw_metric(ax, y_top, title, value_text, bar_frac; highlight=false,
                     subtitle=nothing, value_color=TEXT, is_evpi=false)
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
    clr = highlight ? BAR_DARK : (is_evpi ? ACCENT : BAR_LIGHT)
    if bar_end > 0.075
        rounded_rect!(ax, 0.07, y_bar_bot, bar_end, y_bar_top; r=0.005,
                      color=clr, strokewidth=0)
    end
    if subtitle !== nothing
        text!(ax, 0.07, y_bar_bot-0.012; text=subtitle, align=(:left, :top),
              fontsize=9, color=GRAY_ANN, font=:italic)
    end
end

# m* = 9 values from paper
v_oracle = 1.879
v_adapt  = 1.704
v_fixed  = 1.401
v_evpi   = 0.478
# Normalized to max = v_oracle (2.1 for bar axis headroom)
vmax = 2.1

draw_metric(ax_left, 0.86, "Oracle Value  V_oracle",
            @sprintf("%.3f  nats", v_oracle), v_oracle/vmax;
            subtitle="clairvoyant upper bound (knows true target)")
draw_metric(ax_left, 0.69, "Adaptive Value  V_adaptive  (this work)",
            @sprintf("%.3f  nats", v_adapt), v_adapt/vmax; highlight=true,
            value_color=BAR_DARK,
            subtitle="Bellman-optimal policy on hardware c = top-hat m⋆=9")
# Optimal tag
rounded_rect!(ax_left, 0.62, 0.606, 0.925, 0.650; r=0.008,
              color=YELLOW, strokecolor=ACCENT, strokewidth=1.3)
text!(ax_left, 0.7725, 0.628; text="Optimal Adaptive Value",
      align=(:center, :center), fontsize=10, color=ACCENT, font=:bold)

draw_metric(ax_left, 0.52, "Fixed-Schedule Value  V_fixed",
            @sprintf("%.3f  nats", v_fixed), v_fixed/vmax;
            subtitle="best non-adaptive schedule at c = m⋆=9")
draw_metric(ax_left, 0.35, "Ignorance Gap  E[IG] = V_oracle − V_fixed",
            @sprintf("%.3f  nats", v_evpi), v_evpi/vmax; is_evpi=true,
            subtitle="Expected Value of Perfect Information")

# Headline callout
adv_top = 0.17; adv_bot = 0.02
rounded_rect!(ax_left, 0.04, adv_bot, 0.96, adv_top; r=0.012,
              color=RGBf(0.99, 0.95, 0.80), strokecolor=ACCENT, strokewidth=1.5)
text!(ax_left, 0.50, adv_top-0.03; text="Joint-DP Advantage",
      align=(:center, :top), fontsize=13, color=ACCENT, font=:bold)
text!(ax_left, 0.50, adv_top-0.075;
      text="EVPI-argmax geometry (m=1 or m=16) loses a factor of",
      align=(:center, :top), fontsize=10.5, color=TEXT)
text!(ax_left, 0.50, adv_top-0.115;
      text="2.75× in attainable V_adaptive to the joint-DP optimum",
      align=(:center, :top), fontsize=12, color=ACCENT, font=:bold)
text!(ax_left, 0.50, 0.035;
      text="(EVPI disagrees with V_adaptive argmax; only the joint solve picks m⋆=9)",
      align=(:center, :center), fontsize=8.8, color=GRAY_ANN, font=:italic)

# =============================================================================
# RIGHT: Radar geometry
# =============================================================================
ax_right = Axis(right_col[1, 1]; backgroundcolor=BG,
                leftspinevisible=false, rightspinevisible=false,
                topspinevisible=false, bottomspinevisible=false,
                xticksvisible=false, yticksvisible=false,
                xticklabelsvisible=false, yticklabelsvisible=false,
                limits=(0,1,0,1), aspect=DataAspect())
hidedecorations!(ax_right)

text!(ax_right, 0.5, 0.97;
      text="Radar Beam-Search Geometry  (K=16 target ring,  top-hat beam m)",
      align=(:center, :center), fontsize=18, font=:bold, color=TEXT)

# ---------- phased-array antenna baseplate (bottom of scene, pseudo-3D) ----------
plate = Point2f[(0.25, 0.08), (0.70, 0.08), (0.78, 0.22), (0.33, 0.22)]
poly!(ax_right, plate; color=METAL, strokecolor=RGBf(0.45,0.46,0.50),
      strokewidth=1.0)
# 4x4 grid of gold antenna elements
for i in 0:3, j in 0:3
    # pseudo-3D: x and y skewed by projection
    px = 0.30 + 0.10*j + 0.018*i
    py = 0.11 + 0.024*i
    # element post
    lines!(ax_right, [px, px], [py, py+0.035]; color=GOLD, linewidth=2.5)
    # element head (small diamond)
    poly!(ax_right, Point2f[(px-0.013, py+0.035), (px, py+0.048),
                            (px+0.013, py+0.035), (px, py+0.022)];
          color=GOLD, strokecolor=RGBf(0.55,0.40,0.10), strokewidth=0.5)
end

# Annotation: baseplate label
text!(ax_right, 0.82, 0.12; text="Phased-Array Antenna\n(16 elements)",
      align=(:left, :center), fontsize=10, color=TEXT, font=:italic)

# ---------- target ring (K=16 cells arranged in a circle above antenna) ----------
ring_cx = 0.50; ring_cy = 0.63
ring_r  = 0.28
K = 16
target_idx   = 4   # true target cell (1-indexed)
belief_idx   = 7   # current belief mode
# Ellipse squished for perspective
for k in 1:K
    φ = -π/2 + (k-1) * 2π/K
    x = ring_cx + ring_r * cos(φ)
    y = ring_cy + 0.50*ring_r * sin(φ)
    color = if k == target_idx
        ACCENT
    elseif k == belief_idx
        RGBf(0.30, 0.60, 0.35)
    else
        RGBf(0.80, 0.82, 0.85)
    end
    stroke_c = if k == target_idx
        RGBf(0.60, 0.10, 0.10)
    elseif k == belief_idx
        RGBf(0.10, 0.35, 0.15)
    else
        RGBf(0.55, 0.58, 0.62)
    end
    # pseudo-3D cell as small rounded square
    sx = 0.030; sy = 0.028
    rounded_rect!(ax_right, x-sx/2, y-sy/2, x+sx/2, y+sy/2; r=0.006,
                  color=color, strokecolor=stroke_c, strokewidth=1.2)
    # cell number
    text!(ax_right, x, y; text=string(k), align=(:center, :center),
          fontsize=9, color=k in (target_idx, belief_idx) ? :white : TEXT,
          font=k in (target_idx, belief_idx) ? :bold : :regular)
end

# ---------- beam cone (top-hat angular pattern) ----------
# Emerging from centroid of antenna baseplate toward belief mode
anchor_x = 0.45; anchor_y = 0.18
# belief mode position
φ_b = -π/2 + (belief_idx-1) * 2π/K
bmx = ring_cx + ring_r * cos(φ_b)
bmy = ring_cy + 0.50*ring_r * sin(φ_b)
# Beam cone (triangle-like translucent polygon)
direction = [bmx - anchor_x, bmy - anchor_y]
len = sqrt(direction[1]^2 + direction[2]^2)
u = [-direction[2], direction[1]] ./ len
beam_halfwidth_near = 0.005
beam_halfwidth_far  = 0.048  # cone spread
p1 = [anchor_x + beam_halfwidth_near*u[1], anchor_y + beam_halfwidth_near*u[2]]
p2 = [anchor_x - beam_halfwidth_near*u[1], anchor_y - beam_halfwidth_near*u[2]]
p3 = [bmx - beam_halfwidth_far*u[1], bmy - beam_halfwidth_far*u[2]]
p4 = [bmx + beam_halfwidth_far*u[1], bmy + beam_halfwidth_far*u[2]]
poly!(ax_right, Point2f[(p1[1], p1[2]), (p2[1], p2[2]),
                        (p3[1], p3[2]), (p4[1], p4[2])];
      color=(ACCENT, 0.30), strokewidth=0)

# central beam spine
lines!(ax_right, [anchor_x, bmx], [anchor_y, bmy];
       color=ACCENT, linewidth=1.0, linestyle=:dash)

# ghost beam (next-step adaptive pointing — toward cell adjacent to true target)
φ_next = -π/2 + (target_idx-1) * 2π/K  # bisection step toward true
nxt_x = ring_cx + ring_r * cos(φ_next)
nxt_y = ring_cy + 0.50*ring_r * sin(φ_next)
direction2 = [nxt_x - anchor_x, nxt_y - anchor_y]
len2 = sqrt(direction2[1]^2 + direction2[2]^2)
u2 = [-direction2[2], direction2[1]] ./ len2
p1b = [anchor_x + 0.004*u2[1], anchor_y + 0.004*u2[2]]
p2b = [anchor_x - 0.004*u2[1], anchor_y - 0.004*u2[2]]
p3b = [nxt_x - 0.038*u2[1], nxt_y - 0.038*u2[2]]
p4b = [nxt_x + 0.038*u2[1], nxt_y + 0.038*u2[2]]
poly!(ax_right, Point2f[(p1b[1], p1b[2]), (p2b[1], p2b[2]),
                        (p3b[1], p3b[2]), (p4b[1], p4b[2])];
      color=(BAR_LIGHT, 0.20), strokecolor=BAR_LIGHT, strokewidth=0.8,
      linestyle=:dash)

# =============================================================================
# Annotations around the scene
# =============================================================================
annotate!(ax_right, ring_cx + ring_r*cos(-π/2 + (8-1)*2π/K),
          ring_cy + 0.5*ring_r*sin(-π/2 + (8-1)*2π/K),
          0.52, 0.92,
          "Target cells  (K=16 ring)";
          fontsize=11, color=TEXT)

# true target label
annotate!(ax_right, ring_cx + ring_r*cos(-π/2 + (target_idx-1)*2π/K) + 0.035,
          ring_cy + 0.5*ring_r*sin(-π/2 + (target_idx-1)*2π/K) + 0.015,
          0.88, 0.80,
          "True target state\n(cell  $target_idx)";
          fontsize=11, color=ACCENT, bold=true)

# belief mode label
annotate!(ax_right, ring_cx + ring_r*cos(-π/2 + (belief_idx-1)*2π/K) - 0.035,
          ring_cy + 0.5*ring_r*sin(-π/2 + (belief_idx-1)*2π/K),
          0.16, 0.78,
          "Current belief mode\n(cell  $belief_idx,  π⋆ steers here)";
          fontsize=10, color=GREEN)

# current beam pointing label
annotate!(ax_right, (anchor_x + bmx)/2, (anchor_y + bmy)/2,
          0.17, 0.47,
          "Beam pointing\nat belief mode\n(width  m = 3)";
          fontsize=10, color=ACCENT, bold=true)

# next-step adaptive redirect
annotate!(ax_right, (anchor_x + nxt_x)/2, (anchor_y + nxt_y)/2,
          0.85, 0.45,
          "Next-step beam\n(π⋆ redirects\nafter observation)";
          fontsize=10, color=BAR_LIGHT)

# =============================================================================
# Observation model inset (bottom-right)
# =============================================================================
ox = 0.68; oy = 0.26; ow = 0.28; oh = 0.10
rounded_rect!(ax_right, ox, oy-oh, ox+ow, oy+oh; r=0.008,
              color=PANEL_BG2, strokecolor=FRAME, strokewidth=0.9)
text!(ax_right, ox+ow/2, oy+oh-0.014;
      text="Observation model",
      align=(:center, :top), fontsize=10.5, color=TEXT, font=:bold)
text!(ax_right, ox+ow/2, oy+0.007;
      text="p(y=1 | α) = 1 − (1 − G(α))^N_eff",
      align=(:center, :center), fontsize=9.5, color=TEXT, font=:italic)
text!(ax_right, ox+ow/2, oy-0.025;
      text="top-hat gain  G_c(α) = 1{|α − α_beam| < m·Δα/2}",
      align=(:center, :center), fontsize=9, color=GRAY_ANN)
text!(ax_right, ox+ow/2, oy-0.048;
      text="action  s_k = α_beam  ∈ {α_1, ..., α_K}",
      align=(:center, :center), fontsize=9, color=GRAY_ANN)

# =============================================================================
# Caption band
# =============================================================================
rounded_rect!(ax_right, 0.04, 0.02, 0.96, 0.085; r=0.006,
              color=PANEL_BG, strokecolor=FRAME, strokewidth=0.7)
text!(ax_right, 0.50, 0.053;
      text="Hardware c = beam-width  m,  policy π⋆ = Bellman-optimal bisection over the ring  (Case Study A, §7)",
      align=(:center, :center), fontsize=10.5, color=TEXT, font=:italic)

out = joinpath(@__DIR__, "radar_hero.png")
save(out, fig; px_per_unit=2.0)
println("Saved: $out")
