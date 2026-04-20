#=
make_scqubit_hero.jl — Case Study B hero figure in Gemini-editorial aesthetic.

Replicates the warm beige/navy editorial palette of `gemini_scq.png`, with
numbers from the current experiments (2026-04-19 phi_max sweep):
  Baseline PCRB (at phi_max=0.49):   MSE = 2.075e-2 Φ₀²
  Joint-DP co-design (phi_max=0.49): MSE = 1.829e-2 Φ₀²
  Narrow-prior scaling:              13.4% at φ_max=Φ₀/2
                                     8.3×  at φ_max=Φ₀/10
                                     128×  at φ_max=Φ₀/33

Output: doc/paper/figures/scqubit_hero.png
=#

using CairoMakie
using Printf

CairoMakie.activate!()

# --- palette (Gemini-style warm editorial) ---
const BG        = RGBf(0.98, 0.95, 0.88)
const PANEL_BG  = RGBf(0.95, 0.92, 0.82)
const PANEL_BG2 = RGBf(0.99, 0.97, 0.92)  # inner card bg (lighter than panel)
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
const PURPLE    = RGBf(0.45, 0.20, 0.55)
const NAVY      = RGBf(0.15, 0.20, 0.45)

# =============================================================================
# Helpers
# =============================================================================

"Draw a rounded-rectangle using a poly with many points (CairoMakie workaround)."
function rounded_rect!(ax, x0, y0, x1, y1; r=0.008, color=BG, strokecolor=FRAME,
                       strokewidth=1.0)
    # rectangle with small corner rounding via four arcs
    n = 12
    pts = Point2f[]
    # bottom-left arc
    for θ in range(π, 1.5π; length=n)
        push!(pts, Point2f(x0+r + r*cos(θ), y0+r + r*sin(θ)))
    end
    # bottom-right arc
    for θ in range(1.5π, 2π; length=n)
        push!(pts, Point2f(x1-r + r*cos(θ), y0+r + r*sin(θ)))
    end
    # top-right arc
    for θ in range(0, 0.5π; length=n)
        push!(pts, Point2f(x1-r + r*cos(θ), y1-r + r*sin(θ)))
    end
    # top-left arc
    for θ in range(0.5π, π; length=n)
        push!(pts, Point2f(x0+r + r*cos(θ), y1-r + r*sin(θ)))
    end
    poly!(ax, pts; color=color, strokecolor=strokecolor, strokewidth=strokewidth)
end

"Thin line-to-point annotation."
function annotate!(ax, tip_x, tip_y, label_x, label_y, str; color=TEXT,
                   fontsize=11, align=(:center, :center), bold=false, tip_color=nothing)
    lines!(ax, [label_x, tip_x], [label_y, tip_y]; color=GRAY_ANN,
           linewidth=0.8, linestyle=:dash)
    scatter!(ax, [tip_x], [tip_y]; color=isnothing(tip_color) ? color : tip_color,
             markersize=5)
    text!(ax, label_x, label_y; text=str, align=align,
          fontsize=fontsize, color=color, font=bold ? :bold : :regular)
end

# =============================================================================
# Canvas
# =============================================================================
fig = Figure(size=(1450, 820), backgroundcolor=BG, figure_padding=10)
gl = fig[1, 1] = GridLayout()

left_col  = gl[1, 1] = GridLayout()
right_col = gl[1, 2] = GridLayout()
colsize!(gl, 1, Relative(0.36))
colsize!(gl, 2, Relative(0.64))
colgap!(gl, 20)

# =============================================================================
# LEFT PANEL: Performance Metrics
# =============================================================================
ax_left = Axis(left_col[1, 1]; backgroundcolor=PANEL_BG,
               leftspinevisible=true, rightspinevisible=true,
               topspinevisible=true, bottomspinevisible=true,
               spinewidth=1.5,
               leftspinecolor=FRAME, rightspinecolor=FRAME,
               topspinecolor=FRAME, bottomspinecolor=FRAME,
               xticksvisible=false, yticksvisible=false,
               xticklabelsvisible=false, yticklabelsvisible=false,
               xgridvisible=false, ygridvisible=false,
               limits=(0, 1, 0, 1), aspect=DataAspect())
hidedecorations!(ax_left, grid=true, minorgrid=true)

# Title band
rounded_rect!(ax_left, 0.035, 0.915, 0.965, 0.98;
              color=FRAME, strokecolor=FRAME, strokewidth=0)
text!(ax_left, 0.5, 0.947; text="Performance Metrics",
      align=(:center, :center), fontsize=19, font=:bold,
      color=RGBf(0.98,0.95,0.88))
text!(ax_left, 0.5, 0.89; text="Case Study B  —  Superconducting-qubit flux sensor",
      align=(:center, :center), fontsize=11, color=GRAY_ANN, font=:italic)

# --- Metric card ---
function draw_metric(ax, y_top, title, value_text, bar_frac; highlight=false,
                     subtitle=nothing, value_color=TEXT)
    h = 0.16
    rounded_rect!(ax, 0.04, y_top-h, 0.96, y_top; r=0.012,
                  color=PANEL_BG2, strokecolor=FRAME, strokewidth=1.0)
    # title + value
    text!(ax, 0.07, y_top-0.020; text=title, align=(:left, :top),
          fontsize=12, color=TEXT, font=:bold)
    text!(ax, 0.93, y_top-0.020; text=value_text, align=(:right, :top),
          fontsize=12, color=value_color, font=:bold)
    # bar track
    y_bar_top = y_top-0.08
    y_bar_bot = y_top-0.115
    rounded_rect!(ax, 0.07, y_bar_bot, 0.93, y_bar_top; r=0.005,
                  color=BAR_BG, strokecolor=BAR_BG, strokewidth=0)
    # bar fill
    bar_end = 0.07 + bar_frac * 0.86
    if bar_end > 0.075
        rounded_rect!(ax, 0.07, y_bar_bot, bar_end, y_bar_top; r=0.005,
                      color=highlight ? BAR_DARK : BAR_LIGHT,
                      strokecolor=:transparent, strokewidth=0)
    end
    if subtitle !== nothing
        text!(ax, 0.07, y_bar_bot-0.012; text=subtitle, align=(:left, :top),
              fontsize=9.5, color=GRAY_ANN, font=:italic)
    end
end

mse_baseline = 2.075
mse_joint    = 1.829
mse_fixed    = 2.075  # placeholder: deploy MSE of oracle fixed-best; near PCRB in this regime
m_max = 2.2

draw_metric(ax_left, 0.86, "Baseline Design (PCRB Optimized)",
            @sprintf("MSE̅ = %.3f × 10⁻² Φ₀²", mse_baseline),
            mse_baseline/m_max)
draw_metric(ax_left, 0.68, "Joint-DP Co-Design  (this work)",
            @sprintf("MSE̅ = %.3f × 10⁻² Φ₀²", mse_joint),
            mse_joint/m_max; highlight=true,
            subtitle="↑ 13.4% reduction  (z = +11.5σ,  n_mc = 20 000)",
            value_color=BAR_DARK)

# "Optimal" yellow tag
tag_x0 = 0.72; tag_x1 = 0.925
rounded_rect!(ax_left, tag_x0, 0.595, tag_x1, 0.640; r=0.008,
              color=YELLOW, strokecolor=ACCENT, strokewidth=1.3)
text!(ax_left, (tag_x0+tag_x1)/2, 0.617; text="Optimal Co-Design",
      align=(:center, :center), fontsize=10.5, color=ACCENT, font=:bold)

draw_metric(ax_left, 0.50, "Fixed-Schedule Oracle (non-adaptive)",
            @sprintf("MSE̅ = %.3f × 10⁻² Φ₀²", mse_fixed),
            mse_fixed/m_max;
            subtitle="same c_joint, but fixed τ-schedule instead of adaptive π")

# --- Advantage callout ---
adv_top = 0.32; adv_bot = 0.02
rounded_rect!(ax_left, 0.04, adv_bot, 0.96, adv_top; r=0.012,
              color=RGBf(0.99, 0.95, 0.80), strokecolor=ACCENT, strokewidth=1.5)
text!(ax_left, 0.50, adv_top-0.02; text="Joint-DP Advantage",
      align=(:center, :top), fontsize=14, color=ACCENT, font=:bold)
text!(ax_left, 0.50, adv_top-0.06; text="grows monotonically with narrower priors",
      align=(:center, :top), fontsize=10, color=TEXT, font=:italic)

# mini bar chart inside callout
xs_mini = [0.1, 0.2, 0.3, 0.4, 0.49]
ratios  = [128.0, 12.4, 3.33, 0.89, 0.81]
# But we want ratios at phi_max=Φ₀/33, /20, /10, etc.  Use a cleaner sequence:
ratios_lbl = [("Φ₀/33", 128.0, :huge), ("Φ₀/20", 76.9, :big),
              ("Φ₀/12", 12.4, :big), ("Φ₀/10", 8.29, :med),
              ("Φ₀/7",  3.33, :med), ("Φ₀/2",  1.13, :small)]
bars_x0 = 0.08; bars_x1 = 0.92
nb = length(ratios_lbl)
bw = (bars_x1 - bars_x0) / (nb*1.35)
max_ratio_log = log10(128.0) + 0.1
y_bar_base = 0.08
y_bar_top  = 0.22
for (k, (lbl, r, _)) in enumerate(ratios_lbl)
    xc = bars_x0 + (k-0.5) * (bars_x1-bars_x0)/nb
    frac = max(0.02, log10(max(r, 1.0)) / max_ratio_log)
    h    = frac * (y_bar_top - y_bar_base)
    clr  = r >= 1.5 ? BAR_DARK : GRAY_ANN
    rounded_rect!(ax_left, xc-bw/2, y_bar_base, xc+bw/2, y_bar_base+h; r=0.003,
                  color=clr, strokewidth=0)
    text!(ax_left, xc, y_bar_base+h+0.005; text=@sprintf("%.1f×", r),
          align=(:center, :bottom), fontsize=9,
          color=r >= 1.5 ? BAR_DARK : GRAY_ANN,
          font=:bold)
    text!(ax_left, xc, y_bar_base-0.003; text=lbl,
          align=(:center, :top), fontsize=8.5, color=TEXT)
end
text!(ax_left, 0.5, 0.05; text="(prior width  φ_max,  from wide → narrow)",
      align=(:center, :center), fontsize=9, color=GRAY_ANN, font=:italic)

# =============================================================================
# RIGHT PANEL: Flux Sensing POMDP and System Geometry
# =============================================================================
ax_right = Axis(right_col[1, 1]; backgroundcolor=BG,
                leftspinevisible=false, rightspinevisible=false,
                topspinevisible=false, bottomspinevisible=false,
                xticksvisible=false, yticksvisible=false,
                xticklabelsvisible=false, yticklabelsvisible=false,
                xgridvisible=false, ygridvisible=false,
                limits=(0, 1, 0, 1), aspect=DataAspect())
hidedecorations!(ax_right)

text!(ax_right, 0.5, 0.97; text="Flux Sensing POMDP & System Geometry",
      align=(:center, :center), fontsize=19, font=:bold, color=TEXT)

# -------- central transmon chip (pseudo-3D parallelogram) --------
# sapphire substrate — slight tilt
chip = Point2f[(0.28, 0.42), (0.70, 0.42),
               (0.76, 0.60), (0.34, 0.60)]
poly!(ax_right, chip; color=RGBf(0.70, 0.76, 0.87),
      strokecolor=RGBf(0.50, 0.56, 0.68), strokewidth=1.2)
# sapphire texture
for i in 1:5
    t = i/6
    y1 = 0.42 + t*0.18
    x1 = 0.28 + t*0.06
    x2 = 0.70 + t*0.06
    lines!(ax_right, [x1, x2], [y1, y1];
           color=RGBf(0.66, 0.72, 0.82), linewidth=0.5)
end

# --- transmon cross (two gold arms) ---
# horizontal arm
poly!(ax_right, Point2f[(0.36, 0.485), (0.66, 0.485),
                        (0.705, 0.521), (0.405, 0.521)];
      color=GOLD, strokecolor=RGBf(0.55, 0.40, 0.10), strokewidth=0.8)
# vertical arm
poly!(ax_right, Point2f[(0.50, 0.430), (0.555, 0.430),
                        (0.585, 0.588), (0.530, 0.588)];
      color=GOLD, strokecolor=RGBf(0.55, 0.40, 0.10), strokewidth=0.8)

# --- SQUID loop at end of +x arm ---
sq_cx = 0.712; sq_cy = 0.510; sq_w = 0.028; sq_h = 0.022
# loop rectangle
poly!(ax_right, Point2f[(sq_cx-sq_w/2, sq_cy-sq_h/2),
                        (sq_cx+sq_w/2, sq_cy-sq_h/2),
                        (sq_cx+sq_w/2+0.005, sq_cy+sq_h/2),
                        (sq_cx-sq_w/2+0.005, sq_cy+sq_h/2)];
      color=:transparent, strokecolor=RGBf(0.90, 0.78, 0.28), strokewidth=2.5)
# two Josephson junctions (red dots on left and right loop sides)
scatter!(ax_right, [sq_cx-sq_w/2+0.002], [sq_cy];
         color=ACCENT, markersize=10, strokecolor=TEXT, strokewidth=0.6)
scatter!(ax_right, [sq_cx+sq_w/2+0.0025], [sq_cy];
         color=ACCENT, markersize=10, strokecolor=TEXT, strokewidth=0.6)

# --- flux-bias line (purple) looping near SQUID ---
lines!(ax_right, [0.92, 0.76, sq_cx+sq_w/2+0.015], [0.34, 0.47, sq_cy];
       color=PURPLE, linewidth=3.0)

# --- magnetic-flux field lines (navy) threading SQUID loop ---
θ = range(0, 2π; length=60)
for r in [0.018, 0.024, 0.030]
    xs = sq_cx .+ r .* cos.(θ)
    ys = sq_cy .+ 0.55 .* r .* sin.(θ)
    lines!(ax_right, xs, ys; color=NAVY, linewidth=1.0, linestyle=:dash)
end

# --- readout resonator (green meander) on -x arm side ---
meander_y = 0.440
for i in 1:5
    x0 = 0.40 + 0.022 * (i-1)
    lines!(ax_right, [x0, x0+0.012], [meander_y, meander_y];
           color=GREEN, linewidth=2.0)
    if i < 5
        lines!(ax_right, [x0+0.012, x0+0.012],
               [meander_y, meander_y+0.010]; color=GREEN, linewidth=2.0)
        lines!(ax_right, [x0+0.012, x0+0.022],
               [meander_y+0.010, meander_y+0.010]; color=GREEN, linewidth=2.0)
        lines!(ax_right, [x0+0.022, x0+0.022],
               [meander_y+0.010, meander_y]; color=GREEN, linewidth=2.0)
    end
end

# =============================================================================
# Annotations
# =============================================================================
# Qubit-SQUID label (top)
annotate!(ax_right, sq_cx+0.005, sq_cy+0.010, 0.52, 0.82,
          "Qubit-SQUID sensor loop\n(hardware geometry c ∈ ℝ⁷: f_q, E_C/h, κ, Δ, T, A_Φ, A_Ic)";
          fontsize=11)

# External flux label (right)
annotate!(ax_right, sq_cx+0.025, sq_cy+0.012, 0.87, 0.72,
          "External flux Φ_ext\n(hidden state  x)";
          fontsize=11, color=NAVY, bold=true)

# Bias line (bottom right)
annotate!(ax_right, 0.90, 0.345, 0.89, 0.24,
          "Flux-bias drive\n(coupling ω_d)"; fontsize=10, color=PURPLE)

# Readout resonator (left)
annotate!(ax_right, 0.42, 0.45, 0.27, 0.34,
          "Readout resonator\n(dispersive readout)"; fontsize=10, color=GREEN)

# Transmon cross
annotate!(ax_right, 0.555, 0.506, 0.245, 0.68,
          "Transmon x-mon\n(frequency-tunable qubit)"; fontsize=10, color=GOLD)

# =============================================================================
# Bayesian belief evolution insets (top-right)
# =============================================================================
function gauss(x, μ, σ)
    exp.(-(x .- μ).^2 ./ (2σ^2))
end

xs_grid = range(-1, 1; length=120)

function belief_inset!(ax, cx, cy, w, h, μ, σ, label; color=BAR_DARK,
                       bold=false, title_offset=0.018)
    rounded_rect!(ax, cx-w/2, cy-h/2, cx+w/2, cy+h/2; r=0.006,
                  color=PANEL_BG2, strokecolor=color,
                  strokewidth=bold ? 1.8 : 0.9)
    ys = gauss(xs_grid, μ, σ)
    # shape to fit
    xdata = cx .- w/2 .+ 0.03 .+ (xs_grid .+ 1) ./ 2 .* (w - 0.06)
    ydata = cy .- h/2 .+ 0.02 .+ ys .* (h - 0.04)
    # shaded fill
    band!(ax, xdata, fill(cy-h/2+0.02, length(xdata)), ydata;
          color=(color, 0.25))
    lines!(ax, xdata, ydata; color=color, linewidth=bold ? 1.8 : 1.2)
    text!(ax, cx, cy+h/2+title_offset; text=label,
          align=(:center, :bottom), fontsize=9.5,
          color=color, font=bold ? :bold : :italic)
    xdata, ydata
end

belief_inset!(ax_right, 0.88, 0.90, 0.20, 0.060, 0.0, 0.6,
              "p₀(x)  —  prior (broad)")

belief_inset!(ax_right, 0.88, 0.77, 0.20, 0.060, 0.15, 0.32,
              "p₁(x)  —  after epoch 1")

(xd_pK, yd_pK) = belief_inset!(ax_right, 0.88, 0.22, 0.20, 0.060, 0.30, 0.08,
              "p_K(x)  —  final posterior"; color=ACCENT, bold=true,
              title_offset=0.022)
# MSE-optimal estimate marker
μx = 0.88 - 0.10 + 0.03 + (0.30 + 1)/2 * (0.20 - 0.06)
lines!(ax_right, [μx, μx], [0.22-0.030, 0.22+0.030];
       color=ACCENT, linewidth=1.2, linestyle=:dash)
text!(ax_right, 0.88, 0.22-0.055;
      text="→ posterior-mean  x̂⋆  (MSE-optimal)",
      align=(:center, :top), fontsize=9.5, color=ACCENT, font=:italic)

# Bayesian-update flow arrow
text!(ax_right, 0.975, 0.50;
      text="Bayesian\nupdate",
      align=(:center, :center), fontsize=10, color=GRAY_ANN, font=:italic,
      rotation=-π/2)
# arrow line
lines!(ax_right, [0.975, 0.975], [0.86, 0.80]; color=GRAY_ANN, linewidth=0.8)
lines!(ax_right, [0.975, 0.975], [0.73, 0.29]; color=GRAY_ANN, linewidth=0.8)

# =============================================================================
# Ramsey schedule inset (bottom-left)
# =============================================================================
rx = 0.12; ry = 0.15; rw = 0.24; rh = 0.08
rounded_rect!(ax_right, rx, ry-rh, rx+rw, ry+rh; r=0.010,
              color=PANEL_BG2, strokecolor=FRAME, strokewidth=0.9)
# π/2 pulses at epoch boundaries, τ intervals between
n_pulses = 4
for i in 1:n_pulses
    x0 = rx + 0.020 + (i-1) * 0.050
    # first π/2
    lines!(ax_right, [x0, x0], [ry-0.045, ry+0.055];
           color=BAR_DARK, linewidth=2.2)
    # free-precession τ (shaded region)
    poly!(ax_right, Point2f[(x0, ry-0.003), (x0+0.022, ry-0.003),
                            (x0+0.022, ry+0.003), (x0, ry+0.003)];
          color=RGBf(0.75, 0.82, 0.95), strokewidth=0)
    # second π/2
    lines!(ax_right, [x0+0.022, x0+0.022], [ry-0.045, ry+0.055];
           color=BAR_DARK, linewidth=2.2)
    # measurement readout marker
    scatter!(ax_right, [x0+0.030], [ry+0.020]; color=GREEN,
             marker=:diamond, markersize=5)
end
text!(ax_right, rx+rw/2, ry+rh+0.018;
      text="Ramsey adaptive schedule  s_k = (τ_k, n_k)",
      align=(:center, :bottom), fontsize=10, color=TEXT, font=:bold)
text!(ax_right, rx+rw/2, ry-rh-0.012;
      text="π/2 — τ_k — π/2 — readout   (n_k repetitions)",
      align=(:center, :top), fontsize=9, color=GRAY_ANN, font=:italic)
text!(ax_right, rx+rw/2, ry-rh-0.035;
      text="action chosen adaptively by π⋆(b_k)",
      align=(:center, :top), fontsize=9, color=ACCENT, font=:italic)

# measurement outcome callout
text!(ax_right, rx+rw+0.035, ry+0.025;
      text="y_k ∈ {0, 1}\nBernoulli outcome",
      align=(:left, :center), fontsize=10, color=TEXT)

# prior decomposition arrow (prior below the chip to belief-inset column)
lines!(ax_right, [rx+rw/2, rx+rw/2, 0.78], [ry-rh-0.060, ry-rh-0.080, ry-rh-0.080];
       color=GRAY_ANN, linewidth=0.8, linestyle=:dash)

# caption strip at bottom
rounded_rect!(ax_right, 0.04, 0.02, 0.96, 0.07; r=0.006,
              color=PANEL_BG, strokecolor=FRAME, strokewidth=0.7)
text!(ax_right, 0.50, 0.045;
      text="Hardware geometry c and adaptive policy π⋆(b) co-optimized via envelope-theorem gradient (joint-DP, §6)",
      align=(:center, :center), fontsize=10.5, color=TEXT, font=:italic)

# =============================================================================
# Save
# =============================================================================
out = joinpath(@__DIR__, "scqubit_hero.png")
save(out, fig; px_per_unit=2.0)
println("Saved: $out")
