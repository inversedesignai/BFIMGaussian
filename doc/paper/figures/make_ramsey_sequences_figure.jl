#=
make_ramsey_sequences_figure.jl — publication-quality timeline of three
Ramsey sequences for Case Study B (superconducting-qubit flux sensor).

Each row shows a single shot of the Ramsey pulse sequence at one of the
K=3 decision epochs of the Kitaev-style adaptive protocol:

    [π/2 pulse] ── free evolution τ_k ── [π/2 pulse] ── [readout]

with the "× n_k repetitions" annotation indicating how many independent
shots are aggregated at that epoch (yielding a Binomial(n_k, P_|1>)
observation).

The three rows together depict the (τ_k, n_k) action space of the
adaptive policy: short delay with few reps for coarse phase
disambiguation, longer delay with more reps for refinement.

Dependencies: CairoMakie (no GPU required; pure 2D vector output).

Output: doc/paper/figures/ramsey_sequences.png
=#

using CairoMakie

CairoMakie.activate!(; px_per_unit=2)
set_theme!(theme_light(); fontsize=15)

# ---------- per-epoch parameters (τ_k in ns, n_k repetitions) ----------
const EPOCHS = [
    (k=1, τ=20.0,  n=3,  τ_label="τ_1 = 20 ns"),
    (k=2, τ=80.0,  n=5,  τ_label="τ_2 = 80 ns"),
    (k=3, τ=320.0, n=10, τ_label="τ_3 = 320 ns"),
]

# Visual layout (in axis units; pulse durations are exaggerated relative to
# the inter-pulse delay so that they remain visible at all τ_k).
const PULSE_W = 12.0       # pulse rectangle width
const READOUT_W = 18.0     # readout rectangle width
const ROW_H = 1.0          # row height for pulse rectangles
const ROW_GAP = 1.6        # vertical spacing between rows
const LEFT_PAD = 25.0      # left padding for row label
const PULSE_COLOR = RGBf(0.95, 0.55, 0.15)
const READOUT_COLOR = RGBf(0.20, 0.55, 0.85)
const TAU_LINE_COLOR = RGBf(0.30, 0.30, 0.30)

# Place a rectangle (pulse or readout) centered at (x_center, y_center).
function place_block!(ax, x_center, y_center, width, height; color, label="")
    x0 = x_center - width/2
    x1 = x_center + width/2
    y0 = y_center - height/2
    y1 = y_center + height/2
    poly!(ax, Point2f[(x0,y0),(x1,y0),(x1,y1),(x0,y1)];
        color=color, strokecolor=:black, strokewidth=1.2)
    if !isempty(label)
        text!(ax, x_center, y_center; text=label, align=(:center, :center),
            color=:white, fontsize=12, font=:bold)
    end
end

# Place the "double-arrow with τ label" between two x-coordinates.
function place_tau_arrow!(ax, x_left, x_right, y_center; label="")
    lines!(ax, [Point2f(x_left, y_center), Point2f(x_right, y_center)];
        color=TAU_LINE_COLOR, linewidth=1.5)
    # left arrowhead
    arrowsize = 6.0
    poly!(ax, Point2f[(x_left, y_center),
                      (x_left + arrowsize, y_center - arrowsize/2.5),
                      (x_left + arrowsize, y_center + arrowsize/2.5)];
        color=TAU_LINE_COLOR, strokecolor=TAU_LINE_COLOR, strokewidth=0.5)
    # right arrowhead
    poly!(ax, Point2f[(x_right, y_center),
                      (x_right - arrowsize, y_center - arrowsize/2.5),
                      (x_right - arrowsize, y_center + arrowsize/2.5)];
        color=TAU_LINE_COLOR, strokecolor=TAU_LINE_COLOR, strokewidth=0.5)
    if !isempty(label)
        text!(ax, (x_left + x_right)/2, y_center + 0.35;
            text=label, align=(:center, :bottom),
            color=:black, fontsize=14, font=:bold)
    end
end

function render_ramsey_sequences()
    # Compute total horizontal extent: each row's right edge depends on τ_k.
    # τ_k is mapped to axis units via tau_scale; we cap the longest delay
    # so the figure stays square-ish.
    max_tau = maximum(e.τ for e in EPOCHS)
    tau_scale_unit = 0.45  # axis units per ns
    function row_width(τ)
        LEFT_PAD + PULSE_W + τ * tau_scale_unit + PULSE_W + READOUT_W + 30.0
    end
    figure_x_max = maximum(row_width(e.τ) for e in EPOCHS)

    n_rows = length(EPOCHS)
    fig = Figure(; size=(1500, 700), backgroundcolor=:white)

    ax = Axis(fig[1, 1];
        title="Three Ramsey sequences in the K = 3 adaptive protocol",
        titlesize=20, titlefont=:bold,
        xlabel="time within one shot →",
        ylabel="",
        xlabelsize=15, ylabelsize=15,
        xticksvisible=false, xticklabelsvisible=false,
        yticksvisible=false, yticklabelsvisible=false,
        topspinevisible=false, rightspinevisible=false,
        leftspinevisible=false, bottomspinevisible=false,
        xgridvisible=false, ygridvisible=false,
    )

    xlims!(ax, 0, figure_x_max + 50)
    ylims!(ax, -ROW_GAP * (n_rows + 0.5), ROW_GAP)

    for (i, ep) in enumerate(EPOCHS)
        y_center = -i * ROW_GAP
        # Row label (epoch index)
        text!(ax, LEFT_PAD - 8, y_center;
            text="Epoch k = $(ep.k)", align=(:right, :center),
            color=:black, fontsize=15, font=:bold)

        # Pulse 1 (π/2)
        x1_center = LEFT_PAD + PULSE_W/2
        place_block!(ax, x1_center, y_center, PULSE_W, ROW_H;
            color=PULSE_COLOR, label="π/2")

        # Free evolution τ_k
        x_left = x1_center + PULSE_W/2
        x_right = x_left + ep.τ * tau_scale_unit
        place_tau_arrow!(ax, x_left, x_right, y_center; label=ep.τ_label)

        # Pulse 2 (π/2)
        x2_center = x_right + PULSE_W/2
        place_block!(ax, x2_center, y_center, PULSE_W, ROW_H;
            color=PULSE_COLOR, label="π/2")

        # Readout
        x_ro_center = x2_center + PULSE_W/2 + 4.0 + READOUT_W/2
        place_block!(ax, x_ro_center, y_center, READOUT_W, ROW_H;
            color=READOUT_COLOR, label="readout")

        # × n_k repetitions annotation
        x_rep = x_ro_center + READOUT_W/2 + 8.0
        text!(ax, x_rep, y_center;
            text="× n_$(ep.k) = $(ep.n)",
            align=(:left, :center), color=RGBf(0.85, 0.10, 0.10),
            fontsize=16, font=:bold)
    end

    # Legend at top
    legend_elements = [
        PolyElement(color=PULSE_COLOR, strokecolor=:black, strokewidth=1.2),
        PolyElement(color=READOUT_COLOR, strokecolor=:black, strokewidth=1.2),
        LineElement(color=TAU_LINE_COLOR, linewidth=1.5),
    ]
    legend_labels = [
        "π/2 control pulse",
        "Projective readout",
        "Free evolution (τ_k)",
    ]
    Legend(fig[1, 1], legend_elements, legend_labels;
        tellwidth=false, tellheight=false,
        halign=:right, valign=:top,
        orientation=:horizontal, framevisible=true, backgroundcolor=(:white, 0.9),
        margin=(10, 10, 10, 10), labelsize=13, patchsize=(20, 15))

    # Footer caption
    text!(ax, figure_x_max/2, -ROW_GAP * (n_rows + 0.5) + 0.2;
        text="Action s_k = (τ_k, n_k):  τ_k sets phase-accumulation per shot;  n_k aggregates Binomial(n_k, P_{|1⟩}) per epoch.",
        align=(:center, :bottom), color=RGBf(0.30, 0.30, 0.30),
        fontsize=13, font=:regular)

    out = joinpath(@__DIR__, "ramsey_sequences.png")
    save(out, fig; px_per_unit=2)
    println("Saved: $out")
    return fig
end

render_ramsey_sequences()
