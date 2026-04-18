"""
Numerical computation of IG(x) for the radar beam-search case study.

Setup (matches AdaptiveDesign.tex §9):
- K = 16 cells equally spaced on a ring, bearings φ(z) = 2π(z-1)/K.
- Uniform prior p₀(z) = 1/K.
- N = 4 measurements per episode.
- Binary detection:  P(y=1 | z, θ, c) = p_FA + (p_D − p_FA) · G_c(θ − φ(z)).
- p_D = 0.9, p_FA = 0.05.
- Geometry c: TOP-HAT beam with angular width w. "Narrow" = w spans 1 cell;
  "wide" = w spans K/2 cells.

Under a top-hat beam pointing at cell a, the likelihood for target at cell z is
    p(y=1 | z, a) = p_D   if z is within the beam centered on cell a,
                    p_FA  otherwise.

Outputs:
- V_oracle(x) for each x (should be constant over x by D_K symmetry).
- Φ(x, s★, c) for each x, where s★ is any fixed-optimal sensor sequence.
- IG(x) = V_oracle(x) − Φ(x, s★, c) for each x.
- Statistics: E[IG], std[IG], max[IG], distribution.
"""

using Printf
using Statistics

# ---------- problem setup ----------
const K = 16
const N = 4
const p_D = 0.9
const p_FA = 0.05

# φ(z) for z ∈ 1..K, bearing in radians
φ(z) = 2π * (z - 1) / K

# Top-hat beam: beam centered at cell a covers cells in `cells_in_beam[a]`
# `beam_width_cells` = number of cells in the main lobe
function make_beam_coverage(beam_width_cells::Int)
    coverage = [falses(K) for _ in 1:K]
    half = div(beam_width_cells, 2)   # beam extends ±half cells from center
    for a in 1:K
        for z in 1:K
            # circular distance (in cells) between pointing a and cell z
            d = minimum((mod(z - a, K), mod(a - z, K)))
            if beam_width_cells == 1
                coverage[a][z] = (d == 0)
            else
                # for even width, include cells with d < half (not at exact boundary)
                coverage[a][z] = (d <= half - 1) || (beam_width_cells % 2 == 1 && d == half)
            end
        end
    end
    coverage
end

# Per-cell detection probability when pointing at a with top-hat beam coverage
function p_detect(z, a, coverage)
    coverage[a][z] ? p_D : p_FA
end

# ---------- posterior and information-gain machinery ----------
# For a given sensor sequence s_{1:N} and outcome y_{1:N}, compute:
#   posterior b(z) ∝ ∏_k p(y_k | z, s_k)
#   and its entropy H (nats).
function posterior_entropy(s::Vector{Int}, y::Vector{Int}, coverage)
    logb = zeros(K)
    for z in 1:K
        for k in 1:N
            pk = p_detect(z, s[k], coverage)
            logb[z] += log(y[k] == 1 ? pk : 1 - pk)
        end
    end
    # prior uniform → log prior is constant; ignore for max
    # normalize in log domain
    m = maximum(logb)
    b = exp.(logb .- m)
    b ./= sum(b)
    H = 0.0
    for z in 1:K
        H -= b[z] > 0 ? b[z] * log(b[z]) : 0.0
    end
    H
end

# P(y_{1:N} | x, s_{1:N}) assuming conditional independence across k.
function py_given_xs(y::Vector{Int}, x::Int, s::Vector{Int}, coverage)
    p = 1.0
    for k in 1:N
        pk = p_detect(x, s[k], coverage)
        p *= (y[k] == 1 ? pk : 1 - pk)
    end
    p
end

# Φ(x, s, c) = E_{y | x, s} [ ln K − H(z | y_{1:N}) ]   (nats)
function Φ_value(x::Int, s::Vector{Int}, coverage)
    lnK = log(K)
    val = 0.0
    # enumerate 2^N outcomes
    for bits in 0:(2^N - 1)
        y = [((bits >> (k-1)) & 1) for k in 1:N]
        pyx = py_given_xs(y, x, s, coverage)
        if pyx > 0
            H = posterior_entropy(s, y, coverage)
            val += pyx * (lnK - H)
        end
    end
    val
end

# ---------- oracle ----------
# V_oracle(x) = max over sensor sequences of Φ(x, s, c).
# By symmetry and intuition, for narrow top-hat, oracle points at x every step.
# For wide top-hat, oracle's optimum may differ; we search over all sequences
# of length N with actions from 1..K.  For K=16, N=4 → 16^4 = 65 536 sequences.
function V_oracle(x::Int, coverage)
    best = -Inf
    best_s = zeros(Int, N)
    for s1 in 1:K, s2 in 1:K, s3 in 1:K, s4 in 1:K
        s = [s1, s2, s3, s4]
        v = Φ_value(x, s, coverage)
        if v > best
            best = v
            best_s = copy(s)
        end
    end
    (best, best_s)
end

# ---------- fixed design ----------
# V_fixed = max over sensor sequences of E_x[Φ(x, s, c)].
# Only need to enumerate (ordered) sequences; but by symmetry of the ring +
# uniform prior, the optimum value is attained by any 4-distinct-cell sequence
# (narrow) or any balanced-pointing sequence (wide).  We enumerate and record
# the argmax.
function fixed_design(coverage)
    best = -Inf
    best_s = zeros(Int, N)
    # E_x[Φ] = (1/K) Σ_x Φ(x, s, c)
    for s1 in 1:K, s2 in 1:K, s3 in 1:K, s4 in 1:K
        s = [s1, s2, s3, s4]
        E = 0.0
        for x in 1:K
            E += Φ_value(x, s, coverage) / K
        end
        if E > best
            best = E
            best_s = copy(s)
        end
    end
    (best, best_s)
end

# ---------- main computation ----------
function run_ig_distribution(beam_width::Int, label::String)
    println("\n" * "="^72)
    println("Regime: $label (beam covers $beam_width cells)")
    println("="^72)

    coverage = make_beam_coverage(beam_width)
    cells_covered = sum(coverage[1])
    @printf("Beam pointing at cell 1 covers %d cells: %s\n",
            cells_covered, findall(coverage[1]))

    # 1. oracle values per x
    println("\nComputing V_oracle(x) for each x ∈ 1..K ...")
    V_or = zeros(K)
    s_or = [zeros(Int, N) for _ in 1:K]
    for x in 1:K
        V_or[x], s_or[x] = V_oracle(x, coverage)
    end

    # 2. fixed design
    println("Computing fixed-optimal s★ by enumeration over $(K^N) sequences ...")
    V_fx, s_star = fixed_design(coverage)

    # 3. Φ(x, s★, c) per x
    println("Computing Φ(x, s★, c) for each x ...")
    Φ_at_star = zeros(K)
    for x in 1:K
        Φ_at_star[x] = Φ_value(x, s_star, coverage)
    end

    # 4. IG(x) = V_oracle(x) − Φ(x, s★, c)
    IG = V_or .- Φ_at_star

    # ---------- report ----------
    println("\n--- Oracle values V_oracle(x) ---")
    @printf("%-6s  %-10s  %-s\n", "x", "V_oracle", "oracle sensors s★(x)")
    for x in 1:K
        @printf("%-6d  %-10.6f  %s\n", x, V_or[x], s_or[x])
    end
    @printf("\n  V_oracle mean over x = %.6f nats\n", mean(V_or))
    @printf("  V_oracle std over x  = %.6e nats\n", std(V_or))

    @printf("\n--- Fixed-optimal design ---\n")
    @printf("  s★ = %s\n", s_star)
    @printf("  V_fixed = E_x[Φ(x,s★)]  = %.6f nats\n", V_fx)

    @printf("\n--- Φ(x, s★, c) per x ---\n")
    @printf("%-6s  %-10s\n", "x", "Φ(x,s★)")
    for x in 1:K
        @printf("%-6d  %-10.6f\n", x, Φ_at_star[x])
    end

    @printf("\n--- Ignorance gap IG(x) = V_oracle(x) − Φ(x, s★, c) ---\n")
    @printf("%-6s  %-10s\n", "x", "IG(x)")
    for x in 1:K
        @printf("%-6d  %-10.6f\n", x, IG[x])
    end

    @printf("\n--- Summary statistics of IG(x) ---\n")
    @printf("  E[IG]        = %.6f nats\n", mean(IG))
    @printf("  std[IG]      = %.6e nats\n", std(IG))
    @printf("  max[IG]      = %.6f nats\n", maximum(IG))
    @printf("  min[IG]      = %.6f nats\n", minimum(IG))
    @printf("  #distinct    = %d\n", length(unique(round.(IG, digits=9))))
    @printf("  V_oracle − V_fixed = E[IG] (check) = %.6f nats\n",
            mean(V_or) - V_fx)

    return (; V_or, V_fx, s_star, Φ_at_star, IG)
end

# ---------- run ----------
println("Parameters: K=$K, N=$N, p_D=$p_D, p_FA=$p_FA")

narrow = run_ig_distribution(1, "NARROW (m = 1 cell/beam)")
wide   = run_ig_distribution(div(K, 2), "WIDE (m = K/2 = $(div(K,2)) cells/beam)")

# Save results for subsequent analysis
using Serialization
open("ig_results.jls", "w") do io
    serialize(io, (; narrow, wide))
end
println("\nSaved to ig_results.jls")
