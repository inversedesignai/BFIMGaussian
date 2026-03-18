"""
SimGeomBroadBand.jl

Geometry builders, broadband FDFD solvers, and S-matrix computation for
2-D FDFD photonic structures.

API
  setup_4port_sweep            — construct cross-waveguide geometry, ε_base, and per-ω Laplacians
  batch_solve                  — parallel multi-frequency, multi-RHS FDFD solve (AD-compatible)
  getSmatrices                 — compute normalised 4×4 S-matrices for all frequencies
  calibrate_straight_waveguide — reference amplitude calibration for S-matrix normalisation
"""
module SimGeomBroadBand

using SparseArrays
using LinearAlgebra
using ChainRulesCore
using Distributed
using ImplicitDifferentiation

export setup_4port_sweep, batch_solve, getSmatrices, calibrate_straight_waveguide,
       powers_only, jac_only, jac_and_dirderiv_s

# ═══════════════════════════════════════════════════════════════════════════════
# Grid descriptor
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FDFDGrid(Nx, Ny, dx, dy, n_pml, sigma_max)

Describes the geometry and PML parameters of one 2-D FDFD domain.
The permittivity distribution ε_r is passed separately to the solver.

Fields
- `Nx`, `Ny`    : number of grid points in x and y (including PML layers).
- `dx`, `dy`    : physical grid spacing [same units as 1/ω].
- `n_pml`       : number of PML cells on each side.
- `sigma_max`   : maximum PML conductivity (real, > 0).
"""
struct FDFDGrid
    Nx        :: Int
    Ny        :: Int
    dx        :: Float64
    dy        :: Float64
    n_pml     :: Int
    sigma_max :: Float64
end

const _lattice_index_cache = Dict{Int, Any}()

# ═══════════════════════════════════════════════════════════════════════════════
# PML parameter selection
# ═══════════════════════════════════════════════════════════════════════════════

"""
    pml_sigma_max(ω, n_pml, res; R_target = 1e-8) -> Float64

Compute the peak PML conductivity σ_max required to achieve a target round-trip
amplitude reflection `R_target` for a quadratic stretch-factor profile.

**Derivation**

The stretch factor is  s(x) = 1 + i·σ_max·(x/D)²  over the physical PML
thickness  D = n_pml / res.  For a forward-propagating plane wave at wavenumber
k = ω (cladding, n_clad = 1, c = 1) the one-way attenuation through the PML is:

    A = exp(-ω · ∫₀ᴰ Im(1/s) dx) ≈ exp(-ω · σ_max · D / 3)

where the 1/3 factor is ∫₀¹ t² dt from the quadratic profile.  The round-trip
(wave reflects off the outer PML wall and returns) gives R = A²:

    R = exp(-2·ω·σ_max·D/3) = exp(-2·ω·σ_max·n_pml / (3·res))

Inverting for σ_max:

    σ_max = -3·res·ln(R_target) / (2·ω·n_pml)

**Scaling**

σ_max ∝ res / (ω · n_pml): a finer grid or a thinner PML (fewer cells) requires
a larger peak conductivity to maintain the same absorption.  Conversely, for a
fixed σ_max, increasing n_pml or ω improves absorption exponentially.

**Verification against the hardcoded default**

At the reference parameters (res = 20, n_pml = 20, ω = 2π) the commonly used
value σ_max = 4.0 corresponds to R_target ≈ 5×10⁻⁸ (-146 dB in amplitude).
The default R_target = 1e-8 gives σ_max ≈ 4.4 at those same parameters —
slightly more aggressive but practically equivalent.

Arguments
- `ω`       : angular frequency (= free-space wavenumber since c = 1).
- `n_pml`   : number of PML cells on each side.
- `res`     : grid points per unit length (controls cell size dx = 1/res).

Keyword argument
- `R_target` : target round-trip amplitude reflection (default 1e-8, i.e. −160 dB
               in amplitude or −320 dB in power).
"""
function pml_sigma_max(ω      :: Float64,
                       n_pml  :: Int,
                       res    :: Int;
                       R_target :: Float64 = 1e-8) :: Float64
    return -3.0 * res * log(R_target) / (2.0 * ω * n_pml)
end

# ═══════════════════════════════════════════════════════════════════════════════
# PML stretch factors
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_pml_vectors(N, n_pml, σ_max) -> (s, s_f, s_b)

Return three 1-D complex PML stretch-factor arrays for a domain of N grid points
with absorbing layers of thickness n_pml cells on each side.

The stretch factor at a (possibly fractional) grid index `i_float` is:

    s(i_float) = 1 + im·σ_max·(dist/n_pml)²   (σ_max = peak PML conductivity)

where `dist` measures penetration into the PML:
  - Left  PML (i_float ≤ n_pml + ½):  dist = (n_pml + 1) − i_float
      → dist = n_pml at i=1 (outer wall), dist = 1 at i=n_pml (inner cell)
  - Right PML (i_float > N − n_pml):  dist = i_float − (N − n_pml)
      → dist = 1 at i=N−n_pml+1 (inner cell), dist = n_pml at i=N (outer wall)
  - Interior:  dist = 0  →  s = 1

Both sides reach the same peak conductivity σ_max (at cells 1 and N).

Returns
- `s`   : factors at integer grid points 1…N            (length N)
- `s_f` : factors at forward  half-points i+½, i=1…N-1  (length N-1)
- `s_b` : factors at backward half-points i-½, i=2…N    (length N-1)

Note: s_f[i] == s_b[i] == get_s(i+0.5); the two arrays differ only in how they
are indexed by the Laplacian assembler (s_f[ix] for the forward neighbour of ix,
s_b[ix-1] for the backward neighbour).
"""
function build_pml_vectors(N::Int, n_pml::Int, σ_max::Float64)
    function get_s(i_float)
        dist = 0.0
        if i_float < n_pml + 1          # left PML: cells 1…n_pml (and the adjacent face)
            dist = n_pml + 1 - i_float  # n_pml at outer wall, 1 at innermost cell
        elseif i_float > N - n_pml      # right PML: cells N-n_pml+1…N
            dist = i_float - (N - n_pml)
        end
        return dist > 0 ? ComplexF64(1.0 + im * σ_max * (dist / n_pml)^2) :
                          ComplexF64(1.0)
    end

    s   = [get_s(i)       for i in 1:N]       # length N
    s_f = [get_s(i + 0.5) for i in 1:N-1]    # forward  half-points, length N-1
    s_b = [get_s(i - 0.5) for i in 2:N]      # backward half-points, length N-1

    return s, s_f, s_b
end

# ═══════════════════════════════════════════════════════════════════════════════
# PML-stretched Laplacian
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_stretched_laplacian(grid) -> SparseMatrixCSC{ComplexF64}

Assemble the (Nx·Ny) × (Nx·Ny) sparse PML-stretched Laplacian L for the 2-D
Helmholtz equation.  L contains only the second-derivative (kinetic) terms;
the caller is responsible for adding the ω²ε diagonal:

    A = L + spdiagm(0 => ω² .* vec(ε_r))

where `vec(ε_r)` uses Julia's column-major order, matching the linear index
convention k = (ix-1)·Ny + iy.

The operator applies the PML stretch factor at the *cell centre* and at the
*cell face*:

    (1/sx[i]) · [(E_{i+1}−E_i)/(sx_{i+½}·dx) − (E_i−E_{i-1})/(sx_{i-½}·dx)] / dx

giving off-diagonal coefficients 1/(dx² · sx[i] · sx_{face}).  The resulting
matrix is generally non-symmetric in the PML region.

Because L depends only on the grid geometry and PML profile — not on ε(x,y)
or ω — it can be precomputed once and reused for multiple frequencies or
permittivity patterns.
"""
function build_stretched_laplacian(grid::FDFDGrid)
    (; Nx, Ny, dx, dy, n_pml, sigma_max) = grid
    N = Nx * Ny

    # PML stretch vectors for x and y
    sx, sx_f, sx_b = build_pml_vectors(Nx, n_pml, sigma_max)
    sy, sy_f, sy_b = build_pml_vectors(Ny, n_pml, sigma_max)

    # COO buffers — pre-allocate for up to 5 entries per row
    rows = Vector{Int}(undef, 5N)
    cols = Vector{Int}(undef, 5N)
    vals = Vector{ComplexF64}(undef, 5N)
    ptr  = Ref(0)

    add! = (r, c, v) -> begin
        ptr[] += 1
        rows[ptr[]] = r
        cols[ptr[]] = c
        vals[ptr[]] = v
    end

    # Linear index (column-major: y is the fast index)
    idx(iy, ix) = (ix - 1) * Ny + iy

    for ix in 1:Nx, iy in 1:Ny
        k        = idx(iy, ix)
        diag_val = zero(ComplexF64)

        # ── x-direction:  (1/sx[ix]) · d/dx[(1/sx_face) · dE/dx] ──────────
        if ix < Nx
            coeff = 1.0 / (dx^2 * sx[ix] * sx_f[ix])   # sx_f[ix] = s at ix+½
            add!(k, idx(iy, ix+1), coeff)
            diag_val -= coeff
        end
        if ix > 1
            coeff = 1.0 / (dx^2 * sx[ix] * sx_b[ix-1]) # sx_b[ix-1] = s at ix-½
            add!(k, idx(iy, ix-1), coeff)
            diag_val -= coeff
        end

        # ── y-direction:  (1/sy[iy]) · d/dy[(1/sy_face) · dE/dy] ──────────
        if iy < Ny
            coeff = 1.0 / (dy^2 * sy[iy] * sy_f[iy])
            add!(k, idx(iy+1, ix), coeff)
            diag_val -= coeff
        end
        if iy > 1
            coeff = 1.0 / (dy^2 * sy[iy] * sy_b[iy-1])
            add!(k, idx(iy-1, ix), coeff)
            diag_val -= coeff
        end

        # ── diagonal: Laplacian part only — caller adds ω²ε ─────────────────
        add!(k, k, diag_val)
    end

    n = ptr[]
    return sparse(rows[1:n], cols[1:n], vals[1:n], N, N)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Port data structure
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Port(edge, transverse_range, mode, kz, plane_idx)

Describes one waveguide port or monitor plane in a 2-D FDFD domain.

Fields
- `edge`             : which side the port sits on — one of `:left`, `:right`,
                       `:bottom`, `:top`.
- `transverse_range` : grid-index slice *along the edge* covered by the
                       waveguide cross-section.
                       For `:left`/`:right` ports: range of iy values.
                       For `:bottom`/`:top` ports: range of ix values.
- `mode`             : normalised complex mode profile sampled at the
                       transverse_range grid points (length = length(range)).
                       Normalised so that d·Σ|φⱼ|² = 1, where d = dy for
                       `:left`/`:right` ports and d = dx for `:bottom`/`:top`.
- `kz`               : complex propagation constant β of the mode.
- `plane_idx`        : the ix (for `:left`/`:right`) or iy (for `:bottom`/`:top`)
                       grid index of the injection / extraction plane.
"""
struct Port
    edge             :: Symbol
    transverse_range :: UnitRange{Int}
    mode             :: Vector{ComplexF64}
    kz               :: ComplexF64
    plane_idx        :: Int
end

# ═══════════════════════════════════════════════════════════════════════════════
# Waveguide mode solver
# ═══════════════════════════════════════════════════════════════════════════════

function mismatch_kx_vec(kx_vec, ω, n_core, w)
    kx = kx_vec[1]
    k₀ = ω
    R = k₀ * sqrt(n_core^2 - 1.0)
    γ = sqrt(R^2 - kx^2)
    return [kx * tan(kx * w / 2) - γ]
end

function implicit_κx_vec(ω, n_core, w)
    k₀ = ω
    R = k₀ * sqrt(n_core^2 - 1.0)
    max_kx = min(R, π / w * 0.9999)
    low, high = 1e-9, max_kx
    for _ in 1:100
        mid = (low + high) / 2
        if only(mismatch_kx_vec([mid], ω, n_core, w)) < 0
            low = mid
        else
            high = mid
        end
    end
    return [(low + high) / 2]
end

# Pack [ω, n_core, w] into x so ImplicitFunction differentiates w.r.t. them.
# args are treated as constants — so ω, n_core, w must be in x, not args.
implicit_κx_autodiff = ImplicitFunction(
    x -> (implicit_κx_vec(x[1], x[2], x[3]), nothing),
    (x, y, z) -> mismatch_kx_vec(y, x[1], x[2], x[3])
)

function getκx(ω, n_core, w)
    x = [ω, n_core, w]
    kx_vec, _ = implicit_κx_autodiff(x)
    return only(kx_vec)
end

"""
    compute_waveguide_mode(w, n_core, ω, y_coords) -> (kz, φ)

Compute the fundamental even TE mode of a symmetric dielectric slab waveguide
(core index n_core, cladding index 1) by bisection on the dispersion relation.

Arguments
- `w`        : physical waveguide width (same units as 1/ω).
- `n_core`   : core refractive index (must be > 1 for a guided mode).
- `ω`        : angular frequency.
- `y_coords` : physical transverse coordinates at which to sample the mode
               (centred so that y = 0 is the waveguide centre).

Returns
- `kz` : propagation constant β = sqrt((n_core·ω)² − κx²),  Re(kz) > 0.
- `φ`  : mode profile sampled at `y_coords`, normalised so that
         Σ|φⱼ|²·dy = 1  (dy inferred from the spacing of y_coords).

Dispersion relation (even mode):
    kx · tan(kx · w/2) = γ
where  γ = sqrt(R² − kx²),  R = ω · sqrt(n_core² − 1).
Profile:
    φ(y) = cos(kx · y)                                  |y| ≤ w/2
    φ(y) = cos(kx · w/2) · exp(−γ · (|y| − w/2))       |y| > w/2
"""
function compute_waveguide_mode(w       :: Float64,
                                n_core  :: Float64,
                                ω       :: Float64,
                                y_coords :: AbstractVector{<:Real})
    k₀     = ω
    R      = k₀ * sqrt(n_core^2 - 1.0)          # n_clad = 1
    κx     = getκx(ω, n_core, w)

    γ  = sqrt(R^2 - κx^2)
    kz = sqrt(ComplexF64((n_core * k₀)^2 - κx^2))
    if real(kz) < 0; kz = -kz; end

    half_w = w / 2
    function E_profile(y)
        ay = abs(y)
        return ay <= half_w ? cos(κx * y) :
                              cos(κx * half_w) * exp(-γ * (ay - half_w))
    end

    φ  = ComplexF64.(map(E_profile, y_coords))
    dy = length(y_coords) > 1 ? abs(y_coords[2] - y_coords[1]) : 1.0
    φ ./= sqrt(sum(abs2, φ) * dy)

    return kz, φ
end

# ═══════════════════════════════════════════════════════════════════════════════
# Unidirectional source construction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_source(grid, port) -> Vector{ComplexF64}

Build the RHS source vector for exciting `port` using the combined J+M
unidirectional source.

`grid` must expose the fields `Nx`, `Ny`, `dx`, `dy` (e.g. a `FDFDGrid`).

**Derivation**: a surface electric current J = −(kz/ω)·φ together with a surface
magnetic current M = φ at the injection plane cancel the backward-propagating
wave while doubling the forward-propagating one.  After substituting into the
FDFD RHS the ω factors cancel, leaving:

    S_0  = im·kz·φ    at n0       (volume source at injection plane)
    S_+  = −φ/(2d)    at n0 + 1   (forward  neighbour, from M)
    S_−  = +φ/(2d)    at n0 − 1   (backward neighbour, from M)

where d = dx for `:left`/`:right` ports and d = dy for `:bottom`/`:top` ports,
and n0 = port.plane_idx.  For `:right`/`:top` (backward propagation) the sign
of kz is flipped and the ± neighbours are swapped.
"""
function build_source(grid, port::Port)
    (; Nx, Ny, dx, dy) = grid
    N  = Nx * Ny
    b  = zeros(ComplexF64, N)
    n0 = port.plane_idx
    kz = port.kz
    φ  = port.mode

    if port.edge == :left
        ix0 = n0
        for (j, iy) in enumerate(port.transverse_range)
            b[(ix0 - 1)*Ny + iy] +=  im * kz * φ[j]      # S_0  at ix0
            b[(ix0    )*Ny + iy] += -φ[j] / (2 * dx)     # S_+  at ix0+1
            b[(ix0 - 2)*Ny + iy] +=  φ[j] / (2 * dx)     # S_−  at ix0-1
        end
    elseif port.edge == :right
        ix0 = n0
        for (j, iy) in enumerate(port.transverse_range)
            b[(ix0 - 1)*Ny + iy] += -im * kz * φ[j]      # S_0  at ix0
            b[(ix0 - 2)*Ny + iy] +=  φ[j] / (2 * dx)     # S_+  at ix0-1
            b[(ix0    )*Ny + iy] += -φ[j] / (2 * dx)     # S_−  at ix0+1
        end
    elseif port.edge == :bottom
        iy0 = n0
        for (j, ix) in enumerate(port.transverse_range)
            b[(ix - 1)*Ny + iy0    ] +=  im * kz * φ[j]  # S_0  at iy0
            b[(ix - 1)*Ny + iy0 + 1] += -φ[j] / (2 * dy) # S_+  at iy0+1
            b[(ix - 1)*Ny + iy0 - 1] +=  φ[j] / (2 * dy) # S_−  at iy0-1
        end
    else  # :top
        iy0 = n0
        for (j, ix) in enumerate(port.transverse_range)
            b[(ix - 1)*Ny + iy0    ] += -im * kz * φ[j]  # S_0  at iy0
            b[(ix - 1)*Ny + iy0 - 1] +=  φ[j] / (2 * dy) # S_+  at iy0-1
            b[(ix - 1)*Ny + iy0 + 1] += -φ[j] / (2 * dy) # S_−  at iy0+1
        end
    end

    return b
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mode-overlap extraction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    extract_overlap(Ez, port, d) -> ComplexF64

Compute the mode overlap integral ⟨φ | Ez⟩ at the plane given by `port.plane_idx`.

For `:left`/`:right` ports the field is sampled at column `ix = plane_idx` over
`port.transverse_range` (iy values).  For `:bottom`/`:top` ports the field is
sampled at row `iy = plane_idx` over `port.transverse_range` (ix values).

`d` is the grid spacing along the transverse direction (dy for horizontal ports,
dx for vertical ports).

Returns the complex mode amplitude  a = d · Σⱼ conj(φⱼ) · Ez[j].
"""
function extract_overlap(Ez :: AbstractMatrix, port :: Port, d :: Float64)
    if port.edge in (:left, :right)
        ix = port.plane_idx
        Ez_slice = [Ez[iy, ix] for iy in port.transverse_range]
    else
        iy = port.plane_idx
        Ez_slice = [Ez[iy, ix] for ix in port.transverse_range]
    end
    return d * dot(port.mode, Ez_slice)
end



# ═══════════════════════════════════════════════════════════════════════════════
# 4-port geometry builder
# ═══════════════════════════════════════════════════════════════════════════════

"""
    setup_4port_sweep(ωmin, ωmax, ωnum, n_core, w, d_length, d_width;
                         Lx, Ly, res, n_pml, R_target, port_offset, mon_offset)
    -> (ε_base, ω_array, Ls, Bs, grid_info, ports_array, monitors_array)

Create a 2-D FDFD domain with a rectangular design region at the centre,
four waveguide arms attached to the centre of each side of the design region,
and for each frequency in `range(ωmin, ωmax, length=ωnum)`, a separate set of
injection ports, monitors, and a `ScatteringBlock` whose PML σ_max scales with ω.

The permittivity `ε_base` is ω-independent and is built once.  Everything that
depends on the wave (mode profiles, propagation constants, transverse extents,
PML conductivity) is recomputed per frequency.

Arguments
- `ωmin`, `ωmax` : frequency sweep bounds.
- `ωnum`         : number of frequency points (linearly spaced).
- `n_core`       : core refractive index of the waveguide arms (> 1).
- `w`            : waveguide width (same units as 1/ω).
- `d_length`     : length of the design region (x-direction).
- `d_width`      : width  of the design region (y-direction).
                   The design region is filled with background permittivity ε_base = 1.

Keyword arguments
- `Lx`, `Ly`    : total domain size (default 10 × 10).
- `res`         : grid points per unit length (default 20).
- `n_pml`       : PML thickness in grid cells (default 20).
- `R_target`    : target round-trip PML amplitude reflection (default 1e-8).
                  Passed to `pml_sigma_max(ω, n_pml, res; R_target)` which computes
                  σ_max(ω) = −3·res·ln(R_target)/(2·ω·n_pml) per frequency.
- `port_offset` : cells from PML inner edge to injection plane (default 20).
- `mon_offset`  : cells from PML inner edge to monitor plane (default 5).
                  Must satisfy mon_offset < port_offset.

Returns
- `ε_base`        : (Ny × Nx) ComplexF64 permittivity for the full domain (shared).
- `ω_array`       : length-ωnum Vector{Float64} of angular frequencies.
- `Ls`            : length-ωnum Vector of sparse PML-stretched Laplacian matrices,
                    one per frequency (each built from the ω-scaled σ_max).
- `Bs`            : length-ωnum Vector of (Nx·Ny)×4 sparse RHS matrices, one per
                    frequency.  Column order: left, right, bottom, top port.
                    Built from `build_source(grid, port)` using the ω-dependent `FDFDGrid`.
- `grid_info`     : NamedTuple `(Nx, Ny, dx, dy, n_pml, x_coords, y_coords,
                    design_iy, design_ix)` with the ω-independent grid metadata
                    shared across all frequencies.  `design_iy` and `design_ix`
                    are the UnitRange row/column indices of the design region within
                    the full `(Ny × Nx)` domain; pass them to `getSmatrices` and
                    `batch_solve` as keyword/positional args.
- `ports_array`   : length-ωnum Vector of named tuples `(left, right, bottom, top)`
                    of injection `Port`s, one per frequency.
- `monitors_array`: length-ωnum Vector of named tuples `(left, right, bottom, top)`
                    of monitor `Port`s, one per frequency.

Port / monitor ordering
  left   : injects +x, monitor at smaller ix (ix = n_pml + mon_offset).
  right  : injects −x, monitor at larger  ix (ix = Nx − n_pml − mon_offset + 1).
  bottom : injects +y, monitor at smaller iy (iy = n_pml + mon_offset).
  top    : injects −y, monitor at larger  iy (iy = Ny − n_pml − mon_offset + 1).
"""
function setup_4port_sweep(
        ωmin         :: Number,
        ωmax         :: Number,
        ωnum         :: Int,
        n_core       :: Float64,
        w            :: Float64,
        d_length     :: Float64,
        d_width      :: Float64;
        Lx           :: Float64 = 10.0,
        Ly           :: Float64 = 10.0,
        res          :: Int     = 20,
        n_pml        :: Int     = 20,
        R_target     :: Float64 = 1e-8,
        port_offset  :: Int     = 20,
        mon_offset   :: Int     = 5)

    @assert mon_offset < port_offset "mon_offset must be < port_offset so monitors are behind ports"

    dx = 1.0 / res
    dy = 1.0 / res
    Nx = round(Int, Lx * res)
    Ny = round(Int, Ly * res)

    x_coords = range(-Lx/2, Lx/2, length=Nx)
    y_coords = range(-Ly/2, Ly/2, length=Ny)

    # ── Design region grid extent ────────────────────────────────────────────
    Nx_d = round(Int, d_length * res)
    Ny_d = round(Int, d_width  * res)

    ix_c = Nx ÷ 2
    iy_c = Ny ÷ 2

    ix_d0 = ix_c - Nx_d ÷ 2 + 1
    ix_d1 = ix_d0 + Nx_d - 1
    iy_d0 = iy_c - Ny_d ÷ 2 + 1
    iy_d1 = iy_d0 + Ny_d - 1

    # ── Build ε_base (ω-independent) ────────────────────────────────────────────
    ε_base = ones(ComplexF64, Ny, Nx)

    iy_wg0 = iy_c - round(Int, w/2 * res) + 1
    iy_wg1 = iy_c + round(Int, w/2 * res)
    for iy in iy_wg0:iy_wg1, ix in 1:Nx
        ε_base[iy, ix] = ComplexF64(n_core^2)
    end

    ix_wg0 = ix_c - round(Int, w/2 * res) + 1
    ix_wg1 = ix_c + round(Int, w/2 * res)
    for ix in ix_wg0:ix_wg1, iy in 1:Ny
        ε_base[iy, ix] = ComplexF64(n_core^2)
    end

    # Design region reset to background (ε_base = 1), overwriting any waveguide arm pixels
    for iy in iy_d0:iy_d1, ix in ix_d0:ix_d1
        ε_base[iy, ix] = ComplexF64(1.0)
    end

    # ── Port / monitor plane indices (ω-independent) ─────────────────────────
    ix_left_port  = n_pml + port_offset
    ix_right_port = Nx - n_pml - port_offset + 1
    iy_bot_port   = n_pml + port_offset
    iy_top_port   = Ny - n_pml - port_offset + 1

    ix_left_mon   = n_pml + mon_offset
    ix_right_mon  = Nx - n_pml - mon_offset + 1
    iy_bot_mon    = n_pml + mon_offset
    iy_top_mon    = Ny - n_pml - mon_offset + 1

    # ── Per-ω: mode profiles, PML σ_max, block, ports, monitors ─────────────
    ω_array = collect(range(Float64(ωmin), Float64(ωmax), length = ωnum))

    results = map(ω_array) do ω
        λ         = 2π / ω
        ext_cells = round(Int, 2 * λ * res)

        trange_h = max(n_pml + 1, iy_wg0 - ext_cells) : min(Ny - n_pml, iy_wg1 + ext_cells)
        y_h      = [Float64(y_coords[iy]) for iy in trange_h]
        kz, mode_h = compute_waveguide_mode(w, n_core, ω, y_h)

        trange_v = max(n_pml + 1, ix_wg0 - ext_cells) : min(Nx - n_pml, ix_wg1 + ext_cells)
        x_v      = [Float64(x_coords[ix]) for ix in trange_v]
        _, mode_v = compute_waveguide_mode(w, n_core, ω, x_v)

        port_left   = Port(:left,   trange_h, mode_h, kz, ix_left_port)
        port_right  = Port(:right,  trange_h, mode_h, kz, ix_right_port)
        port_bottom = Port(:bottom, trange_v, mode_v, kz, iy_bot_port)
        port_top    = Port(:top,    trange_v, mode_v, kz, iy_top_port)

        mon_left    = Port(:left,   trange_h, mode_h, kz, ix_left_mon)
        mon_right   = Port(:right,  trange_h, mode_h, kz, ix_right_mon)
        mon_bottom  = Port(:bottom, trange_v, mode_v, kz, iy_bot_mon)
        mon_top     = Port(:top,    trange_v, mode_v, kz, iy_top_mon)

        ports    = (left=port_left, right=port_right, bottom=port_bottom, top=port_top)
        monitors = (left=mon_left,  right=mon_right,  bottom=mon_bottom,  top=mon_top)

        σ_max_ω = pml_sigma_max(ω, n_pml, res; R_target)
        grid    = FDFDGrid(Nx, Ny, dx, dy, n_pml, σ_max_ω)
        L       = build_stretched_laplacian(grid)

        port_vec = [port_left, port_right, port_bottom, port_top]
        B        = reduce(hcat, [build_source(grid, p) for p in port_vec])

        (L, B, ports, monitors)
    end

    Ls             = [r[1] for r in results]
    Bs             = [r[2] for r in results]
    ports_array    = [r[3] for r in results]
    monitors_array = [r[4] for r in results]

    grid_info = (Nx=Nx, Ny=Ny, dx=dx, dy=dy, n_pml=n_pml,
                 x_coords=collect(x_coords), y_coords=collect(y_coords),
                 design_iy=iy_d0:iy_d1, design_ix=ix_d0:ix_d1)

    return ε_base, ω_array, Ls, Bs, grid_info, ports_array, monitors_array
end

function batch_solve(ε_geom, n_geom, Ls, ωs, ε_base, Bs, design_iy, design_ix)
    Ny, Nx = size(ε_base)
    results = pmap(1:length(ωs)) do i
        ε_geom_full = zeros(eltype(ε_geom), Ny, Nx)
        ε_geom_full[design_iy, design_ix] .= ε_geom
        p = ωs[i]^2 .* vec(ε_base .+ (n_geom^2 - 1) .* ε_geom_full)
        A = Ls[i] + spdiagm(0 => p)
        F = lu(A)
        X = F \ Bs[i]
        dA_diag  = ωs[i]^2 .* vec(2 * n_geom .* ε_geom_full)  # ∂diag(A)/∂n_geom
        d2A_diag = ωs[i]^2 .* vec(2 .* ε_geom_full)            # ∂²diag(A)/∂n_geom²
        C1 = -dA_diag .* X
        Y  = F \ C1
        C2 = -(d2A_diag .* X .+ 2 .* dA_diag .* Y)
        Z  = F \ C2
        (X, Y, Z)
    end
    return results
end

# Custom rrule for Zygote/ForwardDiff compatibility
function ChainRulesCore.rrule(::typeof(batch_solve), ε_geom, n_geom, Ls, ωs, ε_base, Bs,
                              design_iy, design_ix)

    # 1. Forward Pass (Runs exactly as defined)
    results = batch_solve(ε_geom, n_geom, Ls, ωs, ε_base, Bs, design_iy, design_ix)

    # 2. Backward Pass (Pullback)
    function batch_solve_pullback(Δresults_raw)
        # Unthunk unwraps Lazy/Zero tangents generated by Zygote
        Δresults = unthunk(Δresults_raw)
        Ny, Nx = size(ε_base)

        # Distribute the adjoint computations across workers
        Δε_geom_full_list = pmap(1:length(ωs)) do i

            # Extract primal solutions and cotangents
            X, Y, Z = results[i]
            Δres_i = unthunk(Δresults[i])
            ΔX = unthunk(Δres_i[1])
            ΔY = unthunk(Δres_i[2])
            ΔZ = unthunk(Δres_i[3])

            # Recompute local variables to avoid serializing UmfpackLU C-pointers
            # across process boundaries (which causes segfaults).
            ε_geom_full = zeros(eltype(ε_geom), Ny, Nx)
            ε_geom_full[design_iy, design_ix] .= ε_geom
            p = ωs[i]^2 .* vec(ε_base .+ (n_geom^2 - 1) .* ε_geom_full)
            F = lu(Ls[i] + spdiagm(0 => p))
            dA_diag  = ωs[i]^2 .* vec(2 * n_geom .* ε_geom_full)
            d2A_diag = ωs[i]^2 .* vec(2 .* ε_geom_full)

            # -----------------------------------------------------------------
            # Reverse-Mode AD Math (Wirtinger Calculus)
            # -----------------------------------------------------------------

            # A) Reverse solve for: Z = F \ C2
            ΔZ_mat = ΔZ isa AbstractZero ? zeros(ComplexF64, size(X)) : ΔZ
            ΔC2 = F' \ ΔZ_mat

            # B) Reverse for: C2 = -(d2A_diag .* X + 2 * dA_diag .* Y)
            Δd2A_diag    = -vec(sum(ΔC2 .* conj.(X), dims=2))
            ΔdA_diag_C2  = -2 .* vec(sum(ΔC2 .* conj.(Y), dims=2))
            ΔX_from_C2   = -ΔC2 .* conj.(d2A_diag)
            ΔY_from_C2   = -2 .* ΔC2 .* conj.(dA_diag)

            # C) Reverse solve for: Y = F \ C1
            ΔY_mat   = ΔY isa AbstractZero ? zeros(ComplexF64, size(X)) : ΔY
            ΔY_total = ΔY_mat .+ ΔY_from_C2
            ΔC1 = F' \ ΔY_total

            # D) Reverse for: C1 = -dA_diag .* X
            ΔdA_diag_C1 = -vec(sum(ΔC1 .* conj.(X), dims=2))
            ΔX_from_C1  = -ΔC1 .* conj.(dA_diag)

            # E) Reverse solve for: X = F \ Bs[i]
            ΔX_total = ΔX .+ ΔX_from_C2 .+ ΔX_from_C1
            ΔB = F' \ ΔX_total

            # F) Gradients w.r.t the diagonal vector `p` of matrix `A`.
            Δp = -vec(sum(ΔB  .* conj.(X), dims=2)) .-
                  vec(sum(ΔC1 .* conj.(Y), dims=2)) .-
                  vec(sum(ΔC2 .* conj.(Z), dims=2))

            # G) Chain rule back to ε_geom_full (full domain)
            # p = ωs[i]^2 * (ε_base + (n_geom^2 - 1) * ε_geom_full)
            term_p = Δp .* conj.(ωs[i]^2 * (n_geom^2 - 1))

            # dA_diag = ωs[i]^2 * 2 * n_geom * ε_geom_full
            ΔdA_diag_total = ΔdA_diag_C1 .+ ΔdA_diag_C2
            term_d = ΔdA_diag_total .* conj.(ωs[i]^2 * 2 * n_geom)

            # d2A_diag = ωs[i]^2 * 2 * ε_geom_full
            term_d2 = Δd2A_diag .* conj.(ωs[i]^2 * 2)

            # Local full-domain gradient contribution for this frequency
            term_p .+ term_d .+ term_d2
        end

        # Sum contributions, reshape to (Ny, Nx), then extract design region.
        # Project to real if ε_geom is real-typed: ∂L/∂ε is mathematically real
        # when ε is real (real loss, real input) — imaginary part is zero in exact arithmetic.
        Δε_geom_full = reshape(sum(Δε_geom_full_list), Ny, Nx)
        Δε_geom_design = Δε_geom_full[design_iy, design_ix]
        Δε_geom_out = eltype(ε_geom) <: Real ? real(Δε_geom_design) : Δε_geom_design

        # Return gradients mapped to:
        # (function, ε_geom, n_geom, Ls, ωs, ε_base, Bs, design_iy, design_ix)
        return (NoTangent(), Δε_geom_out, NoTangent(), NoTangent(), NoTangent(),
                NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return results, batch_solve_pullback
end

# ═══════════════════════════════════════════════════════════════════════════════
# S-matrix extraction
# ═══════════════════════════════════════════════════════════════════════════════

function getSmatrices(ε_geom, n_geom, ε_base, ωs, Ls, Bs, grid_info, monitors_array, a_f_array, a_b_array;
                      design_iy, design_ix)

    results = batch_solve(ε_geom, n_geom, Ls, ωs, ε_base, Bs, design_iy, design_ix)

    Nx = grid_info.Nx
    Ny = grid_info.Ny
    dx = grid_info.dx
    dy = grid_info.dy

    S_and_derivs = map(1:length(ωs)) do k
        X, Y, Z = results[k]

        monitors = monitors_array[k]
        a_f      = a_f_array[k]
        a_b      = a_b_array[k]

        mon_vec = [monitors.left, monitors.right, monitors.bottom, monitors.top]
        mon_d   = [mon.edge in (:left, :right) ? dy : dx for mon in mon_vec]

        overlap(field, i, j) = extract_overlap(reshape(field[:, j], Ny, Nx), mon_vec[i], mon_d[i])

        S_vec = [
            begin
                tmp = overlap(X, i, j)
                i == j ? (tmp + (-1)^i * a_b) / a_f : tmp / a_f
            end
            for i in 1:4, j in 1:4
        ]
        # Y[:,j] = ∂X[:,j]/∂n_geom; Z[:,j] = ∂²X[:,j]/∂n_geom²
        # a_f and a_b are independent of n_geom, so derivatives pass through linearly.
        dSdn_vec   = [overlap(Y, i, j) / a_f for i in 1:4, j in 1:4]
        d2Sdn2_vec = [overlap(Z, i, j) / a_f for i in 1:4, j in 1:4]

        (reshape(S_vec, 4, 4), reshape(dSdn_vec, 4, 4), reshape(d2Sdn2_vec, 4, 4))
    end

    S_array      = [p[1] for p in S_and_derivs]
    dSdn_array   = [p[2] for p in S_and_derivs]
    d2Sdn2_array = [p[3] for p in S_and_derivs]
    return S_array, dSdn_array, d2Sdn2_array
end


# ═══════════════════════════════════════════════════════════════════════════════
# Straight-waveguide calibration  (port / monitor reference amplitudes)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    calibrate_straight_waveguide(ωmin, ωmax, ωnum, n_core, w; kwargs...)
        -> (a_f_array, a_b_array)

Simulate a straight dielectric-slab waveguide excited from the left at each
frequency in `range(ωmin, ωmax, length=ωnum)` and return per-frequency
reference amplitudes needed to normalise the 4-port scattering matrix computed
by `getSmatrices`.

The port and monitor plane positions use **exactly** the same index expressions
as `setup_4port_sweep`; the keyword arguments must be passed with the same
values used to build the 4-port geometry.

Arguments
- `ωmin`, `ωmax` : frequency sweep bounds.
- `ωnum`         : number of frequency points (linearly spaced).
- `n_core`       : core refractive index (> 1).
- `w`            : waveguide width (same units as 1/ω).

Keyword arguments (must match the 4-port geometry setup)
- `Lx`, `Ly`    : total domain size (default 10 × 10).
- `res`         : grid points per unit length (default 20).
- `n_pml`       : PML thickness in grid cells (default 20).
- `R_target`    : target round-trip PML amplitude reflection (default 1e-8).
                  Passed to `pml_sigma_max(ω, n_pml, res; R_target)` per frequency.
- `port_offset` : injection plane offset from PML edge (default 20).
                  Injection plane: ix_port = n_pml + port_offset.
- `mon_offset`  : monitor plane offset from PML edge (default 5).
                  Forward  monitor: ix_fwd = Nx − n_pml − mon_offset + 1.
                  Backward monitor: ix_bwd = n_pml + mon_offset.

Returns
- `a_f_array` : `Vector{Float64}` — forward mode amplitude at each frequency.
- `a_b_array` : `Vector{ComplexF64}` — backward residual amplitude at each frequency.
"""
function calibrate_straight_waveguide(
        ωmin        :: Number,
        ωmax        :: Number,
        ωnum        :: Int,
        n_core      :: Float64,
        w           :: Float64;
        Lx          :: Float64 = 10.0,
        Ly          :: Float64 = 10.0,
        res         :: Int     = 20,
        n_pml       :: Int     = 20,
        R_target    :: Float64 = 1e-8,
        port_offset :: Int     = 20,
        mon_offset  :: Int     = 5)

    @assert mon_offset < port_offset "mon_offset must be < port_offset"

    dx = 1.0 / res
    dy = 1.0 / res
    Nx = round(Int, Lx * res)
    Ny = round(Int, Ly * res)

    y_coords = range(-Ly/2, Ly/2, length=Ny)

    # ── Build ε_base for a straight horizontal waveguide (ω-independent) ────────
    ε_base  = ones(ComplexF64, Ny, Nx)
    iy_c = Ny ÷ 2
    iy_wg0 = iy_c - round(Int, w/2 * res) + 1
    iy_wg1 = iy_c + round(Int, w/2 * res)
    for iy in iy_wg0:iy_wg1
        ε_base[iy, :] .= n_core^2
    end

    # ── Port / monitor plane indices (ω-independent) ─────────────────────────
    ix_port = n_pml + port_offset          # excitation port  (≡ left port in 4-port)
    ix_fwd  = Nx - n_pml - mon_offset + 1 # forward  monitor (≡ right monitor in 4-port)
    ix_bwd  = n_pml + mon_offset           # backward monitor (≡ left monitor in 4-port)

    # ── Per-ω: mode, PML, solve, reference amplitudes ────────────────────────
    ω_array = collect(range(Float64(ωmin), Float64(ωmax), length = ωnum))

    results = map(ω_array) do ω
        λ         = 2π / Float64(ω)
        ext_cells = round(Int, 2 * λ * res)
        trange    = max(n_pml + 1, iy_wg0 - ext_cells) : min(Ny - n_pml, iy_wg1 + ext_cells)
        y_h       = [Float64(y_coords[iy]) for iy in trange]
        kz, mode  = compute_waveguide_mode(w, n_core, Float64(ω), y_h)

        inj_port = Port(:left, trange, mode, kz, ix_port)
        mon_fwd  = Port(:left, trange, mode, kz, ix_fwd)
        mon_bwd  = Port(:left, trange, mode, kz, ix_bwd)

        σ_max_ω = pml_sigma_max(Float64(ω), n_pml, res; R_target)
        grid    = FDFDGrid(Nx, Ny, dx, dy, n_pml, σ_max_ω)
        L       = build_stretched_laplacian(grid)
        A       = L + spdiagm(0 => ω^2 .* vec(ε_base))
        b       = build_source(grid, inj_port)
        Ez_flat = A \ b
        Ez      = reshape(Ez_flat, Ny, Nx)

        a_f = Float64(abs(extract_overlap(Ez, mon_fwd, dy)))
        a_b = ComplexF64(extract_overlap(Ez, mon_bwd, dy))
        (a_f, a_b)
    end

    a_f_array = [r[1] for r in results]
    a_b_array = [r[2] for r in results]
    return a_f_array, a_b_array
end

# ─────────────────────────────────────────────────────────────────────────────
# Index metadata (cached per lattice size n)
# ─────────────────────────────────────────────────────────────────────────────

function _lattice_index_metadata(n::Int)
    get!(_lattice_index_cache, n) do
        gidx(i, j, p) = 4 * ((i-1)*n + (j-1)) + p

        # Port classification: a port (i,j,p) is external iff on the boundary.
        is_ext(i, j, p) = (p == 1 && j == 1) || (p == 2 && j == n) ||
                          (p == 3 && i == n) || (p == 4 && i == 1)

        # Row-major scan, port-number order within each scatterer
        external_idx = Int[gidx(i,j,p) for i in 1:n for j in 1:n
                                        for p in 1:4 if  is_ext(i,j,p)]
        internal_idx = Int[gidx(i,j,p) for i in 1:n for j in 1:n
                                        for p in 1:4 if !is_ext(i,j,p)]
        n_ext = length(external_idx)
        n_int = length(internal_idx)   # = 4n(n-1)

        # Decode a global port index: gidx(i,j,p) = 4*((i-1)*n + (j-1)) + p
        slin(g) = (g - 1) ÷ 4          # 0-based linear scatterer index
        si(g)   = slin(g) ÷ n + 1      # scatterer row    (1-based)
        sj(g)   = slin(g) % n + 1      # scatterer column (1-based)
        port(g) = (g - 1) % 4 + 1      # local port       (1-based)

        ext_slin = Int[slin(g) for g in external_idx]
        int_slin = Int[slin(g) for g in internal_idx]
        ext_si   = Int[si(g)   for g in external_idx]
        ext_sj   = Int[sj(g)   for g in external_idx]
        int_si   = Int[si(g)   for g in internal_idx]
        int_sj   = Int[sj(g)   for g in internal_idx]
        ext_port = Int[port(g) for g in external_idx]
        int_port = Int[port(g) for g in internal_idx]

        # Connection permutation P: build via Dict (not AbstractArray), then
        # materialise as a Matrix{Int} via comprehension.
        port_pos  = Dict(g => k for (k,g) in enumerate(internal_idx))
        connected = Dict{Int,Int}()
        for i in 1:n, j in 1:n-1        # horizontal: (i,j) right ↔ (i,j+1) left
            pa, pb = gidx(i, j, 2), gidx(i, j+1, 1)
            connected[port_pos[pa]] = port_pos[pb]
            connected[port_pos[pb]] = port_pos[pa]
        end
        for i in 1:n-1, j in 1:n        # vertical: (i,j) bottom ↔ (i+1,j) top
            pa, pb = gidx(i, j, 3), gidx(i+1, j, 4)
            connected[port_pos[pa]] = port_pos[pb]
            connected[port_pos[pb]] = port_pos[pa]
        end
        P = Int[get(connected, i, 0) == j ? 1 : 0 for i in 1:n_int, j in 1:n_int]

        # ── Scatterer groupings for block-sum construction of S_cc, S_ec ────────
        # For each scatterer (i,j) precompute constant selection matrices E_ext
        # (n_ext × n_ext_l) and E_int (n_int × n_int_l), so that the partition
        # blocks can be assembled as Σ_l E_l · S_l · E_l' without an O(n⁴)
        # comprehension over all port-pair combinations.
        scat_groups = Vector{Any}(undef, n^2)
        for i in 1:n, j in 1:n
            l  = (i-1)*n + (j-1)
            ea = findall(ext_slin .== l)   # positions in 1:n_ext for scatterer l
            ia = findall(int_slin .== l)   # positions in 1:n_int for scatterer l
            ep = isempty(ea) ? Int[] : ext_port[ea]
            ip = isempty(ia) ? Int[] : int_port[ia]
            E_ext = isempty(ea) ? zeros(Int, n_ext, 0) :
                    Int[a == ea[α] ? 1 : 0 for a in 1:n_ext, α in eachindex(ea)]
            E_int = isempty(ia) ? zeros(Int, n_int, 0) :
                    Int[a == ia[α] ? 1 : 0 for a in 1:n_int, α in eachindex(ia)]
            # F_ext/F_int: 4×|ep| and 4×|ip| port-embedding matrices.
            # F_ext[p,α]=1 iff ep[α]==p — maps ea-indexed amplitudes into a 4-port block vector.
            # Used by powers_and_jac to assemble block inputs and project output perturbations.
            F_ext = zeros(Int, 4, length(ep))
            for (α, p) in enumerate(ep); F_ext[p, α] = 1; end
            F_int = zeros(Int, 4, length(ip))
            for (α, p) in enumerate(ip); F_int[p, α] = 1; end
            scat_groups[(j-1)*n + i] = (; si=i, sj=j, ea, ia, ep, ip, E_ext, E_int, F_ext, F_int)
        end

        (; n_ext, n_int, ext_slin, int_slin, ext_si, ext_sj,
           int_si, int_sj, ext_port, int_port, P, scat_groups)
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Lattice port-power forward model and analytical Jacobian ∂μ/∂vec(Δn)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Shared lattice interconnection core
# ─────────────────────────────────────────────────────────────────────────────
#
# _lattice_core assembles the block S-matrices, solves the interconnection,
# and returns all intermediate quantities needed by the public API functions.
# This avoids triplicating the S-matrix assembly + LU solve across
# powers_only / jac_only / jac_and_dirderiv_s.

function _lattice_freq_core(k, Δn, n_lat, a_in, S_arr, dSdn_arr, d2Sdn2_arr,
                            n_int, P, scat_groups; da_in=nothing)
    S_b  = [S_arr[k]    .+ dSdn_arr[k]     .* Δn[i,j] .+ d2Sdn2_arr[k] .* Δn[i,j]^2
            for i in 1:n_lat, j in 1:n_lat]
    dS_b = [dSdn_arr[k] .+ 2 .* d2Sdn2_arr[k] .* Δn[i,j]
            for i in 1:n_lat, j in 1:n_lat]

    S_ee = sum(sg -> sg.E_ext * S_b[sg.si,sg.sj][sg.ep, sg.ep] * sg.E_ext', scat_groups)
    S_ec = sum(sg -> sg.E_ext * S_b[sg.si,sg.sj][sg.ep, sg.ip] * sg.E_int', scat_groups)
    S_ce = sum(sg -> sg.E_int * S_b[sg.si,sg.sj][sg.ip, sg.ep] * sg.E_ext', scat_groups)
    S_cc = sum(sg -> sg.E_int * S_b[sg.si,sg.sj][sg.ip, sg.ip] * sg.E_int', scat_groups)

    A      = I(n_int) - S_cc * P
    A_lu   = lu(A)                       # factorize once, reuse for all solves
    a_int  = P * (A_lu \ (S_ce * a_in))
    a_out  = S_ee * a_in + S_ec * a_int

    da_int = da_in !== nothing ? P * (A_lu \ (S_ce * da_in)) : nothing
    da_out = da_in !== nothing ? S_ee * da_in + S_ec * da_int : nothing

    return (; S_b, dS_b, S_ec, A_lu, P, a_in, a_int, a_out,
              da_in, da_int, da_out)
end

"""
    jac_only(Δn, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω) -> J

Compute the analytical Jacobian J = ∂μ/∂vec(Δn) (size n_ext × n²_lat).

Compatible with Zygote and ForwardDiff.
"""
function jac_only(Δn::Matrix, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
    n_lat = size(Δn, 1)
    meta = @ignore_derivatives _lattice_index_metadata(n_lat)
    (; n_ext, n_int, P, scat_groups) = meta

    cT   = typeof(cis(float(one(φ₁))))
    a_in = vcat(one(cT), cis(φ₁), cis(φ₂), zeros(cT, n_ext - 3))

    Js = map(eachindex(S_arr)) do k
        fc = _lattice_freq_core(k, Δn, n_lat, a_in, S_arr, dSdn_arr, d2Sdn2_arr,
                                n_int, P, scat_groups)
        Gk_dω = GΔω[k]

        J_k = map(scat_groups) do sg
            a_blk  = sg.F_ext * (sg.E_ext' * fc.a_in) + sg.F_int * (sg.E_int' * fc.a_int)
            δb     = fc.dS_b[sg.si, sg.sj] * a_blk
            δa_out = sg.E_ext * (sg.F_ext' * δb) +
                     fc.S_ec * (fc.P * (fc.A_lu \ (sg.E_int * (sg.F_int' * δb))))
            2 .* real.(conj.(fc.a_out) .* δa_out) .* Gk_dω
        end
        reduce(hcat, J_k)
    end
    return sum(Js)
end

"""
    jac_and_dirderiv_s(Δn, φ₁, φ₂, λ, S_arr, dSdn_arr, d2Sdn2_arr, GΔω) -> (J, dJ_λ)

Compute J = ∂μ/∂vec(Δn) **and** its directional derivative
dJ_λ = Σ_k λ_k · ∂J/∂s_k analytically.  No ForwardDiff types in the computation.
"""
function jac_and_dirderiv_s(Δn::Matrix, φ₁, φ₂, λ, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
    n_lat = size(Δn, 1)
    meta = @ignore_derivatives _lattice_index_metadata(n_lat)
    (; n_ext, n_int, P, scat_groups) = meta

    a_in  = vcat(one(ComplexF64), cis(φ₁), cis(φ₂),
                 zeros(ComplexF64, n_ext - 3))
    da_in = vcat(zero(ComplexF64), im * λ[1] * cis(φ₁), im * λ[2] * cis(φ₂),
                 zeros(ComplexF64, n_ext - 3))

    results = map(eachindex(S_arr)) do k
        fc = _lattice_freq_core(k, Δn, n_lat, a_in, S_arr, dSdn_arr, d2Sdn2_arr,
                                n_int, P, scat_groups; da_in=da_in)
        Gk_dω = GΔω[k]

        cols = map(scat_groups) do sg
            a_blk   = sg.F_ext * (sg.E_ext' * fc.a_in)  + sg.F_int * (sg.E_int' * fc.a_int)
            δb      = fc.dS_b[sg.si, sg.sj] * a_blk
            δa_out  = sg.E_ext * (sg.F_ext' * δb) +
                      fc.S_ec * (fc.P * (fc.A_lu \ (sg.E_int * (sg.F_int' * δb))))

            da_blk  = sg.F_ext * (sg.E_ext' * fc.da_in) + sg.F_int * (sg.E_int' * fc.da_int)
            dδb     = fc.dS_b[sg.si, sg.sj] * da_blk
            dδa_out = sg.E_ext * (sg.F_ext' * dδb) +
                      fc.S_ec * (fc.P * (fc.A_lu \ (sg.E_int * (sg.F_int' * dδb))))

            j_col  = 2 .* real.(conj.(fc.a_out)  .* δa_out) .* Gk_dω
            dj_col = 2 .* real.(conj.(fc.da_out) .* δa_out .+
                                conj.(fc.a_out)  .* dδa_out) .* Gk_dω
            (j_col, dj_col)
        end

        (reduce(hcat, [c[1] for c in cols]),
         reduce(hcat, [c[2] for c in cols]))
    end

    J  = sum(r -> r[1], results)
    dJ = sum(r -> r[2], results)
    return J, dJ
end


function powers_only(Δn::Matrix, φ₁, φ₂, S_arr, dSdn_arr, d2Sdn2_arr, GΔω)
    n_lat = size(Δn, 1)
    meta = @ignore_derivatives _lattice_index_metadata(n_lat)
    (; n_ext, n_int, P, scat_groups) = meta

    # a_in element type tracks φ₁, φ₂ so ForwardDiff Duals propagate correctly.
    cT   = typeof(cis(float(one(φ₁))))
    a_in = vcat(one(cT), cis(φ₁), cis(φ₂), zeros(cT, n_ext - 3))

    μs = map(eachindex(S_arr)) do k
        fc = _lattice_freq_core(k, Δn, n_lat, a_in, S_arr, dSdn_arr, d2Sdn2_arr,
                                n_int, P, scat_groups)
        abs2.(fc.a_out) .* GΔω[k]
    end
    return sum(μs)
end

end  # module SimGeomBroadBand
