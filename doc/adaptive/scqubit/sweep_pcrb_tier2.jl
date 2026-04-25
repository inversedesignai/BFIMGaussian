#=
sweep_pcrb_tier2.jl — tier-2 c (11-D) PCRB baseline.
Matches the 11-D c space of sweep_joint_tier2.jl for a fair §8.7 comparison.
=#
using Printf
using Serialization
using Dates
using Zygote

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt, .PCRB

println("sweep_pcrb_tier2.jl — tier-2 (11-D) c, J=20, L=2, K=3")
println("Threads: $(Threads.nthreads())")

const K_EPOCHS    = 3
const K_PHI       = 64
const J_TAU       = 20
const TAU_GRID    = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID      = (1, 10)
const OUTER_ITERS = parse(Int, get(ENV, "PCRB_ITERS", "400"))
const OUTER_LR    = parse(Float64, get(ENV, "PCRB_LR", "5e-3"))
const REOPT_EVERY = parse(Int, get(ENV, "PCRB_REOPT", "1"))

const T2_FIELD_NAMES = (:f_q_max, :E_C_over_h, :kappa, :Delta_qr, :temperature,
                        :A_phi, :A_Ic, :M, :Mprime, :C_qg, :C_c)

c_as_vec_t2(c::ScqubitParams) = [c.f_q_max, c.E_C_over_h, c.kappa, c.Delta_qr,
                                 c.temperature, c.A_phi, c.A_Ic,
                                 c.M, c.Mprime, c.C_qg, c.C_c]

function vec_as_c_t2(v::AbstractVector{T}) where {T<:Real}
    ScqubitParams{T}(
        f_q_max=v[1], E_C_over_h=v[2], kappa=v[3], Delta_qr=v[4],
        temperature=v[5], A_phi=v[6], A_Ic=v[7],
        M=v[8], Mprime=v[9], C_qg=v[10], C_c=v[11])
end

function tier2_realistic_box(bl::ScqubitParams)
    lo = [ 3.0e9,  0.15e9,  0.1e6,   0.8e9,  bl.temperature,  bl.A_phi,  bl.A_Ic,
           bl.M/3, bl.Mprime/3, bl.C_qg/3, bl.C_c/3]
    hi = [12.0e9,  0.4e9,   5.0e6,   5.0e9,  bl.temperature,  bl.A_phi,  bl.A_Ic,
           bl.M*3, bl.Mprime*3, bl.C_qg*3, bl.C_c*3]
    CBox(lo, hi)
end

grid = make_grid(; K_phi=K_PHI, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)
box = tier2_realistic_box(PAPER_BASELINE)

# ---- tier-2 pcrb_baseline (re-implementation of PCRB.pcrb_baseline with 11-D v) ----
omega_d_fn = make_omega_d_fn()

v = c_as_vec_t2(PAPER_BASELINE)
# project
for i in eachindex(v); v[i] = clamp(v[i], box.lo[i], box.hi[i]); end
scale = max.(box.hi .- box.lo, 0.0)

# Internal Adam state
mutable struct _Adam
    lr::Float64; β1::Float64; β2::Float64; ϵ::Float64
    m::Vector{Float64}; v::Vector{Float64}; t::Int
    scale::Vector{Float64}
end
_Adam(n::Int, lr; scale=ones(n)) = _Adam(lr, 0.9, 0.999, 1e-8, zeros(n), zeros(n), 0, scale)
function _adam!(v::AbstractVector, g::AbstractVector, s::_Adam)
    s.t += 1
    gs = g .* s.scale
    s.m .= s.β1 .* s.m .+ (1 - s.β1) .* gs
    s.v .= s.β2 .* s.v .+ (1 - s.β2) .* gs.^2
    mh = s.m ./ (1 - s.β1^s.t); vh = s.v ./ (1 - s.β2^s.t)
    step_n = s.lr .* mh ./ (sqrt.(vh) .+ s.ϵ)
    v .+= step_n .* s.scale
    v
end

function run_pcrb_t2!(v, grid, box, omega_d_fn, outer_iters, outer_lr, reopt_every)
    state = _Adam(length(v), outer_lr; scale=max.(box.hi .- box.lo, 0.0))
    hist = (log_JP = Float64[], grad_norm = Float64[],
            c_vec = Vector{Vector{Float64}}(),
            sched = Vector{Vector{Tuple{Int,Int}}}(),
            reopt_iter = Int[])
    c_cur = vec_as_c_t2(v); ωd = omega_d_fn(c_cur)
    (sched_idx, lJP0) = argmax_schedule_enumerate(grid, c_cur, ωd, K_EPOCHS)
    push!(hist.reopt_iter, 0)
    @printf("[init] log J_P = %.4f  sched=%s\n", lJP0, string(sched_idx))
    t0 = time()
    for iter in 1:outer_iters
        if (iter - 1) % reopt_every == 0 && iter > 1
            c_cur = vec_as_c_t2(v); ωd = omega_d_fn(c_cur)
            (sched_idx, _) = argmax_schedule_enumerate(grid, c_cur, ωd, K_EPOCHS)
            push!(hist.reopt_iter, iter)
        end
        # Precompute (τ, n) schedule and grid.phi_grid OUTSIDE Zygote tape.
        # The grid is a struct that Zygote trips on (Grid{J>4, L} + StepRangeLen).
        sched_tn = [(Float64(grid.tau_grid[j]), Int(grid.n_grid[ℓ])) for (j, ℓ) in sched_idx]
        pg = collect(Float64, grid.phi_grid)
        g = Zygote.gradient(
            v_ -> log(bim_prior_averaged(sched_tn, vec_as_c_t2(v_),
                                          omega_d_fn(vec_as_c_t2(v_)), pg;
                                          J_0=1e-4)),
            v)[1]
        gn = sqrt(sum(abs2, g))
        _adam!(v, g, state)
        for i in eachindex(v); v[i] = clamp(v[i], box.lo[i], box.hi[i]); end
        cc = vec_as_c_t2(v); ωd_cur = omega_d_fn(cc)
        lJP = log_JP_of_schedule(sched_idx, grid, cc, ωd_cur; J_0=1e-4)
        push!(hist.log_JP, lJP); push!(hist.grad_norm, gn)
        push!(hist.c_vec, copy(v)); push!(hist.sched, copy(sched_idx))
        if iter == 1 || iter % 20 == 0
            @printf("iter %4d log J_P=%.4f |g|=%.3e sched=%s\n",
                    iter, lJP, gn, string(sched_idx))
        end
    end
    @printf("\nTotal elapsed: %.1f min\n", (time() - t0) / 60)
    (v, sched_idx, hist)
end

(v, sched_final, hist) = run_pcrb_t2!(v, grid, box, omega_d_fn,
                                       OUTER_ITERS, OUTER_LR, REOPT_EVERY)
c_final = vec_as_c_t2(v)

outdir = joinpath(@__DIR__, "results", "pcrb_tier2")
isdir(outdir) || mkpath(outdir)
open(joinpath(outdir, "final.jls"), "w") do io
    serialize(io, (; c_final, v_final=c_as_vec_t2(c_final), sched_final,
                     history=hist, K_EPOCHS, K_PHI, TAU_GRID, N_GRID,
                     timestamp=now(), baseline=PAPER_BASELINE,
                     T2_FIELD_NAMES))
end

imax = argmax(hist.log_JP)
@printf("log J_P best iter %d = %.4f (init %.4f)\n", imax, hist.log_JP[imax], hist.log_JP[1])
println("best sched: ", hist.sched[imax])
println("best c (tier-2):")
for (n, vv) in zip(T2_FIELD_NAMES, hist.c_vec[imax])
    @printf("  %-12s = %+.4e\n", n, vv)
end
