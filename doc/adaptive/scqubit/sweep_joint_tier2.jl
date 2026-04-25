#=
sweep_joint_tier2.jl — tier-2 c (11-dim: tier-1 + 4 SQUID-loop geometric dims).

Extends the design vector c to include:
  M       — SQUID-loop mutual inductance (flux-bias line to loop)
  Mprime  — parasitic mutual to full x-mon
  C_qg    — x-mon to ground plane capacitance
  C_c     — x-mon to xy control-line capacitance

These enter Γ1_ind (M, Mprime) and Γ1_cap (C_qg, C_c), the relaxation rates
in the Ramsey envelope, so they ARE differentiable geometric design knobs.

Uses MSE terminal reward, J=20 fine-τ grid (matching §14 of scqubit_results).
=#
using Printf
using Serialization
using Dates

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt

println("sweep_joint_tier2.jl — tier-2 (11-D) c, J=20, L=2, K=3, MSE terminal")
println("Threads: $(Threads.nthreads())")

const K_EPOCHS    = 3
const K_PHI       = 64
const J_TAU       = 20
const TAU_GRID    = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID      = (1, 10)
const OUTER_ITERS = parse(Int, get(ENV, "JOINT_ITERS", "40"))
const OUTER_LR    = parse(Float64, get(ENV, "JOINT_LR", "5e-4"))
const REOPT_EVERY = parse(Int, get(ENV, "JOINT_REOPT", "5"))

# ---- tier-2 c vector mapping (11-D) ----
# layout: [f_q_max, E_C_over_h, kappa, Delta_qr, temperature, A_phi, A_Ic,
#          M, Mprime, C_qg, C_c]
const T2_FIELD_NAMES = (:f_q_max, :E_C_over_h, :kappa, :Delta_qr, :temperature,
                        :A_phi, :A_Ic, :M, :Mprime, :C_qg, :C_c)
const T2_DIM = length(T2_FIELD_NAMES)

function c_as_vec_t2(c::ScqubitParams)
    [c.f_q_max, c.E_C_over_h, c.kappa, c.Delta_qr, c.temperature,
     c.A_phi, c.A_Ic, c.M, c.Mprime, c.C_qg, c.C_c]
end

function vec_as_c_t2(v::AbstractVector{T}) where {T<:Real}
    ScqubitParams{T}(
        f_q_max     = v[1],
        E_C_over_h  = v[2],
        kappa       = v[3],
        Delta_qr    = v[4],
        temperature = v[5],
        A_phi       = v[6],
        A_Ic        = v[7],
        M           = v[8],
        Mprime      = v[9],
        C_qg        = v[10],
        C_c         = v[11],
    )
end

"Realistic geometric-only box: tier-1 circuit ranges + tier-2 SQUID-loop
dimensions with a factor-of-3 range around paper baseline for M, Mprime,
C_qg, C_c.  Noise and temperature pinned at paper baseline."
function tier2_realistic_box(bl::ScqubitParams)
    lo = [ 3.0e9,  0.15e9,  0.1e6,   0.8e9,  bl.temperature,  bl.A_phi,  bl.A_Ic,
           bl.M/3, bl.Mprime/3, bl.C_qg/3, bl.C_c/3]
    hi = [12.0e9,  0.4e9,   5.0e6,   5.0e9,  bl.temperature,  bl.A_phi,  bl.A_Ic,
           bl.M*3, bl.Mprime*3, bl.C_qg*3, bl.C_c*3]
    CBox(lo, hi)
end

# ---------------------------------------------------------------
# Tier-2-aware joint_opt (parallel to JointOpt.joint_opt but with
# 11-D c vector translation; everything else identical).
# ---------------------------------------------------------------
function joint_opt_t2(c0::ScqubitParams;
                      grid::Main.Belief.Grid, K_epochs::Int,
                      outer_iters::Int, outer_lr::Float64,
                      policy_reopt_every::Int, ckpt_every::Int,
                      ckpt_dir::String, cbox::CBox,
                      terminal::Symbol=:mse, verbose::Bool=true)
    isdir(ckpt_dir) || mkpath(ckpt_dir)
    v = c_as_vec_t2(c0)
    project_c!(v, cbox)
    scale = max.(cbox.hi .- cbox.lo, 0.0)
    state = AdamState(length(v); lr=outer_lr, scale=scale)

    hist = (V_adaptive = Float64[], grad_norm = Float64[],
            c_vec = Vector{Vector{Float64}}(),
            reopt_iter = Int[], omega_d = Float64[],
            memo_size = Int[], elapsed = Float64[])

    omega_d_fn = make_omega_d_fn()
    c_cur = vec_as_c_t2(v); ω_d = omega_d_fn(c_cur)
    (V_cur, memo, st) = solve_bellman_full(grid, K_epochs, c_cur, ω_d; terminal=terminal)
    push!(hist.reopt_iter, 0); push!(hist.memo_size, st.memo_size)
    verbose && @printf("[init] V=%.6e memo=%d ω_d=%.3e %.2fs\n",
                       V_cur, st.memo_size, ω_d, st.elapsed)

    for iter in 1:outer_iters
        t0 = time()
        refreshed = false
        if (iter - 1) % policy_reopt_every == 0 && iter > 1
            c_cur = vec_as_c_t2(v); ω_d = omega_d_fn(c_cur)
            (V_cur, memo, st) = solve_bellman_full(grid, K_epochs, c_cur, ω_d; terminal=terminal)
            push!(hist.reopt_iter, iter); push!(hist.memo_size, st.memo_size)
            refreshed = true
        end
        # Envelope gradient with 11-D c
        g = Zygote.gradient(
            v_ -> V_adaptive_policy_exact(vec_as_c_t2(v_), memo, grid, ω_d, K_epochs; terminal=terminal),
            v)[1]
        gn = sqrt(sum(abs2, g))
        adam_update!(v, g, state)
        project_c!(v, cbox)

        push!(hist.V_adaptive, V_cur); push!(hist.grad_norm, gn)
        push!(hist.c_vec, copy(v)); push!(hist.omega_d, ω_d)
        push!(hist.elapsed, time() - t0)

        if verbose && (iter == 1 || iter % max(1, policy_reopt_every ÷ 2) == 0)
            marker = refreshed ? "[reopt]" : "       "
            @printf("%s iter %4d V=%.4e |g|=%.3e Δt=%.2fs ω_d=%.4e\n",
                    marker, iter, V_cur, gn, hist.elapsed[end], ω_d)
        end
        if ckpt_every > 0 && iter % ckpt_every == 0
            open(joinpath(ckpt_dir, @sprintf("ckpt_%06d.jls", iter)), "w") do io
                serialize(io, (; v=copy(v), iter=iter, history=hist, timestamp=now()))
            end
        end
    end
    (vec_as_c_t2(v), hist)
end

using Zygote

grid = make_grid(; K_phi=K_PHI, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)
box  = tier2_realistic_box(PAPER_BASELINE)

println(@sprintf("Config: K=%d K_Φ=%d J=%d L=%d iters=%d lr=%.1e reopt=%d dim(c)=%d",
                 K_EPOCHS, K_PHI, length(TAU_GRID), length(N_GRID),
                 OUTER_ITERS, OUTER_LR, REOPT_EVERY, T2_DIM))

# Memory probe
t_probe = time()
(V0, memo0, st0) = solve_bellman_full(grid, K_EPOCHS, PAPER_BASELINE,
                                       omega_q(0.442, PAPER_BASELINE); terminal=:mse)
@printf("Probe Bellman at c₀: V=%.4e memo=%d %.2fs\n", V0, st0.memo_size, time()-t_probe)
if st0.memo_size > 5_000_000
    println("!! memo too large, aborting")
    exit(1)
end

t0 = time()
(c_final, hist) = joint_opt_t2(PAPER_BASELINE;
    grid=grid, K_epochs=K_EPOCHS,
    outer_iters=OUTER_ITERS, outer_lr=OUTER_LR,
    policy_reopt_every=REOPT_EVERY,
    ckpt_every=10,
    ckpt_dir=joinpath(@__DIR__, "results", "joint_tier2"),
    cbox=box, terminal=:mse, verbose=true)
@printf("\nTotal elapsed: %.1f min\n", (time() - t0) / 60)

out_path = joinpath(@__DIR__, "results", "joint_tier2", "final.jls")
isdir(dirname(out_path)) || mkpath(dirname(out_path))
open(out_path, "w") do io
    serialize(io, (; c_final, v_final=c_as_vec_t2(c_final),
                     history=hist, K_EPOCHS, K_PHI,
                     TAU_GRID, N_GRID, timestamp=now(),
                     baseline=PAPER_BASELINE,
                     terminal=:mse,
                     T2_FIELD_NAMES))
end
println("Saved to $out_path")

nj = length(hist.c_vec)
va = hist.V_adaptive[1:nj]
ibest = argmax(va)
@printf("-E[Var_post](c₀)   = %.4e\n", va[1])
@printf("-E[Var_post](best) = %.4e (iter %d; Var=%.4e)\n", va[ibest], ibest, -va[ibest])
println("Best c (tier-2, 11-D):")
for (n, vv) in zip(T2_FIELD_NAMES, hist.c_vec[ibest])
    @printf("  %-12s = %+.4e\n", n, vv)
end
