"""
JointOpt.jl

Outer Adam loop on c with periodic Bellman policy re-solution.

At each outer iteration:
  1. If `outer_iter % policy_reopt_every == 1` (or iter == 1): re-solve the
     Bellman DP at the current c to refresh the policy memo.
  2. Compute the envelope-theorem gradient ∂V_adaptive/∂c via
     `grad_c_exact(v, memo, grid, ω_d, K)` (exact tree traversal, fixed policy).
  3. Adam step on the 7-vector; project into box bounds.

History recorded per iter:
  - V_adaptive (computed at the refreshed policy, equals the full DP value)
  - grad_norm
  - c_vec (full history if small)
"""
module JointOpt

using Main.ScqubitModel
using Main.Belief
using Main.Bellman
using Main.Gradient
using LinearAlgebra
using Serialization
using Printf
using Dates

export joint_opt, AdamState, adam_update!, project_c!, CBox, default_cbox,
       omega_d_baseline_ωd_of_c,
       make_phi_star_fn, make_omega_d_fn

# ---------------------------------------------------------------
# Box constraints for tier-1 c = (f_q_max, E_C/h, κ, Δ_qr, T, A_φ, A_Ic)
# Units follow ScqubitModel: f_q_max, E_C/h, κ, Δ_qr in Hz (linear).
# T in K.  A_φ, A_Ic dimensionless fractions of Φ₀, I_c.
# ---------------------------------------------------------------
struct CBox
    lo::Vector{Float64}
    hi::Vector{Float64}
end

function default_cbox()
    lo = [ 1.0e9,   0.1e9,  0.01e6,   0.5e9,   5e-3,   1e-7,  1e-7]
    hi = [30.0e9,   1.0e9,  10.0e6,  10.0e9, 100e-3,   1e-4,  1e-4]
    CBox(lo, hi)
end

"In-place projection of v onto the box."
function project_c!(v::AbstractVector, box::CBox)
    @inbounds for i in eachindex(v)
        v[i] = clamp(v[i], box.lo[i], box.hi[i])
    end
    v
end

# ---------------------------------------------------------------
# Adam state on Vector{Float64}
# ---------------------------------------------------------------
mutable struct AdamState
    lr::Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
    scale::Vector{Float64}     # per-component scale (default ones); step is multiplied by this
end

AdamState(dim::Int; lr=1e-3, β1=0.9, β2=0.999, ϵ=1e-8, scale=ones(dim)) =
    AdamState(lr, β1, β2, ϵ, zeros(dim), zeros(dim), 0, scale)

"""
    adam_update!(v, grad, state)

In-place Adam step using the gradient of a MAXIMISATION objective
(i.e., we step v ← v + lr * adam_scaled_grad to increase V).
When state.scale != 1, Adam operates on the gradient expressed in
box-normalised units: internally uses g̃ = grad .* scale, then converts
the step back to v-space by multiplying the adam-normalised step
by scale again.  Result: per-iter step magnitude in v-space is
≈ lr * scale * sign(grad), which is box-aware when scale = box width.
Returns the step vector Δv.
"""
function adam_update!(v::AbstractVector, grad::AbstractVector, s::AdamState)
    s.t += 1
    g = grad .* s.scale                 # box-normalised gradient
    s.m .= s.β1 .* s.m .+ (1 - s.β1) .* g
    s.v .= s.β2 .* s.v .+ (1 - s.β2) .* g.^2
    m_hat = s.m ./ (1 - s.β1^s.t)
    v_hat = s.v ./ (1 - s.β2^s.t)
    step_norm = s.lr .* m_hat ./ (sqrt.(v_hat) .+ s.ϵ)   # in normalised space
    step = step_norm .* s.scale                           # back to v-space
    v .+= step
    step
end

# ---------------------------------------------------------------
# φ* and ω_d as (differentiable if needed) functions of c
# The paper's operating point is chosen by a 1-D single-shot sensitivity
# argmax; we cache it as a plain float (non-differentiable) here.
# ---------------------------------------------------------------
"""
    phi_star_of_c(c; n_phi_grid=400) -> (phi_star, amp_star)

Grid-argmax of the single-shot Ramsey sensitivity envelope
(paper Eq. 12/16).  Non-differentiable — used to CHOOSE the operating point.
"""
function make_phi_star_fn(; n_phi_grid::Int=400)
    (c::ScqubitParams) -> begin
        φ_grid = range(0.05, 0.485; length=n_phi_grid)
        best = -Inf
        best_phi = 0.442
        for phi in φ_grid
            A = A_coef(phi, c)
            B = B_coef(phi, c)
            τopt = (-A + sqrt(A^2 + 8*B^2)) / (4*B^2)
            th = thermal_factor(phi, c)
            dωdφ = abs(domega_q_dphi(phi, c))
            amp = (τopt/2) * th * exp(-A*τopt - B^2 * τopt^2) * dωdφ
            if amp > best
                best = amp
                best_phi = phi
            end
        end
        (best_phi, best)
    end
end

"Construct a ω_d(c) closure that evaluates ω_d at the current operating-point φ*."
function make_omega_d_fn(; phi_star_fn=make_phi_star_fn())
    (c::ScqubitParams) -> begin
        (φ_star, _) = phi_star_fn(c)
        omega_q(φ_star, c)
    end
end

# ---------------------------------------------------------------
# joint_opt
# ---------------------------------------------------------------
"""
    joint_opt(c0; ...) -> (c_final::ScqubitParams, history::NamedTuple)

Outer Adam on c with envelope-theorem gradient and periodic Bellman policy
re-solution.

Arguments:
  - c0                 : initial ScqubitParams.
  - grid               : Belief.Grid (fixes K_Φ, τ_grid, n_grid).
  - K_epochs           : horizon.
  - outer_iters        : number of Adam steps.
  - outer_lr           : Adam learning rate.
  - policy_reopt_every : re-solve Bellman every N iters (at iter 1, 1+N, ...).
  - ckpt_every         : checkpoint period (0 disables).
  - ckpt_dir           : output directory.
  - cbox               : box constraints (default from default_cbox()).
  - omega_d_fn         : function c → ω_d (defaults to paper operating point).
  - verbose            : print per-iter progress if true.

The function treats c.beta, c.Z0, c.M, c.Mprime, c.C_qg, c.C_c as frozen at
their ScqubitParams defaults — only the 7 tier-1 fields are updated.
"""
function joint_opt(c0::ScqubitParams;
                   grid::Main.Belief.Grid,
                   K_epochs::Int = 3,
                   outer_iters::Int = 200,
                   outer_lr::Float64 = 1e-3,
                   policy_reopt_every::Int = 10,
                   ckpt_every::Int = 50,
                   ckpt_dir::String = joinpath(@__DIR__, "results", "joint"),
                   cbox::CBox = default_cbox(),
                   omega_d_fn = make_omega_d_fn(),
                   terminal::Symbol = :mi,
                   verbose::Bool = true)
    isdir(ckpt_dir) || mkpath(ckpt_dir)
    v = c_as_vec(c0)
    project_c!(v, cbox)
    # Use box width as per-component scale so Adam's step is ≈ lr * box_width.
    # Zero-width (pinned) components get scale 0 → step 0.
    scale = max.(cbox.hi .- cbox.lo, 0.0)
    state = AdamState(length(v); lr=outer_lr, scale=scale)

    hist = (V_adaptive = Float64[],
            grad_norm  = Float64[],
            c_vec      = Vector{Vector{Float64}}(),
            reopt_iter = Int[],
            omega_d    = Float64[],
            memo_size  = Int[],
            elapsed    = Float64[])

    # Initial policy solve
    c_cur = vec_as_c(v)
    ω_d   = omega_d_fn(c_cur)
    (V_cur, memo, st) = solve_bellman_full(grid, K_epochs, c_cur, ω_d; terminal=terminal)
    push!(hist.reopt_iter, 0)
    push!(hist.memo_size, st.memo_size)
    verbose && @printf("[init] V_adaptive = %.6f  memo = %d  ω_d = %.4e  %.2f s\n",
                       V_cur, st.memo_size, ω_d, st.elapsed)

    for iter in 1:outer_iters
        t0 = time()
        # Refresh policy?
        refreshed = false
        if (iter - 1) % policy_reopt_every == 0 && iter > 1
            c_cur = vec_as_c(v)
            ω_d   = omega_d_fn(c_cur)
            (V_cur, memo, st) = solve_bellman_full(grid, K_epochs, c_cur, ω_d; terminal=terminal)
            push!(hist.reopt_iter, iter)
            push!(hist.memo_size, st.memo_size)
            refreshed = true
        end

        # Envelope gradient at current c with fixed policy memo.
        g = grad_c_exact(v, memo, grid, ω_d, K_epochs; terminal=terminal)
        gn = norm(g)

        # Adam (maximise)
        adam_update!(v, g, state)
        project_c!(v, cbox)

        # Record
        push!(hist.V_adaptive, V_cur)
        push!(hist.grad_norm,  gn)
        push!(hist.c_vec,      copy(v))
        push!(hist.omega_d,    ω_d)
        push!(hist.elapsed,    time() - t0)

        if verbose && (iter == 1 || iter % max(1, policy_reopt_every ÷ 2) == 0)
            marker = refreshed ? "[reopt]" : "       "
            @printf("%s iter %4d  V=%.6f  |grad|=%.3e  Δt=%.2f s  ω_d=%.4e\n",
                    marker, iter, V_cur, gn, hist.elapsed[end], ω_d)
        end

        if ckpt_every > 0 && iter % ckpt_every == 0
            ckpt_path = joinpath(ckpt_dir, @sprintf("ckpt_%06d.jls", iter))
            open(ckpt_path, "w") do io
                serialize(io, (; v=copy(v), c=vec_as_c(v), iter=iter,
                                 state=deepcopy(state), history=hist,
                                 timestamp=now()))
            end
            verbose && println("  → checkpoint $ckpt_path")
        end
    end

    # Final policy solve at last v for reporting.
    c_final = vec_as_c(v)
    ω_d_final = omega_d_fn(c_final)
    (V_final, memo_final, st_final) = solve_bellman_full(grid, K_epochs, c_final, ω_d_final; terminal=terminal)
    push!(hist.V_adaptive, V_final)
    push!(hist.reopt_iter, outer_iters)
    push!(hist.memo_size,  st_final.memo_size)
    verbose && @printf("[final] V_adaptive = %.6f  memo = %d  ω_d = %.4e  %.2f s\n",
                       V_final, st_final.memo_size, ω_d_final, st_final.elapsed)

    (c_final, hist, memo_final)
end

end # module
