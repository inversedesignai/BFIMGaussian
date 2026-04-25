#=
sweep_joint_narrow_v3.jl — converged rerun using threaded Bellman + threaded
ForwardDiff gradient.

Same problem as sweep_joint_narrow_v2.jl (K=4, J=10, L=2, phi_max=0.1,
MSE-terminal Bellman, realistic_box, init at PAPER_BASELINE) but with the
inner `joint_opt`-style loop hand-rolled here so we can call:

  - BellmanThreaded.solve_bellman_threaded_full         (replaces solve_bellman_full)
  - GradientThreaded.grad_c_exact_fd_threaded           (replaces grad_c_exact)

Per-state arithmetic in BellmanThreaded matches single-threaded Bellman bit-
for-bit (validated, max |Δ| = 0). The threaded ForwardDiff gradient matches
single-threaded ForwardDiff bit-exactly on small cases. Both use only the
read-only policy memo, so threading is safe.

Defaults (same as v2):
  JOINT_LR    = 1e-4   (5x smaller than v1 default)
  JOINT_ITERS = 100    (4x longer)
  JOINT_REOPT = 5

Output: results/joint_narrow_v3/{ckpt_NNNNNN.jls, final.jls}
=#
using Printf
using Serialization
using Dates
using LinearAlgebra

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "BellmanThreaded.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "GradientThreaded.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))   # for AdamState/adam_update!/CBox/project_c!
using .ScqubitModel, .Belief, .Bellman, .BellmanThreaded, .Gradient, .GradientThreaded, .JointOpt

println("sweep_joint_narrow_v3.jl — threaded Bellman + threaded FD gradient")
println("Threads: $(Threads.nthreads())"); flush(stdout)

const K_EPOCHS    = 4
const K_PHI       = 128
const J_TAU       = 10
const PHI_MAX     = 0.1
const TAU_GRID    = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID      = (1, 10)
const OUTER_ITERS = parse(Int, get(ENV, "JOINT_ITERS", "100"))
const OUTER_LR    = parse(Float64, get(ENV, "JOINT_LR", "5e-4"))
const REOPT_EVERY = parse(Int, get(ENV, "JOINT_REOPT", "5"))
const PARALLEL_DEPTH = parse(Int, get(ENV, "JOINT_PD", string(K_EPOCHS - 1)))
# Step-decay schedule: halve lr at each step in the schedule.
# Default: (15, 35, 60) -> initial 5e-4 -> 2.5e-4 (iter 15) -> 1.25e-4 (iter 35) -> 6.25e-5 (iter 60).
const LR_DECAY_STEPS = let raw = get(ENV, "JOINT_LR_DECAY", "15,35,60")
    isempty(raw) ? Int[] : parse.(Int, split(raw, ","))
end
const LR_DECAY_FACTOR = parse(Float64, get(ENV, "JOINT_LR_DECAY_FACTOR", "0.5"))

function realistic_box(bl::ScqubitParams)
    lo = [ 3.0e9,   0.15e9,  0.1e6,   0.8e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    hi = [12.0e9,   0.4e9,   5.0e6,   5.0e9,  bl.temperature,  bl.A_phi,  bl.A_Ic]
    CBox(lo, hi)
end

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)
box  = realistic_box(PAPER_BASELINE)

println(@sprintf("Config: K=%d K_Φ=%d J=%d L=%d iters=%d lr=%.1e reopt=%d pd=%d phi_max=%.3f",
                 K_EPOCHS, K_PHI, length(TAU_GRID), length(N_GRID),
                 OUTER_ITERS, OUTER_LR, REOPT_EVERY, PARALLEL_DEPTH, PHI_MAX))
println("LR decay steps = ", LR_DECAY_STEPS, "  factor = ", LR_DECAY_FACTOR)
println("τ grid (ns): ", round.(collect(TAU_GRID) .* 1e9; digits=1))
flush(stdout)

# ---------------- Probe: threaded Bellman to confirm setup ----------------
t_probe = time()
(V0, memo0, st0) = solve_bellman_threaded_full(grid, K_EPOCHS, PAPER_BASELINE,
                                                omega_q(0.442, PAPER_BASELINE);
                                                terminal=:mse)
@printf("Probe Bellman_threaded at c₀: V=%.4e, memo=%d, %.2fs (%d threads)\n",
        V0, st0.memo_size, st0.elapsed, st0.n_threads)
flush(stdout)
memo0 = nothing; GC.gc()

# ---------------- Hand-rolled outer Adam loop ----------------
phi_star_fn = make_phi_star_fn()
omega_d_fn = make_omega_d_fn(; phi_star_fn=phi_star_fn)

ckpt_dir = joinpath(@__DIR__, "results", "joint_narrow_v3")
isdir(ckpt_dir) || mkpath(ckpt_dir)

function run_optimization(grid, box, omega_d_fn, ckpt_dir;
                          K_EPOCHS, OUTER_ITERS, OUTER_LR, REOPT_EVERY,
                          PARALLEL_DEPTH, LR_DECAY_STEPS, LR_DECAY_FACTOR)
    v = c_as_vec(PAPER_BASELINE)
    project_c!(v, box)
    scale = max.(box.hi .- box.lo, 0.0)
    state = AdamState(length(v); lr=OUTER_LR, scale=scale)

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
    (V_cur, memo, st) = solve_bellman_threaded_full(grid, K_EPOCHS, c_cur, ω_d; terminal=:mse)
    push!(hist.reopt_iter, 0)
    push!(hist.memo_size, st.memo_size)
    @printf("[init] V_adaptive = %.6f  memo = %d  ω_d = %.4e  %.2f s (%d threads)\n",
            V_cur, st.memo_size, ω_d, st.elapsed, st.n_threads)
    flush(stdout)

    t_run0 = time()
    for iter in 1:OUTER_ITERS
        t_iter = time()

        # Refresh policy?
        refreshed = false
        if (iter - 1) % REOPT_EVERY == 0 && iter > 1
            c_cur = vec_as_c(v)
            ω_d   = omega_d_fn(c_cur)
            (V_cur, memo, st) = solve_bellman_threaded_full(grid, K_EPOCHS, c_cur, ω_d; terminal=:mse)
            push!(hist.reopt_iter, iter)
            push!(hist.memo_size, st.memo_size)
            refreshed = true
        end

        # Threaded ForwardDiff gradient at current c with fixed policy memo.
        g = grad_c_exact_fd_threaded(v, memo, grid, ω_d, K_EPOCHS;
                                     terminal=:mse, parallel_depth=PARALLEL_DEPTH)
        gn = norm(g)

        # Adam (maximise)
        adam_update!(v, g, state)
        project_c!(v, box)

        # Step decay on lr
        if iter in LR_DECAY_STEPS
            state.lr *= LR_DECAY_FACTOR
            @printf("       [lr decay] iter %d  state.lr -> %.3e\n", iter, state.lr)
            flush(stdout)
        end

        push!(hist.V_adaptive, V_cur)
        push!(hist.grad_norm,  gn)
        push!(hist.c_vec,      copy(v))
        push!(hist.omega_d,    ω_d)
        push!(hist.elapsed,    time() - t_iter)

        if iter == 1 || iter % max(1, REOPT_EVERY ÷ 2) == 0
            marker = refreshed ? "[reopt]" : "       "
            @printf("%s iter %4d  V=%.6f  |grad|=%.3e  Δt=%.2f s  ω_d=%.4e\n",
                    marker, iter, V_cur, gn, hist.elapsed[end], ω_d)
            flush(stdout)
        end

        if iter % 10 == 0
            ckpt_path = joinpath(ckpt_dir, @sprintf("ckpt_%06d.jls", iter))
            open(ckpt_path, "w") do io
                serialize(io, (; v=copy(v), c=vec_as_c(v), iter=iter,
                                 history=hist, timestamp=now()))
            end
            println("  → checkpoint $ckpt_path")
            flush(stdout)
        end
    end
    @printf("\nTotal elapsed: %.1f min\n", (time() - t_run0) / 60)
    flush(stdout)

    # Final policy solve
    c_final = vec_as_c(v)
    ω_d_final = omega_d_fn(c_final)
    (V_final, memo_final, st_final) = solve_bellman_threaded_full(grid, K_EPOCHS, c_final, ω_d_final; terminal=:mse)
    push!(hist.V_adaptive, V_final)
    push!(hist.reopt_iter, OUTER_ITERS)
    push!(hist.memo_size,  st_final.memo_size)
    @printf("[final] V_adaptive = %.6f  memo = %d  ω_d = %.4e  %.2f s\n",
            V_final, st_final.memo_size, ω_d_final, st_final.elapsed)
    flush(stdout)

    (v, c_final, hist)
end

(v, c_final, hist) = run_optimization(grid, box, omega_d_fn, ckpt_dir;
                                       K_EPOCHS=K_EPOCHS, OUTER_ITERS=OUTER_ITERS,
                                       OUTER_LR=OUTER_LR, REOPT_EVERY=REOPT_EVERY,
                                       PARALLEL_DEPTH=PARALLEL_DEPTH,
                                       LR_DECAY_STEPS=LR_DECAY_STEPS,
                                       LR_DECAY_FACTOR=LR_DECAY_FACTOR)

out_path = joinpath(ckpt_dir, "final.jls")
open(out_path, "w") do io
    serialize(io, (; c_final, v_final=c_as_vec(c_final),
                     history=hist, K_EPOCHS, K_PHI, PHI_MAX,
                     TAU_GRID, N_GRID, timestamp=now(),
                     baseline=PAPER_BASELINE,
                     fixed_components=(:temperature, :A_phi, :A_Ic),
                     terminal=:mse,
                     parallel_depth=PARALLEL_DEPTH))
end
println("Saved to $out_path")

nj = length(hist.c_vec)
va = hist.V_adaptive[1:nj]
ibest = argmax(va)
@printf("V at init           = %.4e\n", va[1])
@printf("V at best (iter %d) = %.4e (improvement %.2f%% vs init)\n",
        ibest, va[ibest], 100*(va[ibest]-va[1])/abs(va[1]))
@printf("V at last iter      = %.4e\n", va[end])
println("Last 5 V_adaptive: ", va[max(1, end-4):end])
println("Last 5 |grad|:     ", hist.grad_norm[max(1, end-4):end])
