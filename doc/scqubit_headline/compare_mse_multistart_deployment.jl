#=
compare_mse_multistart_deployment.jl — paired Monte Carlo deployment of
the global best joint-DP and PCRB optima, where "best" is selected by
**deployed MSE at K_PHI=256**, not by the training-time metric at K_PHI=128.

Why the deployment-criterion variant matters:
The training V_adaptive at K_PHI=128 can be over-optimistic for some c
regions because the coarse grid under-resolves the posterior. Selecting
the global-best joint-DP `c` by training V_best can therefore pick a
geometry whose actual deployed MSE at K_PHI=256 is worse than another
candidate's. Deploying every candidate at K_PHI=256 first, then picking
by MSE, is the honest "global by deployment" answer.

Reads: results/joint_multistart/<id>.jls, results/pcrb_multistart/<id>.jls
Writes: results/compare_mse_multistart_deployment.jls
Wall-clock: ~5-7 min on 64 threads (4 Bellman re-solves at K_PHI=256
plus 8 deployment-MC sweeps).
=#
using Printf, Random, Serialization, Dates

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "BellmanThreaded.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel, .Belief, .Bellman, .BellmanThreaded, .Gradient, .JointOpt, .PCRB

const N_MC       = parse(Int, get(ENV, "MSE_N", "20000"))
const K_PHI_POST = parse(Int, get(ENV, "MSE_K_PHI", "256"))
const J_TAU      = 10
const TAU_GRID   = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID     = (1, 10)
const PHI_MAX    = 0.1
const K_EPOCHS   = 4

const INIT_LIST = let raw = get(ENV, "INITS", "paper,naive,rand_1,rand_2")
    isempty(raw) ? String[] : split(raw, ",")
end

println("compare_mse_multistart_deployment.jl"); flush(stdout)
println("Threads: $(Threads.nthreads())"); flush(stdout)
println("Inits to consider: ", INIT_LIST); flush(stdout)

grid = make_grid(; K_phi=K_PHI_POST, phi_max=PHI_MAX,
                   tau_grid=TAU_GRID, n_grid=N_GRID)
phi_star_fn = make_phi_star_fn()

# ---------- Deploy every joint-DP candidate ----------
println("\n" * "="^72)
println("Deploying joint-DP candidates at K_PHI=$K_PHI_POST")
println("="^72)
joint_results = NamedTuple[]
for id in INIT_LIST
    path = joinpath(@__DIR__, "results", "joint_multistart", "$(id).jls")
    isfile(path) || (@warn "missing $path  – skipping"; continue)
    j = deserialize(path)
    v = j.v_best
    c = ScqubitParams(f_q_max=v[1], E_C_over_h=v[2], kappa=v[3], Delta_qr=v[4],
                      temperature=v[5], A_phi=v[6], A_Ic=v[7])
    ωd = omega_q(phi_star_fn(c)[1], c)
    @printf("\n[joint/%s] V_train(K_PHI=128) = %.4e   c=(%.4f, %.4f, %.4f MHz, %.4f)\n",
            id, j.V_best, v[1]/1e9, v[2]/1e9, v[3]/1e6, v[4]/1e9); flush(stdout)
    t = time()
    (V256, memo, st) = solve_bellman_threaded_full(grid, K_EPOCHS, c, ωd; terminal=:mse)
    @printf("  Re-solve V(K_PHI=256) = %.4e  memo=%d  %.1fs\n",
            V256, length(memo), time()-t); flush(stdout)
    rng = MersenneTwister(2026); t = time()
    (mse, se) = deployed_mse_adaptive(c, memo, ωd, grid, K_EPOCHS; n_mc=N_MC, rng=rng)
    @printf("  Deployed MSE = %.4e ± %.2e   (%.1fs)\n", mse, se, time()-t); flush(stdout)
    push!(joint_results, (init_id=id, V_train=j.V_best, V_deploy=V256,
                          MSE=mse, se=se, c=c, v=v, ωd=ωd))
end

# ---------- Deploy every PCRB candidate ----------
println("\n" * "="^72)
println("Deploying PCRB candidates at K_PHI=$K_PHI_POST")
println("="^72)
pcrb_results = NamedTuple[]
for id in INIT_LIST
    path = joinpath(@__DIR__, "results", "pcrb_multistart", "$(id).jls")
    isfile(path) || (@warn "missing $path  – skipping"; continue)
    p = deserialize(path)
    v = p.v_best
    c = ScqubitParams(f_q_max=v[1], E_C_over_h=v[2], kappa=v[3], Delta_qr=v[4],
                      temperature=v[5], A_phi=v[6], A_Ic=v[7])
    sched = p.sched_best
    ωd = omega_q(phi_star_fn(c)[1], c)
    @printf("\n[pcrb/%s] log_JP(K_PHI=128) = %.4f   c=(%.4f, %.4f, %.4f MHz, %.4f)\n",
            id, p.logJP_best, v[1]/1e9, v[2]/1e9, v[3]/1e6, v[4]/1e9); flush(stdout)
    rng = MersenneTwister(2026); t = time()
    (mse, se) = deployed_mse_fixed(c, sched, ωd, grid; n_mc=N_MC, rng=rng)
    @printf("  Deployed MSE = %.4e ± %.2e   (%.1fs)\n", mse, se, time()-t); flush(stdout)
    push!(pcrb_results, (init_id=id, log_JP=p.logJP_best,
                         MSE=mse, se=se, c=c, v=v, sched=sched, ωd=ωd))
end

# ---------- Pick deployment-best of each ----------
i_j = argmin(i -> joint_results[i].MSE, 1:length(joint_results))
i_p = argmin(i -> pcrb_results[i].MSE,  1:length(pcrb_results))
J = joint_results[i_j]; P = pcrb_results[i_p]

println("\n" * "="^72)
println("PER-INIT DEPLOYMENT MSE SUMMARY")
println("="^72)
println("Joint-DP:")
for r in joint_results
    star = (r.init_id == J.init_id) ? " <- best" : ""
    @printf("  %-10s V_train=%+.4e  V_deploy=%+.4e  MSE=%.4e ± %.2e%s\n",
            r.init_id, r.V_train, r.V_deploy, r.MSE, r.se, star)
end
println("PCRB:")
for r in pcrb_results
    star = (r.init_id == P.init_id) ? " <- best" : ""
    @printf("  %-10s log_JP =%+.4f      MSE=%.4e ± %.2e%s\n",
            r.init_id, r.log_JP, r.MSE, r.se, star)
end

ratio = P.MSE / J.MSE
z = (P.MSE - J.MSE) / sqrt(J.se^2 + P.se^2)
JP2 = exp(log_JP_of_schedule(P.sched, grid, P.c, P.ωd; J_0=1e-4))
crb = 1 / JP2

println("\n" * "="^72)
@printf("HEADLINE — multistart, select by DEPLOYMENT MSE (K=4, J=10, L=2, phi_max=%.3f)\n", PHI_MAX)
@printf("  joint-DP best init: %s   PCRB best init: %s\n", J.init_id, P.init_id)
println("-"^72)
@printf("  c_joint*: f_q=%.4f  E_C=%.4f  κ=%.4f MHz  Δ=%.4f\n",
        J.c.f_q_max/1e9, J.c.E_C_over_h/1e9, J.c.kappa/1e6, J.c.Delta_qr/1e9)
@printf("  c_pcrb*:  f_q=%.4f  E_C=%.4f  κ=%.4f MHz  Δ=%.4f  sched=%s\n",
        P.c.f_q_max/1e9, P.c.E_C_over_h/1e9, P.c.kappa/1e6, P.c.Delta_qr/1e9,
        string(P.sched))
@printf("  MSE̅₁ (joint-DP) = %.4e ± %.2e\n", J.MSE, J.se)
@printf("  MSE̅₂ (PCRB)     = %.4e ± %.2e\n", P.MSE, P.se)
@printf("  CRB lower bound  = %.4e\n", crb)
@printf("  ratio MSE̅₂/MSE̅₁ = %.3f\n", ratio)
@printf("  z-score          = %+.2f σ\n", z)
println("="^72); flush(stdout)

open(joinpath(@__DIR__, "results", "compare_mse_multistart_deployment.jls"), "w") do io
    serialize(io, (; MSE_1=J.MSE, se_1=J.se, MSE_2=P.MSE, se_2=P.se,
                     pcrb_bound=crb, ratio, z,
                     c_1_star=J.c, c_2_star=P.c, sched_2_star=P.sched,
                     joint_winner=J.init_id, pcrb_winner=P.init_id,
                     joint_results, pcrb_results,
                     N_MC, K_PHI_POST, PHI_MAX,
                     omega_d_1=J.ωd, omega_d_2=P.ωd,
                     timestamp=now()))
end
println("Saved.")
