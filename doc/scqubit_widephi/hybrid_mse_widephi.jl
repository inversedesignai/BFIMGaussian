#=
hybrid_mse_widephi.jl — hybrid Bellman-then-CE policy at phi_max = 0.5

Pre-existing: K=4 BayesOpt winners at phi_max=0.5 (joint c_1* and PCRB c_2*).
Question: if we extend horizon past K=4 by appending certainty-equivalent
(Fisher-info-at-posterior-mean) epochs to the K=4 Bellman policy, does the
joint-DP / PCRB ratio improve?

Hybrid policy at K_total ≥ K_dp = 4:
  1. First K_dp = 4 epochs: Bellman policy from memo solved at c_1*
  2. Remaining (K_total - K_dp) epochs: pick action (τ_j, n_ℓ) that maximizes
     n_ℓ · J_F(φ̂_k, τ_j; c_1*) at the current posterior mean

Compared against:
  PCRB extended schedule: (τ_J, n=10)^K_total at c_2*

Reads (with PHI_TAG=phi$(round(Int, PHI_MAX*100))):
  results/bayesopt_joint_$(PHI_TAG)_K4/result.jls
  results/bayesopt_pcrb_$(PHI_TAG)_K4/result.jls

Writes:
  results/hybrid_mse_$(PHI_TAG)_K{K_total}.jls
=#
using Printf, Random, Serialization, Dates, ForwardDiff

include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "BellmanThreaded.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel, .Belief, .Bellman, .BellmanThreaded, .Gradient, .JointOpt, .PCRB

const PHI_MAX     = parse(Float64, get(ENV, "PHI_MAX", "0.5"))
const PHI_TAG     = "phi$(round(Int, PHI_MAX*100))"
const CE_TAG      = get(ENV, "CE_TAG", "MI")  # tag for output naming (e.g. "FI", "MI")
const K_DP        = 4
const K_TOTAL_LIST = parse.(Int, split(get(ENV, "K_TOTALS", "4,5,6,8,10"), ","))
const K_PHI       = parse(Int, get(ENV, "K_PHI", "256"))
const J_TAU       = 10
const TAU_GRID    = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
const N_GRID      = (1, 10)
const N_MC        = parse(Int, get(ENV, "MSE_N", "20000"))
const TWO_PI      = 2π

println("hybrid_mse_widephi.jl  (PHI_MAX=$(PHI_MAX))"); flush(stdout)
println("Threads: $(Threads.nthreads())"); flush(stdout)
println("K_dp=$(K_DP), K_total list=$(K_TOTAL_LIST), K_phi=$(K_PHI), n_mc=$(N_MC)"); flush(stdout)

j4 = deserialize(joinpath(@__DIR__, "results", "bayesopt_joint_$(PHI_TAG)_K4", "result.jls"))
p4 = deserialize(joinpath(@__DIR__, "results", "bayesopt_pcrb_$(PHI_TAG)_K4",  "result.jls"))

v1 = j4.v_best;  c1 = vec_as_c(v1);  ωd1 = j4.omega_d_best
v2 = p4.v_best;  c2 = vec_as_c(v2);  ωd2 = p4.omega_d_best

@printf("c_joint*: f_q=%.4f  E_C=%.4f  κ=%.4f MHz  Δ=%.4f   ω_d/(2π)=%.4f GHz\n",
        c1.f_q_max/1e9, c1.E_C_over_h/1e9, c1.kappa/1e6, c1.Delta_qr/1e9, ωd1/TWO_PI/1e9)
@printf("c_pcrb*:  f_q=%.4f  E_C=%.4f  κ=%.4f MHz  Δ=%.4f   ω_d/(2π)=%.4f GHz\n",
        c2.f_q_max/1e9, c2.E_C_over_h/1e9, c2.kappa/1e6, c2.Delta_qr/1e9, ωd2/TWO_PI/1e9)
flush(stdout)

grid = make_grid(; K_phi=K_PHI, phi_max=PHI_MAX, tau_grid=TAU_GRID, n_grid=N_GRID)

# Solve Bellman once at K_dp for the joint geometry; reuse memo across all K_totals
println("\n[bellman] solving K_dp=$K_DP at K_phi=$K_PHI for c_joint*..."); flush(stdout)
t = time()
(V1_dp, memo1, _) = solve_bellman_threaded_full(grid, K_DP, c1, ωd1; terminal=:mse)
@printf("  V_adaptive(K=%d) = %.6e   memo=%d   %.1fs\n", K_DP, V1_dp, length(memo1), time()-t)
flush(stdout)

# Single-shot binary entropy at point estimate (Phi_MI rung-2 specialization).
# For Bernoulli observation y at known phi=phi_hat, H(p) is maximized at p=0.5;
# the oracle's argmax-expected-information-gain action under Phi_MI reduces to
# argmax binary entropy of the predicted outcome at phi_hat.  Distinct from
# Fisher-info-based CE (which weights by 1/[p(1-p)]) when the slope dp/dphi is
# small but p is far from 0.5.
function H_binary_at(phi::Float64, tau::Float64, c::ScqubitParams, ωd::Float64)
    p = clamp(P1_ramsey(phi, tau, c, ωd), 1e-12, 1 - 1e-12)
    -p * log(p) - (1 - p) * log1p(-p)
end

# Certainty-equivalent action under Phi_MI: argmax_{(j,ℓ)} n_ℓ * H_binary(p(phi_hat, τ_j))
function ce_action(phi_hat::Float64, c::ScqubitParams, ωd::Float64,
                   grid_local::Main.Belief.Grid{J, L}) where {J, L}
    best_j = 1; best_ℓ = 1; best_val = -Inf
    for j in 1:J, ℓ in 1:L
        τ = grid_local.tau_grid[j]
        n = grid_local.n_grid[ℓ]
        v = n * H_binary_at(phi_hat, τ, c, ωd)
        if v > best_val
            best_val = v; best_j = j; best_ℓ = ℓ
        end
    end
    (best_j, best_ℓ)
end

posterior_mean(logb, phi_grid) = begin
    mx = maximum(logb)
    w = exp.(logb .- mx); Z = sum(w)
    sum((w ./ Z) .* phi_grid)
end

# Hybrid MC: K_dp Bellman, then K_total - K_dp certainty-equivalent
function hybrid_mse(c::ScqubitParams, memo::Dict, ωd::Float64,
                    grid_local::Main.Belief.Grid{J, L},
                    K_total::Int, K_dp::Int;
                    n_mc::Int, rng::AbstractRNG) where {J, L}
    errs = Vector{Float64}(undef, n_mc)
    counts0 = ntuple(_ -> (0, 0), J)
    for t in 1:n_mc
        phi_true = rand(rng) * grid_local.phi_max
        counts = counts0
        logb = zeros(Float64, length(grid_local.phi_grid))
        # phase 1: Bellman
        for r in K_dp:-1:1
            node = get(memo, (counts, r), nothing)
            node === nothing && error("policy memo miss at counts=$counts r=$r")
            j, ℓ = node.action
            τ = grid_local.tau_grid[j]; n = grid_local.n_grid[ℓ]
            p_true = clamp(P1_ramsey(phi_true, τ, c, ωd), 1e-300, 1 - 1e-16)
            m = 0
            for _ in 1:n; rand(rng) < p_true && (m += 1); end
            @inbounds for i in eachindex(logb)
                p = clamp(P1_ramsey(grid_local.phi_grid[i], τ, c, ωd), 1e-300, 1 - 1e-16)
                logb[i] += m * log(p) + (n - m) * log1p(-p)
            end
            counts = ntuple(k -> k == j ? (counts[k][1] + n, counts[k][2] + m) : counts[k], J)
        end
        # phase 2: certainty-equivalent
        for k in K_dp+1:K_total
            phi_hat = posterior_mean(logb, grid_local.phi_grid)
            (j, ℓ) = ce_action(phi_hat, c, ωd, grid_local)
            τ = grid_local.tau_grid[j]; n = grid_local.n_grid[ℓ]
            p_true = clamp(P1_ramsey(phi_true, τ, c, ωd), 1e-300, 1 - 1e-16)
            m = 0
            for _ in 1:n; rand(rng) < p_true && (m += 1); end
            @inbounds for i in eachindex(logb)
                p = clamp(P1_ramsey(grid_local.phi_grid[i], τ, c, ωd), 1e-300, 1 - 1e-16)
                logb[i] += m * log(p) + (n - m) * log1p(-p)
            end
        end
        phi_hat = posterior_mean(logb, grid_local.phi_grid)
        errs[t] = (phi_hat - phi_true)^2
    end
    mse = sum(errs) / n_mc
    se = sqrt(sum((errs .- mse).^2) / ((n_mc - 1) * n_mc))
    (mse, se)
end

results = Vector{NamedTuple}(undef, length(K_TOTAL_LIST))
for (idx, K_total) in enumerate(K_TOTAL_LIST)
    @printf("\n%s\n=== K_total = %d  (K_dp=%d Bellman + %d CE) ===\n%s\n",
            "="^60, K_total, K_DP, K_total - K_DP, "="^60); flush(stdout)

    # Hybrid MC for joint-DP geometry
    rng = MersenneTwister(2026); t = time()
    (MSE_hyb, se_hyb) = hybrid_mse(c1, memo1, ωd1, grid, K_total, K_DP; n_mc=N_MC, rng=rng)
    @printf("  hybrid MSE_1 = %.4e ± %.2e   (%.1fs)\n", MSE_hyb, se_hyb, time()-t)
    flush(stdout)

    # PCRB extended schedule: (τ_J, n=10)^K_total at PCRB c
    pcrb_sched = [(J_TAU, 2) for _ in 1:K_total]
    rng = MersenneTwister(2026); t = time()
    (MSE_pcrb, se_pcrb) = deployed_mse_fixed(c2, pcrb_sched, ωd2, grid; n_mc=N_MC, rng=rng)
    @printf("  PCRB MSE_2  = %.4e ± %.2e   (%.1fs)\n", MSE_pcrb, se_pcrb, time()-t)
    flush(stdout)

    ratio = MSE_pcrb / MSE_hyb
    z = (MSE_pcrb - MSE_hyb) / sqrt(se_hyb^2 + se_pcrb^2)
    @printf("\n  --- K_total=%d summary ---\n", K_total)
    @printf("    hybrid MSE (joint-DP K_dp=4 + CE)        = %.4e ± %.2e\n", MSE_hyb, se_hyb)
    @printf("    PCRB MSE   ((τ_max, n=10)^%d)              = %.4e ± %.2e\n", K_total, MSE_pcrb, se_pcrb)
    @printf("    ratio MSE_pcrb / MSE_hyb                  = %.3f\n", ratio)
    @printf("    z-score                                   = %+.2f σ\n", z)
    flush(stdout)

    results[idx] = (; K_total, K_dp=K_DP, MSE_hyb, se_hyb, MSE_pcrb, se_pcrb, ratio, z)
end

println("\n", "="^72)
@printf("HYBRID-CE SUMMARY  (PHI_MAX=%.2f, K_dp=%d Bellman, then CE)\n", PHI_MAX, K_DP)
println("-"^72)
@printf("%-8s  %14s  %14s  %8s  %8s\n", "K_total", "MSE_hyb", "MSE_pcrb", "ratio", "z-score")
for r in results
    @printf("%-8d  %14.4e  %14.4e  %8.3f  %+8.2f σ\n",
            r.K_total, r.MSE_hyb, r.MSE_pcrb, r.ratio, r.z)
end
println("="^72); flush(stdout)

for r in results
    open(joinpath(@__DIR__, "results", "hybrid_mse_$(CE_TAG)_$(PHI_TAG)_K$(r.K_total).jls"), "w") do io
        serialize(io, (; r..., PHI_MAX, c_joint=c1, c_pcrb=c2,
                         omega_d_joint=ωd1, omega_d_pcrb=ωd2,
                         K_PHI, N_MC, timestamp=now()))
    end
end
println("Saved.")
