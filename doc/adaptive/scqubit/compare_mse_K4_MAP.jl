#=
compare_mse_K4_MAP.jl — Test MAP estimator (posterior mode) at K=4 baseline.

If posterior is multimodal, posterior-MEAN can have large bias (lands between
modes).  MAP picks the dominant mode, which may be closer to true φ.

We use the existing adaptive policy (no retraining), just change the estimator
in the MC deployment.
=#
using Printf, Random
include(joinpath(@__DIR__, "ScqubitModel.jl"))
include(joinpath(@__DIR__, "Belief.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))
include(joinpath(@__DIR__, "Bellman.jl"))
include(joinpath(@__DIR__, "Gradient.jl"))
include(joinpath(@__DIR__, "JointOpt.jl"))
include(joinpath(@__DIR__, "PCRB.jl"))
using .ScqubitModel, .Belief, .Bellman, .Gradient, .JointOpt, .PCRB

const K_EPOCHS = parse(Int, get(ENV, "K_EPOCHS", "4"))
const J_TAU = parse(Int, get(ENV, "J_TAU", "10"))
const N_MC = 20000
const K_PHI_POST = 256

TAU_GRID = ntuple(k -> 10e-9 * (32.0)^((k-1)/(J_TAU-1)), J_TAU)
N_GRID = (1, 10)

c = PAPER_BASELINE
phi_star_fn = make_phi_star_fn()
ωd = omega_q(phi_star_fn(c)[1], c)

grid = make_grid(; K_phi=K_PHI_POST, phi_max=0.49, tau_grid=TAU_GRID, n_grid=N_GRID)

@printf("Config: K=%d J=%d K_PHI=%d\n", K_EPOCHS, J_TAU, K_PHI_POST)
flush(stdout)

t0 = time()
(V, memo, st) = solve_bellman_full(grid, K_EPOCHS, c, ωd; terminal=:mse)
@printf("V_train=%.4e memo=%d %.1fs\n", V, st.memo_size, time()-t0)
flush(stdout)

# MAP estimator in adaptive deployment
function deployed_metric_adaptive_MAP(c, policy_memo, ωd, grid, K_epochs; n_mc, rng)
    errs_mean = Vector{Float64}(undef, n_mc)
    errs_map  = Vector{Float64}(undef, n_mc)
    errs_med  = Vector{Float64}(undef, n_mc)
    errs_mode = Vector{Float64}(undef, n_mc)
    J = length(grid.tau_grid)
    counts0 = ntuple(_ -> (0, 0), J)
    for t in 1:n_mc
        phi_true = rand(rng) * grid.phi_max
        counts = counts0
        logb = zeros(Float64, length(grid.phi_grid))
        for r in K_epochs:-1:1
            node = get(policy_memo, (counts, r), nothing)
            node === nothing && error("miss at counts=$counts r=$r")
            j, ℓ = node.action
            τ = grid.tau_grid[j]; n = grid.n_grid[ℓ]
            p_true = clamp(P1_ramsey(phi_true, τ, c, ωd), 1e-300, 1-1e-16)
            m = 0
            for _ in 1:n
                rand(rng) < p_true && (m += 1)
            end
            @inbounds for i in eachindex(logb)
                p = clamp(P1_ramsey(grid.phi_grid[i], τ, c, ωd), 1e-300, 1-1e-16)
                logb[i] += m * log(p) + (n - m) * log1p(-p)
            end
            counts = ntuple(k -> k == j ? (counts[k][1] + n, counts[k][2] + m) : counts[k], J)
        end
        mx = maximum(logb)
        w = exp.(logb .- mx); Z = sum(w)
        p = w ./ Z
        phi_mean = sum(p .* grid.phi_grid)
        phi_map  = grid.phi_grid[argmax(p)]
        # median: find grid point where CDF crosses 0.5
        cdf = cumsum(p)
        i_med = searchsortedfirst(cdf, 0.5)
        phi_med = grid.phi_grid[clamp(i_med, 1, length(grid.phi_grid))]
        errs_mean[t] = (phi_mean - phi_true)^2
        errs_map[t]  = (phi_map  - phi_true)^2
        errs_med[t]  = (phi_med  - phi_true)^2
    end
    (mean_mse(errs_mean), mean_mse(errs_map), mean_mse(errs_med))
end
mean_mse(x) = (m = sum(x)/length(x); se = sqrt(sum((x .- m).^2)/((length(x)-1)*length(x))); (m, se))

rng = MersenneTwister(2026)
@printf("Deploying adaptive policy with 3 estimators...\n"); flush(stdout)
(m_mean, m_map, m_med) = deployed_metric_adaptive_MAP(c, memo, ωd, grid, K_EPOCHS; n_mc=N_MC, rng=rng)
@printf("  MSE(mean) = %.4e ± %.2e\n", m_mean[1], m_mean[2])
@printf("  MSE(MAP)  = %.4e ± %.2e\n", m_map[1], m_map[2])
@printf("  MSE(med)  = %.4e ± %.2e\n", m_med[1], m_med[2])
flush(stdout)

# Same 3 estimators for PCRB fixed schedule
function deployed_metric_fixed_MAP(c, sched, ωd, grid; n_mc, rng)
    errs_mean = Vector{Float64}(undef, n_mc)
    errs_map  = Vector{Float64}(undef, n_mc)
    errs_med  = Vector{Float64}(undef, n_mc)
    for t in 1:n_mc
        phi_true = rand(rng) * grid.phi_max
        logb = zeros(Float64, length(grid.phi_grid))
        for (j, ℓ) in sched
            τ = grid.tau_grid[j]; n = grid.n_grid[ℓ]
            p_true = clamp(P1_ramsey(phi_true, τ, c, ωd), 1e-300, 1-1e-16)
            m = 0
            for _ in 1:n
                rand(rng) < p_true && (m += 1)
            end
            @inbounds for i in eachindex(logb)
                p = clamp(P1_ramsey(grid.phi_grid[i], τ, c, ωd), 1e-300, 1-1e-16)
                logb[i] += m * log(p) + (n - m) * log1p(-p)
            end
        end
        mx = maximum(logb); w = exp.(logb .- mx); Z = sum(w); p = w ./ Z
        phi_mean = sum(p .* grid.phi_grid)
        phi_map  = grid.phi_grid[argmax(p)]
        cdf = cumsum(p)
        i_med = searchsortedfirst(cdf, 0.5)
        phi_med = grid.phi_grid[clamp(i_med, 1, length(grid.phi_grid))]
        errs_mean[t] = (phi_mean - phi_true)^2
        errs_map[t]  = (phi_map  - phi_true)^2
        errs_med[t]  = (phi_med  - phi_true)^2
    end
    (mean_mse(errs_mean), mean_mse(errs_map), mean_mse(errs_med))
end

sched_pcrb = [(J_TAU, 2) for _ in 1:K_EPOCHS]
rng = MersenneTwister(2026)
@printf("\nDeploying PCRB schedule with 3 estimators...\n"); flush(stdout)
(m_mean, m_map, m_med) = deployed_metric_fixed_MAP(c, sched_pcrb, ωd, grid; n_mc=N_MC, rng=rng)
@printf("  MSE(mean) = %.4e ± %.2e\n", m_mean[1], m_mean[2])
@printf("  MSE(MAP)  = %.4e ± %.2e\n", m_map[1], m_map[2])
@printf("  MSE(med)  = %.4e ± %.2e\n", m_med[1], m_med[2])
flush(stdout)
