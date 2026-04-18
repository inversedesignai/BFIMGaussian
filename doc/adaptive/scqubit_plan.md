# scqubit Joint-(c, π) Bellman DP: Implementation Plan

**Audience:** a Claude session on a workstation with 388 AMD EPYC cores (2 × EPYC sockets) and 10 × RTX Pro 6000 Blackwell GPUs, working inside the `BFIMGaussian` repo.

**Goal:** implement the exact joint geometry–policy optimization for the Danilin–Nugent–Weides superconducting-qubit flux sensor, replicating the methodology of the radar case study (`doc/adaptive/compute_joint.jl`) while scaling up: continuous `x = Φ`, discrete 2-D action `s = (τ, n)`, continuous high-dim `c`, exact Bellman DP, envelope-theorem `c`-gradient, full Adam outer loop.

**Deliverables (analogs of the radar study outputs):**
1. Exact `V_oracle(Φ)`, `V_fixed(c)`, `V_adaptive(c)`, and `E[IG](c)` for a baseline `c₀`.
2. Joint `(c₁*, π₁*)` via envelope-theorem gradient descent in `c`, alternated with exact Bellman policy re-solution.
3. Within-family sweep over the 1-D slice `c = f_q^max` for direct comparability with the source paper (Fig. 2 of Danilin et al.).
4. Full-dimensional Adam run over 7-to-15-D `c` with convergence history and final design characterization.
5. **PCRB-based joint baseline `(c₂*, s_{1:N}^{(2)*}) = argmax_{c, s} log J_P(s, c)`** (single-level, no adaptive policy). Deployed prior-averaged MSE `MSE̅₂` from Monte Carlo with posterior-mean estimator.
6. **MSE-anchored headline comparison:** `MSE̅₂ / MSE̅₁` where `MSE̅₁` is the deployed MSE under the joint-DP result `(c₁*, π₁*)`. This is the §8.7 headline in `AdaptiveDesign.tex` operationalized for this case study.
7. A LaTeX writeup (`doc/adaptive/scqubit_results.tex`) structured like `doc/adaptive/IG_numerics.tex`, including the MSE comparison as a headline table.

---

## 0. Prerequisite reading (do this first)

Before writing any code, read these files in order. They fully define the problem:

1. `doc/adaptive/scqubit.pdf` — the source paper (Danilin, Nugent & Weides, arXiv:2211.08344v4, 2024). Authoritative reference for all rate formulas, geometry values, noise amplitudes, and the Kitaev-PEA protocol. Cross-check any formula against this PDF before implementing.
2. `doc/adaptive/AdaptiveDesign.tex` — the theoretical framework. §5 (ignorance gap), §7 (joint optimization + envelope theorem + policy iteration), §9 (radar case study — the template to follow), §10 (relaxation hierarchy — scqubit is the exact-DP counterpart to the photonic BFIM which is path 2c).
3. `doc/adaptive/scqubit_model.tex` — distilled measurement model for this problem: the Ramsey likelihood `P_|1⟩(Φ, τ; c)`, all rate formulas, full delineation of `c` and `s` parameters. Derived from scqubit.pdf; use it as a fast-access summary but defer to the PDF for any ambiguity.
4. `doc/adaptive/IG_numerics.tex` — the numerical writeup of the radar case study. Write `scqubit_results.tex` in this exact structure.
5. `doc/adaptive/compute_ig.jl` — radar V_oracle + V_fixed enumeration. Study the symmetry reductions.
6. `doc/adaptive/compute_dp.jl` — radar exact Bellman DP via backward induction on reachable beliefs. Study the memoization pattern.
7. `doc/adaptive/compute_joint.jl` — radar joint sweep + policy iteration.
8. `BFIMGaussian.jl` and `PhEnd2End.jl` — the photonic path (2c) codebase. Reuse the custom-rrule patterns for `ekf_update`-style manual adjoints if needed.
9. `CLAUDE.md` (both root and `~/.claude/CLAUDE.md`) — style and workflow rules. Julia, iteration-driven, correctness-first, concise. **Do not** use mutating operations inside Zygote-traced code. **Do** verify every gradient with finite differences before claiming success.

---

## 1. Problem statement (precise form)

**Unknown:** x = Φ_ext ∈ [0, Φ₀/2], continuous (restrict to first half-period of the Cooper-pair-box spectrum).

**Prior:** uniform on [0, Φ₀/2].

**Action:** s = (τ, n) where
- τ ∈ T_grid ⊂ [τ_min, τ_max], discrete of size J (default 6, Kitaev-ladder τ_j = τ_min · 2^(j-1)).
- n ∈ N_grid ⊂ ℤ₊, discrete of size L (default 4, values {1, 3, 10, 30}).

**Observation:** m ~ Binomial(n, P_|1⟩(Φ, τ; c)), the count of |1⟩-readouts in n repeated Ramsey shots at delay τ with fixed operating flux Φ* and fixed drive ω_d.

**Forward model** (from scqubit_model.tex, Eq. (1)–(14)):
```
P_|1⟩(Φ, τ; c) = 1/2 + (1/2) · thermal(c, Φ) · exp(-A(c,Φ)·τ - B(c,Φ)²·τ²) · cos(Δω(Φ; c)·τ)

f_q(Φ; c)   = (f_q_max + E_C/h)·sqrt(|cos(π Φ/Φ₀)|) - E_C/h
Δω(Φ; c)   = 2π·f_q(Φ; c) - ω_d
thermal    = [exp(h f_q/k_B T) - 1] / [exp(h f_q/k_B T) + 1]
A(c, Φ)    = (Γ₁_cav + Γ₁_ind + Γ₁_cap)/2 + π²·A_Φ²·|∂²ω_q/∂Φ²|
B(c, Φ)    = sqrt( A_Φ²·|∂ω_q/∂Φ|² + A_Ic²·|∂ω_q/∂I_c|² )
```
All rates Γ₁_{cav,ind,cap}, ∂ω_q/∂Φ, ∂²ω_q/∂Φ², ∂ω_q/∂I_c are closed-form in c and Φ per Eqs. 2–10 of the source.

**Horizon:** K decision epochs (default K = 4). Each epoch picks one (τ, n) and observes one count m.

**Terminal reward** (mutual-information form, analogous to `Φ_value` in the radar code):
```
W_K(b_K) = -H(b_K)   plus additive constant ln(Φ₀/2) for comparability with the radar code's ln K.
```
So the objective `Φ` (MI) at a trajectory is `H(b_0) - H(b_K)` averaged over observation trajectories.

**c (design variable):** tier-1 default is 7-D:
```
c = (f_q_max, E_C, κ, Δ_qr, T, A_Φ, A_Ic)
```
Extend to tier-2 (add 5 geometric dims via Biot–Savart-differentiable M, M') if time permits. Tier 3 (capacitances via FEM surrogate) is out of scope for the first implementation.

---

## 2. Directory layout

Create `doc/adaptive/scqubit/` (sibling of the radar scripts):

```
doc/adaptive/scqubit/
├── ScqubitModel.jl          # forward model, rates, likelihood (Phase 1)
├── Belief.jl                # belief grid + count-statistic dual representation (Phase 2)
├── Baselines.jl             # V_oracle, V_fixed enumeration (Phase 3)
├── Bellman.jl               # exact DP via memoized backward induction (Phase 4)
├── Gradient.jl              # envelope-theorem ∂W₁/∂c via pathwise MC (Phase 5)
├── JointOpt.jl              # outer Adam + policy iteration (Phase 6)
├── PCRB.jl                  # Fisher info, J_P, PCRB-baseline joint opt, deployed MSE (Phase 7)
├── tests/
│   ├── test_model.jl        # closed-form rate formulas vs hand-computed values
│   ├── test_belief.jl       # belief update consistency (grid vs count rep)
│   ├── test_baselines.jl    # V_oracle ≥ V_fixed, symmetry checks
│   ├── test_bellman.jl      # monotonicity in K, agreement with brute-force at K=2
│   ├── test_gradient.jl     # gradient check vs finite differences (per c-component)
│   ├── test_joint.jl        # end-to-end sanity: Adam reduces loss monotonically
│   └── test_pcrb.jl         # Fisher-info positivity, J_N additivity, PCRB bound empirical check
├── sweep_fq.jl              # 1-D sweep over f_q_max (paper Fig. 2 analog) — both Phi and PCRB
├── sweep_joint.jl           # full 7-D Adam (joint-DP)
├── sweep_pcrb.jl            # full PCRB-baseline joint optimization
├── compare_mse.jl           # deploy both designs, compute MSE̅₁ and MSE̅₂, headline ratio
├── plot_results.jl          # plotting
└── results/                 # output .jls, .png, tables
```

Every Julia file declares a module matching its basename (e.g., `module ScqubitModel`). Top-level scripts `sweep_*.jl` and `plot_*.jl` `using` those modules.

---

## 3. Numerical defaults (use these unless a specific sweep asks otherwise)

| symbol | default | note |
|---|---|---|
| `K_epochs` | 4 | decision horizon (#actions chosen) |
| `J` | 6 | size of τ-grid |
| `L` | 4 | size of n-grid |
| `τ_min` | 10 ns | set by hardware (FPGA switching) |
| `τ_max` | 5 μs | bounded by typical T₂ at mid-range f_q_max |
| `τ_grid` | `τ_min .* 2 .^ (0:J-1)` | Kitaev-style doubling ladder |
| `n_grid` | `[1, 3, 10, 30]` | log-spaced |
| `K_Φ` | 256 | flux-grid points on [0, Φ₀/2] |
| `n_traj` | 10_000 | Monte Carlo trajectories for gradient estimation |
| `τ_MC` per trajectory | sampled under π* | |
| `outer_lr` | 1e-3 | Adam learning rate |
| `outer_iters` | 500–2000 | depending on convergence |
| `policy_reopt_every` | 10 | re-solve Bellman every N outer steps (policy iteration) |
| `c_tier` | 1 | 7-D by default; tier-2 ≈ 12-D |

**Baseline c₀ (values from Danilin et al.):** `f_q_max = 9 GHz, E_C/h = 0.254 GHz, κ = 0.5 MHz, Δ = 2π·2 GHz, T = 40 mK, A_Φ = 1e-6·Φ₀, A_Ic = 1e-6·I_c`.

Paper finds optimal flux point `Φ* = 0.442·Φ₀` at baseline. Implementation should *reproduce* this value analytically (Eq. 16 of source) as a sanity check during Phase 1 testing.

---

## 4. Phase-by-phase implementation

### Phase 1 — `ScqubitModel.jl`

**Goal:** closed-form forward model with fully-differentiable `c`-dependence.

**Public API:**
```julia
struct ScqubitParams
    f_q_max::Float64  # Hz
    E_C::Float64      # in joules; relate to E_C_over_h = E_C/h (GHz)
    κ::Float64        # Hz (resonator decay)
    Δ_qr::Float64     # rad/s (qubit-resonator detuning, angular)
    T::Float64        # K
    A_Φ::Float64      # flux noise amplitude, units Φ₀
    A_Ic::Float64     # critical-current noise amplitude, units I_c
    # tier-2 (optional extensions):
    # M, M_prime, C_qg, C_c, ... (default to paper values if not varied)
end

"Return angular qubit freq ω_q(Φ; c) in rad/s."
omega_q(Φ::Real, c::ScqubitParams) -> Real

"Return ∂ω_q/∂Φ (and ∂²ω_q/∂Φ², ∂ω_q/∂I_c) at (Φ, c). Use ForwardDiff internally if clean."
domega_q_dΦ(Φ, c), d2omega_q_dΦ2(Φ, c), domega_q_dIc(Φ, c)

"Relaxation / dephasing rates; sums needed for A, B."
Γ1_cav(Φ, c), Γ1_ind(Φ, c), Γ1_cap(Φ, c)
Γφ_flux_linear_factor(Φ, c), Γφ_current(Φ, c)

"Ramsey-envelope coefficients."
A_coef(Φ, c), B_coef(Φ, c)

"Drive frequency chosen by protocol. Default: ω_d = ω_q(Φ*; c) − small detuning for Ramsey off-resonance if desired. Parametrize explicitly:"
omega_d(Φ_star, c) -> Real   # default ω_d = ω_q(Φ_star, c)

"Full Ramsey likelihood P_|1⟩(Φ, τ; c, Φ_star, ω_d). Differentiable in (Φ, c)."
P1_ramsey(Φ, τ, c, Φ_star, ω_d; include_thermal=true) -> Real in (0, 1)

"Binomial log-likelihood at observation m given (Φ, τ, n, c)."
log_binom_like(Φ, τ, n, m, c, Φ_star, ω_d) -> Real
```

**Implementation notes:**
- Use `ForwardDiff` or `Zygote` for `∂ω_q/∂Φ`, `∂²ω_q/∂Φ²`, `∂ω_q/∂I_c`. Validate once against hand-derived formulas (Eqs. 9, and the ∂ω_q/∂Φ, ∂²ω_q/∂Φ² implied by Eq. 8 of source).
- For `|cos(π Φ/Φ₀)|`: use a smooth approximation `sqrt(cos²+ε)` near `Φ = Φ₀/2` to avoid gradient singularities during optimization. Document this in the rate-formula code.
- The `Γ₁_ind` formula (Eq. 4) has `L_J(c, Φ)` which blows up as `Φ → Φ₀/2`. Clip `Φ` away from Φ₀/2 by `ε = 1e-4·Φ₀` during gradient computation.

**Tests (`tests/test_model.jl`):**
- `test_ω_q_at_sweet_spot`: ω_q(Φ=0) = 2π·f_q_max.
- `test_ω_q_at_half`: ω_q(Φ=Φ₀/2) = -2π·E_C/h (gradient singular here).
- `test_A_B_positive`: A ≥ 0, B ≥ 0 for all (Φ, c) in typical range.
- `test_P1_bounds`: 0 ≤ P_|1⟩ ≤ 1 for all inputs.
- `test_P1_at_zero_tau`: P_|1⟩(Φ, τ=0) = 1 (no decoherence, cos(0)=1, thermal=1 for ω_d small).
- `test_derivatives_FD`: `∂ω_q/∂Φ` and `∂²ω_q/∂Φ²` vs finite differences on Φ at 10 random points, rel error < 1e-6.
- `test_optimal_flux_point`: At baseline c₀, solve Eq. 12 (argmax over (Φ, τ) of |∂P_|1⟩/∂Φ|) and check result is Φ*/Φ₀ ≈ 0.442 ± 0.005 — this reproduces paper's Fig. 2.

### Phase 2 — `Belief.jl`

**Goal:** belief representation that is (i) exact, (ii) differentiable in c, (iii) efficiently memoizable for the Bellman backup.

**Dual representation:**
- **Grid rep:** `logb::Vector{Float64}` of length `K_Φ` over `Φ_grid = range(0, Φ₀/2, length=K_Φ)`. Non-mutating updates return new vectors for Zygote compatibility.
- **Count-statistic rep:** `counts::NTuple{J, Tuple{Int, Int}}` = `((n_τ₁, m_τ₁), ..., (n_τ_J, m_τ_J))`. Lossless: every reachable belief has a unique count tuple (order of steps irrelevant under the Bernoulli product).

**Public API:**
```julia
struct Belief
    logb::Vector{Float64}      # K_Φ-vector
    counts::NTuple{J, Tuple{Int, Int}}  # for memoization key
end

"Uniform prior over Φ-grid."
prior_belief(K_Φ::Int, J_::Int) -> Belief

"Bayes update after observing m out of n at delay τ_j (index j)."
update(b::Belief, j::Int, n::Int, m::Int, c, Φ_star, ω_d, Φ_grid) -> Belief

"Shannon entropy in nats, using normalized exp(logb) times grid spacing."
entropy(b::Belief, Φ_grid) -> Float64

"Memoization key: just the counts tuple (grid is deterministic function of counts + c)."
memo_key(b::Belief) = b.counts
```

**Implementation notes:**
- The grid `logb` is determined by `counts` and `c` alone: `logb = sum_j counts[j][2]·log P_|1⟩(Φ_grid, τ_j; c) + counts[j][1]-counts[j][2])·log(1 - P_|1⟩(Φ_grid, τ_j; c))`. Storing both is redundant but the grid cache accelerates entropy evaluation.
- For differentiability: update/entropy must be computable in a Zygote-compatible way. Use broadcasting, not in-place. Log-sum-exp for normalization.
- Entropy integrand: `H = -sum(p .* log.(p .+ 1e-300)) * Δ_Φ` where `p = exp.(logb .- logsumexp(logb)) / Δ_Φ` (density). Numerical-stable.

**Tests (`tests/test_belief.jl`):**
- `test_prior_entropy`: H(uniform over [0, Φ₀/2]) = ln(Φ₀/2) (in nats).
- `test_posterior_narrows`: for a single informative measurement (high-τ, n=10, m=n/2) the entropy decreases.
- `test_counts_sufficient`: two belief updates with the same final counts (different orders) yield the same logb (up to numerical tolerance).
- `test_grid_agreement`: recomputing logb from scratch from counts matches the incrementally-updated logb to 1e-10.

### Phase 3 — `Baselines.jl`

**Goal:** exact enumeration of V_oracle(Φ) and V_fixed(c) for comparability with radar study.

**V_oracle(Φ) definition:**
```
V_oracle(Φ) = max over (τ_k, n_k)_{k=1..K} of E_{m_1..m_K | Φ} [ H(b_0) - H(b_K) ]
            = ln(Φ₀/2) - min_{schedule} E[H(b_K | Φ, schedule)]
```
Enumerate (J·L)^K schedules = 24^4 ≈ 3.3e5. For each schedule and each Φ, enumerate observation trajectories (∏_k (n_k + 1) outcomes) and accumulate expected entropy.

**V_fixed(c) definition:**
```
V_fixed(c) = max over schedule of E_Φ E_{m_1..m_K | Φ} [ H(b_0) - H(b_K) ]
```
Same enumeration, with outer expectation over Φ on the K_Φ-grid.

**Public API:**
```julia
"Enumerate all (J·L)^K schedules; return list of length K tuples."
enumerate_schedules(J, L, K) -> Vector{NTuple{K, Tuple{Int, Int}}}

"Compute Φ_value: expected -H(b_K) under observation distribution for a given x=Φ, schedule, c."
Phi_value(Φ, schedule, c, Φ_star, ω_d, Φ_grid) -> Float64

"V_oracle(Φ) = max over schedules of Φ_value(Φ, ·, c)."
V_oracle(Φ, c, Φ_star, ω_d, Φ_grid; schedules) -> (val::Float64, s_star::Schedule)

"V_fixed(c) = max over schedules of mean_Φ Φ_value(Φ, schedule, c)."
V_fixed(c, Φ_star, ω_d, Φ_grid; schedules) -> (val::Float64, s_star::Schedule, IG_dist::Vector{Float64})

"E[IG] = V_oracle_mean - V_fixed (check identity)."
```

**Parallelism (on the 388-core EPYC):**
- `V_oracle(Φ, c)` for each Φ in Φ_grid: independent → `pmap` over workers, ≥ 256 workers (1-to-1 with Φ-grid). Each computes max over (J·L)^K schedules.
- `V_fixed(c)` enumeration: parallelize over schedules using `Threads.@threads` with chunking to amortize overhead.
- Expected speedup: ~100× from 388 cores on V_oracle (limited by Φ-grid coarseness); ~50× on V_fixed.

**Tests (`tests/test_baselines.jl`):**
- `test_V_oracle_ge_V_fixed`: `V_oracle(Φ) ≥ Φ_value(Φ, s_fixed_star)` for every Φ on the grid.
- `test_V_oracle_ge_zero`: ≥ 0 trivially.
- `test_IG_positive`: `IG(Φ) = V_oracle(Φ) - Φ_value(Φ, s_fixed_star) ≥ -1e-9` for all Φ (small numerical slop allowed).
- `test_V_fixed_equals_mean_IG_identity`: `mean(V_oracle) - V_fixed ≈ mean(IG)` (exact identity, should match to 1e-10).
- `test_small_K`: at K=2, cross-check V_adaptive (Phase 4) against a brute-force enumeration of all (J·L)^K · ∏(n+1) branch leaves — agreement to 1e-9 nats.

### Phase 4 — `Bellman.jl`

**Goal:** exact Bellman DP via memoized backward induction on reachable beliefs.

**Algorithm:**
```
function bellman(b::Belief, k_remaining::Int, c, Φ_grid, memo)
    key = (b.counts, k_remaining)
    haskey(memo, key) && return memo[key]
    if k_remaining == 0
        val = -entropy(b, Φ_grid)
        memo[key] = (val, nothing, nothing)
        return memo[key]
    end
    best_val = -Inf
    best_action = nothing
    for j in 1:J, ℓ in 1:L
        n = n_grid[ℓ]
        τ = τ_grid[j]
        val = 0.0
        # expected value over m ~ Binomial(n, P_marg)
        # P_marg(m) = ∫ P_binom(m | n, P_|1⟩(Φ, τ; c)) b(Φ) dΦ
        for m in 0:n
            b_new = update(b, j, n, m, c, Φ_star, ω_d, Φ_grid)
            p_m = marg_observation_prob(b, j, n, m, c, Φ_star, ω_d, Φ_grid)
            if p_m > 0
                (sub_val, _, _) = bellman(b_new, k_remaining - 1, c, Φ_grid, memo)
                val += p_m * sub_val
            end
        end
        if val > best_val
            best_val = val
            best_action = (j, ℓ)
        end
    end
    memo[key] = (best_val, best_action, nothing)
    return memo[key]
end

V_adaptive(c) = bellman(prior, K_epochs, c, Φ_grid, Dict())[1] + ln(Φ₀/2)
```

**Marginal observation probability:**
```
p(m | b, τ, n, c) = ∫ C(n,m) P_|1⟩(Φ, τ; c)^m (1 - P_|1⟩)^(n-m) b(Φ) dΦ
                  ≈ Δ_Φ * sum_i C(n,m) · P_|1⟩(Φ_i, τ; c)^m · (1 - P_|1⟩(Φ_i, τ; c))^(n-m) · b(Φ_i)
```

**Implementation notes:**
- Use `Dict{NTuple{J, Tuple{Int,Int}}, Tuple{Float64, Tuple{Int,Int}, Nothing}}` for memo. Given ~10⁶ distinct beliefs expected, use `Dict` (hash-based) not sorted structures. Pre-allocate with `sizehint!(memo, 2_000_000)`.
- Memory: each belief stores a 256-vector. 10⁶ beliefs × 256 × 8 bytes = 2 GB. Acceptable on the 2.95 TB machine.
- The recursion is hard to parallelize naively (memo is shared state). Two options:
  (a) **Breadth-first depth-level parallelism.** Compute all beliefs at depth K, then all at K−1, etc. At each depth, parallel map `Threads.@threads` over the set of reachable beliefs (independent). Share the memo after each depth.
  (b) **Concurrent dict** (`ThreadSafeDict.jl` or a sharded lock). More elegant but lock contention.
  Use (a). Pre-compute reachable belief set by breadth-first forward traversal from the prior, then fill Bellman values backward by depth.

**Tests (`tests/test_bellman.jl`):**
- `test_bellman_eq_horizon_1`: at K=1, V_adaptive reduces to a closed-form `max_{j,ℓ} E_m[H(b_0) - H(b_1)]` — check against a direct scalar computation.
- `test_bellman_monotone_K`: V_adaptive(c, K+1) ≥ V_adaptive(c, K). (More epochs never hurt.)
- `test_bellman_le_V_oracle`: V_adaptive(c) ≤ E_Φ[V_oracle(Φ)]. Check at baseline c₀ at K=2.
- `test_bellman_ge_V_fixed`: V_adaptive(c) ≥ V_fixed(c) at baseline.
- `test_bellman_agrees_with_brute_force_K2`: at K=2, enumerate all schedules and trajectories directly; check agreement with Bellman to 1e-9.

### Phase 5 — `Gradient.jl`

**Goal:** efficient, low-variance estimate of `∂W₁/∂c` at the current `π*(c)`.

**Pathwise estimator (preferred):**

By the envelope theorem, at the optimal policy:
```
∂W_1/∂c = E_{traj ~ π*} [ ∂ log L(traj | Φ; c) / ∂c  +  ∂ (-H(b_K; c)) / ∂c ]
```
where `log L(traj | Φ; c) = sum_k log Binom(m_k | n_k, P_|1⟩(Φ, τ_k; c))`.

Sampling a trajectory:
1. Sample Φ ~ prior (uniform on [0, Φ₀/2]).
2. For k = 1..K_epochs:
   - Look up `π*(b_k)` = action-at-this-belief from the memoized Bellman solution.
   - Compute τ_k, n_k.
   - Sample m_k ~ Binomial(n_k, P_|1⟩(Φ, τ_k; c)).
   - Update b_{k+1}.
3. Compute log L along the trajectory; autodiff through to c.

**Implementation:**
```julia
"Run one trajectory forward, return (trajectory, -H(b_K) )."
rollout(c, policy_memo, rng) -> (traj, value)

"Pathwise gradient: autodiff the value function w.r.t. c using Zygote along a batch of trajectories."
function grad_c_pathwise(c, policy_memo; n_traj=10_000, rng=default_rng())
    grads = pmap(1:n_traj) do t
        Zygote.gradient(c_) do c_
            (traj, val) = rollout(c_, policy_memo, rng)
            val
        end |> first
    end
    mean(grads)
end
```

**Parallelism:**
- Trajectories are embarrassingly parallel. Distribute over workers or over GPU batches.
- **GPU batching (preferred for this workload):** implement `rollout_batch(c, policy_memo, n_traj)` that runs all `n_traj` trajectories simultaneously as a vectorized computation on one GPU. Each GPU handles `n_traj / 10` trajectories. With 10 RTX Pro 6000 GPUs, 10⁶-trajectory gradient estimates become cheap.
- The policy `π*(b)` lookup uses the `counts`-based memo key. Since `counts` is a discrete tuple, the policy lookup is just an index-into-a-hash-table — do this on CPU, pre-gather the action sequences, then batch-run the likelihood autodiff on GPU.
- Use `CUDA.jl` for Julia GPU, or call out to JAX via `PythonCall.jl` if Zygote+CUDA is painful. (User preference: Julia.) Test with `CUDA.@threads` first; fall back to `KernelAbstractions.jl` if needed.

**Policy iteration (for handling kinks):**
```
for outer_iter in 1:outer_iters
    if outer_iter % policy_reopt_every == 1
        # Re-solve Bellman DP, refresh policy_memo
        (V_adapt, policy_memo) = solve_bellman(c, K_epochs, Φ_grid)
    end
    grad_c = grad_c_pathwise(c, policy_memo; n_traj=n_traj)
    c = adam_step!(c, grad_c, opt_state)
end
```

**Tests (`tests/test_gradient.jl`):**
- `test_grad_FD_per_component`: at baseline c₀, for each component c_i, compute `∂W₁/∂c_i` analytically via Zygote and vs a central finite difference `(W₁(c+δe_i) - W₁(c-δe_i)) / (2δ)` with `δ = 1e-4·|c_i|`. Relative error should be < 1e-3 (with n_traj ≥ 10⁶ to beat MC noise). **Print AD value, FD value, and relative error with full precision per CLAUDE.md.**
- `test_grad_direction`: random unit direction u, check `u·grad_AD ≈ (W₁(c+δu) - W₁(c-δu))/(2δ)`.
- `test_grad_reduces_envelope_term`: verify that the pathwise gradient does NOT include `∂π*/∂c` terms (they should be zero at π* by Danskin; if test fails, we have a bug).

### Phase 6 — `JointOpt.jl`

**Goal:** outer Adam / MMA loop with periodic Bellman policy re-solution.

**Public API:**
```julia
function joint_opt(c0::ScqubitParams;
                   K_epochs=4, K_Φ=256, J=6, L=4,
                   outer_iters=1000, outer_lr=1e-3, policy_reopt_every=10,
                   n_traj=10_000,
                   ckpt_every=50, ckpt_dir="results/joint/")
    c = c0
    opt_state = Adam(lr=outer_lr)
    history = (W1=Float64[], grad_norm=Float64[], V_fixed=Float64[], EIG=Float64[])
    policy_memo = nothing
    for outer_iter in 1:outer_iters
        if (outer_iter - 1) % policy_reopt_every == 0
            (V_adapt, policy_memo, _) = solve_bellman_full(c, K_epochs, Φ_grid)
            push!(history.W1, V_adapt)
        end
        grad = grad_c_pathwise(c, policy_memo; n_traj=n_traj)
        push!(history.grad_norm, norm(grad))
        # Adam update in ScqubitParams space (flatten to Vector{Float64}, then restructure)
        c = adam_step(c, grad, opt_state)
        # Periodic full evaluation of V_fixed and E[IG]
        if outer_iter % 50 == 0
            V_fix = V_fixed_evaluate(c)
            V_or_mean = V_oracle_mean(c)
            push!(history.V_fixed, V_fix)
            push!(history.EIG, V_or_mean - V_fix)
        end
        if outer_iter % ckpt_every == 0
            serialize(joinpath(ckpt_dir, "ckpt_$(outer_iter).jls"), (c, opt_state, history))
        end
    end
    (c, history)
end
```

**Projection / box constraints:**
- `f_q_max ∈ [1, 30] GHz`
- `E_C/h ∈ [0.1, 1.0] GHz`
- `κ ∈ [0.01, 10] MHz`
- `Δ_qr ∈ [2π · 0.5, 2π · 10] GHz`
- `T ∈ [5, 100] mK`
- `A_Φ ∈ [1e-7, 1e-4] · Φ₀`
- `A_Ic ∈ [1e-7, 1e-4] · I_c`
Project after each Adam step.

**Tests (`tests/test_joint.jl`):**
- `test_joint_monotone_MA50`: a 50-step moving-average of W1 should be monotonically nondecreasing after the first 50 iterations. (Allows for single-step noise from MC gradient but expects overall improvement.)
- `test_joint_reproduces_paper_at_f_q_max_scan`: run the 1-D sweep over f_q_max = [2, 5, 9, 15, 20] GHz with other c components fixed to paper values; check that the in-family optimum is at f_q_max ≈ 9 GHz and the sensitivity value at (f_q_max = 9, Φ* = 0.442) matches the paper's Fig. 2 to within 20% (close-form sensitivity vs our information-theoretic V_adaptive are not identical, but they should rank geometries consistently).

### Phase 7 — `PCRB.jl` (PCRB baseline and MSE-anchored comparison)

**Goal:** compute the joint `(c, s_{1:N})` PCRB fixed-design baseline `(c₂*, s_{1:N}^{(2)*})` and deploy both designs to report an MSE-anchored comparison with the joint-DP result of Phase 6. Operationalizes §8.7 of `AdaptiveDesign.tex` for this case study.

**Mathematical setup.**

For continuous flux `Φ ∈ [0, Φ₀/2]` observed through Ramsey shots with likelihood `P_|1⟩(Φ, τ; c)` (Phase 1), the per-shot Fisher information is scalar:

```
J_F(Φ, τ; c) = (∂P_|1⟩/∂Φ)² / [P_|1⟩ · (1 - P_|1⟩)]
```

Accumulated Fisher information at true Φ across K decision epochs with n_k shots at delay τ_k:

```
J_N(Φ, s_{1:N}, c) = Σ_{k=1..K} n_k · J_F(Φ, τ_k; c)
```

Prior-averaged Bayesian Information Matrix (scalar here since dim(Φ) = 1):

```
J_P(s_{1:N}, c) = J₀ + E_{Φ ~ Uniform[0, Φ₀/2]}[ J_N(Φ, s_{1:N}, c) ]
                = J₀ + (2/Φ₀) · ∫₀^{Φ₀/2} J_N(Φ, s_{1:N}, c) dΦ
```

For the uniform prior on `[0, Φ₀/2]`, `J₀ = 0` in the interior (interior uniform prior has zero Fisher info; the finite-support term vanishes except at the two boundaries). In practice use a small floor `J₀ = ε/Φ₀²` with `ε = 1e-4` to avoid degeneracy when s is uninformative.

The PCRB inequality: `E_{Φ,y}[(Φ̂ - Φ)²] ≥ J_P⁻¹`. The PCRB-baseline joint optimization:

```
(c₂*, s_{1:N}^{(2)*}) = argmax_{c, s_{1:N}} log J_P(s_{1:N}, c)
```

This is a **single-level** optimization in `d_c + 2K` real variables (c plus the K pairs (τ_k, n_k), treated as continuous during the outer Adam and snapped to the discrete grid at the end for the sweep comparison). No belief tree, no policy memoization, no Bellman recursion — dramatically cheaper than Phase 6.

**Public API:**
```julia
"Per-shot Fisher information J_F(Φ, τ; c), closed form from Bernoulli likelihood."
function fisher_per_shot(Φ::Real, τ::Real, c::ScqubitParams, Φ_star, ω_d)
    P = P1_ramsey(Φ, τ, c, Φ_star, ω_d)
    dPdΦ = ForwardDiff.derivative(Φ_ -> P1_ramsey(Φ_, τ, c, Φ_star, ω_d), Φ)
    dPdΦ^2 / (P * (1 - P) + 1e-300)
end

"Accumulated Fisher at Φ: J_N(Φ, s, c) = Σ n_k J_F(Φ, τ_k, c)."
function fisher_accumulated(Φ, s_schedule::Vector{Tuple{Float64,Int}}, c, Φ_star, ω_d)
    sum(n * fisher_per_shot(Φ, τ, c, Φ_star, ω_d) for (τ, n) in s_schedule)
end

"Prior-averaged BIM on uniform prior [0, Φ₀/2] by K_Φ-point quadrature."
function bim_prior_averaged(s_schedule, c, Φ_star, ω_d, Φ_grid; J_0 = 1e-4)
    J_0 + mean(fisher_accumulated(Φ, s_schedule, c, Φ_star, ω_d) for Φ in Φ_grid)
end

"PCRB-baseline joint optimization.  Alternating inner schedule search + outer Adam on c."
function pcrb_baseline(c_init::ScqubitParams;
                       K_epochs=4, J=6, L=4, K_Φ=256,
                       outer_iters=500, outer_lr=1e-3,
                       enumerate_schedule=true)
    c = c_init
    opt_state = Adam(lr=outer_lr)
    history = (log_JP=Float64[], c_trajectory=[])
    for iter in 1:outer_iters
        # inner: find best schedule at current c (enumeration or continuous)
        if enumerate_schedule
            s_star = argmax_schedule_by_enumeration(c, J, L, K_epochs, K_Φ)
        else
            s_star = argmax_schedule_continuous(c, K_epochs, K_Φ)  # gradient descent on (τ_k, n_k)
        end
        log_JP_val = log(bim_prior_averaged(s_star, c, Φ_star(c), ω_d(c), Φ_grid(K_Φ)))
        push!(history.log_JP, log_JP_val)
        # outer: gradient step on c (autodiff through bim_prior_averaged at fixed s_star)
        grad_c = Zygote.gradient(c_ -> log(bim_prior_averaged(s_star, c_, Φ_star(c_), ω_d(c_), Φ_grid(K_Φ))), c)[1]
        c = adam_step(c, grad_c, opt_state)
        push!(history.c_trajectory, c)
    end
    (c, s_star, history)
end

"Deploy a fixed schedule at a fixed geometry; return deployed prior-averaged MSE of the posterior-mean estimator."
function deployed_mse_fixed(c, s_schedule, Φ_star, ω_d; n_mc=10_000, K_Φ=256)
    Φ_grid = range(0, Φ_0/2, length=K_Φ)
    # ... MC loop: sample Φ_true ~ uniform, roll out Binomial observations under s_schedule,
    # compute posterior grid, take posterior-mean, accumulate (Φ_mean - Φ_true)^2.
end

"Deploy an adaptive policy; return deployed prior-averaged MSE of the posterior-mean estimator."
function deployed_mse_adaptive(c, π_policy_memo, Φ_star, ω_d; n_mc=10_000, K_Φ=256)
    # ... MC loop: sample Φ_true, roll out policy π_memo, compute posterior-mean, accumulate sq-err.
end
```

**Implementation notes.**

- **Inner schedule search.** Two modes:
  - `enumerate_schedule=true` (default): for fixed c, enumerate `(J·L)^K = 24⁴ ≈ 3.3×10⁵` discrete schedules. For each, compute `log J_P` (cost: `K_Φ · K` Fisher evaluations). Pick the argmax. Parallelize over schedules with `Threads.@threads` or `pmap`.
  - `enumerate_schedule=false`: treat `(τ_k, n_k)` as continuous (relaxing the integer n_k to ℝ₊), run gradient descent to find the inner optimum, snap back at the end. Faster but may miss discrete combinatorial structure.
- **Outer Adam on c.** Because `log J_P` is differentiable in c (closed form, analytic rate formulas in Phase 1), the outer gradient `∂ log J_P / ∂c` flows via Zygote with no envelope-theorem machinery needed: this is a standard single-level gradient-descent on `c` with the inner `s*(c)` held fixed by the (continuous or integer) inner optimization.
- **Implicit function theorem for continuous inner.** If using continuous relaxation of n, the inner optimum `s*(c)` depends on c. The outer gradient should use the IFT adjoint (or re-solve the inner at each c). For enumeration mode, treat `s*(c)` as piecewise-constant in c and ignore boundary effects (standard bi-level practice for combinatorial inner problems).
- **Gradient check required.** Same rigor as Phase 5: per-component FD check of `∂ log J_P / ∂c` at baseline c₀, AD vs FD, relative error < 1e-5 (cleaner than Phase 5 because no MC variance here).

**Tests (`tests/test_pcrb.jl`):**
- `test_fisher_positive`: J_F ≥ 0 for all valid (Φ, τ, c).
- `test_fisher_vanishes_at_sweet_spot`: at Φ = 0, `∂ω_q/∂Φ = 0`, so J_F = 0 for any τ (per the source paper's Eq. 8). Important sanity check.
- `test_JN_additivity`: `J_N(Φ, concat(s₁, s₂), c) = J_N(Φ, s₁, c) + J_N(Φ, s₂, c)`.
- `test_JP_bound_empirical`: at a random (c, s), sample 10⁴ Φ ~ uniform, run fixed-schedule observations, compute posterior-mean estimator, average `(Φ̂ - Φ)²` over samples. Expect empirical MSE ≥ 1/J_P (PCRB bound holds). The bound may be loose (posterior-mean is biased at finite N; shrinkage typically reduces pointwise MSE at boundary).
- `test_log_JP_differentiable_c`: gradient check on `∂ log J_P/∂c` via FD. **Print AD value, FD value, relative error with full precision.**
- `test_pcrb_vs_joint_DP_radar_sanity`: not applicable here (radar has discrete state, no PCRB). Note in the test comment that this test exists only for the scqubit pipeline.

### Phase 7.1 — MSE-anchored comparison (`compare_mse.jl`)

After both Phase 6 (joint DP → `(c₁*, π₁*)`) and Phase 7 (PCRB baseline → `(c₂*, s₂*)`) complete:

```julia
# Load checkpointed results
(c_1_star, π_1_star) = deserialize("results/joint/ckpt_final.jls")
(c_2_star, s_2_star) = deserialize("results/pcrb/ckpt_final.jls")

# Deploy both: prior-averaged MSE under posterior-mean estimator, 10⁵ MC trials
MSE_1 = deployed_mse_adaptive(c_1_star, π_1_star, Φ_star(c_1_star), ω_d(c_1_star); n_mc=100_000)
MSE_2 = deployed_mse_fixed(c_2_star, s_2_star, Φ_star(c_2_star), ω_d(c_2_star); n_mc=100_000)

# Headline
println("MSE̅₁ (joint DP):      $(MSE_1)")
println("MSE̅₂ (PCRB baseline): $(MSE_2)")
println("MSE gain MSE̅₂/MSE̅₁:   $(MSE_2/MSE_1)")

# Sanity cross-check: MSE̅₂ ≥ 1/J_P(c_2_star, s_2_star) (PCRB bound)
J_P_val = bim_prior_averaged(s_2_star, c_2_star, Φ_star(c_2_star), ω_d(c_2_star), Φ_grid(K_Φ))
println("PCRB bound 1/J_P:     $(1/J_P_val)")
@assert MSE_2 ≥ 1/J_P_val - 1e-10 "PCRB bound violated — check bias model"
```

**Expected:** `MSE̅₂/MSE̅₁ > 1` — the joint-DP beats the PCRB baseline on deployed MSE, quantitatively endorsing the framework of the paper.

**Not guaranteed:** if `MSE̅₂/MSE̅₁ ≤ 1` at the chosen K, J, L, and c_tier, the joint-DP machinery does not pay for itself in this regime. Report honestly in the writeup. Possible causes: K_epochs = 4 is too short for adaptation to overcome PCRB's more careful geometric optimization; the PCRB baseline is within the convex hull of joint-DP-reachable designs; the tier-1 c has insufficient degrees of freedom to give the adaptive policy room to reshape. Expand c_tier or K_epochs as diagnostic.

### Phase 8 — Sweeps (`sweep_fq.jl`, `sweep_joint.jl`, `sweep_pcrb.jl`)

**`sweep_fq.jl`:** 1-D sweep over `f_q_max ∈ {2, 3, 4, ..., 20} GHz`, all other c components at paper values. For each:
- Solve for Φ* (1-D max of |∂P_|1⟩/∂Φ|).
- Compute `V_oracle_mean, V_fixed, V_adaptive, E[IG], saturation` (Φ framework).
- Compute `log J_P^*(c) = max_s log J_P(s, c)` (PCRB framework).
- Compute `1/J_P^*` (PCRB MSE lower bound).
- Tabulate and save to `results/sweep_fq.jls`.

Expected: both frameworks should identify `f_q_max ≈ 9 GHz` as favorable (consistent with paper), but the exact in-family optima may differ — the Φ framework picks the `f_q_max` that maximizes `V_adaptive`, the PCRB framework picks the one that maximizes `log J_P^*`. Reporting both columns gives the reader a direct side-by-side.

**`sweep_joint.jl`:** full 7-D Adam for joint DP from a range of c₀ starts. 500–2000 Adam steps each. Returns `(c₁*, π₁*)`.

**`sweep_pcrb.jl`:** full 7-D Adam for PCRB baseline from the same c₀ starts. 500 iterations each (cheaper per iteration; no Bellman solve). Returns `(c₂*, s₂*)`.

### Phase 9 — Plots (`plot_results.jl`)

Using `Plots.jl` or `Makie.jl`:
- `plot_sweep_fq.png`: six panels — V_adaptive, V_fixed, V_oracle_mean, E[IG], log J_P*, 1/J_P* vs f_q_max. Highlight in-family optimum for each.
- `plot_adam_history_joint.png`: convergence curves for V_adaptive, grad_norm during joint-DP Adam.
- `plot_adam_history_pcrb.png`: convergence curves for log J_P during PCRB-baseline Adam.
- `plot_belief_evolution.png`: typical belief b_k(Φ) at each epoch k for optimal policy at c₁*.
- `plot_policy_tree.png`: first two levels of the optimal policy (action at each observation history).
- `plot_mse_comparison.png`: bar chart of MSE̅₁ vs MSE̅₂ vs PCRB bound 1/J_P(c₂*, s₂*). Annotate ratio.
- Table generation: LaTeX-formatted tables analogous to IG_numerics Table 1, plus an MSE-comparison table (joint-DP vs PCRB baseline, absolute MSE and ratio).

### Phase 10 — Writeup (`doc/adaptive/scqubit_results.tex`)

Mirror `doc/adaptive/IG_numerics.tex` structure, with the MSE-anchored comparison as the headline operational claim:

1. Problem recap (reference scqubit_model.tex).
2. Numerical setup (Φ-grid, τ-grid, n-grid, horizon).
3. V_oracle, V_fixed enumeration results (Φ framework).
4. Bellman DP results (tree size, runtime, memoization statistics).
5. Within-family sweep over f_q_max: side-by-side Φ-framework and PCRB-framework columns (results table, plot). Compare to paper Fig. 2.
6. Joint 7-D optimization (joint DP): Adam trajectory, final c₁*, V_adaptive* vs baseline.
7. PCRB-baseline joint optimization: Adam trajectory, final (c₂*, s₂*), log J_P* vs baseline.
8. **Headline MSE-anchored comparison:** table of MSE̅₁ vs MSE̅₂ vs PCRB bound 1/J_P; ratio MSE̅₂/MSE̅₁. This is the operational §8.7 claim.
9. Within-family boxed three-way comparison (analog of IG_numerics §8.6).
10. Position in the relaxation hierarchy: this is the exact-DP counterpart to the photonic BFIM path-(2c) instance.
11. Caveats and limitations (finite K, potential PCRB bound slack, biased estimator, boundary effects of uniform prior).

---

## 5. Parallelism strategy (388 cores + 10 GPUs)

**Big picture:**
- **CPU-bound:** V_oracle / V_fixed enumeration, Bellman DP expansion, policy memoization, outer Adam bookkeeping.
- **GPU-bound:** Monte Carlo trajectory sampling for gradient estimation (10⁶+ trajectories), since each trajectory is a short sequence of GPU-friendly vectorized ops.

**Worker topology:**
- **Main process:** 1 process. Coordinates.
- **Bellman workers:** `Threads.@threads` with `Threads.nthreads() = 128` (one per physical core on socket 0, following user's NUMA guidance in CLAUDE.md). Used only during Bellman-solve phases.
- **V_oracle/V_fixed workers:** `addprocs(256)` via `Distributed`. BLAS threads = 1 per worker (avoid oversubscription).
- **Gradient workers:** 10 GPU device processes, one per GPU. Managed via `CUDA.device!(i)` + `@spawnat` or equivalent. Each holds a batch of ~10⁵ trajectories.

**Per CLAUDE.md:**
- Use physical core count for `-t` flag (Julia thread count). Do NOT use hyperthreaded count (Threads.maxthreadid() can be 2× on Julia 1.12+).
- Set FFTW threads to 1.
- Never write `var = nothing` after a `@threads` loop (closure boxing).
- Use `Threads.nthreads() + 1` for workspace-pool sizes if needed.

**NUMA awareness:**
- Bellman memo is a single shared object; pin it to one NUMA node by running the Bellman solver on socket 0.
- GPU trajectories: pin each GPU device process to its nearest NUMA socket (Blackwell cards 0–4 → socket 0, cards 5–9 → socket 1). Use `numactl` to enforce.

**Target runtimes on this machine:**

| phase | time | notes |
|---|---|---|
| V_oracle(Φ) at one c, parallel over K_Φ=256 Φ values | 30 s | 256-way parallelism |
| V_fixed(c) at one c | 10 s | 24⁴ = 332k schedules, parallel |
| Bellman DP at one c (K=4) | 20 s | 10⁶ belief memoization |
| grad_c at one c, 10⁶ trajectories across 10 GPUs | 5 s | batched GPU rollouts |
| one outer Adam step (incl. grad only, reuse policy) | 6 s | joint-DP loop |
| one policy-iteration step (full DP re-solve + grad) | 25 s | every 10 steps |
| 1000 outer Adam steps (joint-DP, policy re-opt every 10) | ~2 hours | |
| `J_P(s, c)` at one (s, c), parallel over K_Φ Φ values | 0.1 s | closed-form Fisher, cheap |
| inner schedule enumeration `argmax_s log J_P` at one c | 30 s | 3.3×10⁵ schedules, 256-way parallel |
| one outer Adam step (PCRB baseline: inner enum + Zygote grad) | 30 s | dominated by inner enum |
| 500 outer Adam steps (PCRB baseline) | ~4 hours | |
| deployed MSE (100k MC trials, adaptive or fixed) | 5 min | parallelized over trials |
| `sweep_fq.jl` (19 values of f_q_max, both frameworks) | ~15 min | joint-DP dominates |
| `sweep_joint.jl` (10 random starts × 1000 steps each) | ~20 hours | |
| `sweep_pcrb.jl` (10 random starts × 500 steps each) | ~40 hours | could parallelize across starts |
| `compare_mse.jl` (deploy both, 10⁵ MC each) | ~15 min | |

All comfortably within overnight or a weekend run.

---

## 6. Gradient-check protocol (MANDATORY per CLAUDE.md)

Before starting any Adam run, `test_gradient.jl` MUST pass with:
- Per-component FD check at c₀: AD value, FD value, relative error printed with full precision.
- n_traj ≥ 10⁶ (use GPU batching), so MC noise is < 1e-4 relative.
- Random-direction directional-derivative check over ≥ 10 trials.
- Printed output must show the actual numbers; do not just say "it passes".

If gradient check fails: halt, diagnose (likely cause: Zygote incompatibility with some rate-formula branch, or sign error in envelope theorem derivation). Do NOT proceed to Adam until the check passes.

---

## 7. Risk register and mitigations

| risk | mitigation |
|---|---|
| Memoization table exceeds RAM | Limit K_epochs ≤ 4; monitor `length(memo)`; use `sizehint!` |
| Multi-modal posterior breaks grid | Increase K_Φ adaptively; fallback K_Φ = 512 or 1024 |
| Gradient variance too high | Increase n_traj; use antithetic/control-variate sampling |
| Zygote fails on some rate formula | Rewrite that formula with ForwardDiff-only AD, wrap in custom rrule |
| L_J singularity at Φ = Φ₀/2 | Clip Φ ≤ 0.49·Φ₀ in rate evaluation |
| Adam oscillates near kinks | Decrease lr; increase policy_reopt_every (more frequent re-solves) |
| GPU rollout slow due to kernel launches | Batch all n_traj in one kernel; avoid Python-in-loop |
| Discrepancy with paper's Φ* = 0.442 | Check ω_d convention (rad/s vs Hz), check cos convention, check Eq. 16 derivation step-by-step |
| Bellman agrees with V_fixed (no adaptive gain) | Check that K_epochs > 1 and observation outcome branches actually diverge — diagnose at K=2 by hand |

---

## 8. Definition of done

**Must have:**
- [ ] All tests in `tests/` pass.
- [ ] Gradient check (joint DP): relative error < 1e-3 per component at baseline c₀.
- [ ] Gradient check (PCRB baseline): relative error < 1e-5 per component at baseline c₀ (cleaner because no MC variance).
- [ ] `sweep_fq.jl` reproduces the paper's finding that f_q_max ≈ 9 GHz is the in-family-optimum at paper values of the other c-components — for both Φ framework and PCRB framework.
- [ ] `sweep_joint.jl` produces a c₁* with V_adaptive(c₁*) > V_adaptive(c₀).
- [ ] `sweep_pcrb.jl` produces a (c₂*, s₂*) with log J_P(c₂*, s₂*) > log J_P(c₀, s_random).
- [ ] `compare_mse.jl` reports MSE̅₁, MSE̅₂, and the ratio MSE̅₂/MSE̅₁ with absolute MSE values (not just ratios) at n_mc ≥ 10⁵.
- [ ] PCRB bound sanity check: MSE̅₂ ≥ 1/J_P(c₂*, s₂*) (empirical PCRB not violated).
- [ ] `doc/adaptive/scqubit_results.tex` compiles to PDF with all tables and plots including the MSE-anchored comparison as the headline table.
- [ ] All .jls results are serialized to `results/` for reproducibility.
- [ ] `git commit -m "scqubit joint (c, π) Bellman DP + envelope-theorem gradient results"` and push.

**Nice to have:**
- [ ] Policy-tree visualization at c* (first 2–3 levels).
- [ ] Within-family boxed three-way table (analog of IG_numerics §8.6).
- [ ] Tier-2 extension: 12-D c including SQUID-loop geometric parameters via analytically differentiable Biot–Savart.
- [ ] Comparison of exact Bellman policy with paper's Kitaev PEA — by how much does Bellman beat Kitaev at the same (τ, n) grid?

---

## 9. Quick-start sequence (for the fresh Claude session)

```bash
# 1. Read context
# (see Section 0; do not skip)

# 2. Set up
cd ~/BFIMGaussian
mkdir -p doc/adaptive/scqubit/{tests,results}

# 3. Implement Phase 1 first, test it, then move to Phase 2, etc.
# Each phase ends with passing tests before moving on.

# 4. For Julia launches:
julia -t 128 --project=. doc/adaptive/scqubit/tests/test_model.jl
julia -t 128 --project=. doc/adaptive/scqubit/sweep_fq.jl
# Gradient work:
julia -t 128 --project=. -e 'using Distributed; addprocs(10); @everywhere using CUDA' doc/adaptive/scqubit/sweep_joint.jl

# 5. When results are in, build the tex:
cd doc/adaptive
pdflatex scqubit_results.tex
pdflatex scqubit_results.tex
rm -f *.aux *.log *.out *.toc

# 6. Commit:
git add doc/adaptive/scqubit/
git add doc/adaptive/scqubit_results.tex
git add doc/adaptive/scqubit_results.pdf
git commit -m "Implement scqubit exact joint (c, π) Bellman DP with envelope-theorem gradient"
git push
```

---

## 10. Communication protocol

When reporting progress back to the user (after this Claude session runs), report:

1. Which phases completed and which tests passed.
2. For gradient check: **print** AD value, FD value, relative error with full precision (per CLAUDE.md).
3. Actual runtimes for each phase (seconds, minutes) — absolute numbers, not relative.
4. Final V_adaptive(c*) vs V_adaptive(c₀) — concrete numbers with relative improvement.
5. Comparison with paper Fig. 2 (reproducibility check).
6. Any surprises, numerical issues, or design decisions you had to make.

Do not summarize vaguely. Quote the numbers.

---

## 11. Stylistic reminders

- **Follow CLAUDE.md** (both global and project-level). Especially: no mutating ops in Zygote-traced code; physical-core thread counts; FFTW thread=1; scope precisely.
- **Iteration-driven workflow.** Test Phase 1 before Phase 2. Don't write all 6 phases then debug — the debugging surface becomes too big.
- **Concise responses when reporting progress.** Lead with numbers, not prose.
- **No unnecessary comments or docstrings** on untouched code.
- **Don't refactor the existing photonic BFIM code.** This is a new module set.

---

End of plan.
