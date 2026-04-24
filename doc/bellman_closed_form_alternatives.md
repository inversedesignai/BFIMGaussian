# Closed-form compact-rule alternatives to tabulated Bellman DP

A note on representing Bellman value functions as small exact rules rather than lookup tables or neural parametrizations.

## Context

Exact Bellman DP is expensive because it tabulates `V_k(b)` over a set of reachable beliefs. Even with sufficient-statistic reductions — count-tuples, Gaussian moments, PBVI α-vectors — the representation is still essentially enumerative: a list of beliefs and their values.

Two standard alternatives trade exactness for compactness:
- **Parametric approximation (classical).** `V_k(b) ≈ w_k · φ(b)` with a hand-chosen feature map. Approximate; structurally faithful to Bellman.
- **Neural approximation.** `V_{θ,w}(b, k)` with a trained network. Approximate; very flexible; loses some structural faithfulness.

This note catalogs a third route: **closed-form compact rules** — problem classes where `V_k` is *exactly* representable by a small set of parameters plus a fixed propagation formula, with no tabulation and no trained model. This route is narrow (it requires the problem to have specific structural features), but when it applies it gives the cleanest answer available: the value function is a small object, its recursion is analytical, and no approximation is involved.

## The unifying principle

`V_k` admits a closed-form compact representation iff the Bellman operator maps the chosen function class to itself. Formally, let `𝓕` be a family of functions parameterized by a small number of parameters (quadratics, polynomials of bounded degree, max-of-linears, …). If

```
V_{k+1} ∈ 𝓕   ⟹   TV_{k+1} ∈ 𝓕
```

where `T` is the Bellman backup operator, then the recursion stays inside `𝓕` forever, and the "rule" is just: which element of `𝓕` does `V_k` equal, and how are the parameters of that element updated from `V_{k+1}`?

This is what fails for generic problems. The Bellman backup's combination of max (non-smooth) and expectation (integration) is nonlinear enough to destroy most compact function families. Only problems with specific structural features — linear dynamics, quadratic costs, finite-action belief-linear rewards, specific symmetries — preserve a compact family.

## Catalog

### 1. Linear-Quadratic regulator / Linear-Quadratic Gaussian (LQR / LQG)

**Structure.** Linear dynamics `x_{k+1} = A x_k + B u_k + w_k` with `w_k` Gaussian, quadratic cost `c(x, u) = x^T Q x + u^T R u` per stage, Gaussian prior.

**V_k.** Quadratic: `V_k(x) = x^T P_k x + α_k`.

**Rule.** Riccati equation:

```
P_k  =  Q  +  A^T P_{k+1} A  −  A^T P_{k+1} B (R + B^T P_{k+1} B)^{−1} B^T P_{k+1} A
```

with terminal `P_K = Q_K`. One matrix recursion — that's the entire representation.

**Policy.** Linear feedback `u_k*(x) = −(R + B^T P_{k+1} B)^{−1} B^T P_{k+1} A · x`. Also compact: a matrix `K_k` per stage.

**Why it closes.** The Bellman operator on quadratic `V_{k+1}` produces quadratic `V_k` because: (i) the `min_u` of a quadratic-in-`(x, u)` function is a quadratic function of `x` by block-completion; (ii) the Gaussian expectation of a quadratic is a quadratic plus a constant. The quadratic family is closed under the Bellman backup.

**Where it applies.** Classical optimal control. Belief states under Gaussian priors and linear observations give LQG — same story with the belief covariance evolving separately (Kalman).

### 2. Linear-exponential-quadratic (LEQG, risk-sensitive LQR)

**Structure.** LQR with a risk-sensitive exponential objective, `E[exp(−γ · cost)]` for some `γ`.

**V_k.** Exponential-of-quadratic: `V_k(x) = exp(x^T P_k x + …)`.

**Rule.** Modified Riccati, with a risk-sensitivity correction term that can destabilize the recursion at large `γ` (the "neurotic breakdown" of Whittle).

**Note.** Useful when one wants to penalize high-variance outcomes disproportionately; still closed-form.

### 3. POMDP α-vectors (Sondik's theorem)

**Structure.** Finite discrete-state POMDP with finite action and observation sets, prescribed reward.

**V_k.** Piecewise-linear-convex in the belief `b ∈ Δ^{|𝒳|−1}`:

```
V_k(b) = max_{α ∈ Γ_k}  α · b
```

where `Γ_k` is a finite set of `|𝒳|`-dimensional vectors.

**Rule.** For each action `a` and observation `y`, combine each `α' ∈ Γ_{k+1}` with the reward function and the belief-update operator to produce a new α-vector. Take the union across all `(a, y, α')` combinations; prune dominated α-vectors to get `Γ_k`.

**Sondik 1971** proves that exact finite-horizon POMDP value functions are PWLC — this is an exact, no-approximation statement. The representation is compact in the sense of being expressible by a finite set of vectors; it is not bounded in the sense that `|Γ_k|` can grow exponentially with horizon in the worst case. **Point-Based Value Iteration (PBVI, Pineau–Gordon–Thrun 2003)** trades exactness for size by sampling a finite set of representative beliefs and keeping only the α-vectors best at those beliefs. SARSOP, HSVI, Perseus refine this with bounds and trust regions.

**Why it closes.** The Bellman backup on a PWLC function produces a PWLC function because: (i) finite-action max over PWLC functions is PWLC (the max of max-of-linears is a max-of-linears); (ii) finite-observation expectation is a linear combination of PWLC functions, which is PWLC. The PWLC family is closed under the Bellman backup for finite-state POMDPs.

**Where it applies.** Discrete-state POMDPs of moderate size. The canonical non-LQR closed-form DP.

### 4. Polynomial DPs

**Structure.** Polynomial dynamics, polynomial cost, polynomial noise covariance. Generalizes LQR to higher orders.

**V_k.** Polynomial of bounded degree: `V_k(x) = Σ_{|α| ≤ d} c_{k, α} x^α`.

**Rule.** Polynomial arithmetic in the Bellman backup: multiplying, integrating, maximizing over a polynomial structure. The coefficient vector `c_k` is the compact parameter; its update is a set of multilinear relations.

**Caveat.** The degree `d` grows under each Bellman backup unless truncated; truncation reintroduces approximation. Exact polynomial DP works cleanly only when the problem structure caps the degree (e.g., quadratic-cost LQR caps at 2).

**Software.** Sum-of-squares optimization tools (SOSTOOLS, SumOfSquares.jl) automate polynomial Bellman backups.

### 5. Symmetry-reduced DPs

**Structure.** Problem has a symmetry group `G` acting on the state space such that dynamics, cost, and prior are `G`-invariant.

**V_k.** Constant on orbits: `V_k(x) = V_k(g · x) ∀ g ∈ G`. Equivalently, `V_k` is a function on the quotient space `𝒳 / G`.

**Rule.** Bellman on `𝒳 / G`, which has smaller effective dimension. Symmetry reduction can shrink the state space by factors of `|G|` or more.

**Where it applies.** Problems with translational, rotational, or permutation invariance. Multi-agent problems with exchangeable agents. The paper's radar case (Case A) has rotational symmetry `ℤ_16` that could in principle be exploited to reduce the 16-cell state to an orbit space of size 1 (by rotational equivalence of initial beliefs) — though this specific case is already tractable by direct tabulation.

### 6. Hamilton-Jacobi-Bellman with analytical solutions

**Structure.** Continuous-time optimal control / stochastic control with specific dynamics and rewards.

**V(x, t).** Closed-form from the HJB PDE:

```
−∂V/∂t  =  min_u { c(x, u)  +  (∇V)^T f(x, u)  +  ½ tr(σ σ^T ∇²V) }
```

**Examples with closed form:**
- **Merton's portfolio optimization:** `V` is explicit (power-of-wealth / log-wealth), policy is constant proportion.
- **Black-Scholes option pricing:** European option value from the Black-Scholes PDE.
- **Heston stochastic volatility:** closed-form characteristic function.
- **Linear-Gaussian control in continuous time:** quadratic `V` (continuous-time LQR).

**Where it applies.** Small set of classical problems, mostly in mathematical finance and linear-Gaussian control.

### 7. Monotone / convex DPs with threshold or bang-bang policies

**Structure.** Problems where the optimal policy is known by structure to be a threshold rule or a bang-bang rule (e.g., inventory problems with base-stock policies, optimal stopping, Gittins indexes for bandits).

**V_k.** Characterized by its "sufficient profile" — a threshold value, a critical state, an index function — plus a rule for computing the rest.

**Rule.** Problem-specific algebra (the Gittins index for multi-armed bandits; the `(s, S)` rule for inventory).

**Classical results.** Topkis (1978) for monotone policies; Gittins (1979) for bandits. Well-developed for specific problem families.

## Where the paper's rungs sit

The companion paper's hierarchy (§5) moves through exact tabulation → prescribed heuristics → Gaussian surrogate. The closed-form compact-rule regime is essentially "rung 0": when the problem's structure gives `V_k` a closed-form representation, none of rungs 1–4 is needed, and the DP collapses to a finite parameter recursion.

| Regime | Representation of V_k | Exactness | Scales to |
|---|---|---|---|
| **Rung 0 (closed form):** LQR, α-vectors, polynomial, symmetric, HJB | Small parameter set + analytical rule | Exact (no approximation) | Any problem in the class |
| **Rung 1 (exact tabulation)** | Lookup table over reachable beliefs | Exact | Moderate state/action/horizon |
| **Rung 1.5 (PBVI / linear FVI)** | Truncated α-vector set / linear-in-features | Nearly exact (truncation error) | Larger than rung 1 |
| **Rung 1.7 (NN-FVI, Option B)** | NN value function | Approximate | Any problem if NN has capacity |
| **Rungs 2–4 (prescribed)** | Implicit in the prescribed rule | Approximate | Large/continuous |

Rung 0 dominates every other rung when it applies. Its applicability is the binding constraint.

## The photonic problem: is there a closed-form rule available?

The photonic topology problem of Case C has structure worth examining through this lens:

- **Dynamics are trivial** (the state `x = Δn` is static over the horizon). So the "dynamics" part of LQR is `A = I`, `B = 0`.
- **Observation model** is the port-power mapping. The Taylor-expanded S-matrix is quadratic in `x`:
  ```
  S(x) ≈ S_0 + (∂S/∂n) x + ½ (∂²S/∂n²) x²
  ```
  The port powers `y = |S(x) a|² + noise` are quartic in `x`. This is **not** linear-Gaussian, so strict LQG does not apply.
- **Prior** is Gaussian on `x`.
- **Reward** is quadratic (posterior-mean MSE).

So the problem is linear-in-prior, quartic-in-observation, quadratic-in-reward. The Bellman operator on a quadratic `V_{k+1}(b)` (quadratic in the belief moments) does not stay quadratic: the quartic observation introduces higher-order terms in the belief update (moments of order > 2 appear), and the max over `s` typically doesn't close the algebra either.

**The upshot.** The photonic problem is *just outside* the clean LQG class. The natural escape routes:

1. **Polynomial DP with truncation.** Represent `V_k` as a polynomial in the belief's first `M` moments (truncated at some order). The Bellman backup then involves polynomial arithmetic on those moments. Exact up to the truncation; approximate beyond it. This is a structured generalization of the rung 4 EKF approach, which is the special case `M = 2` (mean + covariance).

2. **Moment closure via Gaussian assumption.** Rung 4 of the paper: assume the belief stays Gaussian, propagate moments via EKF, compute the Fisher-information surrogate. Exact for the Gaussian projection; approximate for the true posterior.

3. **Numerical HJB on the belief manifold.** Solve the Bellman equation numerically on a discretization of belief space (grid + PBVI-style compression). Feasible for the `x ∈ ℝ^4` state of Case C.

4. **Compact exact rule on a truncated physics.** If we accept that the physics is Taylor-truncated, an *exact* closed-form rule exists: represent `V_k` as a polynomial of bounded degree in the belief moments, track enough moments that the Bellman backup closes (probably 4–6 orders), and propagate by explicit multilinear algebra. This would be an exact DP on the Taylor-truncated model and an approximation of the Bellman optimum for the original non-Taylor-truncated problem. Whether the bookkeeping is tractable at 20 frequencies × 4 cells × 10 measurement epochs is an open question that would reward someone willing to work out the multilinear algebra.

Option 4 is the spiritual cousin of rung 0 for the photonic problem: it exists in principle, but whether the rule is *compact enough to be practical* depends on how aggressively the moment hierarchy can be truncated while preserving closure. A first-principles investigation would (i) write out the Bellman backup for the Taylor-truncated S-matrix model under a fourth-moment belief representation, (ii) verify the family is closed under the backup (i.e., fourth moments map to fourth moments, or identify what higher moments get generated), (iii) write the parameter recursion.

## When to reach for rung 0

The checklist, in decreasing order of preference:

1. Does the problem have linear dynamics, quadratic costs, Gaussian noise? → **LQR / LQG.**
2. Is the state discrete and finite, with finite action and observation spaces? → **PBVI on α-vectors** (exact variant or truncated approximation).
3. Does the problem have a symmetry group? → **Symmetry reduction** to quotient space, then apply one of the other rungs on the reduced problem.
4. Does the problem admit a closed-form HJB solution (e.g., Merton-like continuous-time control)? → **Analytical HJB.**
5. Is the optimal policy known by structure to be monotone / threshold / bang-bang? → **Structural DP** via the problem-specific parametric family.
6. Is the forward model polynomial in the state with the Bellman backup closing under a finite-degree polynomial family? → **Polynomial DP.**
7. None of the above → fall back to the paper's §5 hierarchy (rungs 1–4) or Option B (NN-FVI).

## What this means for Option B

Option B (the NN-based end-to-end approach) is not the only path beyond tabulation — it is the general-purpose path for problems *without* the structural features that enable rungs 0a–0f above. The photonic problem sits right at the edge: it has a lot of structure (Taylor-truncated polynomial physics, Gaussian prior, quadratic reward), but the combination does not quite admit an off-the-shelf closed-form rule. So one either bends the problem into a clean closed-form class (e.g., assume Gaussian posterior → rung 4) and pays the approximation bias, or reaches for a general-purpose flexible approximator (NN-FVI = Option B).

A well-invested research direction: work out the polynomial-DP rule for the photonic Taylor-truncated model (option 4 above) and compare against Option B. If the polynomial DP produces a tractable rule, it would be the *first-principles-closest* non-NN, non-tabulated representation for the photonic problem — a genuine rung 0 entry for the paper's hierarchy.

## References

- **LQR / LQG.** Anderson & Moore, *Optimal Control* (1989). Bertsekas, *DP and Optimal Control*, Vol. I chapter 4.
- **Risk-sensitive LQR.** Whittle, *Risk-Sensitive Optimal Control* (1990).
- **Sondik α-vectors.** Sondik, *The optimal control of partially observable Markov processes* (PhD thesis, 1971); Smallwood & Sondik, *The optimal control of partially observable Markov processes over a finite horizon* (OR 1973).
- **PBVI and descendants.** Pineau, Gordon & Thrun, *PBVI* (IJCAI 2003). Smith & Simmons, *HSVI* (UAI 2004). Spaan & Vlassis, *Perseus* (JAIR 2005). Kurniawati, Hsu & Lee, *SARSOP* (RSS 2008).
- **Polynomial DP / sum-of-squares.** Parrilo, *Structured semidefinite programs and semialgebraic geometry methods* (PhD thesis, 2000). Henrion, Lasserre, et al., various SOS-for-control papers.
- **Symmetry reduction.** Zinkevich & Balch, *Symmetry in Markov decision processes* (UAI 2001). Narayanamurthy & Ravindran, *On the hardness of finding symmetries in Markov decision processes* (ICML 2008).
- **HJB with analytical solutions.** Merton, *Optimum consumption and portfolio rules in a continuous-time model* (1971). Duffie, *Dynamic Asset Pricing Theory* (2001).
- **Monotone / threshold DPs.** Topkis, *Minimizing a submodular function on a lattice* (OR 1978). Gittins, *Bandit processes and dynamic allocation indices* (JRSS-B 1979).
- **Companion paper:** `doc/paper/paper.pdf` — the hierarchy of §5 where this note would slot in as "rung 0."
- **Option B note:** `doc/option_b_end2end_implementation.md` — the NN-based general-purpose alternative for problems without a rung-0 structure.
