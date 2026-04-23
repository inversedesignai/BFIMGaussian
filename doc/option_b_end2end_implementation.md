# End-to-end joint-DP co-design with a neural value function (Option B)

Implementation note for the photonic topology optimization problem.

## Context

Starting point: the joint-DP framework of the companion paper. Inner problem is a finite-horizon Bayesian DP; outer problem is gradient descent on the physical geometry `c` (here, a 90,000-pixel permittivity distribution). When exact DP over reachable beliefs is infeasible, the paper's §5 hierarchy offers four relaxations (rungs 1–4), all prescribing a closed-form inner policy.

Option B is a different kind of relaxation: instead of prescribing the inner policy, **parameterize the value function with a neural network and train the network, the hardware, and the inner policy jointly end-to-end** against the deployment-aligned loss. The Bellman structure is preserved (backward induction, Bayesian belief update, argmax policy); only the value function's representation is parametric.

This note is deliberately agnostic to the choice of belief representation. The framework works with any Bayesian filter `T` that is differentiable end-to-end: exact posterior on a grid, particle filter, moment-propagation filter (EKF/UKF), Fourier-basis filter, or a learned sufficient-statistic encoder. The only requirements are: (i) the belief is summarizable by a finite-dimensional vector of known shape; (ii) `T` is differentiable (via closed form, via a reparameterization trick, or via a custom rrule).

This note is a pragmatic implementation sketch specialized to photonic topology optimization. It assumes familiarity with the existing FDFD + Taylor-expansion pipeline in [SimGeomBroadBand.jl](../SimGeomBroadBand.jl).

## Summary

Parameters optimized jointly:

| Parameter | Shape | Role |
|---|---|---|
| `c` | `300 × 300` pixels (90,000 scalars) | Physical geometry (permittivity distribution). |
| `θ` | NN weights (small MLP) | Shared feature encoder: maps belief summary → learned feature vector. |
| `w_k` | one vector per epoch `k = 0, ..., K-1` | Per-layer linear value-function head. |

Loss (end-to-end):

```
L(θ, w, c) = E_{x_0, y_{1:K}} [ Φ_0(x_0, b_K; c) ]
```

the expected pointwise reward at the terminal belief `b_K`, under the inner argmax policy derived from the NN value function. The reward `Φ_0` is whatever the deployment metric of the application dictates — posterior-variance (Bayesian MSE), log-determinant of the posterior Fisher matrix, negative posterior entropy, etc.

Training:

```
for outer_iter in 1..N:
    (θ, w, c) <- (θ, w, c) - η · ∇_{(θ, w, c)} L̂
```

where `L̂` is a Monte-Carlo estimate of `L` over a mini-batch of episodes, and the gradient is computed by reverse-mode AD through the fully-unrolled differentiable DP pipeline, with envelope-theorem handling of the per-step argmax.

## Architecture

### 1. Belief representation

The belief `b_k` at epoch `k` is an application-chosen object. Four common choices and their belief-summary vectors `b_vec ∈ ℝ^{d_b}`:

| Belief representation | `b_vec` | Differentiable? |
|---|---|---|
| Exact posterior on a state grid | Values of `b_k` at `N_grid` points | Yes, straightforward. |
| Gaussian moments (EKF/UKF) | `(μ_k, vec(chol Σ_k))`, `d_b = d_x + d_x(d_x+1)/2` | Yes, via Joseph-form updates. |
| Particle set `{(x_i, w_i)}_{i=1}^N` | Sorted summary: sample quantiles + moments; or permutation-invariant pooling (Deep Sets) | Partial — resampling step needs Gumbel or straight-through. |
| Orthogonal-basis expansion | Coefficients in a chosen basis (e.g., Fourier, Hermite) | Yes, via closed-form projection updates. |

Pick one representation and stick with it for the entire training run; the NN encoder is tied to the chosen `b_vec` layout. Mixing representations across training phases requires retraining the encoder.

For the photonic topology problem, we flag the three most relevant options:

- **Particle filter** with `N_particles ≈ 10^3–10^4`, if the true posterior is multimodal (e.g., under degenerate scattering geometries). Permutation-invariant pooling on the particle set gives `b_vec`.
- **Grid** with `N_grid = 256` points per state dimension, if `d_x` is small (`d_x ≤ 4` in Case C).
- **Moment-based** (Gaussian), if the posterior is well-approximated by a single Gaussian. Cheapest; fewer dimensions in `b_vec`.

The rest of this note treats `b_vec` as a black-box `d_b`-dim vector produced by the chosen representation. Swap implementations without touching the downstream architecture.

### 2. Shared feature encoder `φ_θ`

A small MLP:

```
φ_θ :  ℝ^{d_b}   →   ℝ^{d_feat}
```

with `d_feat = 64` or `128`. Architecture: 3–4 hidden layers, width 128–256, with GELU or SiLU activations. No dropout (deterministic value approximation).

**Why a shared encoder.** The value function at different `k` should share structure — the belief means the same thing at every epoch; what differs is the continuation horizon. A shared feature encoder `φ_θ` with per-epoch linear heads `w_k` captures this without forcing all `V_k` to be the same function.

**Hardware `c` does not enter `φ_θ`.** `c` enters the computation graph through the Bayesian update (via the FDFD-derived likelihood) and the action-selection argmax. The feature encoder sees only the belief summary; it does not see `c` directly. This matters for generalization: you want `φ_θ` trained once to work across all `c` visited during outer-loop optimization, not retrained at every outer step.

### 3. Per-layer linear heads `w_k`

For each epoch `k`:

```
V_{θ, w}(b_k, k)  =  w_k · φ_θ(b_vec(b_k))
```

`w_k ∈ ℝ^{d_feat}`, so `K` heads total. Storage is negligible compared to `θ` or `c`.

### 4. Policy

The inner policy is the Bellman argmax over the NN Q-function:

```
Q_{θ, w}(b_k, s, k)  =  E_{y_{k+1} | b_k, s, c} [ w_{k+1} · φ_θ(b_vec(T(b_k, s, y_{k+1}; c))) ]

s_k*  =  argmax_{s ∈ 𝒮} Q_{θ, w}(b_k, s, k)
```

For continuous action space `𝒮`, use grid search + L-BFGS local refinement (same as the current pipeline). For discrete `𝒮`, enumerate.

The inner expectation over `y_{k+1}` is computed either by closed-form marginalization (available when the belief representation and likelihood are conjugate, e.g., Gaussian-Gaussian), by low-order quadrature (Gauss–Hermite for continuous `y` with smooth likelihoods), or by Monte Carlo with a modest sample count (≈ 8–16 reparameterized samples).

## Loss function

### Deployment-aligned objective

```
L(θ, w, c)  =  E_{x_0 ~ p_0}\, E_{y_{1:K} | x_0, π_{θ,w}, c}
                  [ Φ_0(x_0, b_K; c) ]
```

— the expected pointwise reward at the terminal belief `b_K`, averaged over the prior and the data, under the NN-argmax policy `π_{θ, w}`.

`Φ_0` is the application-chosen reward that the deployment metric reduces to. For a few common deployment metrics:

- **Bayesian MSE** (posterior-mean estimator deployed on a continuous state): `Φ_0 = -(μ̂_K - x_0)² / x_0²` (relative-squared error, following the existing Case C convention).
- **Mutual information** (discrete state, classification-like deployment): `Φ_0 = log b_K(x_0) - log p_0(x_0)`.
- **Fisher-log-det** (asymptotic-regime surrogate): `Φ_0 = ½ log det J_K(x_0, s_{1:K}, c)` with `J_K` the accumulated Fisher-information matrix (belief-augmentation required, see §7 of the companion tutorial).

Pick whichever matches the deployment metric of the application. All three fit the same differentiable-DP machinery.

### Optional auxiliary Bellman residual

For stability, add a regularizer:

```
L_bellman(θ, w, c)  =  (1/K) Σ_{k=0}^{K-1} E_{b_k} [ (w_k · φ_θ(b_vec(b_k)) - ỹ_k)² ]

ỹ_k  =  max_s E_{y_{k+1} | b_k, s, c} [ w_{k+1} · φ_θ(b_vec(T(b_k, s, y_{k+1}; c))) ]    (target; detached from gradient)
```

with `ỹ_K := E_{x | b_K}[Φ_0(x, b_K; c)]` (terminal reward). Total loss:

```
L_total  =  L_deployment  +  λ · L_bellman
```

with `λ ~ 0.01 – 0.1` (tune).

**Why the auxiliary loss helps.** Without it, `V_{θ, w}` only has to produce the right argmax — its absolute values are unconstrained, and the network can drift into regimes where the value landscape is non-informative even though the argmax is correct. The Bellman residual anchors the value function to be Bellman-consistent, which keeps the gradient signal stable across outer iterations. Early in training the deployment loss has high variance; the Bellman residual provides a dense, low-variance signal that regularizes the feature encoder.

The target `ỹ_k` is computed with the *current* `(θ, w, c)` but treated as a constant (detached / `stop_gradient`) when computing the gradient — standard bootstrapped-target practice. In Julia, wrap the target computation in `Zygote.ignore(...)`.

## Differentiable DP unrolling

The full computation graph from `(c, θ, w)` to `L_deployment`:

```
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT:  c (90K pixels),  θ (NN weights),  w_{0:K-1} (per-k heads)   │
└──────────────────────────────────────────────────────────────────────┘
          │
          │  density filter + tanh(β·) projection              ← standard
          ▼
      c_proj (binary-ish pattern)
          │
          │  FDFD solve at 20 frequencies, per-cell S-matrix Taylor expansion
          ▼
      {S_0, ∂S/∂n, ∂²S/∂n²} at each cell, each frequency      ← c-dependent primitive
          │
          │  sample x_0 ~ p_0
          │  initialize b_0 = p_0
          ▼
┌──────── DP loop (k = 0, 1, ..., K-1) ────────────────────────────────┐
│                                                                      │
│  for each candidate s in grid × LBFGS over actions:                  │
│      compute E_{y|b_k, s, c}[w_{k+1} · φ_θ(b_vec(T(b_k, s, y; c)))]  │
│          via conjugate marginalization, quadrature, or MC            │
│      Q_k(s) = that expectation                                       │
│  s_k* = argmax_s Q_k(s)               ← record, freeze in gradient   │
│                                                                      │
│  sample y_{k+1} ~ p(·|b_k, s_k*, c)   ← reparameterized              │
│  b_{k+1} = T(b_k, s_k*, y_{k+1}; c)   ← Bayesian update              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
          │
          ▼
      Φ_0(x_0, b_K; c)
          │
          ▼
      L_deployment
```

Three points where differentiability is non-trivial.

### Argmax over `s` (envelope theorem, sharp max)

Use the paper's §4 recipe:

1. **Forward (no AD tape):** grid search + LBFGS to find `s_k*`.
2. **Gradient (AD tape):** re-evaluate `Q_k(s_k*)` with `s_k*` substituted for the max, inside an AD-traced region.

This avoids differentiating through the argmax and gives the exact envelope-theorem gradient for the outer parameters `(θ, w, c)` at the Bellman-optimal `s_k*`.

In Julia/Zygote:

```julia
# Forward pass: find argmax, no AD tape active
s_star = Zygote.ignore() do
    find_argmax(b_k, c, w_next, θ, k)  # grid + LBFGS
end

# Gradient pass: evaluate integrand with s_star substituted, AD active
Q_at_sstar = expected_next_value(b_k, s_star, c, w_next, θ)
```

### Observation sampling `y_{k+1}`

Sample `y_{k+1}` via reparameterization appropriate to the likelihood family, so gradient flows through `c` via the likelihood's parameters:

- **Continuous `y` with Gaussian likelihood.** `y_{k+1} = mean_y(b_k, s, c) + chol(Σ_y(b_k, s, c)) · ε` with `ε ~ N(0, I)` drawn outside the tape.
- **Continuous `y` with non-Gaussian likelihood.** Use inverse-CDF if the CDF is differentiable: `y_{k+1} = F^{-1}(u; b_k, s, c)` with `u ~ U(0, 1)`.
- **Discrete `y`.** Use Gumbel-softmax with a small temperature τ (anneal τ → 0 across training), or marginalize analytically if the outcome alphabet is small.

Draw the random noise once per episode outside the AD tape; compute `y_{k+1}` as a deterministic function of `(b_k, s_k*, c, noise)`. Gradient flows through the likelihood parameters, which depend on `c` through the FDFD S-matrices.

### Belief update `T` (Bayesian)

Whatever filter you chose in §2.1, it must be differentiable in `(b_k, s_k*, y_{k+1}, c)`. Four options recap:

- **Grid / exact.** Bayes' rule as pointwise multiplication with the likelihood, followed by normalization. Fully differentiable under Zygote; no custom rrule needed.
- **Particle filter.** Weight update is differentiable; resampling is not. Use differentiable resampling (Gumbel-softmax over categorical) or straight-through estimator during backprop. Alternatively, avoid resampling during training (accept particle-weight degeneracy for the training objective).
- **Moment-propagation (EKF/UKF).** Closed-form Jacobian / sigma-point updates; fully differentiable via standard linear algebra rrules.
- **Basis expansion.** Project the posterior onto the basis after multiplying by the likelihood. Differentiable if the projection is closed-form.

Write `T` as a clean Julia function `belief_update(b_k, s_k, y_k, c) -> b_{k+1}` and Zygote will handle the rest (possibly with a custom rrule for resampling if you go the particle-filter route).

## On the implicit function theorem

The existing Case C pipeline in [BFIMGaussian.jl](../BFIMGaussian.jl) uses an IFT-based custom rrule for the inner sensor-parameter argmax (`_get_sopt` and its pullback). A natural question: is IFT involved in Option B?

**Short answer: not strictly, though it could be.**

In Option B as described, the envelope-theorem argmax-freeze recipe (§5.1) handles the inner optimization without IFT. The forward pass records `s_k*` with no tape; the gradient pass substitutes it in and runs reverse-mode AD. No Hessian-vector products; no custom rrule for the argmax.

**When envelope-freeze is exact.** At convergence of the NN, if `Q_{θ,w}` is the true Bellman Q-function for the deployment loss, then `s_k* = argmax Q_{θ,w}` is the true Bellman-optimal action. The envelope theorem then gives the exact outer gradient with `s_k*` frozen — identical to what IFT would produce at the inner argmax. No contribution from `ds_k*/dc` is needed.

**When envelope-freeze is approximate.** Away from NN convergence, `Q_{θ,w}` is an approximation to the true Bellman Q, and `s_k* = argmax Q_{θ,w}` is a sub-optimal action w.r.t. the deployment loss. The exact outer gradient has a correction term `(∂L_deployment/∂s_k*) · (ds_k*/dc)` that the envelope-freeze recipe drops. This correction is first-order small in the NN approximation error (paper §4.3 point 2) and shrinks as training converges, but it is not literally zero at intermediate iterations.

**When to switch to IFT.** If you want exactness even at non-converged NN, differentiate through the inner argmax via the implicit function theorem, reproducing Case C's machinery:

```
ds_k*/dc       =  -(∂²Q_{θ,w}/∂s²)⁻¹ · (∂²Q_{θ,w}/∂s∂c)
ds_k*/dθ       =  -(∂²Q_{θ,w}/∂s²)⁻¹ · (∂²Q_{θ,w}/∂s∂θ)
ds_k*/dw_{k+1} =  -(∂²Q_{θ,w}/∂s²)⁻¹ · (∂²Q_{θ,w}/∂s∂w_{k+1})
```

and chain-rule these into the outer gradient. Trades the simplicity of the envelope-freeze recipe for an exact gradient at the cost of a Hessian-vector product at every inner argmax (computable via `Zygote.hessian` or Hessian-free methods). The `get_sopt` pullback in [BFIMGaussian.jl](../BFIMGaussian.jl) already does this for Case C's BFIM-trace objective; for Option B you'd generalize its inner objective to be the NN Q-function instead.

**Recommendation.** Start with the envelope-freeze recipe. Switch to IFT only if training is unstable in a way that suggests gradient bias from sub-optimal inner argmax — usually visible as outer-loop oscillation that does not decrease when you lower the learning rate. For most problems, the first-order correction that IFT would add is smaller than the Monte-Carlo noise in the batch-wise deployment loss estimate, and IFT's overhead buys nothing.

**Not an IFT setup: the training is joint, not bi-level.** `(θ, w, c)` are all free variables updated simultaneously. There is no outer-over-inner loop that would make `(θ, w)` implicit functions of `c`. If you instead chose a bi-level setup — inner: fit `(θ, w)` to convergence given `c`; outer: update `c` given the converged inner fit — then `dθ*/dc` and `dw*/dc` would be proper IFT gradients through the inner fit's first-order condition `∂L_bellman/∂(θ, w) = 0`. This is the classical hyperparameter-optimization recipe (Pedregosa 2016, Blondel et al. 2022's JAXopt), and works but adds a layer of complexity that joint training usually avoids without sacrificing optimality.

**Not an IFT setup: the belief update `T`.** `T` is closed-form Bayes' rule — pointwise multiplication of the prior by the likelihood, followed by normalization. No inner optimization, no implicit function, no IFT. Zygote's default rules handle it via division-through-normalizer. The only exception is particle-filter resampling, which requires a custom rule for different reasons (categorical sampling), not IFT.

## Integration with the existing photonic forward model

The current pipeline in [SimGeomBroadBand.jl](../SimGeomBroadBand.jl) provides:

1. **`batch_solve`** — parallel multi-frequency FDFD with the custom `rrule` for Wirtinger adjoint. Returns S-matrices; gradient w.r.t. `ε_geom` available.
2. **`getSmatrices`** — extracts normalized S, dS/dn, d2S/dn2.
3. **`powers_only`** — per-cell powers from the Taylor expansion.
4. Existing rungs' belief-update primitives (EKF in [BFIMGaussian.jl](../BFIMGaussian.jl); A-optimal covariance update in [PosteriorCovGaussian.jl](../PosteriorCovGaussian.jl)).
5. **`get_sopt`** + IFT rrule — inner sensor-parameter argmax, differentiated via the implicit function theorem.

Option B reuses (1)–(3) as-is (the photonic physics is unchanged). It replaces (5)'s objective with a callback-style NN objective, and replaces (4) with whichever filter the chosen belief representation requires:

```
s_opt_NN(b_k, c, w_{k+1}, θ)  =  argmax_s E_{y | b_k, s, c}[ w_{k+1} · φ_θ(b_vec(T(b_k, s, y; c))) ]
```

The IFT rrule structure is the same — implicit differentiation at the inner argmax — but the cotangent propagation now flows through the NN and into `(θ, w_{k+1})` in addition to `c`.

**Action items for integration:**

1. Port `get_sopt` to a version that takes a callback objective as argument. Grid search + LBFGS unchanged.
2. Ensure the custom IFT rrule's Hessian computation handles the NN objective's second derivatives (use a Hessian-vector product via Zygote-over-Zygote or a structured approximation).
3. Implement the belief-update function `T` for whichever representation you chose in §2.1. Register a custom rrule if reverse-mode AD through it needs help (e.g., particle-filter resampling).
4. Replace the terminal reward with the application-chosen `Φ_0`.

## Training procedure

### Optimizer

- **Outer**: Adam, learning rate `1e-3` to `1e-4`, with per-component box-width preconditioning on `c` (as in existing Case C). Adam also on `θ` and `w`.
- **Joint or alternating?** Start jointly — all three parameter groups updated every step. If training is unstable, switch to alternating: 1 outer `c` step per 10 inner `(θ, w)` steps (inner steps use fresh episodes but fixed `c`).

### Mini-batching

- **Inner loop:** `n_episodes = 20` per gradient estimate (same as existing Case C). Each episode is one draw of `(x_0, noise_{1:K})` and one forward pass through the DP.
- **Outer loop:** `N_outer = 10^3 – 10^4` iterations.

### Resampling schedule

- Fresh `(x_0, noise_{1:K})` every outer iteration (`resample_every=1`).
- Fresh FDFD solve every outer iteration (can't be avoided — `c` changes, so S-matrices change).

### Initialization

- `c`: random binary-like permittivity, passed through the density filter at `β = 16` (warm start).
- `θ`: Xavier / Glorot initialization for MLP weights.
- `w_k`: zero-initialized. This starts the NN with a trivial value function; the first few outer iterations bootstrap it via the Bellman residual.

### Continuation on `β`

Same as the current Case C pipeline: `β` increases along a schedule `16 → 32 → 64 → 128 → 256` triggered by loss plateaus. This controls the density-filter + tanh projection sharpness and is essential for producing a manufacturable binary final geometry.

### Warm-start from an existing rung

One practical shortcut: initialize `θ` and `w_k` by first training a few hundred iterations of an existing rung (e.g., rung 4's BFIM-trace + EKF) to convergence, then switch to Option B training. This gives the NN a sensible starting value function based on the rung's surrogate reward, after which end-to-end training refines it toward the true deployment objective. Expect a 2× speedup in convergence.

## Gradient flow

```
L_deployment  ← Φ_0(x_0, b_K; c)
    ↑
    │  through:  b_K (terminal belief)
    │            ← belief updates T at k=K-1, K-2, ..., 0
    │            ← each T depends on:  b_{k-1}, s_k*, y_k, c
    │            ← s_k* is frozen (argmax recorded in forward, substituted in gradient)
    │            ← y_k = reparameterized sample from the likelihood at c
    │            ← the likelihood depends on c via S-matrix Taylor coefficients
    │
    │  through:  the Q-function Q_k(s_k*) = w_{k+1} · φ_θ(b_vec(T(b_k, s_k*, y_k; c)))
    │            (appears implicitly via the argmax-frozen recipe)
    │
    ├→ ∇_c L   ... propagates through T and the S-matrix chain
    │            (uses FDFD adjoint)
    │
    ├→ ∇_θ L   ... propagates through φ_θ wherever the NN value is invoked
    │            (standard reverse-mode AD; one backward pass per Zygote call)
    │
    └→ ∇_{w_k} L  ... through the k-th linear head
                 (outer product with the incoming cotangent)
```

## Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| **Hardware-light pathology** | `c` stays near initialization while NN overfits. | Bellman residual regularizer; per-component box-width preconditioning on `c`; two-timescale training (`c` on faster clock than `θ`). |
| **Value-function collapse** | `V_{θ,w}(b) ≈ const` across beliefs — no signal for argmax. | Auxiliary Bellman residual loss; terminal anchor `ỹ_K = E_{x \| b_K}[Φ_0]`. |
| **Non-monotone loss** | Deployment loss oscillates as outer iterations progress. | Lower `η`; policy-iteration warm-starting; larger `n_episodes`. |
| **FDFD gradient explosion** | `∇_c L` has occasional huge values. | Gradient clipping; density filter acts as a regularizer; check for resonances in the geometry. |
| **Binary-gap**: loss spikes during β-continuation | Loss plateaus then spikes at each β doubling. | Expected — existing autotune schedule handles this. |
| **Argmax-frozen gradient mismatch** | Gradient doesn't match finite-differences. | Verify with FD check at small `c` perturbation; the envelope theorem gradient is exact only at the true argmax; ensure the LBFGS inner-step converges tightly. |
| **Particle-filter gradient leak** (if using particle belief) | Backprop returns NaN or zero through the resampling step. | Use Gumbel-softmax resampling with small τ; or skip resampling during training and accept particle-weight degeneracy. |
| **Grid-belief memory blow-up** (if using grid belief) | OOM on larger `d_x` or finer `N_grid`. | Fall back to moment-based or particle representation for larger `d_x`. |

## Diagnostics

Track these every outer iteration:

1. **Deployment loss** — primary metric, moving average over 20 iterations.
2. **Bellman residual** — if using auxiliary loss, should decrease roughly monotonically.
3. **Terminal belief quality** — application-specific: posterior variance, entropy, or `Φ_0` value.
4. **Mean argmax action `s_k*`** — should shift toward a structured pattern; watch for policy collapse (same `s*` for all `b`).
5. **Binary fraction of `c_proj`** — at each β, track fraction of pixels that are `> 0.99` or `< 0.01` (cleanly binary). Target `> 95%` at `β = 256`.
6. **Gradient norms `||∇_c L||, ||∇_θ L||, ||∇_w L||`** — relative magnitudes inform preconditioning.

Monte-Carlo evaluation every 100 outer iterations:

- Deploy current `(c, θ, w)` on 200 episodes with `K_deploy` measurement steps.
- Compare to the best existing rung on the same episode seeds.
- Report the deployment metric (MSE, MI, log-det-FIM — whichever matches `Φ_0`).

## Code scaffold

Pseudo-Julia, with the belief representation treated as a plug-in interface:

```julia
# --- plug-in belief interface ---
abstract type AbstractBelief end

belief_to_vector(b::AbstractBelief) = # returns b_vec ∈ ℝ^{d_b}
belief_update(b::AbstractBelief, s, y, c) = # returns b_next  (differentiable)
belief_to_reward(b::AbstractBelief, x0, c) = # Φ_0 evaluated at terminal belief
sample_observation(b::AbstractBelief, s, c, noise) = # reparameterized y

# --- value function ---
struct ValueNN
    encoder    # Flux.Chain for φ_θ: ℝ^{d_b} → ℝ^{d_feat}
    heads      # Vector of Vector{Float32}, one per k
end

v_value(vnn, b, k) = dot(vnn.heads[k+1], vnn.encoder(belief_to_vector(b)))

function q_value(vnn, b, s, c, k; n_samples = 8)
    # Estimate E_{y | b, s, c}[V(T(b, s, y; c), k+1)]
    # by reparameterized quadrature or Monte Carlo
    return mean([begin
        noise = sample_noise()  # outside tape, in caller
        y = sample_observation(b, s, c, noise)
        b_next = belief_update(b, s, y, c)
        v_value(vnn, b_next, k+1)
    end for _ in 1:n_samples])
end

# --- inner argmax with envelope-theorem freeze ---
function inner_argmax_NN(b, c, vnn, k)
    s_star = Zygote.ignore() do
        s_grid = action_grid()
        s0 = argmax(s -> q_value(vnn, b, s, c, k), s_grid)
        lbfgs_refine(s -> -q_value(vnn, b, s, c, k), s0)
    end
    q_at_sstar = q_value(vnn, b, s_star, c, k)
    return s_star, q_at_sstar
end

# --- episode forward pass ---
function episode_loss(vnn, c, x0, noise_traj)
    b = prior_belief(x0)  # initial belief (state-independent in practice)
    for k in 0:(K-1)
        s_star, _ = inner_argmax_NN(b, c, vnn, k)
        y = sample_observation(b, s_star, c, noise_traj[k+1])
        b = belief_update(b, s_star, y, c)
    end
    return -belief_to_reward(b, x0, c)  # negate if minimizing
end

# --- outer loop ---
function outer_loop(vnn, c, opt_state, n_outer; λ_bellman = 0.05)
    for it in 1:n_outer
        x0_batch = [sample_prior() for _ in 1:n_episodes]
        noise_batch = [[sample_noise() for _ in 1:K] for _ in 1:n_episodes]

        loss_fn = (vnn, c) -> begin
            c_proj = project_and_filter(c, current_β)
            # Recompute FDFD + Taylor at new c
            smatrices = batch_solve(c_proj)
            L_dep = mean(episode_loss(vnn, smatrices, x0_batch[i], noise_batch[i])
                         for i in 1:n_episodes)
            if λ_bellman > 0
                L_bel = bellman_residual(vnn, smatrices, x0_batch, noise_batch)
                return L_dep + λ_bellman * L_bel
            end
            return L_dep
        end

        grads = Zygote.gradient(loss_fn, vnn, c)
        update!(opt_state, (vnn, c), grads)
        maybe_advance_β!()
    end
end
```

The five `belief_*` functions are the only things that change when swapping belief representations. Everything downstream is agnostic.

## Extensions

- **Larger `K_train`.** Current Case C uses `K_train = 3`. End-to-end training scales more gracefully with `K` than exact DP does, so `K_train = 10` or more is plausible.
- **Multimodal belief.** Particle filter with permutation-invariant encoder (Deep Sets) as `belief_to_vector`. Requires differentiable resampling (Gumbel-softmax) or straight-through estimator.
- **Dynamical state.** Extend `T` with a state-transition model for drifting perturbations; the framework survives unchanged if the belief update remains differentiable.
- **Continuous action space with reparameterized policy.** Replace grid + LBFGS argmax with a stochastic policy parameterized by its own small NN, trained by reparameterization + Gumbel-softmax. Moves toward actor-critic but keeps the exact value-function structure.
- **Learned sufficient-statistic encoder.** If the chosen belief representation does not fit the problem well, train a separate encoder that maps raw observation histories directly to `b_vec`. Sacrifices exact Bayesian-filter interpretability for representational flexibility.

## Pre-implementation checklist

- [ ] Choose a belief representation (§2.1) and implement the four plug-in functions: `belief_to_vector`, `belief_update`, `belief_to_reward`, `sample_observation`. Finite-difference check each.
- [ ] Verify that `belief_update` propagates gradients cleanly end-to-end (write an FD check on a toy `(b, c)` pair).
- [ ] Profile `q_value` — the inner expectation over `y` is the hottest loop. 8-point Gauss–Hermite should give adequate accuracy for smooth continuous likelihoods; benchmark vs. MC with 50+ samples.
- [ ] Confirm that `Zygote.ignore()` around the LBFGS inner-step actually prevents the tape from growing (Zygote has had known bugs here in older versions).
- [ ] Match the current Case C's preconditioner for `c` (box-width normalization); confirm it's correct for the new loss.
- [ ] Plan a baseline comparison: Option B vs. the existing best rung on identical physical parameters, 200-episode MC deployment. Expect Option B to match or beat the baseline at sufficient training.

## References

- Companion paper: `doc/paper/paper.pdf` — especially §4 (differentiable DP via envelope theorem) and §5 (relaxation hierarchy).
- Companion tutorial: `doc/tutorial.pdf` — §3.3 Step 4 for the envelope identity, §7 for belief augmentation, §8.2 for the `∂V/∂c` unfolding.
- Case C codebase: [BFIMGaussian.jl](../BFIMGaussian.jl), [PosteriorCovGaussian.jl](../PosteriorCovGaussian.jl), [SimGeomBroadBand.jl](../SimGeomBroadBand.jl), [PhEnd2End.jl](../PhEnd2End.jl).
- Classical FVI and its deep variants: Bertsekas & Tsitsiklis, *Neuro-Dynamic Programming* (1996); Riedmiller, *Neural Fitted Q Iteration* (2005); Ernst et al., *Tree-Based Batch Mode RL* (2005).
- Envelope theorem: Milgrom & Segal, *Envelope Theorems for Arbitrary Choice Sets* (2002).
