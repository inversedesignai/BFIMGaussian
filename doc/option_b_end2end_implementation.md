# End-to-end joint-DP co-design with a neural value function (Option B)

Implementation note for the photonic topology optimization problem.

## Context

Starting point: the joint-DP framework of the companion paper. Inner problem is a finite-horizon Bayesian DP; outer problem is gradient descent on the physical geometry `c` (here, a 90,000-pixel permittivity distribution). When exact DP over reachable beliefs is infeasible, the paper's §5 hierarchy offers four relaxations (rungs 1–4), all prescribing a closed-form inner policy.

Option B is a different kind of relaxation: instead of prescribing the inner policy, **parameterize the value function with a neural network and train the network, the hardware, and the inner policy jointly end-to-end** against the deployment-aligned loss. The Bellman structure is preserved (backward induction, Bayesian belief update, argmax policy); only the value function's representation is parametric.

This note is a pragmatic implementation sketch specialized to photonic topology optimization. It assumes familiarity with the existing FDFD + Taylor-expansion + EKF pipeline in [BFIMGaussian.jl](../BFIMGaussian.jl) and [SimGeomBroadBand.jl](../SimGeomBroadBand.jl).

## Summary

Parameters optimized jointly:

| Parameter | Shape | Role |
|---|---|---|
| `c` | `300 × 300` pixels (90,000 scalars) | Physical geometry (permittivity distribution). |
| `θ` | NN weights (small MLP) | Shared feature encoder: maps belief summary → learned feature vector. |
| `w_k` | one vector per epoch `k = 0, ..., K-1` | Per-layer linear value-function head. |

Loss (end-to-end):

```
L(θ, w, c) = E_{x_0, y_{1:K}} [ || μ_K - x_0 ||² / || x_0 ||² ]
```

the expected relative squared error of the EKF posterior-mean estimator after `K` measurement epochs, under the inner argmax policy derived from the NN value function.

Training:

```
for outer_iter in 1..N:
    (θ, w, c) <- (θ, w, c) - η · ∇_{(θ, w, c)} L̂
```

where `L̂` is a Monte-Carlo estimate of `L` over a mini-batch of episodes, and the gradient is computed by reverse-mode AD through the fully-unrolled differentiable DP + EKF pipeline, with envelope-theorem handling of the per-step argmax.

## Architecture

### 1. Belief summary

The belief `b_k` at epoch `k` is represented by the EKF's Gaussian moments:

```
b_k  =  (μ_k, Σ_k)      μ_k ∈ ℝ⁴,   Σ_k ∈ ℝ^{4×4} symmetric PSD
```

This is the same representation used in Case C of the paper. For a 4-dim state, this is a 4 + 10 = 14-dim summary (mean + 10 unique entries of the Cholesky factor of `Σ_k`).

**Recommended input encoding for the NN:**

```
b_k_vec  =  concat(μ_k, vec_lower_tri(chol(Σ_k)), log(trace(Σ_k)))
```

— flatten the mean and the Cholesky factor of the covariance, plus one auxiliary log-scale feature to give the network an easy handle on uncertainty magnitude.

### 2. Shared feature encoder `φ_θ`

A small MLP:

```
φ_θ :  ℝ^{14}   →   ℝ^{d_feat}
```

with `d_feat = 64` or `128`. Architecture: 3–4 hidden layers, width 128–256, with activation functions that preserve positivity/monotonicity where useful (e.g., GELU or SiLU). No dropout (deterministic value approximation).

**Why a shared encoder.** The value function at different `k` should share structure — the belief means the same thing at every epoch; what differs is the continuation horizon. A shared feature encoder `φ_θ` with per-epoch linear heads `w_k` captures this without forcing all `V_k` to be the same function.

**Hardware `c` does not enter `φ_θ`.** `c` enters the computation graph through the EKF (via the FDFD-derived S-matrix derivatives) and the action-selection argmax. The feature encoder sees only the belief summary; it does not see `c` directly. This matters for generalization: you want `φ_θ` trained once to work across all `c` visited during outer-loop optimization, not retrained at every outer step.

### 3. Per-layer linear heads `w_k`

For each epoch `k`:

```
V_{θ, w}(b_k, k)  =  w_k · φ_θ(b_k)
```

`w_k ∈ ℝ^{d_feat}`, so `K` heads total (`K` = 3 in Case C). Storage is negligible compared to `θ` or `c`.

### 4. Policy

The inner policy is the Bellman argmax over the NN Q-function:

```
Q_{θ, w}(b_k, s, k)  =  E_{y_{k+1} | b_k, s, c} [ w_{k+1} · φ_θ(T(b_k, s, y_{k+1}; c)) ]

s_k*  =  argmax_{s ∈ 𝒮} Q_{θ, w}(b_k, s, k)
```

Computed by grid search + L-BFGS local refinement (same as the current Case C pipeline for the BFIM-trace argmax). The inner optimization is `c`-dependent via `T`.

## Loss function

### Deployment-aligned objective

```
L(θ, w, c)  =  E_{x_0 ~ p_0} E_{y_{1:K} | x_0, π_{θ,w}, c}
                  [ || μ_K - x_0 ||² / || x_0 ||² ]
```

— the relative squared error of the posterior-mean estimator at the terminal belief, averaged over the prior and the data, under the NN-argmax policy.

Relative (not absolute) squared error is already the Case C convention. It handles the small-`Dn` regime where absolute error would be dominated by the problem's overall scale.

### Optional auxiliary Bellman residual

For stability, add a regularizer:

```
L_bellman(θ, w, c)  =  (1/K) Σ_{k=0}^{K-1} E_{b_k} [ (w_k · φ_θ(b_k) - ỹ_k)² ]

ỹ_k  =  max_s E_{y_{k+1} | b_k, s, c} [ w_{k+1} · φ_θ(T(b_k, s, y_{k+1}; c)) ]    (target; detached from gradient)
```

with `ỹ_K := -trace(Σ_K)` (terminal reward, negative-posterior-variance). Total loss:

```
L_total  =  L_deployment  +  λ · L_bellman
```

with `λ ~ 0.01 – 0.1` (tune).

**Why the auxiliary loss helps.** Without it, `V_{θ, w}` only has to produce the right argmax — its absolute values are unconstrained, and the network can drift into regimes where the value landscape is non-informative even though the argmax is correct. The Bellman residual anchors the value function to be Bellman-consistent, which keeps the gradient signal stable across outer iterations. Early in training the deployment loss has high variance; the Bellman residual provides a dense, low-variance signal that regularizes the feature encoder.

The target `ỹ_k` is computed with the *current* `(θ, w, c)` but treated as a constant (detached / `stop_gradient`) when computing the gradient — this is standard bootstrapped-target practice. In Julia, use `Zygote.ignore(...)` or `StopGradient` around the target computation.

## Differentiable DP unrolling

The full computation graph from `c, θ, w` to `L_deployment`:

```
┌───────────────────────────────────────────────────────────────────────┐
│  INPUT:  c (90K pixels),  θ (NN weights),  w_{0:K-1} (per-k heads)    │
└───────────────────────────────────────────────────────────────────────┘
          │
          │  density filter + tanh(β·) projection              ← standard
          ▼
      c_proj (binary-ish pattern)
          │
          │  FDFD solve at 20 frequencies, per-cell S-matrix Taylor expansion
          ▼
      {S_0, ∂S/∂n, ∂²S/∂n²} at each cell, each frequency       ← c-dependent primitive
          │
          │  sample x_0 ~ p_0 (prior over Δn ∈ ℝ⁴)
          │  initialize b_0 = p_0 (Gaussian)
          ▼
┌─────── DP loop (k = 0, 1, ..., K-1) ──────────────────────────────────┐
│                                                                       │
│  for each candidate s in grid × LBFGS over (φ_1, φ_2):                │
│      compute E_{y|b_k, s, c}[w_{k+1} · φ_θ(T(b_k, s, y; c))]          │
│          via Gaussian-Gaussian marginalization (closed form for EKF)  │
│      Q_k(s) = that expectation                                        │
│  s_k* = argmax_s Q_k(s)               ← record, freeze in gradient    │
│                                                                       │
│  sample y_{k+1} ~ p(·|b_k, s_k*, c)   ← stochastic, reparameterize    │
│  b_{k+1} = EKF_update(b_k, s_k*, y_{k+1}; c)                          │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
          │
          ▼
      μ_K = mean of terminal belief
          │
          │  L = ||μ_K - x_0||² / ||x_0||²
          ▼
      L_deployment
```

Three points where differentiability is non-trivial:

### Argmax over `s` (envelope theorem, sharp max)

Use the paper's §4 recipe:

1. **Forward (no AD tape):** grid search + LBFGS to find `s_k*`.
2. **Gradient (AD tape):** re-evaluate `Q_k(s_k*)` with `s_k*` substituted for the max, inside an AD-traced region.

This avoids differentiating through the argmax and gives the exact envelope-theorem gradient for the outer parameters `(θ, w, c)` at the Bellman-optimal `s_k*`.

In Julia/Zygote pseudocode:

```julia
# Forward pass: find argmax, no AD tape active
s_star = Zygote.ignore() do
    find_argmax(b_k, c, w_next, θ, k)  # grid + LBFGS
end

# Gradient pass: evaluate integrand with s_star substituted, AD active
Q_at_sstar = expected_next_value(b_k, s_star, c, w_next, θ)  # this is what carries gradient
```

### Observation sampling `y_{k+1}`

The observation is `y_{k+1} ∈ ℝ^8` (8 port powers) with additive Gaussian noise. Reparameterize:

```
y_{k+1}  =  mean_y(b_k, s_k*, c)  +  Σ_y(b_k, s_k*, c)^{1/2} · ε,       ε ~ N(0, I_8)
```

Draw `ε` once per episode (outside the AD tape), then compute `y_{k+1}` as a deterministic function of `(b_k, s_k*, c, ε)`. Gradient flows through `mean_y` and `Σ_y^{1/2}`, which are closed-form functions of the S-matrix coefficients (and hence of `c`).

### Belief update `T` (EKF)

The EKF update is a closed-form sequence of matrix operations (Kalman gain, posterior mean, posterior covariance via Joseph form). Fully differentiable in `(b_k, s_k*, y_{k+1}, c)`. No custom rule needed — Zygote's default rules for linear algebra compose correctly through the Joseph form, matching the paper's existing custom `rrule` for the simple case.

## Integration with the existing photonic forward model

The current pipeline in [SimGeomBroadBand.jl](../SimGeomBroadBand.jl) + [BFIMGaussian.jl](../BFIMGaussian.jl) already provides:

1. **`batch_solve`** — parallel multi-frequency FDFD with the custom `rrule` for Wirtinger adjoint. Returns S-matrices; gradient w.r.t. `ε_geom` available.
2. **`getSmatrices`** — extracts normalized S, dS/dn, d2S/dn2.
3. **`powers_only`** — per-cell powers from the Taylor expansion.
4. **`ekf_update`** — EKF step with custom rrule.
5. **`get_sopt` + IFT rrule** — inner sensor-parameter argmax, differentiated via the implicit function theorem.

Option B replaces `get_sopt` (and its surrogate `log det J` reward) with:

```
s_opt_NN(b_k, c, w_{k+1}, θ)  =  argmax_s E_{y|b_k, s, c}[ w_{k+1} · φ_θ(T(b_k, s, y; c)) ]
```

That is: the objective that `get_sopt` optimizes becomes `E[NN value at next belief]` instead of `log det J at current belief`. The IFT rrule structure is the same — implicit differentiation at the inner argmax — but the cotangent propagation now flows through the NN and into `(θ, w_{k+1})` in addition to `c`.

**Action items for integration:**

1. Port `get_sopt` to a version that takes a callback objective `φ_θ` as argument. Grid search + LBFGS unchanged.
2. Ensure the custom IFT rrule's Hessian computation handles the NN objective's second derivatives (use a Hessian-vector product via Zygote-over-Zygote or use a structured approximation).
3. Replace the terminal reward `log det J_N` with the posterior-mean MSE. This loses the closed-form `log det` gradient but gains the deployment-aligned objective.

## Training procedure

### Optimizer

- **Outer**: Adam, learning rate `1e-3` to `1e-4`, with per-component box-width preconditioning on `c` (as in existing Case C). Adam also on `θ` and `w`.
- **Joint or alternating?** Start jointly — all three parameter groups updated every step. If training is unstable, switch to alternating: 1 outer `c` step per 10 inner `(θ, w)` steps (inner steps use fresh episodes but fixed `c`).

### Mini-batching

- **Inner loop:** `n_episodes = 20` per gradient estimate (same as existing Case C). Each episode is one draw of `(x_0, ε_{1:K})` and one forward pass through the DP.
- **Outer loop:** `N_outer = 10^3 – 10^4` iterations.

### Resampling schedule

- Fresh `(x_0, ε_{1:K})` every outer iteration (`resample_every=1`, as in existing Case C).
- Fresh FDFD solve every outer iteration (can't be avoided — `c` changes, so S-matrices change).

### Initialization

- `c`: random binary-like permittivity, passed through the density filter at `β = 16` (warm start).
- `θ`: Xavier / Glorot initialization for MLP weights.
- `w_k`: zero-initialized. This starts the NN with a trivial value function; the first few outer iterations bootstrap it via the Bellman residual.

### Continuation on `β`

Same as the current Case C pipeline: β increases along a schedule `16 → 32 → 64 → 128 → 256` triggered by loss plateaus. This controls the density-filter + tanh projection sharpness and is essential for producing a manufacturable binary final geometry.

### Warm-start from rung 4

One practical shortcut: initialize `θ` and `w_k` by first training a few hundred iterations of rung 4 (the existing Case C BFIM-trace EKF pipeline) to convergence, then switch to Option B training. This gives the NN a sensible starting value function based on the Fisher-information surrogate, after which end-to-end training refines it toward the true MSE objective. Expect ~2× speedup in convergence.

## Gradient flow diagram

```
L_deployment  ← ||μ_K - x_0||² / ||x_0||²
    ↑
    │  through:  μ_K (terminal belief mean)
    │            ← EKF updates at k=K-1, K-2, ..., 0
    │            ← each EKF update depends on:  b_{k-1}, s_k*, y_k, c
    │            ← s_k* is frozen (argmax recorded in forward, substituted in gradient)
    │            ← y_k = mean_y + Σ_y^{1/2} · ε  (reparameterized)
    │            ← mean_y, Σ_y depend on c via S-matrix Taylor coefficients
    │
    │  through:  the Q-function Q_k(s_k*) = w_{k+1} · φ_θ(T(b_k, s_k*, y_k; c))
    │            (appears implicitly via the argmax-frozen recipe)
    │
    ├→ ∇_c L   ... propagates through EKF, Σ_y, and S-matrix chain
    │            (uses FDFD adjoint; identical to existing Case C)
    │
    ├→ ∇_θ L   ... propagates through φ_θ wherever the NN value is invoked
    │            (standard reverse-mode AD; one backward pass per Zygote call)
    │
    └→ ∇_{w_k} L  ... through the k-th linear head, per-epoch
                 (small; just an outer product with the incoming cotangent)
```

## Key potential failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| **Hardware-light pathology** | `c` stays near initialization while NN overfits. | Bellman residual regularizer; per-component box-width preconditioning on `c`; two-timescale training (`c` on faster clock than `θ`). |
| **Value-function collapse** | `V_{θ,w}(b) ≈ const` across beliefs — no signal for argmax. | Auxiliary Bellman residual loss; terminal-reward anchor `ỹ_K = -trace(Σ_K)`. |
| **Non-monotone loss** | Deployment loss oscillates as outer iterations progress. | Lower `η`; add policy-iteration warm-starting (re-solve inner argmax less frequently); larger `n_episodes`. |
| **FDFD gradient explosion** | `∇_c L` has occasional huge values. | Gradient clipping; density filter acts as a regularizer; check for resonances in the geometry. |
| **Binary-gap**: loss drops during β-continuation | Loss plateaus then spikes at each β doubling. | Expected — existing autotune schedule handles this. |
| **Argmax-frozen gradient mismatch** | Gradient doesn't match finite-differences. | Verify with FD check at small `c` perturbation; the envelope theorem gradient is exact only at the true argmax; ensure the LBFGS inner-step converges tightly. |

## Diagnostics

Track these every outer iteration:

1. **Deployment loss (MSE relative error)** — primary metric, moving average over 20 iterations.
2. **Bellman residual** — if using auxiliary loss, this should decrease roughly monotonically.
3. **EKF posterior-variance at step K** — should decrease with training (proxy for informativeness).
4. **Mean argmax action `s_k*`** — should shift toward a structured pattern; check for policy collapse (same `s*` for all `b`).
5. **Binary fraction of `c_proj`** — at each β, track fraction of pixels that are `> 0.99` or `< 0.01` (cleanly binary). Target `> 95%` at `β = 256`.
6. **Gradient norms `||∇_c L||, ||∇_θ L||, ||∇_w L||`** — relative magnitudes inform preconditioning.

Monte-Carlo evaluation every 100 outer iterations:

- Deploy current `(c, θ, w)` on 200 episodes with `K_deploy = 10` EKF steps.
- Compare to rung 4 (existing Case C) on the same episode seeds.
- Report posterior-mean relative error, BFIM trace, and episode loss.

## Code scaffold

Pseudo-Julia, sketching the pieces (names matching the existing codebase):

```julia
# --- value function architecture ---
struct ValueNN
    encoder    # Flux.Chain for φ_θ
    heads      # Vector of Vector{Float32}, one per k
end

function v_value(vnn::ValueNN, b::Belief, k::Int)
    b_vec = belief_to_vector(b)
    feats = vnn.encoder(b_vec)
    return dot(vnn.heads[k+1], feats)  # 1-indexed in Julia
end

function q_value(vnn, b::Belief, s::Action, c, k::Int)
    # Expected next-belief value under Gaussian outcome distribution
    # E_{y | b, s, c} [ V(T(b, s, y; c), k+1) ]
    # Closed-form under EKF: propagate b → b_next given predicted y
    μ_y, Σ_y = predicted_observation(b, s, c)
    # For Gaussian integrand, marginalize via Gauss-Hermite quadrature or MC
    return mc_expectation(ε -> begin
        y = μ_y + cholesky(Σ_y).L * ε
        b_next = ekf_update(b, s, y, c)
        return v_value(vnn, b_next, k+1)
    end, n_samples=8)
end

# --- inner argmax with envelope-theorem freeze ---
function inner_argmax_NN(b::Belief, c, vnn, k::Int)
    # Forward (no AD): find s*
    s_star = Zygote.ignore() do
        s_grid = action_grid()
        s_coarse = argmax(s -> q_value(vnn, b, s, c, k), s_grid)
        s_star = lbfgs_refine(s -> -q_value(vnn, b, s, c, k), s_coarse)
        return s_star
    end
    # Gradient (AD active): re-evaluate at s*, freeze s*
    q_at_sstar = q_value(vnn, b, s_star, c, k)
    return s_star, q_at_sstar
end

# --- episode forward pass ---
function episode_loss(vnn, c, x0, ε_traj)
    b = prior_belief()
    for k in 0:(K-1)
        s_star, _ = inner_argmax_NN(b, c, vnn, k)
        y = predicted_observation_reparam(b, s_star, c, ε_traj[k+1])
        b = ekf_update(b, s_star, y, c)
    end
    μ_K = mean(b)
    return sum(((μ_K .- x0) ./ x0) .^ 2)
end

# --- outer loop ---
function outer_loop(vnn, c, opt_state, n_outer)
    for it in 1:n_outer
        x0_batch = sample_prior(n_episodes)
        ε_batch = [sample_noise(K) for _ in 1:n_episodes]
        
        loss_fn = (vnn, c) -> begin
            c_proj = project_and_filter(c, current_β)
            # Recompute FDFD + Taylor at new c
            smatrices = batch_solve(c_proj)
            L_dep = mean(episode_loss(vnn, smatrices, x0_batch[i], ε_batch[i]) 
                         for i in 1:n_episodes)
            if λ_bellman > 0
                L_bel = bellman_residual(vnn, smatrices, x0_batch, ε_batch)
                return L_dep + λ_bellman * L_bel
            end
            return L_dep
        end
        
        grads = Zygote.gradient(loss_fn, vnn, c)
        update!(opt_state, (vnn, c), grads)
        
        # β continuation
        maybe_advance_β!()
    end
end
```

## Extensions worth exploring

- **Larger `K_train`.** Current Case C uses `K_train = 3`. End-to-end training scales more gracefully with `K` than exact DP does, so `K_train = 10` or more is plausible.
- **Non-Gaussian belief.** Replace EKF with a particle filter for multimodal posteriors; the NN value function can take a particle-set-based belief encoding (permutation-invariant network, e.g., Deep Sets).
- **Dynamical state.** Extend `T` with a state-transition model for drifting perturbations; the framework survives unchanged if the belief update is differentiable.
- **Continuous action space with reparameterized policy.** Replace grid + LBFGS argmax with a stochastic policy parameterized by its own small NN, trained by reparameterization + Gumbel-softmax. Moves toward actor-critic but keeps the exact value-function structure.

## Reference checklist before implementation

- [ ] Verify that the existing `ekf_update` rrule propagates gradients cleanly when the inner argmax is frozen (write a finite-difference check on a toy `(b, c)` pair).
- [ ] Profile `q_value` — the inner expectation over `y` is the hottest loop. 8-point Gauss-Hermite should give adequate accuracy; benchmark vs. MC with 50+ samples.
- [ ] Confirm that `Zygote.ignore()` around the LBFGS inner-step actually prevents the tape from growing (Zygote has had known bugs here in older versions).
- [ ] Match the current Case C's preconditioner for `c` (box-width normalization); confirm it's correct for the new loss.
- [ ] Plan a baseline comparison: Option B vs. current Case C (rung 4) on identical physical parameters, 200-episode MC deployment. Expect Option B to match or beat rung 4 at sufficient training.

## References

- Companion paper: `doc/paper/paper.pdf` — especially §4 (differentiable DP via envelope theorem) and §5 (relaxation hierarchy).
- Companion tutorial: `doc/tutorial.pdf` — §3.3 Step 4 for the envelope identity, §7 for belief augmentation, §8.2 for the `∂V/∂c` unfolding.
- Case C codebase: [BFIMGaussian.jl](../BFIMGaussian.jl), [PosteriorCovGaussian.jl](../PosteriorCovGaussian.jl), [SimGeomBroadBand.jl](../SimGeomBroadBand.jl), [PhEnd2End.jl](../PhEnd2End.jl).
- Classical FVI and its deep variants: Bertsekas & Tsitsiklis, *Neuro-Dynamic Programming* (1996); Riedmiller, *Neural Fitted Q Iteration* (2005); Ernst et al., *Tree-Based Batch Mode RL* (2005).
- Envelope theorem: Milgrom & Segal, *Envelope Theorems for Arbitrary Choice Sets* (2002).
