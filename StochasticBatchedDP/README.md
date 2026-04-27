# Stochastic Batched DP

A scaling strategy for joint hardware-policy co-design when exact Bellman dynamic programming is
tractable only up to a horizon ceiling `K_batch` but the deployment horizon `K_total` is much
larger.  Combines:

1. **Batched DP**: instead of solving exact Bellman for the full horizon `K_total` (intractable
   due to memoization blow-up), solve it in `n` consecutive batches of length `K_batch`, each
   starting from the previous batch's terminal belief.

2. **Stochastic outer optimization**: instead of enumerating all reachable batch-boundary
   beliefs (curse of dimensionality at multimodal posteriors), sample a small set of
   trajectories under the current policy, evaluate each per-sample value via fresh Bellman
   solves, and use the empirical mean as an unbiased stochastic estimate.  Take SGD-style
   steps on the outer hardware parameter from the per-sample envelope-theorem gradients.

Together: a method that extends exact-inner-DP joint hardware-policy co-design from
`K_total ≈ 4` to arbitrary horizons, with unbiased gradients per sampled trajectory and a
linear cost in sample size rather than in reachable-belief count.

## Files

- `README.md` — this file (high-level overview)
- `StochasticBatchedDP.tex` — formal derivation and algorithm
- `StochasticBatchedDP.pdf` — compiled article

## Idea in one paragraph

Exact Bellman DP for POMDPs with a reachable-belief memo grows superlinearly in the horizon
`K`; for many problems this caps tractable `K` at 4-5 epochs.  *Batched DP* breaks the long
horizon into batches of length `K_batch`, each one a fresh exact-Bellman solve from its
predecessor's terminal belief; this is locally Bellman-optimal, globally a receding-horizon
approximation.  The new bottleneck moves from "memo size" to "the number of distinct
batch-boundary beliefs", which is essentially the same blow-up under another name.
*Stochastic batched DP* dispenses with enumeration: at each outer optimization step we
sample `M` trajectories under the current policy, solve a fresh `K_batch` Bellman from each
sample's terminal belief, and aggregate `M` envelope-theorem-exact gradients into one SGD
step.  The outer-hardware optimum found this way is a stationary point of the true batched
value, with statistical noise scaling as `1/√M` and the per-sample gradient bias-free for
the *sampled* sub-problem.

## What this generalises

The framework specialises:
- **NN training**: minibatch SGD on a sum over a finite (huge) training set is an unbiased
  estimator of full-batch gradient.  Same logic applies here, with "training points"
  replaced by "reachable batch-boundary beliefs" and "loss" replaced by "exact Bellman value".
- **Stochastic Dual Dynamic Programming (SDDP)**: scenario-sampled forward-backward sweeps
  for stochastic programs, classical in operations research.  This is the structurally
  closest cousin, but SDDP works on convex stochastic programs with cutting-plane value
  approximations, while the construction here works on POMDPs with exact memoized Bellman.
- **Sample-based POMDP solvers** (POMCP, DESPOT): Monte Carlo tree search over belief
  space; here the sampling is at *outer-hardware* gradient time rather than at policy
  inference time.

## What is new (to our knowledge)

The specific combination of (i) continuous physical hardware as outer optimization
variable, (ii) exact Bellman dynamic programming as inner solver within each batch, (iii)
stochastic Monte-Carlo sampling of batch-boundary trajectories to bypass the
reachable-belief enumeration, and (iv) sharp-max envelope-theorem gradients producing
unbiased per-sample outer-hardware gradients, is not present in the literature as a single
framework.  Pieces of it have been used in adjacent communities: SDDP for stochastic
programming, simulation-based sequential OED for adaptive design, deep RL for
NN-policy POMDPs, end-to-end optical co-design for hardware-NN coupled systems.  The
synthesis is the contribution.

## Pseudocode

```
Stochastic Batched DP for joint outer-c, inner-π optimization

Input:
  c_0 : initial outer parameter
  K_total : total horizon
  K_batch : per-batch horizon (chosen so exact Bellman is tractable)
  M : sample size per gradient step
  η : learning rate
  T_outer : number of outer iterations

Initialize c ← c_0

for t = 1 ... T_outer:
    g ← 0
    for m = 1 ... M (parallel):
        # Sample one trajectory of length K_total under current c
        b ← prior
        for batch_id = 1 ... n = K_total / K_batch:
            (V_batch, π_batch, memo_batch) ← solve_bellman(b, K_batch, c)
            for k = 1 ... K_batch:
                a ← π_batch(b)
                y ← sample_observation(b, a, c, x ~ b)
                b ← bayes_update(b, a, y, c)
        # The per-sample value
        V_m ← terminal_reward(b, c)
        # Per-sample gradient via envelope theorem
        # (freeze argmax actions and observation argmaxes along trajectory,
        #  reverse-mode AD through frozen recursion)
        g_m ← envelope_grad(V_m, c, frozen_choices)
        g += g_m
    g /= M
    c ← Adam_step(c, g, η)

return c
```

## Cost summary

| operation | cost |
|---|---|
| per Bellman solve at K_batch | one tractable DP solve (problem-dependent) |
| per trajectory sample | n_batches × Bellman solve = O(K_total / K_batch) Bellman solves |
| per outer gradient step | M × per-trajectory cost (parallelizable in M) |
| total at T_outer steps | T_outer × M × n_batches × K_batch_solve |

The wall-clock scales linearly in M and `n_batches`, not in the reachable-belief count.
For typical problems where one K_batch Bellman takes ~30 s and M = 10, T_outer = 500,
n_batches = 4, total wall-clock is ~17 hours on a single node, dominated by the
M × n_batches Bellman solves per gradient step.

## Convergence

Standard SGD theory applies.  With unbiased per-sample gradients and bounded variance,
the iterates converge to a stationary point of the batched-value objective at the standard
1/√t rate.  The key requirement is that each per-sample gradient is unbiased for the
trajectory's contribution to V_batched.  This holds because the sharp-max envelope theorem
makes the inner Bellman backup gradient exact at the optimal argmax, and the sampling
distribution for the trajectory is exactly the policy-induced distribution under current c.
