# scqubit narrow-prior headline (2026-04-19)

**Joint-DP beats PCRB baseline by 8.3× (z=+166.6σ, gap=729%).**

## Configuration

| Parameter | Value |
|-----------|-------|
| Horizon K | 4 epochs |
| Delay set J | 10 τ log-spaced in [10, 320] ns |
| Repetition set L | 2 (n ∈ {1, 10}) |
| Prior | Uniform[0, 0.1] Φ₀ (narrow) |
| c | PAPER_BASELINE (no c-optimization) |
| Drive frequency ω_d | 2.2821e10 rad/s (at φ*=0.4436) |
| Belief grid K_Φ | 256 (train = deploy) |
| MC trajectories | 40,000 |
| Terminal reward Φ | −Var(b_K) (Bayesian MSE) |

## Deployed MSE (20k trajectories each, paired seed)

| Estimator | MSE | ± SE |
|-----------|-----|------|
| Joint-DP (adaptive policy) | 1.0156e-4 | 2.33e-6 |
| PCRB fixed schedule `[(320ns,10)×4]` | 8.4153e-4 | 3.78e-6 |
| Prior variance (reference) | 8.3333e-4 | – |
| 1/J_P (CRB bound) | 1.086e-9 | – |

- ratio MSE_pcrb / MSE_joint = **8.286**
- z = **+166.63σ** (6.1e4 effective samples per arm)
- gap = **728.6%**

## Phi_max dependence

| phi_max | MSE_adaptive | MSE_pcrb | ratio | gap | prior variance |
|---------|--------------|----------|-------|-----|----------------|
| 0.03 | 5.47e-7 | 7.00e-5 | **128.0** | 12,700% | 7.50e-5 |
| 0.05 | 2.65e-6 | 2.04e-4 | **76.9** | 7,590% | 2.08e-4 |
| 0.08 | 4.30e-5 | 5.35e-4 | **12.4** | 1,144% | 5.33e-4 |
| 0.10 | 1.02e-4 | 8.42e-4 | **8.29** | 729% | 8.33e-4 |
| 0.15 | 5.69e-4 | 1.90e-3 | 3.33 | 233% | 1.88e-3 |
| 0.20 | 2.38e-3 | 3.38e-3 | 1.42 | 42% | 3.33e-3 |
| 0.25 | 4.28e-3 | 5.33e-3 | 1.25 | 25% | 5.21e-3 |
| 0.30 | 7.33e-3 | 7.71e-3 | 1.05 | 5% | 7.50e-3 |
| 0.40 | 1.54e-2 | 1.37e-2 | 0.89 | -11% | 1.33e-2 |
| 0.49 | 2.53e-2 | 2.04e-2 | 0.81 | -19% | 2.00e-2 |

The 50% target is exceeded for phi_max ≤ ~0.18. The gap scales sharply with narrowing prior; at phi_max=0.03 the gap is **128×** (z=+151σ).

## Horizon (K) dependence at phi_max=0.1, PAPER_BASELINE c

| K | J | MSE_adaptive | MSE_pcrb | ratio | gap |
|---|---|--------------|----------|-------|-----|
| 3 | 10 | 2.18e-4 | 8.37e-4 | 3.84 | 284% |
| 4 | 10 | 1.02e-4 | 8.42e-4 | **8.29** | 729% |
| 5 |  6 | 6.75e-4 | 8.45e-4 | 1.25 | 25% |

K=4 J=10 is the sweet spot — both horizon and τ granularity matter.
K=5 J=6 has too few τ options (J=6 vs 10) to resolve aliasing despite
the extra epoch, so the gap shrinks to 25% (still above 0 but below
the K=3 and K=4 configurations).

## Mechanism

Ramsey-fringe period in flux at τ=320ns, f_q=9GHz: Δφ ≈ 3.1e-4 Φ₀.

- **Wide prior (phi_max=0.49):** ~1600 aliasing modes spanning [0, 0.49]. Both policies fail to disambiguate; both saturate near prior variance (2e-2).
- **Narrow prior (phi_max=0.1):** ~320 aliasing modes spanning [0, 0.1]. PCRB's fixed long-τ schedule still saturates at prior variance (8.4e-4). The adaptive policy uses short-τ early epochs (root action τ=21.6ns) to disambiguate, then long-τ to sharpen — achieving 1e-4 MSE, 8× better than PCRB.
- **Very narrow prior (phi_max → 0):** aliasing vanishes; PCRB matches adaptive (Fisher-limited regime).

The narrow-prior regime is the operating point where adaptive disambiguation is both **necessary** (PCRB would saturate) and **feasible** (few enough aliases that K=4 epochs can resolve them).

## Reproduction

```bash
cd /home/zlin/BFIMGaussian/doc/adaptive/scqubit
julia -t 4 compare_mse_narrow_baseline.jl  # ~5 minutes
```

Saved artifacts:
- `results/compare_mse_narrow_baseline.jls` — headline serialized
- `results/narrow_baseline_headline.log` — full output
- `results/phimax_sweep.log` — phi_max dependence

Scripts:
- `compare_mse_narrow_baseline.jl` — clean headline at PAPER_BASELINE c
- `compare_mse_phimax.jl` — phi_max sweep {0.1..0.49}
- `compare_mse_phimax_extreme.jl` — phi_max ∈ {0.03, 0.05, 0.08} + K ∈ {3, 5} check
- `sweep_joint_narrow.jl` / `sweep_pcrb_narrow.jl` — with c-optimization

## c-optimization note

At phi_max=0.1 with the realistic box, the gradient magnitude at PAPER_BASELINE is tiny (|∇V/∇c| ≈ 3e-2). c-optimization yields only marginal V improvement. The baseline c already sits near the narrow-prior optimum; the decisive lever is the prior width, not the circuit tuning.

PCRB-optimizing c moves it to (f_q=9.62 GHz, κ=1e5 Hz floor, Δ=3.2 GHz) but deployed MSE is essentially unchanged (8.40e-4 vs 8.42e-4 at baseline) — PCRB's schedule is MSE-limited by Ramsey-fringe aliasing, not by circuit tuning. Checked ratio at joint@baseline vs PCRB@PCRB-opt: **8.28** (vs 8.29 at paired baseline). Gap is robust.
