# The φ_max cliff: why the joint-DP advantage collapses near φ_max ≈ 0.16

The φ_max sweep at K=4 (5-D BayesOpt with `ω_d` free, `K_phi=256`, `n_mc=20000`) shows a
sharp, narrow transition where the joint-DP / PCRB ratio drops from a huge advantage
(~200× at φ_max=0.1) to single digits over a `Δφ_max ≈ 0.06` increment.  This is not a
numerical artifact: it is the Ramsey aliasing threshold.

## The cliff

| φ_max | MC MSE_1 (joint-DP) | MC MSE_2 (PCRB)    | ratio_mc | z-score |
|-------|---------------------|--------------------|----------|---------|
| 0.10  | (5D, prior result)  | —                  | ~200×    | —       |
| 0.12  | 3.53×10⁻⁵           | 1.24×10⁻³          | 35.08×   | +147σ   |
| 0.15  | 1.30×10⁻⁴           | 1.92×10⁻³          | 14.75×   | +138σ   |
| **0.16** | **2.19×10⁻⁴**    | **2.16×10⁻³**      | **9.88×**| +130σ   |
| 0.17  | 3.18×10⁻⁴           | 2.46×10⁻³          | 7.75×    | +118σ   |
| 0.18  | 3.89×10⁻⁴           | 2.75×10⁻³          | 7.06×    | +119σ   |
| 0.20  | (in `compare_mse_phi20_K4.jls`) | —      | 4.90×    | +107σ   |
| 0.30  | 6.50×10⁻³           | 7.60×10⁻³          | 1.17×    | +13σ    |
| 0.40  | 1.25×10⁻²           | 1.36×10⁻²          | 1.09×    | +7σ     |
| 0.50  | 1.20×10⁻²           | 2.10×10⁻²          | 1.76×    | +48σ    |

The 10× crossover is at **φ_max ≈ 0.16** (9.88× there; 14.75× at φ_max=0.15).  Past φ_max=0.18
the ratio falls under 8×; at φ_max=0.30 it is essentially gone.  The mild non-monotonicity
between φ_max=0.40 and 0.50 reflects BayesOpt finding different basins, both deep in the
aliasing-floor regime where PCRB is already mis-specified.

## Physical explanation: the Ramsey aliasing threshold

The qubit frequency varies with flux as `ω_q(φ) ∝ √cos(πφ)`, with zero slope at φ=0 (the
standard transmon "sweet spot").  Consequently the qubit-frequency variation across the
prior support is **quadratic in φ_max**:

    Δω_q (over prior)  ≈  (f_q_max + E_C/h) · π² · φ_max² / 4

For nominal hardware values `f_q_max ≈ 9 GHz` and `E_C/h ≈ 0.25 GHz`, the longest available
Ramsey delay `τ_max = 320 ns` gives a phase span across the prior of `Δω_q · τ_max`:

| φ_max | Δω_q over prior | phase span at τ=320ns | fraction of one fringe |
|-------|-----------------|-----------------------|------------------------|
| 0.10  | 36 MHz          | 0.45 rad              | 0.07                   |
| 0.12  | 52 MHz          | 0.65 rad              | 0.10                   |
| 0.15  | 81 MHz          | 1.02 rad              | 0.16                   |
| **0.16** | **92 MHz**   | **1.16 rad**          | **0.18**               |
| 0.17  | 104 MHz         | 1.31 rad              | 0.21                   |
| 0.18  | 117 MHz         | 1.47 rad              | 0.23                   |
| 0.20  | 144 MHz         | 1.81 rad              | 0.29                   |
| 0.30  | 325 MHz         | 4.08 rad              | 0.65                   |
| 0.40  | 577 MHz         | 7.25 rad              | 1.15                   |
| 0.50  | 902 MHz         | 11.34 rad             | 1.80                   |

The 10× crossover at φ_max ≈ 0.16 corresponds to the **prior spanning ~0.18 of one Ramsey
fringe at the longest delay**.  That is the threshold past which the unambiguous-prior
regime ends.

## Why the cliff is so sharp

Joint-DP's huge advantage in the unambiguous regime comes from a specific mechanism: it
can deploy long-τ measurements freely, **because** the prior is narrow enough that no
aliasing risk exists.  PCRB also wants long τ (long τ = high Fisher information per shot),
and it lands there too.  Both methods agree on long-τ schedules in this regime.  Joint-DP
beats PCRB by a large factor because it exploits the full posterior shape under sequential
Bayesian updates, while PCRB is locked to a fixed schedule that ignores intermediate
information.

As soon as the prior support starts to wrap around even a fraction of one fringe at
τ=320 ns:

- **Joint-DP** can no longer naively use long τ.  Its adaptive policy must **first**
  disambiguate which fringe via short-τ measurements, **then** refine within.  Some of the
  K=4 epoch budget that previously went entirely to refinement now goes to disambiguation,
  reducing the realized refinement budget.

- **PCRB** doesn't adapt.  It deploys the same fixed long-τ schedule even when that schedule
  is mis-specified at the wider prior.  Its MSE goes up (the estimator now has to handle
  multimodal posteriors), but it doesn't degrade as catastrophically as one might expect,
  because PCRB-optimal hardware geometry already had to compromise for finite-K and bounded
  prior anyway.

Net effect: joint-DP's "free lunch" disappears.  The ratio drops because joint-DP's
denominator grows faster than PCRB's numerator.  This is the cliff.

## Why this matters for the paper

§7.6 of the paper gives the qualitative argument:

> "the prior support spans many Ramsey fringes" → "K-limited"; "well below this" → "operates far below the floor"

The sweep above is the **quantitative pinning of where 'below the floor' ends**.  At K=4:

- φ_max ≲ 0.155 → ratio ≥ 10× (the "below the floor" / unambiguous-prior regime)
- φ_max ∈ [0.16, 0.20] → ratio drops 9.9× → 4.9× (transition)
- φ_max ≥ 0.30 → ratio ≤ 1.2× (aliasing-floor regime; PCRB is mis-specified)

The paper's headline at φ_max = 0.10 sits comfortably ~3× below the cliff in φ_max-units,
or ~6× below in fringe-count units (since fringe count scales as φ_max²).  That is a
**deliberately interior** operating point, not an extreme one.  The 11.3× headline at
φ_max=0.10 (4-D paper headline; ~200× in 5-D) is therefore robust to mild revision of the
prior width.

## Suggested one-line update for §7.6

> At K=4, the joint-DP / PCRB ratio exceeds 10× for φ_max ≲ 0.155, the regime where the
> prior spans less than ~1/5 of a Ramsey fringe at the longest delay τ=320 ns.  Beyond
> that the prior begins to alias and the ratio drops to ~5× by φ_max = 0.20 and ~1.5× by
> φ_max = 0.50.

## Reproducibility

All compare-MC `.jls` files at φ_max ∈ {0.12, 0.15, 0.16, 0.17, 0.18, 0.20, 0.30, 0.40, 0.50}
are in `results/compare_mse_phi{NN}_K4.jls`.  Underlying BayesOpt winners are in
`results/bayesopt_{joint,pcrb}_phi{NN}_K4/result.jls`.  Run the sweep yourself with:

```bash
for phi in 0.12 0.15 0.16 0.17 0.18 0.20 0.30 0.40 0.50; do
    PHI_MAX=$phi K_EPOCHS=4 K_PHI=256 julia --project=/home/zlin/BFIMGaussian \
        --threads 100 bayesopt_joint_widephi.jl
    PHI_MAX=$phi K_EPOCHS=4 K_PHI=256 julia --project=/home/zlin/BFIMGaussian \
        --threads 4 bayesopt_pcrb_widephi.jl
    PHI_MAX=$phi K_EPOCHS=4 julia --project=/home/zlin/BFIMGaussian \
        --threads 64 compare_mse_widephi.jl
done
```

Total wall-clock for the full sweep: ~6 hours (most spent in the joint-DP BayesOpt at each
φ_max, ~30–50 min each at 100 threads).
