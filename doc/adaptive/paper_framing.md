# Paper Framing Language

Working framing for the joint $(c, \pi)$ sensor co-design paper. Target: **PRX, Nature Computational Science, Nature Machine Intelligence.** Companion to [paper_title_candidates.md](paper_title_candidates.md).

## Positioning (what this paper actually contributes)

The paper sits at the intersection of two bodies of prior work, each of which addresses one half of the problem:

- **End-to-end optics / deep optical learning** (Sitzmann 2018; Chang & Wetzstein 2019; Metzler; Kellman et al.): hardware + **one-step reconstruction** co-designed via differentiable simulation. Policy is non-adaptive over time.
- **Adaptive-sensing POMDP** (Kaelbling-Littman-Cassandra 1998; Kreucher-Hero 2005; Hernandez 2004; Krause-Guestrin 2005; Chen-Krause 2015): **multi-step adaptive policy** via Bellman dynamic programming on fixed hardware.

**The gap:** joint optimization of *continuous physical hardware* with a *multi-step Bellman-optimal adaptive policy*. Existing close relatives (cognitive radar with MIMO waveform adaptation, quantum metrology with scanned hardware + Kitaev PEA, variational quantum sensing with fixed protocols) are either decoupled / sequential or parametrize the hardware as a neural-network ansatz without a physical-geometry interpretation.

**The contribution:** close the gap — jointly and rigorously — with the envelope theorem as the enabling structural observation, and a principled relaxation hierarchy making the formulation portable across problem scales.

**Defensible claims (do not overclaim beyond these):**

1. *Unification.* A single pointwise information-gain objective $\Phi(x, s, c)$ unifies hardware co-design and adaptive sensor management, valid for both discrete and continuous state spaces.
2. *Structural insight.* The envelope theorem makes the joint $(c, \pi)$ gradient tractable even when the inner Bellman solver is approximate.
3. *Quantitative finding.* Classical information-bound-guided geometry selection loses a factor of $\sim\!2.8$ in adaptive value to the joint-DP optimum on a radar benchmark — the Expected Value of Perfect Information is a necessary but not sufficient design target.
4. *Portability.* A principled relaxation hierarchy (exact DP $\to$ deep RL $\to$ certainty-equivalent $\to$ myopic Bayesian D-optimal $\to$ Gaussian-trace with EKF) extends the framework from discrete POMDPs to high-dimensional continuous photonic design.

**What to avoid claiming:**

- "We derive Bellman's equation from first principles" — textbook since Bellman 1957, Sondik 1971.
- "We introduce mutual information as the reward" — standard since Lindley 1956.
- "We do joint hardware-policy optimization" — partially done in end-to-end optics for one-step policies.
- "No Bellman's equation exists for adaptive sensing" — **factually wrong**; reviewers will reject immediately.

---

## Framing language, by use case

### (1) One-sentence hook (for cover letter, invited talks, social media)

> **Hardware that doesn't know how it will be used is hardware designed for the way it won't be used.** We formulate the first end-to-end joint optimization of continuous physical sensor geometry with a multi-step Bellman-optimal adaptive measurement policy, and show it strictly outperforms the classical decoupled blueprint by a factor of 2.8 on a radar benchmark.

### (2) Elevator pitch (50 words, for introductions at conferences)

> Sensor hardware is conventionally optimized against a fixed measurement policy; adaptive measurement policies are conventionally run on fixed hardware. Neither half converges to the right answer. We close the gap with joint end-to-end optimization, enabled by an envelope-theorem gradient that eliminates policy Jacobians and a principled relaxation hierarchy that scales from exact Bellman to Gaussian-posterior surrogates.

### (3) Abstract-length framing (150 words)

> End-to-end optics co-designs hardware with a **one-step** reconstruction. Adaptive-sensing POMDP theory co-designs **nothing** with a multi-step measurement policy. Neither addresses what a real reconfigurable sensor is: *physical geometry whose operator will react to data as it arrives*. We formulate the joint optimization of continuous hardware and multi-step Bellman-optimal adaptive policy end-to-end, using the pointwise expected information gain $\Phi(\bm{x}, \bm{s}, \bm{c})$ as the reward. An envelope-theorem identity eliminates policy Jacobians from the geometry gradient, making the outer loop tractable even when the inner Bellman is approximate. A principled relaxation hierarchy — exact DP, deep RL, certainty-equivalent control, myopic Bayesian D-optimal, Gaussian-trace with EKF — makes the formulation portable across problem scales. On a radar beam-search benchmark the classical information-bound-guided geometry loses $2.8\times$ in adaptive value to the joint-DP optimum, establishing that Expected Value of Perfect Information is a necessary but grossly insufficient design criterion.

### (4) Provocative variant (for a title + subtitle, Nature-style)

> **"Beyond the Information Bound: End-to-End Co-Design of Sensor Hardware and Adaptive Measurement Policy"**
>
> Subtitle options:
> - *The envelope theorem, a tractability ladder, and a 2.8× gap nobody was measuring.*
> - *From exact dynamic programming to differentiable physics.*
> - *Why hardware and policy must be designed together.*

### (5) Contrarian variant (highest editor-stopping power, highest rejection risk)

> In the classical separation, the sensor is designed by the hardware engineer and the measurement policy by the statistician; each optimizes against the other's default. We prove this separation is strictly suboptimal — in our benchmark, a factor of $2.8$ in adaptive value. We unify hardware and policy into a single joint optimization via the envelope theorem, which cleanly eliminates the policy Jacobian from the outer geometry gradient, and provide a tractability ladder spanning exact Bellman dynamic programming down to Fisher-trace surrogates with extended Kalman filtering, each rung naming its failure mode. Optimizing the classical information bound — the quantity an experimental-design textbook would point at — loses nearly $3\times$ in attainable sensor value.

---

## Recommended compound framing for the paper

**Title:** *Beyond the Information Bound: End-to-End Co-Design of Sensor Hardware and Adaptive Measurement Policy* (from [paper_title_candidates.md](paper_title_candidates.md) lead recommendation).

**Opening line of abstract:** *End-to-end optics co-designs hardware with a one-step reconstruction; adaptive-sensing POMDP theory co-designs nothing with a multi-step measurement policy — neither addresses what a real reconfigurable sensor is.*

**Second sentence:** *We formulate the joint optimization end-to-end, enabled by an envelope-theorem gradient that removes the policy Jacobian from the outer loop.*

**Contribution three-bullet pitch:**
- *Unification.* A single information-gain objective unifying hardware co-design and adaptive sensor management, valid from discrete to continuous state spaces.
- *Structural insight.* The envelope theorem makes joint $(c, \pi)$ gradient-tractable even when the inner policy solver is approximate.
- *Quantitative finding.* Classical information-bound-guided geometry selection loses $\sim 3\times$ in adaptive value to the joint-DP optimum — the EVPI is a necessary but not sufficient design target.

**Closing punch:** *For any sensor whose hardware is designed once but whose policy runs forever — programmable metasurfaces, cognitive radars, tunable quantum sensors — this is not an incremental refinement but the minimum principled procedure.*

---

## Narrative arc for the paper body

1. **Setup.** Real sensors are reconfigurable hardware + adaptive measurement operator. Existing literature treats one or the other but not both.
2. **Reward.** The pointwise expected information gain $\Phi(x, s, c)$ is the right foundational object: exact integrand of MI, valid discrete and continuous, estimator-agnostic, supports pointwise oracle / ignorance-gap analysis (PCRB does not).
3. **Fixed vs oracle vs adaptive.** Decomposition via $\Phi$ gives $V_{\text{fixed}}, V_{\text{oracle}}(x), V_{\text{adaptive}}$ with the EVPI inequality as a consequence, not a new result.
4. **The surprising finding.** EVPI is necessary but not sufficient for adaptive gain. The geometry that maximizes the information bound generically does *not* maximize the adaptive value; the gap is operationally large ($2.8\times$ on the radar benchmark).
5. **Joint DP.** The envelope theorem eliminates the policy Jacobian from the outer geometry gradient; policy iteration alternates exact (or approximate) Bellman with gradient steps on $c$.
6. **Radar case study.** Full exact DP, verified numerically, every claim quantified.
7. **Relaxation hierarchy.** Deep RL, certainty-equivalent, myopic Bayesian D-optimal, Gaussian-trace + EKF — with each layer's failure mode explicit.
8. **Photonic case study.** Worked example at the cheapest rigorous rung (Gaussian-trace + EKF), demonstrating end-to-end joint optimization on a high-dimensional physical design problem.
9. **PCRB as complementary MSE baseline.** For continuous-state problems, the joint PCRB fixed-design $c_2^\star$ anchors the MSE ratio $\overline{\text{MSE}}_2 / \overline{\text{MSE}}_1$ as a deployment-friendly comparison.
10. **Conclusion.** Joint optimization of $(c, \pi)$ is the minimum principled procedure for adaptive-sensing hardware design.

---

## Review-defense notes

Anticipated objections and pre-prepared rebuttals:

| objection | response |
|---|---|
| "Bellman's equation for sensor POMDPs is not new" | Agreed; we do not claim otherwise. Our contribution is the **joint** optimization with continuous hardware, enabled by the envelope theorem. |
| "End-to-end optics already does this" | Only for one-step reconstruction policies. We address multi-step Bellman-optimal adaptive policies, which require a different technical machinery (envelope theorem + policy iteration). |
| "MI-reward is standard" | Agreed; our contribution is the pointwise decomposition $\Phi(x)$, which enables the oracle / ignorance-gap decomposition (standard bulk MI cannot support this). |
| "Why not PCRB?" | Section 8 of the paper: PCRB has no rigorous pointwise summand, no non-asymptotic pointwise MSE bound for biased Bayesian estimators, and is undefined for discrete state spaces — while $\Phi$ handles all three cases. |
| "EVPI is a known upper bound" | Agreed; we do not claim EVPI is new. Our contribution is the empirical demonstration that **$\argmax_c \E[IG](c) \neq \argmax_c V_{\text{adaptive}}(c)$** — the geometry selected by maximizing the information bound is strictly suboptimal by $\sim 3\times$ in attainable adaptive value. |
| "The radar case is small" | Yes, by design: small enough to solve exactly, large enough to exhibit non-trivial geometry-policy coupling. The photonic case demonstrates the same framework at scale via the cheapest principled relaxation. |
| "The photonic case uses approximations" | Yes, Gaussian-trace + EKF, path (2c) of the relaxation hierarchy — explicitly named and its failure modes cataloged. |

---

## Changelog

- 2026-04-19: Initial framing drafted; companion to paper_title_candidates.md. Lead title recommendation unchanged: *Beyond the Information Bound: End-to-End Co-Design of Sensor Hardware and Adaptive Measurement Policy.*
