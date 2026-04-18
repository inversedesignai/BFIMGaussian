# Paper Title Candidates

Working titles for the joint-$(c, \pi)$ sensor co-design paper, targeting **PRX**, **Nature Computational Science**, or **Nature Machine Intelligence**.

The paper's contribution:
- Joint optimization of sensor geometry $c$ and adaptive measurement policy $\pi$ via Bellman dynamic programming with envelope-theorem gradients.
- Finding: $\E[IG]$ (EVPI / information upper bound) is necessary but not sufficient for adaptive gain; the $c$ that maximizes $\E[IG]$ is typically the $c$ where adaptation has nothing to exploit. Numerics show a $2.8\times$ gap in $V_\text{adaptive}$ between EVPI-guided and joint-DP-guided geometry selection.
- A hierarchy of tractable relaxations (exact Bellman DP → deep RL → oracle CE → myopic Bayesian D-optimal → Gaussian-trace surrogate / EKF) with case studies at each level: radar beam-search (exact DP), superconducting-qubit flux sensor (exact DP, continuous state), photonic 2D metasensor (BFIM + EKF).

---

## Lead recommendation

> **Beyond the Information Bound: End-to-End Co-Design of Sensor Hardware and Adaptive Policy**

**Why it works.** "Beyond the Information Bound" signals the counter-intuitive finding ($\E[IG]$ is not the right design objective) — the hook for senior editors across all three target journals. "End-to-End" earns ML-community buy-in. "Co-Design" is a recognized technical term in computer architecture (HW–SW co-design) and cleanly captures the geometry–policy duality. Works equally well for PRX, NatCompSci, and NatMI.

---

## Alternative leads (descriptive / method-forward)

1. **"Co-Designing Hardware and Policy for Adaptive Sensing: A Unified Dynamic-Programming Framework"**
   More descriptive, journal-standard. Best for NatCompSci.

2. **"Joint Optimization of Sensor Geometry and Adaptive Policy via Envelope-Theorem Gradients"**
   Method-forward, PRX-flavored. Foregrounds the technical enabler.

3. **"The Envelope Theorem for Sensor Design"** / *Jointly Optimal Hardware and Adaptive Policy*
   Nature-style short main title with a long-form subtitle. Evocative, memorable.

4. **"Adaptive Sensors by Design: End-to-End Joint Optimization of Hardware and Inference"**
   Emphasizes the end-to-end ML framing. Best for NatMI.

---

## Provocative variants (higher risk, higher reward)

5. **"Maximizing Information Yields Suboptimal Sensors"** / *Joint Hardware–Policy Optimization with Envelope-Theorem Gradients*
   Contrarian; makes editors stop scrolling. Numerics back it up ($2.8\times$ gap in our benchmark). Defensible because we have the exact-DP comparison.

6. **"Hardware That Knows How It Will Be Used"** / *Joint Co-Design of Sensors and Adaptive Policies*
   Philosophical, very Nature. Highest risk of desk rejection as "too soft."

7. **"The Sensor Design Problem Is a Joint Optimization"** / *A Unified Bellman Framework with Case Studies in Radar, Quantum, and Photonic Sensors*
   Assertive, thesis-statement style. Works if the paper is positioned as foundational.

8. **"Fisher Information Is the Wrong Objective"** / *Joint Hardware–Policy Co-Design for Adaptive Bayesian Sensing*
   Most aggressive; would need very careful framing in the abstract to avoid antagonizing classical-estimation reviewers.

---

## Journal fit matrix

| title | PRX | NatCompSci | NatMI |
|---|---|---|---|
| Beyond the Information Bound | **strong** | **strong** | **strong** |
| Co-Designing Hardware and Policy | medium | **strong** | medium |
| Envelope-Theorem Gradients | **strong** | medium | medium |
| The Envelope Theorem for Sensor Design | **strong** | medium | medium |
| Adaptive Sensors by Design | medium | medium | **strong** |
| Maximizing Information Yields Suboptimal Sensors | **strong** | **strong** | **strong** |
| Hardware That Knows How It Will Be Used | medium | medium | **strong** |
| The Sensor Design Problem Is a Joint Optimization | medium | medium | medium |
| Fisher Information Is the Wrong Objective | medium | medium | **strong** |

Legend: **strong** = title matches the journal's recent style and this editor's likely taste; medium = plausible but not tailored.

---

## Recommendation

**Working title:** *Beyond the Information Bound: End-to-End Co-Design of Sensor Hardware and Adaptive Policy*

**Switch to** *Maximizing Information Yields Suboptimal Sensors* **if** a specific senior editor at PRX or NatMI is known to favor provocative titles (editor-specific call). Both are defensible from the $2.8\times$ numerical finding.

**Do not use** the pure-method titles (*Joint Optimization via Envelope-Theorem Gradients*) as the main title — they bury the headline finding. Keep them as subtitle material.

---

## Notes for title finalization

- Lock the title only after the full list of case studies (radar + scqubit + photonic BFIM) is numerically complete and the table of $2.8\times$-style gaps is written. If the scqubit joint-DP result turns out to show a larger factor than the radar case, update the lead hook.
- Abstract's first sentence must contain the contrarian finding, whichever title is chosen.
- If the editor pre-screens the abstract, the first sentence is functionally the title. Spend time on it.
- Consider an arXiv-first strategy: post with the more provocative title to gauge community reception, then revise for journal submission.
