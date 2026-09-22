<div align="center">

# Thermodynamic Intelligence — Computation Code

**Compute useful internal structure for a physics-grounded definition of intelligence**

<br />

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/fti?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/fti/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

<br />

[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
&nbsp;
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
&nbsp;
[![Paper: Preprint](https://img.shields.io/badge/Paper-Preprint-4B8BBE?style=for-the-badge)](https://github.com/199-biotechnologies/fti)

---

Companion code for **"Intelligence as Useful Structure: A Thermodynamic Perspective"** (Djordjevic, 2026, preprint). Exact computations and two measurable testbeds.

[The Problem](#the-problem) | [Quick Start](#quick-start) | [What's Inside](#whats-inside) | [Results](#results) | [Citation](#citation)

</div>

## The Problem

Most thermodynamic intelligence metrics count all stored information equally, or count bits about the environment without asking whether they do anything for the agent. A system that perfectly memorises noise, or learns the statistics of an irrelevant channel, scores well on raw bits per joule while gaining nothing for its own viability.

The paper proposes that a thermodynamic definition of intelligence needs a *usefulness* filter, and gives it a formal layer. Useful structure has a **value form** (the viability an agent loses when its memory's correlation with the world is destroyed) and a **bits form** (information about viability-relevant environmental structure that the agent's own readout can use):

$$V_e(M) = \mathbb{E}[U_e \mid \pi \text{ uses } M] - \mathbb{E}[U_e \mid \pi \text{ uses } \tilde M], \qquad C_u = I_{\mathcal V}(M \to \theta^V_e)$$

Bits bound value, $V \le \sqrt{I(M;\theta_e)/2}$. Acquisition efficiency $\eta = \Delta C_u / W_{\text{diss}}$ obeys $\eta^* \le C_u/H(M) \le 1$ in closed-cycle operation, so at the Landauer floor efficiency *is* compression.

## Quick Start

```bash
git clone https://github.com/199-biotechnologies/fti.git
cd fti
pip install numpy matplotlib

# Exact five-agent example (reproduces Table I)
python toy_environment_latent.py

# Figures (needs the geb_synth repository for the two testbeds)
python make_paper_figures.py /path/to/logic-synthetic-data
```

The two testbeds live in the companion `geb_synth` repository: `experiments/probe_induce.py` (ECA rule induction, usable-information probe), `geb_synth/analysis/coarse_grain.py` (Israeli-Goldenfeld coarse-graining), `geb_synth/systems/resource_world.py` and `experiments/resource_world_sweep.py` (driven-dissipative resource world).

## Results

**Exact example.** Six agents learn an environment with a viability-relevant rule bit and a learnable but irrelevant bias bit. Epiplexity ranks the agent that learns everything highest and credits the noise memoriser with 0.8 bits; Shannon $C_u$ zeroes the noise memoriser but gives the keyed learner a full bit it cannot use; the readout-relative $C_u^{\mathcal V}$ and the value $V$ zero it exactly; the reflex ties the noise memoriser on reach with zero structure. Both value bounds (per environment and averaged) hold on every row.

| Agent | $H(M)$ | $I(M;e)$ | $C_u$ | $C_u^{\mathcal V}$ | $V$ | $A$ | $\eta^*_{\text{floor}}$ |
|---|---|---|---|---|---|---|---|
| Rule learner | 1.00 | 1.00 | **1.00** | **1.00** | **0.50** | **1.00** | **1.00** |
| Rule + bias learner | 2.00 | 1.80 | 1.00 | 1.00 | 0.50 | 1.00 | 0.50 |
| Lookup table | 3.79 | 1.58 | 1.00 | 0.72 | 0.45 | 0.95 | 0.26 |
| Noise memoriser | 1.00 | 0.80 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 |
| Keyed learner | 2.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.50 | 0.50 |
| Reflex | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | -- |

**ECA rule induction.** Three encodings of the same data, same 0.8M-parameter transformer, three seeds each: Shannon information about the rule is identical, usable information is not. With the probe fitted on training-rule representations and scored on unseen rules, the aligned tokeniser puts 6.6 bit-summed usable bits (ceiling 7.7; joint entropy of the rule set 5.8) into the representation before training and reaches 100% held-out functional accuracy by step 250; the flat encoding starts at 0.5, gains 0.4 in 2000 steps, and stays at 5% accuracy. Holding out whole symmetry classes changes nothing. The representation's value to the deployed head (permutation control) is 0.98 for aligned and 0.03 for flat.

**Resource world.** One designed memory bit carries 0.76-0.91 bits about the environment's drift direction, loses 0.25-0.26 intake per step when redrawn from its own marginal (a gross benefit of about 25x its 0.01 maintenance cost; 0.20 when its own trace is time-shifted, so the value is in alignment with the world), and takes over the population in 10/10 seeds at every per-bit cost up to 0.02, in 27, 19, 13 and 11 of 30 at 0.025-0.04, and never at 0.06; a memory-only population survives at every cost tested, so the exclusion is competitive, not economic, and it persists without mutation. The threshold moves with the switching period in the direction Kussell-Leibler predict (at cost 0.03: 5/5, 3/5, 1/5, 0/5 for periods of 6, 9, 12, 18 day-night cycles). The result requires regime changes to fall in the first half of a day. In 1 of 30 runs from random controllers, at zero bit cost, memory use evolved.

## What changed

An earlier version of this repository defined $C_u = I(M_t; Z^V_{t+\tau} \mid H_t)$ conditioned on the raw history. That quantity is identically zero for deterministic learners with a fixed initial memory; its toy example evaded this by redefining $M$ as the current percept. The neural experiments assumed an untrained network carries no useful structure (a random wide network already carries 0.87 bits under the same probe) and evaluated probes on their training data (exceeding the data-processing ceiling). A second round of review found that the first revision's ECA probe was fitted and scored on trajectories of the same held-out rules, that its 'Shannon ceiling' was the sum of the eight bit entropies rather than the joint entropy, and that its resource-world control replaced memory uniformly rather than from its marginal. A third round found that the value bound had been stated without its retained-context condition (a one-time-pad memory with the key observed at decision time breaks it), per environment where only the average is bounded, and with usable information substituted into a Shannon bound; that the Landauer bound needed its accounting conventions (heat over a full cycle, side-information-blind reset); that conditional acquisition is not net retained gain; and that the keyed learner's key was off-balance by 0.0084. All are fixed in the current scripts and manuscript, and the numbers above are from the corrected protocols. Those scripts (`compute_cu.py`, `neural_experiment.py`, `validation_suite.py`, `hypothesis_battery.py`, `ib_validation.py`, `edge_cases.py`, `multi_seed_experiment.py`) are kept for the record and superseded by the files above.

## Citation

```bibtex
@misc{djordjevic2026intelligence,
  title={Intelligence as Useful Structure: A Thermodynamic Perspective},
  author={Djordjevic, Boris},
  year={2026},
  note={Preprint}
}
```

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built by [Boris Djordjevic](https://github.com/longevityboris) at [Paperfoot AI](https://paperfoot.com)

<br />

**If this is useful to you:**

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/fti?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/fti/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

</div>
