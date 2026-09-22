#!/usr/bin/env python3
"""Figures for 'Intelligence as Useful Structure: A Thermodynamic Perspective'.

Reads:
  toy_environment_latent.json                                   (python toy_environment_latent.py)
  <GEB>/experiments/results_probe_{flat,2d,aligned}.json        (geb_synth: experiments/probe_induce.py)
  <GEB>/experiments/results_resource_world.json                 (geb_synth: experiments/resource_world_sweep.py)
  <GEB>/paper/figures/coarse_graining.png                       (geb_synth: scripts/make_figures.py)
Writes figures/*.png.  GEB defaults to ../../Projects/logic-synthetic-data relative to this file.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
GEB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/biobook/Projects/logic-synthetic-data")
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "figure.dpi": 220, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5})
C = {"flat": "#1f77b4", "2d": "#ff7f0e", "aligned": "#2ca02c", "reflex_L": "#9467bd", "reflex_R": "#d62728", "memory": "#2ca02c"}


# --------------------------------------------------------------------------- 1. toy profile
def fig_toy():
    rows = json.load(open(HERE / "toy_environment_latent.json"))
    names = [r["agent"].replace(" (a = z)", "").replace(" + ", "+") for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), gridspec_kw={"wspace": 0.55})
    y = np.arange(len(rows))[::-1]
    # (a) bits: H(M), I(M;e), C_u (Shannon), C_u for the agent's own readout
    ax = axes[0]
    h = 0.2
    ax.barh(y + 1.5 * h, [r["H_M"] for r in rows], h, color="#bbbbbb", label="$H(M)$ stored")
    ax.barh(y + 0.5 * h, [r["I_M_env"] for r in rows], h, color="#7f7f7f", label="$I(M;e)$ epiplexity")
    ax.barh(y - 0.5 * h, [r["C_u"] for r in rows], h, color="#2ca02c", label="$C_u=I(M;\\rho)$ (Shannon)")
    ax.barh(y - 1.5 * h, [r["C_u_readout"] for r in rows], h, color="#98df8a", label="$C_u^\\mathcal{V}$: bits the agent's own readout uses")
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlabel("bits"); ax.set_title("(a) structure")
    hnd, lab = ax.get_legend_handles_labels()
    fig.legend(hnd, lab, loc="lower center", ncol=2, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, -0.22), handlelength=1.2)
    # (b) value
    ax = axes[1]
    ax.barh(y, [r["V"] for r in rows], 0.6, color="#2ca02c")
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlabel("$V$ (gain vs scrambled $M$)"); ax.set_title("(b) value")
    ax.set_xlim(0, 0.6)
    # (c) reach
    ax = axes[2]
    ax.barh(y, [r["A_mean"] for r in rows], 0.6, color="#1f77b4")
    ax.axvline(0.5, color="k", lw=0.5, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlabel("$A$ (mean over $\\mu$)"); ax.set_title("(c) adaptive reach")
    ax.set_xlim(0, 1.05)
    # (d) efficiency at the Landauer floor
    ax = axes[3]
    eta = [0 if np.isnan(r["eta_floor"]) else r["eta_floor"] for r in rows]
    ax.barh(y, eta, 0.6, color="#d62728")
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlabel("$\\eta^*_{floor}=C_u/H(M)$"); ax.set_title("(d) acquisition efficiency")
    ax.set_xlim(0, 1.05)
    ax.text(0.05, y[-1], "no memory", va="center", fontsize=6.5, color="#555555")
    fig.savefig(OUT / "toy_profile.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 2. ECA induction: usable bits
def _load_probe_runs(split="random"):
    runs = {}
    for k in ("flat", "2d", "aligned"):
        files = sorted((GEB / "experiments").glob(f"results_probe_{k}_s*.json"))
        js = [json.load(open(f)) for f in files]
        js = [j for j in js if "V_rep" in j["log"][0] and j.get("split", "random") == split]
        if js:
            runs[k] = js
    return runs


def _series(js, key):
    """Return (flops, mean, min, max, n_seeds) across seeds for a per-checkpoint accessor."""
    steps = [r["step"] for r in js[0]["log"]]
    fl = np.array([r["flops"] for r in js[0]["log"]])
    vals = np.array([[key(r) for r in j["log"]] for j in js if len(j["log"]) == len(steps)])
    return fl, vals.mean(0), vals.min(0), vals.max(0), vals.shape[0]


def fig_eca():
    runs = _load_probe_runs()
    if not runs:
        print("probe results not found; skipping"); return
    labels = {"flat": "flat (learned positions)", "2d": "row + column positions", "aligned": "aligned (l, c, r, next) tokens"}
    panels = [
        ("(a) reach on 56 unseen rules", lambda r: r["heldout"]["functional"], "held-out-rule functional accuracy", (-0.02, 1.05)),
        ("(b) usable bits, summed over the 8 rule bits", lambda r: r["probe_heldout"]["usable_bits"], "usable bits about the rule", (0, 8.6)),
        ("(c) probe exact-table accuracy", lambda r: r["probe_heldout"]["probe_exact"], "all eight rule bits read correctly", (-0.02, 1.05)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"wspace": 0.42})
    for ax, (title, key, ylab, ylim) in zip(axes, panels):
        for k, js in runs.items():
            fl, m, lo, hi, n = _series(js, key)
            ax.plot(fl / 1e13, m, "-o", ms=2.5, color=C[k], label=f"{labels[k]} (n={n})")
            if n > 1:
                ax.fill_between(fl / 1e13, lo, hi, color=C[k], alpha=0.18, lw=0)
        ax.set_xlabel(r"training FLOPs ($\times 10^{13}$)"); ax.set_ylabel(ylab); ax.set_ylim(*ylim); ax.set_title(title, fontsize=7.5)
    # deployed head's exact-table accuracy, dashed, in (c)
    for k, js in runs.items():
        fl, m, lo, hi, n = _series(js, lambda r: r["heldout"]["exact"])
        axes[2].plot(fl / 1e13, m, "--", lw=0.9, color=C[k])
    axes[2].plot([], [], "-", color="k", lw=1, label="fitted probe"); axes[2].plot([], [], "--", color="k", lw=0.9, label="deployed head")
    axes[2].legend(frameon=False, fontsize=5.6, loc="center left", bbox_to_anchor=(0.02, 0.45))
    j0 = runs["flat"][0]
    ph = j0["log"][0]["probe_heldout"]
    ax = axes[1]
    xr = ax.get_xlim()[1]
    for yv, txt, col, ls, va, dy in ((ph["sum_bit_entropies"], rf"$\sum_k H(\theta_k)$ = {ph['sum_bit_entropies']:.2f}, ceiling of this readout", "k", "-", "bottom", 0.1),
                                     (np.mean([j["baseline_aligned_symbol_counts"]["usable_bits"] for j in runs["flat"]]), "linear reader on aligned symbol counts, no network", C["aligned"], "--", "top", -0.1),
                                     (ph["joint_ceiling_log2_rules"], rf"$\log_2 56$ = {ph['joint_ceiling_log2_rules']:.2f}, joint entropy of the rule set", "k", ":", "top", -0.1),
                                     (np.mean([j["baseline_raw_flat_input"]["usable_bits"] for j in runs["flat"]]), "linear reader on raw cells, no network", C["flat"], "--", "bottom", 0.1)):
        ax.axhline(yv, color=col, lw=0.7, ls=ls)
        ax.text(xr * 0.99, yv + dy, txt, ha="right", va=va, fontsize=5.2, color=col)
    axes[0].legend(frameon=False, loc="center right", fontsize=5.6)
    fig.savefig(OUT / "eca_usable_bits.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 3. coarse graining (two examples)
def fig_coarse():
    sys.path.insert(0, str(GEB))
    try:
        from geb_synth.systems.eca import run_eca, single_seed, random_state
        from geb_synth.analysis.coarse_grain import find_coarse_grainings, project
    except Exception as e:  # noqa: BLE001
        print("geb_synth not importable; copying old figure", e)
        src_png = GEB / "paper/figures/coarse_graining.png"
        if src_png.exists():
            shutil.copy(src_png, OUT / "coarse_graining.png")
        return
    examples = [(146, 3, 128, r"rule 146 $\to$ 128 ($N=3$): everything the projection discards is nuisance, and the coarse level is trivial"),
                (105, 2, 150, r"rule 105 $\to$ 150 ($N=2$): the coarse level keeps a chaotic dynamics of its own")]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), gridspec_kw={"width_ratios": [3, 1, 1], "hspace": 0.6, "wspace": 0.12})
    rng = np.random.default_rng(3)
    for row, (r, N, B, title) in enumerate(examples):
        cg = next(c for c in find_coarse_grainings(r, N) if c.coarse_rule == B)
        W, T = 96, 48 * N
        if r == 146:
            init = np.zeros(W, dtype=np.uint8); init[W // 2 - 6: W // 2 + 6] = 1
            init[rng.integers(0, W, 6)] = 1                       # a seed block plus a few stray cells
        else:
            init = random_state(W, rng, 0.35)
        micro = run_eca(r, init, T)
        proj = np.array([project(micro[t], cg) for t in range(0, T + 1, N)])
        macro = run_eca(B, project(init, cg), T // N)
        assert np.array_equal(proj, macro), (r, N, B)
        for ax, img, lab in zip(axes[row], (micro, proj, macro),
                                (f"(a) rule {r}: {W} cells, {T} steps", f"(b) $P$(micro), every {N}{'rd' if N == 3 else 'nd'} step", f"(c) rule {B} on $P$(init)")):
            ax.imshow(img, cmap="binary", aspect="auto", interpolation="nearest")
            ax.set_title(lab, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        fig.text(0.5, 0.94 - 0.485 * row, title, ha="center", fontsize=7.5)
    fig.savefig(OUT / "coarse_graining.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 4/5. resource world
def fig_resource_world():
    path = GEB / "experiments/results_resource_world.json"
    if not path.exists():
        print("resource world results not found; skipping")
        return
    R = json.load(open(path))
    # ---- spacetime
    S = R["spacetime"]
    food = np.array(S["food"])                     # (T, L)
    T, L = food.shape
    p = S["params"]
    # crop a window that follows nothing: show the whole ring section the patch visits
    occupied = np.flatnonzero(food.max(0) > 0.05)
    lo, hi = max(0, occupied.min() - 22), min(L, occupied.max() + 12)
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    ax.imshow(1 - np.clip(food[:, lo:hi], 0, 1), cmap="gray", aspect="auto", extent=[lo, hi, T, 0], vmin=0, vmax=1, interpolation="nearest")
    for t, ags in enumerate(S["agents"]):
        for x, ty in ags:
            if lo <= x < hi:
                ax.plot(x + 0.5, t + 0.5, ".", ms=1.6, color=C[["reflex_L", "reflex_R", "memory"][ty]], alpha=0.9)
    nights = [t for t in range(T) if not S["day"][t]]
    for t in nights:
        ax.axhspan(t, t + 1, xmin=0, xmax=0.012, color="#333333", lw=0)
    flips = [t for t in range(1, T) if S["direction"][t] != S["direction"][t - 1]]
    for t in flips:
        ax.axhline(t, color="#d62728", lw=0.7, ls="--")
    ax.set_xlabel("position on the ring"); ax.set_ylabel("time step (night marked at left edge; flip dashed)")
    ax.set_title("food (dark = present) and agents", fontsize=8)
    for name, col in (("reflex L", C["reflex_L"]), ("reflex R", C["reflex_R"]), ("memory", C["memory"])):
        ax.plot([], [], ".", color=col, label=name, ms=5)
    ax.legend(loc="lower right", frameon=True, fontsize=6)
    fig.savefig(OUT / "resource_world_spacetime.png", bbox_inches="tight")
    plt.close(fig)

    # ---- sweep
    costs = R["costs"]
    comp = R["competition"]
    comp_all = comp + R.get("competition_extra", [])
    memo = R["memory_only"]
    ref_mem = [r for r in R.get("reference", []) if r["type"] == "memory"]
    bite = R["params_default"]["bite"]

    def wilson(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0)
        p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, c - h), min(1.0, c + h))

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.5), gridspec_kw={"wspace": 0.42, "hspace": 0.62})
    # (a) who is left, with Wilson intervals for "memory is the whole surviving population"
    ax = axes[0, 0]
    ks = [sum(1 for r in comp_all if r["c_bit"] == c and r["final"]["n"] and r["final"].get("memory", 0) == 1.0) for c in costs]
    ns = [sum(1 for r in comp_all if r["c_bit"] == c) for c in costs]
    ps = [k / n for k, n in zip(ks, ns)]
    ci = [wilson(k, n) for k, n in zip(ks, ns)]
    ax.errorbar(costs, ps, yerr=[[p - lo for p, (lo, hi) in zip(ps, ci)], [hi - p for p, (lo, hi) in zip(ps, ci)]], fmt="-o", ms=3, color=C["memory"], capsize=2,
                label="runs in which memory is the whole population (Wilson 95%)")
    surv = [np.mean([1.0 if r["final"].get("n", 0) > 0 else 0.0 for r in memo if r["c_bit"] == c]) for c in costs]
    ax.plot(costs, surv, "--s", ms=3, color="#7f7f7f", label="memory-only population survives")
    ax.set_xlabel("memory cost $c_{bit}$ (energy / bit / step)"); ax.set_ylabel("fraction of runs after 6000 steps")
    ax.set_ylim(-0.05, 1.14); ax.set_title("(a) who is left", fontsize=8)
    ax.legend(frameon=False, fontsize=5.4, loc="lower left", bbox_to_anchor=(0.0, 0.05))
    # (b) V vs I(m;d) for dominant evolved genomes with the (inapplicable) single-time Pinsker curve for reference
    ax = axes[0, 1]
    pts = [(r["dominant"][0]["I_mem_direction"], r["dominant"][0]["V_marginal"], r["c_bit"]) for r in comp if r["dominant"]]
    sc = ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=[p[2] for p in pts], cmap="viridis", s=14, alpha=0.85, edgecolors="none")
    ii = np.linspace(0, 1, 200)
    ax.plot(ii, bite * np.sqrt(ii * np.log(2) / 2), "--", color="#d62728", lw=0.9, label="Pinsker curve (reference)")
    if ref_mem:
        ax.plot(np.mean([r["I_mem_direction"] for r in ref_mem]), np.mean([r["V_marginal"] for r in ref_mem]), "*", ms=9, color="k", label="designed founder")
    above = sum(1 for I_, V_, _ in pts if V_ > bite * np.sqrt(I_ * np.log(2) / 2))
    ax.text(0.02, 0.97, f"{above} of {len(pts)} above the curve", transform=ax.transAxes, va="top", fontsize=6)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("$I(m;d)$ of the dominant genome (bits)"); ax.set_ylabel("$V$ (energy / step)")
    ax.set_title("(b) evolved dominants", fontsize=8); ax.set_xlim(-0.03, 1.0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05); cb.set_label("$c_{bit}$", fontsize=6, labelpad=1); cb.ax.tick_params(labelsize=5.5)
    ax.legend(frameon=False, fontsize=5.2, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    # (c) time-shift vs redraw value for the same dominants
    ax = axes[0, 2]
    pv = [(r["dominant"][0]["V_marginal"], r["dominant"][0].get("V_shift", np.nan)) for r in comp if r["dominant"] and "V_shift" in r["dominant"][0]]
    if pv:
        ax.scatter([p[0] for p in pv], [p[1] for p in pv], s=14, color=C["memory"], alpha=0.75, edgecolors="none", label="dominant evolved genome")
        lim = max(0.3, max(max(abs(p[0]), abs(p[1])) for p in pv) * 1.05)
        ax.plot([-0.1, lim], [-0.1, lim], ":", color="k", lw=0.7, label="equal")
        if ref_mem:
            ax.plot(np.mean([r["V_marginal"] for r in ref_mem]), np.mean([r["V_shift"] for r in ref_mem]), "*", ms=9, color="k", label="designed founder")
        ax.set_xlabel("$V$: redraw memory each step"); ax.set_ylabel("$V$: own trace, time-shifted")
        ax.set_title("(c) shifted-trace vs redraw value", fontsize=8); ax.legend(frameon=False, fontsize=5.4, loc="upper left")
    # (d) one run
    ax = axes[1, 0]
    r = next(r for r in comp if r["c_bit"] == 0.01 and r["seed"] == 0)
    t = [s_["t"] for s_ in r["series"]]
    for key in ("reflex_L", "reflex_R", "memory"):
        ax.plot(t, [s_.get(key, 0) * s_["n"] if s_["n"] else 0 for s_ in r["series"]], color=C[key], label=key.replace("_", " "))
    Tsw = R["params_default"]["T_switch"]
    for k in range(1, r["series"][-1]["t"] // Tsw + 1):
        ax.axvline(k * Tsw, color="#d62728", lw=0.5, ls="--")
    ax.set_xlabel("time step (drift flips dashed)"); ax.set_ylabel("agents"); ax.set_title("(d) one run, $c_{bit}$ = 0.01", fontsize=8)
    ax.legend(frameon=True, fontsize=5.8, loc="center right", framealpha=0.9, edgecolor="none")
    # (e) when the regime change falls
    ax = axes[1, 1]
    rob = R.get("robustness", [])
    conds = list(dict.fromkeys(rr["condition"] for rr in rob))
    for i, cond in enumerate(conds):
        rows = [rr for rr in rob if rr["condition"] == cond]
        ys = [rr["final"].get("memory", 0.0) if rr["final"]["n"] else 0.0 for rr in rows]
        ax.plot([i] * len(ys), ys, "o", ms=3, color=C["memory"], alpha=0.6)
        ax.plot(i, np.mean([rr["founder_memory_intake"] for rr in rows]) / bite, "s", ms=4, color="#1f77b4")
    ax.plot([], [], "o", ms=3, color=C["memory"], alpha=0.6, label="memory-descended fraction (per seed)")
    ax.plot([], [], "s", ms=4, color="#1f77b4", label="designed founder alone: intake / bite")
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([c.replace(" (default)", "").replace(", d0 = -1", "\n$d_0=-1$") for c in conds], fontsize=6, rotation=15)
    ax.set_ylim(-0.05, 1.12); ax.set_ylabel("fraction"); ax.set_title("(e) timing of change, $c_{bit}$ = 0.01", fontsize=8)
    ax.legend(frameon=False, fontsize=5.4, loc="center")
    # (f) switching-period sweep (Test 5)
    ax = axes[1, 2]
    tsw = R.get("tswitch_sweep", [])
    if tsw:
        Ts = sorted({rr["T_switch"] for rr in tsw}); cs = sorted({rr["c_bit"] for rr in tsw})
        cmap = plt.get_cmap("plasma")
        for i, T_ in enumerate(Ts):
            fr = [np.mean([1.0 if rr["final"]["n"] and rr["final"].get("memory", 0) == 1.0 else 0.0 for rr in tsw if rr["T_switch"] == T_ and rr["c_bit"] == c]) for c in cs]
            ax.plot(cs, fr, "-o", ms=3, color=cmap(i / max(1, len(Ts) - 1)), label=f"$T_{{switch}}$ = {T_} ({T_ // 44} cycles)")
        ax.set_xlabel("memory cost $c_{bit}$"); ax.set_ylabel("runs won by memory")
        ax.set_ylim(-0.05, 1.12); ax.set_title("(f) switching period, 5 seeds", fontsize=8)
        ax.legend(frameon=False, fontsize=5.4, loc="lower left")
    fig.savefig(OUT / "resource_world_sweep.png", bbox_inches="tight")
    plt.close(fig)

    # ---- naive detector vs scrambled control (all stored seeds)
    seeds = R.get("naive_detector_seeds", [R["naive_detector"]])
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.9), gridspec_kw={"wspace": 0.5})
    types = ("reflex_L", "reflex_R", "memory")
    ax = axes[0]
    for i, t in enumerate(types):
        ys = [N[t]["corr"] for N in seeds if N[t].get("corr") is not None]
        ax.plot([i] * len(ys), ys, "o", ms=3.5, color=C[t], alpha=0.7)
        ax.hlines(np.mean(ys), i - 0.25, i + 0.25, color=C[t], lw=1.2)
    ax.set_xticks(range(3)); ax.set_xticklabels([t.replace("_", " ") for t in types])
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("corr(spend on moving, next intake)")
    ax.set_title(f"(a) naive detector ({len(seeds)} seeds)", fontsize=8)
    ax = axes[1]
    for i, t in enumerate(types):
        ys = [N[t]["V_intake"] for N in seeds]
        ax.plot([i] * len(ys), ys, "o", ms=3.5, color=C[t], alpha=0.7)
        ax.hlines(np.mean(ys), i - 0.25, i + 0.25, color=C[t], lw=1.2)
    ax.set_xticks(range(3)); ax.set_xticklabels([t.replace("_", " ") for t in types])
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("$V$ (intake lost, memory scrambled)")
    ax.set_title("(b) scrambled-memory control", fontsize=8)
    ax.text(0.5, 0.02, "zero by construction\n(no memory)", ha="center", fontsize=5.5, color="#555555")
    fig.savefig(OUT / "naive_detector.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_toy()
    fig_eca()
    fig_coarse()
    fig_resource_world()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
