"""
Generate publication-quality figures for IRL policy evaluation results.

Figures:
    1. Model-based policy evaluation (V_hat and P(survive))
    2. Policy performance tradeoff scatter
    3. WIS off-policy evaluation with bootstrap CIs
    4. Single-column IRL policy comparison

Style: matplotlib-only, serif font, paper-conference conventions
       (cf. Ziebart 2008, Booth 2023, Ma 2024).

Usage:
    python plot_results.py --outdir figures/
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Style and palette
# ---------------------------------------------------------------------------

# Wong (2011) colorblind-safe palette.
# Reference: Wong, B. "Points of view: Color blindness." Nature Methods 8, 441 (2011).
WONG = {
    "gray":   "#999999",
    "blue":   "#0072B2",
    "green":  "#009E73",
    "orange": "#D55E00",
    "yellow": "#E69F00",
    "purple": "#CC79A7",
    "skyblue": "#56B4E9",
}


@dataclass(frozen=True)
class PolicyStyle:
    label: str
    color: str
    marker: str  # for scatter plots
    category: str  # 'baseline' | 'irl' | 'rl'


POLICIES: dict[str, PolicyStyle] = {
    "Clinician":     PolicyStyle("Clinician",     WONG["gray"],   "o", "baseline"),
    "BC":            PolicyStyle("BC",            WONG["blue"],   "o", "baseline"),
    "VI Komorowski": PolicyStyle("VI Komorowski", WONG["green"],  "*", "irl"),
    "VI IQ-Learn":   PolicyStyle("VI IQ-Learn",   WONG["orange"], "*", "irl"),
    "VI MaxEnt":     PolicyStyle("VI MaxEnt",     WONG["yellow"], "*", "irl"),
    "PPO MaxEnt":    PolicyStyle("PPO MaxEnt",    WONG["purple"], "D", "rl"),
}


def setup_style() -> None:
    """Apply paper style globally. Call once at script start."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset":  "dejavuserif",
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   8.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.titleweight":  "bold",
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "pdf.fonttype":      42,  # Type-42 (TrueType) for paper submission
        "ps.fonttype":       42,
    })


# ---------------------------------------------------------------------------
# Results (replace with load-from-disk if you serialize evals)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Results:
    v_hat: dict[str, float]
    p_survive: dict[str, float]
    wis_mean: dict[str, float]
    wis_ci_low: dict[str, float]
    wis_ci_high: dict[str, float]
    ess_one: dict[str, bool]  # True if effective sample size collapsed to 1


def load_results() -> Results:
    """Hard-coded results from eval runs. Swap for JSON/CSV loader as needed."""
    return Results(
        v_hat={
            "Clinician":     59.4,
            "BC":            36.7,
            "VI Komorowski": 72.9,
            "VI IQ-Learn":   72.9,
            "VI MaxEnt":     48.5,
            "PPO MaxEnt":     7.3,
        },
        p_survive={
            "Clinician":     0.797,
            "BC":            0.684,
            "VI Komorowski": 0.864,
            "VI IQ-Learn":   0.864,
            "VI MaxEnt":     0.742,
            "PPO MaxEnt":    0.537,
        },
        wis_mean={
            "Clinician":      44.0,
            "BC":            -83.0,
            "VI Komorowski":  86.0,
            "VI IQ-Learn":    73.0,
            "VI MaxEnt":      80.0,
            "PPO MaxEnt":     81.0,
        },
        wis_ci_low={
            "Clinician":      40.0,
            "BC":            -85.0,
            "VI Komorowski": -90.0,
            "VI IQ-Learn":    65.0,
            "VI MaxEnt":     -90.0,
            "PPO MaxEnt":     81.0,
        },
        wis_ci_high={
            "Clinician":      48.0,
            "BC":             85.0,
            "VI Komorowski":  92.0,
            "VI IQ-Learn":    82.0,
            "VI MaxEnt":      92.0,
            "PPO MaxEnt":     81.0,
        },
        ess_one={
            "Clinician":     False,
            "BC":            True,
            "VI Komorowski": True,
            "VI IQ-Learn":   True,
            "VI MaxEnt":     True,
            "PPO MaxEnt":    True,
        },
    )


# ---------------------------------------------------------------------------
# Figure 1 — Model-based evaluation
# ---------------------------------------------------------------------------

def fig1_model_based_eval(r: Results, outdir: Path) -> None:
    policies = ["Clinician", "BC", "VI Komorowski", "VI IQ-Learn", "VI MaxEnt", "PPO MaxEnt"]
    colors = [POLICIES[p].color for p in policies]
    y = np.arange(len(policies))[::-1]  # top-to-bottom order

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7, 3.8), sharey=True,
                                     gridspec_kw={"wspace": 0.08})

    # Left: V_hat
    v_vals = [r.v_hat[p] for p in policies]
    ax_l.barh(y, v_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax_l.set_xlabel(r"$\hat{V}(\mu_0)$")
    ax_l.set_xlim(0, max(v_vals) * 1.18)
    for yi, v in zip(y, v_vals):
        ax_l.text(v + max(v_vals) * 0.015, yi, f"{v:.1f}",
                  va="center", ha="left", fontsize=8.5)

    # Right: P(survive) with clinician reference line
    p_vals = [r.p_survive[p] for p in policies]
    clinician_p = r.p_survive["Clinician"]
    ax_r.barh(y, p_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax_r.axvline(clinician_p, ls="--", color=WONG["gray"], lw=1.0,
                 label=f"Clinician ({clinician_p:.3f})")
    ax_r.set_xlabel(r"P(survive)")
    ax_r.set_xlim(0, 1.05)
    for yi, v in zip(y, p_vals):
        ax_r.text(v + 0.015, yi, f"{v:.3f}",
                  va="center", ha="left", fontsize=8.5)

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(policies)
    # Place legend ABOVE the right axis to avoid bar overlap
    ax_r.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                frameon=False, fontsize=8.5)

    fig.suptitle("Model-Based Policy Evaluation", y=1.02, fontsize=12, fontweight="bold")
    fig.savefig(outdir / "fig1_model_based_eval.pdf")
    fig.savefig(outdir / "fig1_model_based_eval.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Tradeoff scatter
# ---------------------------------------------------------------------------

def fig2_tradeoff_scatter(r: Results, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))

    clin_v = r.v_hat["Clinician"]
    clin_p = r.p_survive["Clinician"]

    # Reference cross-hairs at clinician
    ax.axhline(clin_p, ls="--", color=WONG["gray"], lw=0.9, zorder=1)
    ax.axvline(clin_v, ls="--", color=WONG["gray"], lw=0.9, zorder=1)

    # Plot points by category for legend grouping
    marker_size = 140
    for name, style in POLICIES.items():
        ax.scatter(r.v_hat[name], r.p_survive[name],
                   marker=style.marker, s=marker_size,
                   color=style.color, edgecolor="black", linewidth=0.6,
                   zorder=3)

    # Annotation offsets tuned to prevent overlap.
    # VI Komorowski and VI IQ-Learn share coords (72.9, 0.864) — split vertically.
    annotations = {
        "Clinician":     (8,  -14, "left"),
        "BC":            (10,   0, "left"),
        "VI Komorowski": (-8,  10, "right"),  # above-left
        "VI IQ-Learn":   (8,  -14, "left"),   # below-right
        "VI MaxEnt":     (10,   0, "left"),
        "PPO MaxEnt":    (10,   0, "left"),
    }
    for name, (dx, dy, ha) in annotations.items():
        ax.annotate(name,
                    (r.v_hat[name], r.p_survive[name]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center",
                    fontsize=9, color=POLICIES[name].color, fontweight="bold")

    # Label the clinician reference line near the top of the plot, not the bottom
    ax.text(clin_v + 0.5, ax.get_ylim()[1] if False else 0.905, r"Clinician $\hat{V}$",
            color=WONG["gray"], fontsize=8.5, ha="left", va="center", style="italic")

    ax.set_xlabel(r"$\hat{V}(\mu_0)$")
    ax.set_ylabel(r"P(survive)")
    ax.set_xlim(-3, 90)
    ax.set_ylim(0.50, 0.92)

    # Legend by marker shape (category), placed outside lower right of data
    legend_handles = [
        plt.scatter([], [], marker="o", s=80, color=WONG["gray"],
                    edgecolor="black", linewidth=0.6, label="Baseline / RL"),
        plt.scatter([], [], marker="*", s=120, color=WONG["gray"],
                    edgecolor="black", linewidth=0.6, label="IRL method"),
        plt.scatter([], [], marker="D", s=70, color=WONG["purple"],
                    edgecolor="black", linewidth=0.6, label="PPO MaxEnt"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              frameon=True, framealpha=0.9, edgecolor="0.85", fontsize=8.5)

    ax.set_title("Policy Performance Tradeoff")
    fig.savefig(outdir / "fig2_tradeoff_scatter.pdf")
    fig.savefig(outdir / "fig2_tradeoff_scatter.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — WIS off-policy evaluation
# ---------------------------------------------------------------------------

def fig3_wis_eval(r: Results, outdir: Path) -> None:
    policies = ["Clinician", "BC", "VI Komorowski", "VI IQ-Learn", "VI MaxEnt", "PPO MaxEnt"]
    y = np.arange(len(policies))[::-1]

    fig, ax = plt.subplots(figsize=(7, 4.0))

    means = np.array([r.wis_mean[p] for p in policies])
    lows  = np.array([r.wis_ci_low[p] for p in policies])
    highs = np.array([r.wis_ci_high[p] for p in policies])

    # Asymmetric error bars; clip to >= 0 where bootstrap CIs are degenerate
    err_low  = np.maximum(means - lows, 0.0)
    err_high = np.maximum(highs - means, 0.0)

    for yi, p, m, el, eh in zip(y, policies, means, err_low, err_high):
        style = POLICIES[p]
        hatch = "///" if r.ess_one[p] else None
        ax.barh(yi, m, color=style.color, edgecolor="black", linewidth=0.7,
                hatch=hatch, alpha=0.95)
        ax.errorbar(m, yi, xerr=[[el], [eh]],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=3)

    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(policies)
    ax.set_xlabel("WIS Estimate")
    ax.set_title(r"WIS Off-Policy Evaluation ($\epsilon = 0.01$)", pad=10)

    # Legend in the lower-left corner inside the axes — that region is empty
    # because no policy has a strongly negative WIS lower than BC's bar end,
    # so it doesn't collide with bars, error bars, or the title.
    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Normal (ESS > 1)"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///",
                       label="Degenerate (ESS = 1)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              frameon=True, framealpha=0.95, edgecolor="0.85", fontsize=8.5)

    # Footnote below the figure
    fig.text(0.5, -0.04,
             "$^\\dagger$ Hatched bars: ESS = 1 because deterministic ($\\epsilon$-softened) "
             "target policies collapse\n"
             "WIS to a single-trajectory estimate; 95% bootstrap CI (2,000 resamples) "
             "is unreliable for these policies.",
             ha="center", va="top", fontsize=8, style="italic", color="0.3")

    fig.savefig(outdir / "fig3_wis_eval.pdf")
    fig.savefig(outdir / "fig3_wis_eval.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Single-column IRL comparison
# ---------------------------------------------------------------------------

def fig4_irl_comparison(r: Results, outdir: Path) -> None:
    policies = ["BC", "VI Komorowski", "VI IQ-Learn", "VI MaxEnt"]
    colors = [POLICIES[p].color for p in policies]
    y = np.arange(len(policies))[::-1]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(3.5, 3.2), sharey=True,
                                     gridspec_kw={"wspace": 0.12})

    # Left: V_hat
    v_vals = [r.v_hat[p] for p in policies]
    ax_l.barh(y, v_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax_l.set_xlabel(r"$\hat{V}(\mu_0)$", fontsize=10)
    ax_l.set_xlim(0, max(v_vals) * 1.25)
    for yi, v in zip(y, v_vals):
        ax_l.text(v + max(v_vals) * 0.02, yi, f"{v:.1f}",
                  va="center", ha="left", fontsize=9)

    # Right: P(survive) with clinician reference and ≈ bracket
    p_vals = [r.p_survive[p] for p in policies]
    clinician_p = r.p_survive["Clinician"]
    ax_r.barh(y, p_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax_r.axvline(clinician_p, ls="--", color=WONG["gray"], lw=0.9)
    ax_r.set_xlabel(r"P(survive)", fontsize=10)
    ax_r.set_xlim(0, 1.15)
    for yi, v in zip(y, p_vals):
        ax_r.text(v + 0.02, yi, f"{v:.3f}",
                  va="center", ha="left", fontsize=9)

    # ≈ bracket annotation between VI Komorowski (idx 1) and VI IQ-Learn (idx 2)
    y_kom = y[1]
    y_iql = y[2]
    bracket_x = 1.10
    ax_r.annotate("", xy=(bracket_x, y_iql), xytext=(bracket_x, y_kom),
                  arrowprops=dict(arrowstyle="<->", color=WONG["gray"], lw=1.0))
    ax_r.text(bracket_x + 0.015, (y_kom + y_iql) / 2, r"$\approx$",
              fontsize=12, color=WONG["gray"], va="center", ha="left")

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(policies, fontsize=9)

    fig.suptitle("IRL Policy Comparison", y=1.02, fontsize=11, fontweight="bold")
    fig.savefig(outdir / "fig4_irl_comparison.pdf")
    fig.savefig(outdir / "fig4_irl_comparison.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("figures"),
                        help="Directory where figures are written (PDF + PNG).")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    setup_style()
    results = load_results()

    fig1_model_based_eval(results, args.outdir)
    fig2_tradeoff_scatter(results, args.outdir)
    fig3_wis_eval(results, args.outdir)
    fig4_irl_comparison(results, args.outdir)

    print(f"Wrote 4 figures (PDF + PNG) to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()