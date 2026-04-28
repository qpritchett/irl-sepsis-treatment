"""
Per-policy cumulative reward distribution histograms, stratified by outcome.

Generates two style variants:
    fig5a_reward_dist_original.{pdf,png} — red/green default matplotlib style
    fig5b_reward_dist_paper.{pdf,png}    — serif + Wong palette, paper style

Five panels (top-to-bottom): Clinician, BC, PPO MaxEnt, VI IQ-Learn, VI MaxEnt.

NOTE: Trajectory-level rewards are SIMULATED here for layout testing. Replace
`simulate_trajectories()` with a loader that reads your real eval outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Wong palette (shared with plot_results.py)
# ---------------------------------------------------------------------------

WONG = {
    "gray":   "#999999",
    "blue":   "#0072B2",
    "green":  "#009E73",
    "orange": "#D55E00",
    "yellow": "#E69F00",
    "purple": "#CC79A7",
}


# ---------------------------------------------------------------------------
# Data — simulated trajectory rewards stratified by outcome
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryRewards:
    """Per-trajectory cumulative rewards for a single policy."""
    survived: np.ndarray  # (n_survived,) cumulative reward per trajectory
    died: np.ndarray      # (n_died,)
    survived_mean: float
    died_mean: float


def simulate_trajectories(seed: int = 42) -> dict[str, TrajectoryRewards]:
    """
    Simulate per-trajectory cumulative rewards for each policy.

    Patterns chosen to match the qualitative behavior in the original figure:
    - Clinician: broad right-skewed distribution, clear separation between outcomes
    - BC / PPO: large spike at 0 (short-circuited episodes) + long right tail
    - IQ-Learn / VI MaxEnt: intermediate — partial spike, broader distribution

    Replace this function with a real loader once eval data is serialized:
        with open("results/trajectory_rewards.json") as f:
            data = json.load(f)
        return {p: TrajectoryRewards(np.array(d["survived"]), np.array(d["died"]),
                                     d["survived_mean"], d["died_mean"])
                for p, d in data.items()}
    """
    rng = np.random.default_rng(seed)

    def make_clinician(n_surv: int = 7000, n_died: int = 1000) -> TrajectoryRewards:
        # Right-skewed lognormal-like distribution centered around 2-3
        surv = rng.lognormal(mean=0.9, sigma=0.55, size=n_surv) - 0.3
        died = rng.lognormal(mean=0.85, sigma=0.65, size=n_died) - 0.3
        return TrajectoryRewards(surv, died, float(surv.mean()), float(died.mean()))

    def make_spiked(n_surv: int, n_died: int,
                    spike_frac_surv: float, spike_frac_died: float,
                    tail_scale: float) -> TrajectoryRewards:
        # Mixture: large spike near 0 + heavy right tail
        n_spike_s = int(n_surv * spike_frac_surv)
        n_tail_s  = n_surv - n_spike_s
        surv = np.concatenate([
            rng.normal(0.0, 0.15, n_spike_s),
            rng.exponential(tail_scale, n_tail_s),
        ])

        n_spike_d = int(n_died * spike_frac_died)
        n_tail_d  = n_died - n_spike_d
        died = np.concatenate([
            rng.normal(0.0, 0.15, n_spike_d),
            rng.exponential(tail_scale * 0.9, n_tail_d),
        ])
        return TrajectoryRewards(surv, died, float(surv.mean()), float(died.mean()))

    return {
        "Clinician (Actual Trajectories)": make_clinician(),
        "BC Policy (Simulated)":            make_spiked(7000, 1000, 0.35, 0.40, 3.5),
        "PPO MaxEnt Policy (Simulated)":    make_spiked(6500, 1500, 0.30, 0.45, 4.0),
        "VI IQ-Learn Policy (Simulated)":   make_spiked(7200,  800, 0.20, 0.30, 3.8),
        "VI MaxEnt Policy (Simulated)":     make_spiked(6800, 1200, 0.25, 0.35, 3.6),
    }


def trim_outliers(data: TrajectoryRewards,
                  lo_pct: float = 1.0,
                  hi_pct: float = 99.0) -> TrajectoryRewards:
    """Trim to [lo_pct, hi_pct] percentile range across the combined sample."""
    combined = np.concatenate([data.survived, data.died])
    lo = np.percentile(combined, lo_pct)
    hi = np.percentile(combined, hi_pct)
    return TrajectoryRewards(
        survived=data.survived[(data.survived >= lo) & (data.survived <= hi)],
        died=data.died[(data.died >= lo) & (data.died <= hi)],
        survived_mean=data.survived_mean,  # keep original means for annotation
        died_mean=data.died_mean,
    )


# ---------------------------------------------------------------------------
# Variant A — Original style (red/green, default matplotlib)
# ---------------------------------------------------------------------------

def plot_original_style(rewards: dict[str, TrajectoryRewards], outdir: Path) -> None:
    """Match the screenshot: red/green colors, default matplotlib styling."""
    plt.rcdefaults()  # reset to defaults so paper-style rcParams don't leak in

    fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)

    # Common bin edges across panels for honest comparison
    all_data = np.concatenate([
        np.concatenate([d.survived, d.died]) for d in rewards.values()
    ])
    bins = np.linspace(np.percentile(all_data, 1),
                       np.percentile(all_data, 99), 50)

    for ax, (title, data) in zip(axes, rewards.items()):
        trimmed = trim_outliers(data)

        ax.hist(trimmed.survived, bins=bins, color="#7FB77E", alpha=0.85,
                label=f"Survived Mean: {data.survived_mean:.2f}", edgecolor="none")
        ax.hist(trimmed.died, bins=bins, color="#C75555", alpha=0.85,
                label=f"Died Mean: {data.died_mean:.2f}", edgecolor="none")

        ax.axvline(data.survived_mean, color="#2D7A2D", ls="--", lw=1.2)
        ax.axvline(data.died_mean,     color="#8B1A1A", ls="--", lw=1.2)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Patient Count")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Cumulative Reward (Outliers Trimmed)")
    fig.tight_layout()
    fig.savefig(outdir / "fig5a_reward_dist_original.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig5a_reward_dist_original.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Variant B — Paper style (serif, Wong palette, matches figs 1-4)
# ---------------------------------------------------------------------------

def setup_paper_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset":  "dejavuserif",
        "axes.titlesize":    10,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   8.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "axes.titleweight":  "bold",
        "savefig.dpi":       300,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def plot_paper_style(rewards: dict[str, TrajectoryRewards], outdir: Path) -> None:
    """Paper style: serif, Wong palette, clean spines, mean values annotated on lines."""
    setup_paper_style()

    fig, axes = plt.subplots(5, 1, figsize=(7, 8.5), sharex=True)

    color_surv = WONG["blue"]
    color_died = WONG["orange"]

    all_data = np.concatenate([
        np.concatenate([d.survived, d.died]) for d in rewards.values()
    ])
    bins = np.linspace(np.percentile(all_data, 1),
                       np.percentile(all_data, 99), 50)

    for i, (ax, (title, data)) in enumerate(zip(axes, rewards.items())):
        trimmed = trim_outliers(data)

        ax.hist(trimmed.survived, bins=bins, color=color_surv, alpha=0.7,
                label="Survived" if i == 0 else None, edgecolor="none")
        ax.hist(trimmed.died, bins=bins, color=color_died, alpha=0.75,
                label="Died" if i == 0 else None, edgecolor="none")

        ax.axvline(data.survived_mean, color=color_surv, ls="--", lw=1.0, alpha=0.9)
        ax.axvline(data.died_mean,     color=color_died, ls="--", lw=1.0, alpha=0.9)

        # Annotate mean values on the lines themselves (cleaner than legend)
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
        ax.text(data.survived_mean, ymax * 0.95, f" μ={data.survived_mean:.2f}",
                color=color_surv, fontsize=8, va="top", ha="left", fontweight="bold")
        ax.text(data.died_mean, ymax * 0.85, f" μ={data.died_mean:.2f}",
                color=color_died, fontsize=8, va="top", ha="left", fontweight="bold")

        ax.set_title(title, loc="left", pad=4)
        ax.set_ylabel("Patient count")

        # Single shared legend at top, frameless
        if i == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=9)

    axes[-1].set_xlabel(r"Cumulative reward")
    fig.text(0.5, -0.005,
             r"Distributions trimmed to 1st–99th percentile. Dashed lines mark per-outcome means.",
             ha="center", va="top", fontsize=8, style="italic", color="0.3")

    fig.tight_layout()
    fig.savefig(outdir / "fig5b_reward_dist_paper.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig5b_reward_dist_paper.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rewards = simulate_trajectories(seed=args.seed)

    plot_original_style(rewards, args.outdir)
    plot_paper_style(rewards, args.outdir)

    print(f"Wrote 2 variants (PDF + PNG each) to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()