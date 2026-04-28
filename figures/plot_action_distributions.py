"""
figures/plot_action_distributions.py

Reproduces and extends the action distribution figure with all 6 policies.

Usage:
    python -m figures.plot_action_distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
MDP_DIR = ROOT / "models" / "mdp"
BC_DIR  = ROOT / "models" / "bc"
RL_DIR  = ROOT / "models" / "rl"
IQ_DIR  = ROOT / "models" / "iq_learn"
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STATES  = 750
N_ACTIONS = 25

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})

BLUE = "#4878CF"


def greedy_to_stochastic(pi_greedy, n_actions):
    S = pi_greedy.shape[0]
    out = np.zeros((S, n_actions))
    out[np.arange(S), pi_greedy] = 1.0
    return out


def load_policies():
    with open(MDP_DIR / "transitions.pkl", "rb") as f:
        mdp = pickle.load(f)
    pi_b = mdp["pi_b"][:N_STATES]

    with open(BC_DIR / "bc_policy.pkl", "rb") as f:
        bc = pickle.load(f)["policy"]

    with open(RL_DIR / "vi_komorowski.pkl", "rb") as f:
        vi_k = pickle.load(f)["pi"][:N_STATES]

    with open(RL_DIR / "vi_maxent.pkl", "rb") as f:
        vi_m = pickle.load(f)["pi"][:N_STATES]

    with open(RL_DIR / "ppo_maxent.pkl", "rb") as f:
        ppo_m = pickle.load(f)["pi"][:N_STATES]

    with open(IQ_DIR / "iq_learn.pkl", "rb") as f:
        vi_iq = pickle.load(f)["pi"][:N_STATES]

    return {
        "Clinician":        pi_b,
        "BC":               bc,
        "VI (Komorowski)":  greedy_to_stochastic(vi_k, N_ACTIONS),
        "VI (MaxEnt IRL)":  greedy_to_stochastic(vi_m, N_ACTIONS),
        "PPO (MaxEnt)":     greedy_to_stochastic(ppo_m, N_ACTIONS),
        "VI (IQ-Learn)":    greedy_to_stochastic(vi_iq, N_ACTIONS),
    }


def plot_action_distributions():
    print("Generating action distribution plots...")

    policies = load_policies()
    names    = list(policies.keys())
    n        = len(names)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        pi = policies[name]           # (750, 25)
        mean_prob = pi.mean(axis=0)   # average probability per action across states

        ax.bar(np.arange(N_ACTIONS), mean_prob,
               color=BLUE, alpha=0.85, width=0.7, edgecolor="white", linewidth=0.3)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Action (0 = no treatment)", fontsize=8)
        ax.set_ylabel("Mean probability", fontsize=8)
        ax.set_xlim(-1, N_ACTIONS)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Action Distributions Across Policies\n"
                 "(mean probability per action, averaged over 750 states)",
                 fontsize=11)
    fig.tight_layout()

    out = OUT_DIR / "action_distributions"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved: {out}.pdf / .png")
    plt.close(fig)


if __name__ == "__main__":
    plot_action_distributions()