"""
Tabular value iteration on the discretized sepsis MDP.

Given a reward vector r(s) and transition dynamics T(s'|s,a), computes the
optimal Q-function and greedy policy via value iteration. Used for two
downstream models:

  1. Komorowski baseline: r = +100 survive, -100 death, 0 elsewhere.
  2. MaxEnt IRL policy: r loaded from models/irl/maxent.pkl.

Unobserved (s,a) pairs are masked out so the policy can only take
actions actually seen from each state in the training data.

Usage:
    uv run python -m src.rl.value_iteration --reward komorowski
    uv run python -m src.rl.value_iteration --reward maxent
"""

import argparse
import numpy as np
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MDP_DIR = ROOT / "models" / "mdp"
IRL_DIR = ROOT / "models" / "irl"
OUT_DIR = ROOT / "models" / "rl"

GAMMA = 0.99
TOL = 1e-6
MAX_ITERS = 5000


def value_iteration(r: np.ndarray, T: np.ndarray, action_mask: np.ndarray,
                    gamma: float, tol: float, max_iters: int):
    """Standard tabular VI. Returns (V, Q, pi_greedy)."""
    S, A, _ = T.shape
    neg_inf = np.full((S, A), -1e18)
    V = np.zeros(S)
    for it in range(max_iters):
        Q = r[:, None] + gamma * (T @ V)
        Q_masked = np.where(action_mask, Q, neg_inf)
        V_new = Q_masked.max(axis=1)
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            print(f"  converged in {it + 1} iters (delta={delta:.2e})")
            break
    else:
        print(f"  hit max_iters={max_iters} (delta={delta:.2e})")

    Q = r[:, None] + gamma * (T @ V)
    Q_masked = np.where(action_mask, Q, neg_inf)
    pi = Q_masked.argmax(axis=1)
    return V, Q, pi


def build_action_mask(T: np.ndarray):
    row_sums = T.sum(axis=2)
    observed = row_sums > 0
    s_idx, a_idx = np.where(~observed)
    T[s_idx, a_idx, s_idx] = 1.0  # self-loop so T stays stochastic
    dead_states = ~observed.any(axis=1)
    mask = observed.copy()
    mask[dead_states] = True
    return mask


def komorowski_reward(n_total: int, s_discharge: int, s_death: int):
    r = np.zeros(n_total)
    r[s_discharge] = 100.0
    r[s_death] = -100.0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["komorowski", "maxent"], required=True)
    args = ap.parse_args()

    with open(MDP_DIR / "transitions.pkl", "rb") as f:
        mdp = pickle.load(f)
    T = mdp["T"]
    n_total = mdp["n_total"]
    s_discharge = mdp["s_discharge"]
    s_death = mdp["s_death"]

    action_mask = build_action_mask(T)
    print(f"Observed (s,a): {action_mask.sum()} / {action_mask.size}")

    if args.reward == "komorowski":
        r = komorowski_reward(n_total, s_discharge, s_death)
    else:
        with open(IRL_DIR / "maxent.pkl", "rb") as f:
            irl = pickle.load(f)
        r = irl["r"]
    print(f"Reward: {args.reward}  range=[{r.min():.3f}, {r.max():.3f}]")

    print("Running value iteration...")
    V, Q, pi = value_iteration(r, T, action_mask, GAMMA, TOL, MAX_ITERS)

    # Policy summary
    from collections import Counter
    top_actions = Counter(pi.tolist()).most_common(5)
    print(f"V range: [{V.min():.2f}, {V.max():.2f}]")
    print(f"Top actions: {top_actions}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"vi_{args.reward}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "V": V, "Q": Q, "pi": pi,
            "reward_type": args.reward, "gamma": GAMMA,
        }, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
