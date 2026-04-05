"""
Estimate transition dynamics and behavior policy from training data.

Counts (s, a, s') triples across patient trajectories, then normalizes
to get T(s'|s,a). Also computes the empirical clinician behavior policy
pi_b(a|s) from action counts per state.

Absorbing states: 750 = discharge (survived), 751 = death.
The last timestep of each trajectory transitions to the appropriate
absorbing state. Absorbing states self-loop under all actions.

Usage:
    uv run python -m src.mdp.transitions
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parents[2] / "models" / "mdp"

N_STATES = 750
N_ACTIONS = 25
S_DISCHARGE = 750  # absorbing state for survival
S_DEATH = 751      # absorbing state for death
N_TOTAL = 752      # 750 clinical + 2 absorbing

# Transitions seen fewer than this many times are zeroed out (sparse data)
MIN_COUNT = 5


def build_trajectories(meta):
    """Group metadata rows into per-patient trajectories, ordered by bloc."""
    trajectories = []
    for _, group in meta.groupby("icustayid"):
        group = group.sort_values("bloc")
        trajectories.append(group)
    return trajectories


def main():
    meta = pd.read_csv(DATA_DIR / "train" / "metadata.csv")
    print(f"Training data: {len(meta)} rows, {meta['icustayid'].nunique()} patients")

    # Count (s, a, s') transitions across all trajectories
    counts = np.zeros((N_TOTAL, N_ACTIONS, N_TOTAL))

    trajectories = build_trajectories(meta)
    for traj in trajectories:
        states = traj["state"].values
        actions = traj["action"].values
        outcome = traj["outcome"].values[0]  # same for all rows in trajectory
        terminal = S_DEATH if outcome == 1 else S_DISCHARGE

        # Intermediate transitions: s_t --a_t--> s_{t+1}
        for t in range(len(states) - 1):
            counts[states[t], actions[t], states[t + 1]] += 1

        # Final transition: last state --> absorbing state
        counts[states[-1], actions[-1], terminal] += 1

    # Absorbing states self-loop under all actions
    for a in range(N_ACTIONS):
        counts[S_DISCHARGE, a, S_DISCHARGE] = 1
        counts[S_DEATH, a, S_DEATH] = 1

    # Zero out transitions seen fewer than MIN_COUNT times
    rare = (counts > 0) & (counts < MIN_COUNT)
    n_rare = rare.sum()
    counts[rare] = 0
    print(f"Zeroed {n_rare} rare transitions (count < {MIN_COUNT})")

    # Normalize to get T(s'|s,a)
    row_sums = counts.sum(axis=2, keepdims=True)
    # For (s,a) pairs never observed, leave as zero (no valid transition)
    nonzero = row_sums > 0
    T = np.zeros_like(counts)
    T[nonzero.squeeze(axis=2)] = counts[nonzero.squeeze(axis=2)]
    row_sums_safe = np.where(row_sums > 0, row_sums, 1)
    T = T / row_sums_safe

    # Behavior policy: pi_b(a|s) = count(s,a) / count(s)
    sa_counts = counts.sum(axis=2)  # shape (752, 25)
    s_counts = sa_counts.sum(axis=1, keepdims=True)
    pi_b = np.zeros_like(sa_counts)
    seen = (s_counts > 0).squeeze()
    pi_b[seen] = sa_counts[seen] / s_counts[seen]
    pi_b[~seen] = 1.0 / N_ACTIONS  # uniform for unseen states

    # Summary stats
    nonzero_sa = (sa_counts > 0).sum()
    print(f"Non-zero (s,a) pairs: {nonzero_sa} / {N_TOTAL * N_ACTIONS}")
    print(f"Transition matrix shape: {T.shape}")
    print(f"Behavior policy shape: {pi_b.shape}")

    # Sanity check: rows of T should sum to 1 where observed, 0 where not
    row_check = T.sum(axis=2)
    observed = row_check > 0
    print(f"T row sums (observed): min={row_check[observed].min():.6f}, "
          f"max={row_check[observed].max():.6f}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "transitions.pkl", "wb") as f:
        pickle.dump({
            "T": T,
            "pi_b": pi_b,
            "n_states": N_STATES,
            "n_actions": N_ACTIONS,
            "n_total": N_TOTAL,
            "s_discharge": S_DISCHARGE,
            "s_death": S_DEATH,
            "min_count": MIN_COUNT,
        }, f)
    print(f"\nSaved to {OUT_DIR / 'transitions.pkl'}")


if __name__ == "__main__":
    main()
