"""
Behavioral Cloning: tabular policy from clinician action frequencies.

For each discrete state, the policy is the empirical action frequency
distribution observed from clinician decisions in the training data.

Usage:
    uv run python -m src.bc.train_bc
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parents[2] / "models" / "bc"

N_STATES = 750
N_ACTIONS = 25


def main():
    # metadata.csv has one row per patient-timestep with columns:
    # bloc (timestep), icustayid, outcome (0=survived, 1=died),
    # action (0-24, discretized fluid/vasopressor), state (0-749, k-means cluster)
    train_meta = pd.read_csv(DATA_DIR / "train" / "metadata.csv")
    test_meta = pd.read_csv(DATA_DIR / "test" / "metadata.csv")

    # Count how many times clinicians chose each action in each state.
    # Result is a 750x25 matrix of raw counts.
    policy = np.zeros((N_STATES, N_ACTIONS))
    for s, a in zip(train_meta["state"], train_meta["action"]):
        if s < N_STATES:  # skip absorbing states
            policy[s, a] += 1

    # Normalize rows to probabilities (uniform for unseen states)
    row_sums = policy.sum(axis=1, keepdims=True)
    unseen = (row_sums == 0).squeeze()
    policy[~unseen] /= row_sums[~unseen]
    policy[unseen] = 1.0 / N_ACTIONS

    # Evaluate: greedy accuracy on test set
    test_s = test_meta["state"].values
    test_a = test_meta["action"].values
    mask = test_s < N_STATES  # exclude absorbing states (discharge/death)
    greedy_actions = policy.argmax(axis=1)
    preds = greedy_actions[test_s[mask]]
    acc = (preds == test_a[mask]).mean() * 100

    print(f"States: {N_STATES}, Actions: {N_ACTIONS}")
    print(f"Unseen states: {unseen.sum()}")
    print(f"Test accuracy (greedy): {acc:.1f}%")

    # Top actions by frequency
    total = policy.sum(axis=0) * row_sums.sum()  # unnormalized
    freq = train_meta["action"].value_counts().sort_index()
    print("\nTop 5 actions (train):")
    for a in freq.nlargest(5).index:
        print(f"  action {a:>2}: {freq[a]:>6} ({freq[a] / len(train_meta) * 100:.1f}%)")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "bc_policy.pkl", "wb") as f:
        pickle.dump({"policy": policy, "n_states": N_STATES, "n_actions": N_ACTIONS}, f)
    print(f"\nPolicy saved to {OUT_DIR / 'bc_policy.pkl'}")


if __name__ == "__main__":
    main()
