"""
Pre-compute actions and state clusters for train/test splits.

Adds 'action' and 'state' columns to metadata.csv files.
Fits action bins and k-means on training data, applies to both splits.

Action discretization copied from CMU AI-Clinician-MIMICIV common.py
to avoid pulling in their full dependency chain (numba, etc.).

Usage:
    uv run python -m src.precompute
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

N_CLUSTERS = 750
N_ACTION_BINS = 5


def fit_action_bins(input_amounts, vaso_doses, n_action_bins=5):
    """
    Groups fluid inputs and vasopressor doses into discrete bins based on
    percentile ranks. Returns (actions, medians, cutoffs).

    Copied from AI-Clinician-MIMICIV/ai_clinician/modeling/models/common.py
    """
    bin_percentiles = np.linspace(0, 100, n_action_bins - 1, endpoint=False)

    input_cutoffs = [0.0] + np.percentile(
        input_amounts[input_amounts > 0], bin_percentiles
    ).tolist()
    io = np.digitize(input_amounts, input_cutoffs)
    median_inputs = [
        np.median(input_amounts[io == bin_num])
        for bin_num in range(1, n_action_bins + 1)
    ]

    vaso_cutoffs = [0.0] + np.percentile(
        vaso_doses[vaso_doses > 0], bin_percentiles
    ).tolist()
    vc = np.digitize(vaso_doses, vaso_cutoffs)
    median_vaso = [
        np.median(vaso_doses[vc == bin_num])
        for bin_num in range(1, n_action_bins + 1)
    ]

    med = np.array([io, vc])
    actions = (med[0] - 1) * n_action_bins + (med[1] - 1)

    return actions, (median_inputs, median_vaso), (input_cutoffs, vaso_cutoffs)


def transform_actions(input_amounts, vaso_doses, cutoffs):
    """
    Transforms continuous fluid and vasopressor values into discrete actions
    using pre-fitted cutoffs.

    Copied from AI-Clinician-MIMICIV/ai_clinician/modeling/models/common.py
    """
    input_cutoffs, vaso_cutoffs = cutoffs
    return (
        len(input_cutoffs) * (np.digitize(input_amounts, input_cutoffs) - 1)
        + (np.digitize(vaso_doses, vaso_cutoffs) - 1)
    )


def main():
    # Load data
    train_raw = pd.read_csv(DATA_DIR / "train" / "MIMICraw.csv")
    train_zs = pd.read_csv(DATA_DIR / "train" / "MIMICzs.csv")
    train_meta = pd.read_csv(DATA_DIR / "train" / "metadata.csv")

    test_raw = pd.read_csv(DATA_DIR / "test" / "MIMICraw.csv")
    test_zs = pd.read_csv(DATA_DIR / "test" / "MIMICzs.csv")
    test_meta = pd.read_csv(DATA_DIR / "test" / "metadata.csv")

    # --- Actions ---
    print("Fitting action bins on training data...")
    train_actions, medians, cutoffs = fit_action_bins(
        train_raw["input_step"].values,
        train_raw["max_dose_vaso"].values,
        n_action_bins=N_ACTION_BINS,
    )
    test_actions = transform_actions(
        test_raw["input_step"].values,
        test_raw["max_dose_vaso"].values,
        cutoffs,
    )

    # Clamp out-of-range actions (23 rows with negative input_step)
    train_valid = (train_actions >= 0) & (train_actions < 25)
    test_valid = (test_actions >= 0) & (test_actions < 25)
    print(f"  Train: {(~train_valid).sum()} invalid actions clamped to 0")
    print(f"  Test:  {(~test_valid).sum()} invalid actions clamped to 0")
    train_actions = np.clip(train_actions, 0, 24)
    test_actions = np.clip(test_actions, 0, 24)

    # --- State clusters ---
    print(f"Fitting k-means ({N_CLUSTERS} clusters) on training data...")
    sample_idx = np.random.RandomState(42).choice(
        len(train_zs), size=len(train_zs) // 4, replace=False
    )
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, n_init=32, max_iter=30, random_state=42
    ).fit(train_zs.values[sample_idx])

    train_states = kmeans.predict(train_zs.values)
    test_states = kmeans.predict(test_zs.values)

    # --- Save ---
    train_meta["action"] = train_actions
    train_meta["state"] = train_states
    train_meta.to_csv(DATA_DIR / "train" / "metadata.csv", index=False)

    test_meta["action"] = test_actions
    test_meta["state"] = test_states
    test_meta.to_csv(DATA_DIR / "test" / "metadata.csv", index=False)

    with open(DATA_DIR / "kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(DATA_DIR / "action_bins.pkl", "wb") as f:
        pickle.dump({"cutoffs": cutoffs, "medians": medians}, f)

    print(f"\nSaved to metadata.csv (train/test), kmeans.pkl, action_bins.pkl")
    print(f"  Actions: 0-24 ({N_ACTION_BINS}x{N_ACTION_BINS} grid)")
    print(f"  States: 0-{N_CLUSTERS-1} ({N_CLUSTERS} clusters)")
    print(f"  + 2 absorbing: {N_CLUSTERS} (discharge), {N_CLUSTERS+1} (death)")


if __name__ == "__main__":
    main()
