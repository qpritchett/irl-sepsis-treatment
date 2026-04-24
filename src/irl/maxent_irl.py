"""
Tabular MaxEnt IRL on the discretized sepsis MDP.

Recovers a linear reward r(s) = phi(s) @ theta that makes the clinician
behavior look soft-optimal. The update loop is:

    r = phi @ theta
    pi = soft_value_iteration(r)          # forward RL under current reward
    D = forward_visitation(pi)            # state occupancy under pi
    grad = expert_fe - phi.T @ D - L2 * theta
    theta += lr * grad

At convergence, the model's discounted feature expectations match the expert's.
Terminal weights are frozen at Komorowski's +100/-100 to anchor the reward
scale (MaxEnt IRL is otherwise under-identified up to affine reparameterization).

Usage:
    uv run python -m src.irl.maxent_irl
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MDP_DIR = ROOT / "models" / "mdp"
OUT_DIR = ROOT / "models" / "irl"

GAMMA = 0.99
HORIZON = 20      # patients max out at 18 4-hour steps
N_ITERS = 100
LR = 0.5
L2 = 1e-2


def build_features(meta: pd.DataFrame, zs: pd.DataFrame, n_total: int,
                   s_discharge: int, s_death: int) -> np.ndarray:
    """phi(s) = [mean z-scored physiology | survive indicator | death indicator | bias]."""
    d_phys = zs.shape[1]
    d = d_phys + 3

    phi = np.zeros((n_total, d), dtype=np.float64)
    states = meta["state"].values
    X = zs.values.astype(np.float64)

    # Mean physiology per cluster. np.add.at handles repeated indices correctly.
    sums = np.zeros((n_total, d_phys))
    counts = np.zeros(n_total)
    np.add.at(sums, states, X)
    np.add.at(counts, states, 1)
    seen = counts > 0
    phi[seen, :d_phys] = sums[seen] / counts[seen, None]

    phi[s_discharge, d_phys + 0] = 1.0
    phi[s_death, d_phys + 1] = 1.0
    phi[:, -1] = 1.0  # bias
    return phi


def expert_feature_expectations(meta: pd.DataFrame, phi: np.ndarray,
                                s_discharge: int, s_death: int,
                                gamma: float, horizon: int) -> np.ndarray:
    """Discounted sum of phi(s_t) per trajectory, averaged across trajectories.

    Trajectories are padded to `horizon` with their terminal state (discharge
    or death based on outcome), matching the model's forward pass where
    absorbing states self-loop. Both sides must accumulate the same way for
    the gradient (expert_fe - model_fe) to be meaningful.
    """
    d = phi.shape[1]
    fe = np.zeros(d)
    n_traj = 0
    for _, g in meta.groupby("icustayid"):
        g = g.sort_values("bloc")
        states = list(g["state"].values)
        terminal = s_death if g["outcome"].values[0] == 1 else s_discharge
        states.append(terminal)
        while len(states) < horizon:
            states.append(terminal)
        states = states[:horizon]

        discount = 1.0
        for s in states:
            fe += discount * phi[s]
            discount *= gamma
        n_traj += 1
    return fe / n_traj


def initial_distribution(meta: pd.DataFrame, n_total: int) -> np.ndarray:
    """Empirical distribution over first-bloc states."""
    first_rows = meta.loc[meta.groupby("icustayid")["bloc"].idxmin()]
    counts = np.bincount(first_rows["state"].values, minlength=n_total)
    return counts / counts.sum()


def soft_value_iteration(r: np.ndarray, T: np.ndarray, gamma: float,
                         horizon: int, action_mask: np.ndarray) -> np.ndarray:
    """Finite-horizon soft VI. Returns pi(a|s) = softmax_a Q(s, a).

    Soft max (logsumexp) replaces hard max in the Bellman backup this is
    the computational expression of MaxEnt's "be as stochastic as possible
    subject to matching expert feature expectations".

    Forbidden (unobserved) (s, a) pairs are masked to -inf so they get zero
    probability under softmax.
    """
    S = T.shape[0]
    V = np.zeros(S)
    Q = None
    for _ in range(horizon):
        Q = r[:, None] + gamma * (T @ V)
        Q = np.where(action_mask, Q, -np.inf)
        V = logsumexp(Q, axis=1)
    return softmax(Q, axis=1)


def forward_visitation(pi: np.ndarray, T: np.ndarray, mu0: np.ndarray,
                       horizon: int, gamma: float) -> np.ndarray:
    """D(s) = sum_t gamma^t P(s_t = s | s_0 ~ mu0, pi). Must match the
    discount scheme used in expert_feature_expectations."""
    P = np.einsum("sa,sat->st", pi, T)  # marginalize actions
    d_t = mu0.copy()
    D = np.zeros_like(mu0)
    discount = 1.0
    for _ in range(horizon):
        D += discount * d_t
        d_t = d_t @ P
        discount *= gamma
    return D


def main():
    with open(MDP_DIR / "transitions.pkl", "rb") as f:
        mdp = pickle.load(f)
        
    T = mdp["T"]
    n_total = mdp["n_total"]
    s_discharge = mdp["s_discharge"]
    s_death = mdp["s_death"]

    meta = pd.read_csv(DATA_DIR / "train" / "metadata.csv")
    zs = pd.read_csv(DATA_DIR / "train" / "MIMICzs.csv")

    # Action mask: only (s, a) pairs observed in training data are allowed.
    # Unobserved pairs get a self-loop so T stays row-stochastic (else T @ V
    # is garbage for those rows). States with zero observed actions get a
    # uniform mask so soft VI can still produce a valid distribution.
    row_sums = T.sum(axis=2)
    observed = row_sums > 0
    s_idx, a_idx = np.where(~observed)
    T[s_idx, a_idx, s_idx] = 1.0
    dead_states = ~observed.any(axis=1)
    action_mask = observed.copy()
    action_mask[dead_states] = True
    print(f"  Observed (s,a): {observed.sum()} / {observed.size}; "
          f"dead states: {dead_states.sum()}")

    phi = build_features(meta, zs, n_total, s_discharge, s_death)
    d = phi.shape[1]
    print(f"  phi shape: {phi.shape}")

    print("Expert feature expectations...")
    expert_fe = expert_feature_expectations(
        meta, phi, s_discharge, s_death, GAMMA, HORIZON
    )
    print("Initial state distribution...")
    mu0 = initial_distribution(meta, n_total)

    # Freeze terminal weights at Komorowski values to anchor the reward scale.
    d_phys = d - 3
    IDX_DISCHARGE, IDX_DEATH = d_phys, d_phys + 1
    theta = np.zeros(d)
    theta[IDX_DISCHARGE] = 100.0
    theta[IDX_DEATH] = -100.0
    frozen = np.zeros(d, dtype=bool)
    frozen[[IDX_DISCHARGE, IDX_DEATH]] = True
    print(f"  Learning {(~frozen).sum()} / {d} weights "
          f"(terminal weights frozen at +100 / -100)")

    print(f"Running MaxEnt IRL for {N_ITERS} iterations...")
    for it in range(N_ITERS):
        r = phi @ theta
        pi = soft_value_iteration(r, T, GAMMA, HORIZON, action_mask)
        D = forward_visitation(pi, T, mu0, HORIZON, GAMMA)
        model_fe = phi.T @ D
        grad = expert_fe - model_fe - L2 * theta
        grad[frozen] = 0.0
        theta += LR * grad

        if it % 10 == 0 or it == N_ITERS - 1:
            err = np.linalg.norm(expert_fe - model_fe)
            print(f"  iter {it:4d}  ||fe_err||={err:.4f}  "
                  f"r[survive]={r[s_discharge]:+.2f}  r[death]={r[s_death]:+.2f}")

    r_final = phi @ theta

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "maxent.pkl", "wb") as f:
        pickle.dump({
            "theta": theta, "phi": phi, "r": r_final,
            "gamma": GAMMA, "horizon": HORIZON,
        }, f)
    print(f"\nSaved to {OUT_DIR / 'maxent.pkl'}")
    print(f"Reward range: [{r_final.min():.3f}, {r_final.max():.3f}]")
    print(f"  survive: {r_final[s_discharge]:+.3f}")
    print(f"  death:   {r_final[s_death]:+.3f}")


if __name__ == "__main__":
    main()
