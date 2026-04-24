"""
Off-policy evaluation of learned policies.

Two estimators are reported:

1. Model-based direct evaluation (the trustworthy one here).
   We have a tabular T estimated from training data and the reward is
   terminal-only (+100 survive, -100 death). For each target policy pi_e,
   iterate V = r + gamma * P_pi @ V to convergence, then report V under
   the test-set initial state distribution mu_0. This is exact given T.

2. Weighted importance sampling (sanity check; degenerate for
   near-deterministic target policies because the effective sample size
   collapses to ~1). Kept for comparison and to show the ESS problem.

All target policies are softened: (1-eps) * greedy + eps * uniform. The
behavior policy pi_b is the empirical clinician policy from training.

Evaluated policies:
  - clinician   (pi_b, sanity check)
  - bc          (models/bc/bc_policy.pkl)
  - vi_komorowski (models/rl/vi_komorowski.pkl)
  - vi_maxent     (models/rl/vi_maxent.pkl)

Usage:
    uv run python -m src.eval.evaluate
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MDP_DIR = ROOT / "models" / "mdp"
BC_DIR = ROOT / "models" / "bc"
RL_DIR = ROOT / "models" / "rl"

GAMMA = 0.99
EPS = 0.01             # softening: (1-eps) * greedy/empirical + eps * uniform
N_ACTIONS = 25
N_STATES = 750         # clinical clusters (absorbing handled separately)
N_BOOTSTRAP = 2000
SEED = 42


def soften(policy: np.ndarray, eps: float) -> np.ndarray:
    """(1-eps) * policy + eps * uniform. Row-stochastic in -> out."""
    A = policy.shape[1]
    return (1 - eps) * policy + eps / A


def greedy_to_stochastic(pi_greedy: np.ndarray, n_actions: int) -> np.ndarray:
    S = pi_greedy.shape[0]
    out = np.zeros((S, n_actions))
    out[np.arange(S), pi_greedy] = 1.0
    return out


def build_test_trajectories(meta: pd.DataFrame):
    """Return list of (states, actions, outcome) per patient."""
    trajs = []
    for _, g in meta.groupby("icustayid"):
        g = g.sort_values("bloc")
        trajs.append((
            g["state"].values.astype(int),
            g["action"].values.astype(int),
            int(g["outcome"].values[0]),
        ))
    return trajs


def wis_estimate(trajs, pi_e: np.ndarray, pi_b: np.ndarray, gamma: float):
    """
    Standard WIS with terminal-reward returns (+100/-100).
    Returns (per-trajectory weights, per-trajectory returns).
    """
    rhos = np.empty(len(trajs))
    returns = np.empty(len(trajs))
    for i, (states, actions, outcome) in enumerate(trajs):
        # Importance ratio over the trajectory.
        # Any state >= N_STATES (absorbing) is skipped — actions aren't taken there.
        clinical = states < N_STATES
        s = states[clinical]
        a = actions[clinical]
        log_rho = (np.log(pi_e[s, a]) - np.log(pi_b[s, a])).sum()
        rhos[i] = np.exp(log_rho)

        # Discounted return: all reward is terminal.
        H = len(s)  # number of steps taken
        terminal_r = -100.0 if outcome == 1 else 100.0
        returns[i] = (gamma ** H) * terminal_r
    return rhos, returns


def wis_value(rhos: np.ndarray, returns: np.ndarray) -> float:
    denom = rhos.sum()
    if denom <= 0:
        return float("nan")
    return float((rhos * returns).sum() / denom)


def bootstrap_ci(rhos, returns, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(rhos)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = wis_value(rhos[idx], returns[idx])
    return np.nanpercentile(vals, [2.5, 50, 97.5])


def model_based_eval(pi_e_full: np.ndarray, T: np.ndarray, r: np.ndarray,
                     mu0: np.ndarray, gamma: float,
                     tol: float = 1e-8, max_iters: int = 5000) -> float:
    """
    Exact iterative policy evaluation on the tabular MDP.
    pi_e_full: (N_TOTAL, A). T: (N_TOTAL, A, N_TOTAL). r: (N_TOTAL,).
    Returns V^pi under the initial distribution mu0.
    """
    # P_pi[s, s'] = sum_a pi(a|s) T(s, a, s')
    P_pi = np.einsum("sa,sat->st", pi_e_full, T)
    V = np.zeros_like(r)
    for _ in range(max_iters):
        V_new = r + gamma * (P_pi @ V)
        if np.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new
    return float(mu0 @ V)


def test_initial_distribution(meta: pd.DataFrame, n_total: int) -> np.ndarray:
    mu0 = np.zeros(n_total)
    firsts = meta.sort_values(["icustayid", "bloc"]).groupby("icustayid").first()
    for s in firsts["state"].values:
        mu0[int(s)] += 1
    return mu0 / mu0.sum()


def extend_to_full(pi_clinical: np.ndarray, n_total: int) -> np.ndarray:
    """Extend a (N_STATES, A) policy to (N_TOTAL, A) with uniform over the
    two absorbing states (they self-loop so the choice is irrelevant)."""
    A = pi_clinical.shape[1]
    full = np.zeros((n_total, A))
    full[:pi_clinical.shape[0]] = pi_clinical
    full[pi_clinical.shape[0]:] = 1.0 / A
    return full


def load_policies():
    with open(MDP_DIR / "transitions.pkl", "rb") as f:
        mdp = pickle.load(f)
    pi_b_full = mdp["pi_b"]  # (752, 25)
    pi_b = pi_b_full[:N_STATES]

    with open(BC_DIR / "bc_policy.pkl", "rb") as f:
        bc = pickle.load(f)["policy"]

    with open(RL_DIR / "vi_komorowski.pkl", "rb") as f:
        vi_k = pickle.load(f)["pi"][:N_STATES]
    with open(RL_DIR / "vi_maxent.pkl", "rb") as f:
        vi_m = pickle.load(f)["pi"][:N_STATES]

    return {
        "clinician":     soften(pi_b, EPS),
        "bc":            soften(bc, EPS),
        "vi_komorowski": soften(greedy_to_stochastic(vi_k, N_ACTIONS), EPS),
        "vi_maxent":     soften(greedy_to_stochastic(vi_m, N_ACTIONS), EPS),
    }, soften(pi_b, EPS)


def main():
    print("Loading test trajectories...")
    test_meta = pd.read_csv(DATA_DIR / "test" / "metadata.csv")
    trajs = build_test_trajectories(test_meta)
    print(f"  {len(trajs)} trajectories")
    mortality = np.mean([t[2] for t in trajs]) * 100
    mean_return = np.mean([
        (GAMMA ** (t[0] < N_STATES).sum()) * (-100 if t[2] == 1 else 100)
        for t in trajs
    ])
    print(f"  observed mortality: {mortality:.1f}%")
    print(f"  mean discounted return (on-policy, clinician): {mean_return:.2f}")

    policies, pi_b = load_policies()

    # ---- Model-based direct evaluation ----
    with open(MDP_DIR / "transitions.pkl", "rb") as f:
        mdp = pickle.load(f)
    T = mdp["T"]
    n_total = mdp["n_total"]
    s_discharge = mdp["s_discharge"]
    s_death = mdp["s_death"]

    # Komorowski ground-truth reward: +100 survive, -100 die, 0 elsewhere.
    r_true = np.zeros(n_total)
    r_true[s_discharge] = 100.0
    r_true[s_death] = -100.0

    mu0 = test_initial_distribution(test_meta, n_total)

    print(f"\nModel-based direct evaluation (gamma={GAMMA})")
    print(f"{'policy':>15}  {'V_hat(mu0)':>12}  {'P(survive)':>12}")
    print("-" * 45)
    for name, pi_e in policies.items():
        pi_full = extend_to_full(pi_e, n_total)
        v = model_based_eval(pi_full, T, r_true, mu0, GAMMA)
        # V in [-100, 100]/(1-gamma)... really V in ~[-100, 100] because
        # all reward is absorbed within finite hitting time. Convert to
        # implied survival probability: V ~ 100*(2p - 1)*gamma^E[H]
        # Report raw V and simple bound estimate of P(survive).
        p_surv = (v + 100) / 200  # approx, tightest when gamma^H -> 1
        print(f"{name:>15}  {v:>12.3f}  {p_surv:>12.3f}")

    # ---- WIS (kept as sanity check / comparison) ----
    print(f"\nWIS (eps={EPS}, n_boot={N_BOOTSTRAP}) — degenerate for deterministic pi_e")
    print(f"{'policy':>15}  {'V_WIS':>10}  {'95% CI':>22}  {'ESS':>8}")
    print("-" * 60)
    for name, pi_e in policies.items():
        rhos, returns = wis_estimate(trajs, pi_e, pi_b, GAMMA)
        v = wis_value(rhos, returns)
        lo, med, hi = bootstrap_ci(rhos, returns, N_BOOTSTRAP, SEED)
        ess = (rhos.sum() ** 2) / (rhos ** 2).sum() if (rhos ** 2).sum() > 0 else 0.0
        print(f"{name:>15}  {v:>10.3f}  [{lo:>8.2f}, {hi:>8.2f}]  {ess:>8.1f}")


if __name__ == "__main__":
    main()
