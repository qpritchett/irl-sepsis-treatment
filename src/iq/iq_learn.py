"""
Tabular IQ-Learn on the discretized sepsis MDP (Garg et al., 2021).

Recovers an implicit reward and Q-function directly from expert demonstrations
using the chi-squared IQ-Learn objective (Eq. 12 in the paper). Unlike MaxEnt
IRL, no inner RL loop is needed: the reward and policy are derived jointly from
a single Q-table updated via gradient descent on:

    L(Q) = -E_E[Q(s,a) - V*(s)] + (1/4α) * E_E[r(s,a)²]

where V*(s) = logsumexp_a Q(s,a) and r(s,a) = Q(s,a) - γ * E_{s'}[V*(s')].

Terminal state Q-values are frozen at ±(100 - log A) so that logsumexp over
actions gives V*(s_terminal) = ±100, matching the Komorowski scale.

Usage:
    python -m src.iq_learn.iq_learn
    python -m src.iq_learn.iq_learn --epochs 500 --lr 0.01 --alpha 1.0
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MDP_DIR = ROOT / "models" / "mdp"
OUT_DIR = ROOT / "models" / "iq_learn"

GAMMA = 0.99
BATCH_SIZE = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500,
                    help="Number of gradient descent iterations (default: 500)")
    ap.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate for Q update (default: 0.01)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Chi-squared regularization coefficient (default: 1.0)")
    args = ap.parse_args()

    # ---- Load MDP ----
    with open(MDP_DIR / "transitions.pkl", "rb") as f:
        mdp = pickle.load(f)
    T = mdp["T"]                      # (752, 25, 752)
    n_total = mdp["n_total"]          # 752
    s_discharge = mdp["s_discharge"]  # 750
    s_death = mdp["s_death"]          # 751
    pi_b = mdp["pi_b"]               # (752, 25) empirical clinician policy

    S, A = n_total, T.shape[1]

    # Action mask: only (s,a) pairs observed in training data
    action_mask = pi_b > 0  # (752, 25) bool

    # ---- Load expert transitions ----
    meta = pd.read_csv(DATA_DIR / "train" / "metadata.csv")
    expert_states = meta["state"].values.astype(int)
    expert_actions = meta["action"].values.astype(int)

    # Drop any pairs that fall outside the action mask (safety check)
    observed = action_mask[expert_states, expert_actions]
    expert_states = expert_states[observed]
    expert_actions = expert_actions[observed]

    print(f"IQ-Learn: {len(expert_states)} expert transitions "
          f"({(~observed).sum()} unobserved filtered)")
    print(f"  LR={args.lr}  alpha={args.alpha}  gamma={GAMMA}  epochs={args.epochs}")
    print(f"  States={S}  Actions={A}  "
          f"Observed (s,a)={action_mask.sum()} / {action_mask.size}")

    # ---- Initialise Q ----
    # Warm-start from soft VI under Komorowski terminal reward so that
    # r(s,a) = Q - γ·EV_next is small at init, preventing the r² term
    # from dominating the IQ-Learn gradient.
    r_komorowski = np.zeros(S)
    r_komorowski[s_discharge] = 100.0
    r_komorowski[s_death] = -100.0

    Q = np.zeros((S, A), dtype=np.float64)
    for _ in range(50):
        V_warm = logsumexp(Q, axis=1)
        Q = r_komorowski[:, None] + GAMMA * (T @ V_warm)
        Q[s_discharge, :] = 100.0 - np.log(A)
        Q[s_death, :]     = -100.0 - np.log(A)

        # Freeze terminal states: uniform Q so logsumexp gives V*(terminal) = ±100.
        Q[s_discharge, :] = 100.0 - np.log(A)   # logsumexp → +100.0
        Q[s_death, :]     = -100.0 - np.log(A)  # logsumexp → -100.0

    rng = np.random.default_rng(42)

    # ---- Training loop ----
    for it in range(args.epochs):
        # Step 1 — soft value function and policy
        V = logsumexp(Q, axis=1)          # (S,)
        pi_soft = softmax(Q, axis=1)      # (S, A)

        # Step 2 — expected next-state value E_{s'}[V*(s')] for each (s,a)
        EV_next = T @ V                   # (S, A, 752) @ (752,) → (S, A)

        # Step 3 — implicit reward r(s,a) = Q(s,a) - γ * E_{s'}[V*(s')]
        r_arr = Q - GAMMA * EV_next       # (S, A)

        # Step 4 — sample minibatch from expert transitions
        idx = rng.integers(0, len(expert_states), size=min(BATCH_SIZE, len(expert_states)))
        s_b = expert_states[idx]
        a_b = expert_actions[idx]
        N = len(s_b)

        # Step 5 — compute loss for logging
        r_b = r_arr[s_b, a_b]
        loss = -np.mean(Q[s_b, a_b] - V[s_b]) + (1.0 / (4.0 * args.alpha)) * np.mean(r_b ** 2)

        # Step 6 — gradient of L w.r.t. Q (stop-gradient on V inside EV_next)
        #
        # ∂L/∂Q[s,a] = -count_sa/N           (direct from -Q term)
        #             + pi[s,a]*count_s/N     (through V*(s) = logsumexp Q[s,:])
        #             + r[s,a]*count_sa/(2αN) (through r² term, Q direct only)
        G = np.zeros((S, A), dtype=np.float64)
        state_counts = np.bincount(s_b, minlength=S).astype(np.float64)

        np.add.at(G, (s_b, a_b), -1.0 / N)                          # -count_sa / N
        G += pi_soft * (state_counts[:, None] / N)                   # +pi * count_s / N
        np.add.at(G, (s_b, a_b), r_b / (2.0 * args.alpha * N))      # r * count_sa / (2αN)

        # Zero gradient only for frozen terminal states.
        # NOTE: do NOT zero unobserved (s,a) pairs here — the gradient through
        # V*(s) = logsumexp Q[s,:] is valid for all actions and zeroing it
        # kills learning by cancelling the -Q and +V terms.
        G[s_discharge, :] = 0.0
        G[s_death, :] = 0.0

        if it % 10 == 0 or it == args.epochs - 1:
            G_norm = np.abs(G).max()
            Q_nonterm = Q[:s_discharge]   # exclude absorbing states
            V_temp = logsumexp(Q, axis=1)
            print(f"  iter {it:4d}  loss={loss:+.4f}  "
                  f"|G|_max={G_norm:.6f}  "
                  f"Q_nonterm=[{Q_nonterm.min():.3f}, {Q_nonterm.max():.3f}]  "
                  f"r[discharge]={r_arr[s_discharge, 0]:+.2f}  "
                  f"V*[discharge]={V_temp[s_discharge]:+.2f}  "
                  f"V*[death]={V_temp[s_death]:+.2f}")

        # Gradient descent step
        Q -= args.lr * G

        # Re-freeze terminal state Q-values
        Q[s_discharge, :] = 100.0 - np.log(A)
        Q[s_death, :]     = -100.0 - np.log(A)

    # ---- Extract final quantities ----
    V_final = logsumexp(Q, axis=1)
    pi_soft_final = softmax(Q, axis=1)
    EV_next_final = T @ V_final
    r_sa_final = Q - GAMMA * EV_next_final   # (S, A)

    # Per-state reward: marginalize r(s,a) over the soft policy π*(a|s)
    r_final = (pi_soft_final * r_sa_final).sum(axis=1)  # (S,)

    # Greedy policy respecting the action mask
    Q_masked = np.where(action_mask, Q, -np.inf)
    pi_greedy = Q_masked.argmax(axis=1).astype(int)  # (S,)

    # ---- Save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "iq_learn.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "r": r_final,
            "Q": Q,
            "pi": pi_greedy,
            "gamma": GAMMA,
            "epochs": args.epochs,
            "lr": args.lr,
            "alpha": args.alpha,
        }, f)

    print(f"\nSaved to {out_path}")
    print(f"Reward range: [{r_final.min():.3f}, {r_final.max():.3f}]")
    print(f"  discharge: {r_final[s_discharge]:+.3f}")
    print(f"  death:     {r_final[s_death]:+.3f}")

    from collections import Counter
    top_actions = Counter(pi_greedy.tolist()).most_common(5)
    print(f"Top actions: {top_actions}")


if __name__ == "__main__":
    main()