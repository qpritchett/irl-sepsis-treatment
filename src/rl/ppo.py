"""
PPO forward RL policy trained on the MaxEnt IRL recovered reward function.

Wraps the discretized sepsis MDP as a Gymnasium environment and trains
a PPO agent using the learned reward r(s) = Phi @ theta from MaxEnt IRL.
Unobserved (s,a) pairs are masked so the agent only takes clinically
observed actions from each state.

Usage:
    python -m src.rl.ppo --timesteps 50000
    python -m src.rl.ppo --timesteps 500000
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MDP_DIR = ROOT / "models" / "mdp"
IRL_DIR = ROOT / "models" / "irl"
OUT_DIR = ROOT / "models" / "rl"

N_STEPS = 8192


def load_components():
    """Load transition matrix, reward function, and action mask."""
    # Transition matrix
    with open(MDP_DIR / "transitions.pkl", "rb") as f:
        mdp = pickle.load(f)
    T = mdp["T"]
    s_discharge = mdp["s_discharge"]
    s_death = mdp["s_death"]
    n_total = mdp["n_total"]

    # MaxEnt IRL reward
    with open(IRL_DIR / "maxent.pkl", "rb") as f:
        irl = pickle.load(f)
    r = irl["r"]  # shape (752,)

    # Normalize reward to [-1, 1] range to stabilize PPO training
    r_min, r_max = r.min(), r.max()
    r = 2 * (r - r_min) / (r_max - r_min) - 1
    # Preserve terminal signal magnitude
    r[s_discharge] = 1.0
    r[s_death] = -1.0

    # Action mask — only allow clinically observed (s,a) pairs
    row_sums = T.sum(axis=2)
    action_mask = row_sums > 0  # shape (752, 25)

    return T, r, action_mask, s_discharge, s_death, n_total


class SepsisMDPEnv(gym.Env):
    """
    Discretized sepsis MDP environment for PPO training.

    States: 0-749 clinical states (KMeans clusters) + 750 discharge + 751 death
    Actions: 0-24 (5x5 fluid/vasopressor grid)
    Reward: r(s) from MaxEnt IRL, with terminal bonus at absorbing states
    """

    metadata = {"render_modes": []}

    def __init__(self, T, r, action_mask, s_discharge, s_death, start_states=None):
        super().__init__()
        self.T = T
        self.r = r
        self.action_mask = action_mask  # (752, 25) bool
        self.s_discharge = s_discharge
        self.s_death = s_death
        self.n_states = T.shape[0]
        self.n_actions = T.shape[1]
        self.n_clinical = s_discharge  # 750 non-terminal states

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)
        self.state = 0
        self.start_states = start_states  # actual patient initial states from data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.start_states is not None:
            idx = self.np_random.integers(0, len(self.start_states))
            self.state = int(self.start_states[idx])
        else:
            self.state = int(self.np_random.integers(0, self.n_clinical))
        return self.state, {}

    def step(self, action):
        # If action not observed from this state, redirect to first observed action
        if not self.action_mask[self.state, action]:
            action = int(np.where(self.action_mask[self.state])[0][0]) \
                if self.action_mask[self.state].any() else 0

        probs = self.T[self.state, action]
        if probs.sum() <= 0:
            next_state = self.state
        else:
            next_state = int(np.random.choice(self.n_states, p=probs / probs.sum()))

        reward = float(self.r[next_state])
        self.state = next_state
        terminated = bool(next_state >= self.n_clinical)
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def action_masks(self):
        """Returns valid action mask for current state (for MaskablePPO if needed)."""
        return self.action_mask[self.state]


def extract_policy(model, n_states):
    """Extract greedy policy pi(s) from trained PPO model."""
    pi = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action, _ = model.predict(s, deterministic=True)
        pi[s] = int(action)
    return pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50_000,
                    help="Total PPO training timesteps (default: 50000)")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="PPO learning rate (default: 3e-4)")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor (default: 0.99)")
    ap.add_argument("--lr-schedule", choices=["linear", "constant"], default="linear",
                    help="LR schedule: linear decay with 1e-5 floor, or constant (default: linear)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume training from existing ppo_maxent_sb3 checkpoint")
    args = ap.parse_args()

    print("Loading MDP components...")
    T, r, action_mask, s_discharge, s_death, n_total = load_components()
    print(f"  States: {n_total}, Actions: {T.shape[1]}")
    print(f"  Reward range: [{r.min():.3f}, {r.max():.3f}]")
    print(f"  Observed (s,a): {action_mask.sum()} / {action_mask.size}")

    # Load actual patient initial states from training data
    train_meta = pd.read_csv(ROOT / "data/processed/train/metadata.csv")
    start_states = (
        train_meta.sort_values(["icustayid", "bloc"])
        .groupby("icustayid")
        .first()["state"]
        .values
    )
    print(f"  Start states loaded: {len(start_states)}")

    # Build environment initialised from real patient starting states
    env = SepsisMDPEnv(T, r, action_mask, s_discharge, s_death, start_states=start_states)
    env = Monitor(env)

    # Learning rate function: linear decay with 1e-5 floor so updates stay alive at end of run
    if args.lr_schedule == "linear":
        lr_fn = lambda progress: max(3e-4 * progress, 1e-5)
    else:
        lr_fn = args.lr

    # Train PPO
    # n_steps=8192 >> mean episode length (~21), satisfying Schulman et al. recommendation
    print(f"\nTraining PPO for {args.timesteps:,} timesteps (lr_schedule={args.lr_schedule}, resume={args.resume})...")
    if args.resume:
        resume_path = str(OUT_DIR / "ppo_maxent_sb3")
        print(f"  Resuming from {resume_path}...")
        model = PPO.load(resume_path, env=env)
        model.learning_rate = lr_fn
        model.lr_schedule = get_schedule_fn(lr_fn)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr_fn,
            gamma=args.gamma,
            n_steps=N_STEPS,
            batch_size=128,
            n_epochs=10,
            ent_coef=0.05,  # encourage exploration in 750-state space
        )
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume)

    # Extract and save greedy policy over all 752 states (including absorbing)
    print("\nExtracting greedy policy...")
    pi = extract_policy(model, n_total)

    top_actions = Counter(pi.tolist()).most_common(5)
    print(f"Top actions: {top_actions}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ppo_maxent.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "pi": pi,
            "reward_type": "maxent_ppo",
            "timesteps": args.timesteps,
            "gamma": args.gamma,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "n_steps": N_STEPS,
            "start_states_count": len(start_states),
            "resumed": args.resume,
        }, f)

    model.save(str(OUT_DIR / "ppo_maxent_sb3"))
    print(f"Saved policy to {out_path}")
    print(f"Saved SB3 model to {OUT_DIR / 'ppo_maxent_sb3'}")


if __name__ == "__main__":
    main()
