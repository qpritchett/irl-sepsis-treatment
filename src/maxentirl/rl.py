"""
PPO Forward Reiforcemnt learning using the reward function recovered from MaxEnt IRL.

usage:

    python -m src.maxentirl.rl
    
"""

import torch
import pickle
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from pathlib import Path

# --- Paths ---
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/mdp")

# Load the Learned Reward Function
with open(MODEL_DIR / "learned_reward_weights.pkl", "rb") as f:
    learned_theta = pickle.load(f)

# Print Thetas at the beginning
print("="*30)
print("RECOVERED REWARD WEIGHTS (THETA)")
print("="*30)
for i, weight in enumerate(learned_theta.flatten()):
    print(f"Feature {i:02d}: {weight:.6f}")
print("="*30 + "\n")

class MIMICEnv(gym.Env):
    """
    A custom Environment that uses the Transition Matrix T 
    and the Learned Reward Function.
    """
    def __init__(self, T, phi, theta):
        super(MIMICEnv, self).__init__()
        self.T = T.numpy() if torch.is_tensor(T) else T
        self.n_states = self.T.shape[0]
        self.n_actions = self.T.shape[1]
        
        # Calculate R(s) once: R = Phi * Theta
        self.rewards = (phi @ theta).flatten()
        
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states)
        self.state = 0 # Default start state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly sample an initial state from clinical states (0-749)
        self.state = np.random.randint(0, 750)
        return self.state, {}

    def step(self, action):
        # Sample next state from T(s, a, :)
        probs = self.T[self.state, action]
        
        # Handle cases where T was zeroed out (ensure probs sum to 1)
        if probs.sum() <= 0:
            next_state = self.state # Stay put if no transition exists
        else:
            next_state = np.random.choice(self.n_states, p=probs/probs.sum())
        
        reward = self.rewards[next_state]
        self.state = next_state
        
        # Done if we hit Discharge (750) or Death (751)
        terminated = bool(next_state >= 750)
        truncated = False
        
        return self.state, float(reward), terminated, truncated, {}

def load_components():
    # Load Transition Matrix
    with open(MODEL_DIR / "transitions.pkl", "rb") as f:
        mdp_data = pickle.load(f)
    T = mdp_data["T"]

    # Load State Features (Phi) - Same logic as your IRL script
    with open(DATA_DIR / "kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    centroids = kmeans.cluster_centers_
    n_f = centroids.shape[1]
    
    # Reconstruct Phi (including terminal flags)
    phi = np.zeros((752, n_f + 2))
    phi[:750, :n_f] = centroids
    phi[750, -2] = 1.0 # Discharge
    phi[751, -1] = 1.0 # Death
    
    return T, phi

def main():
    # Prepare Environment Components
    T, phi = load_components()
    env = MIMICEnv(T, phi, learned_theta)

    # Initialize PPO
    # MlpPolicy is the built-in PyTorch MLP policy in Stable Baselines3
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, gamma=0.99)

    # Train
    print("Training PPO Policy on Recovered Rewards...")
    model.learn(total_timesteps=50000)

    # Save the Policy
    model.save(MODEL_DIR / "ppo_clinician_policy")
    print(f"\nPolicy saved to {MODEL_DIR / 'ppo_clinician_policy'}")

if __name__ == "__main__":
    main()