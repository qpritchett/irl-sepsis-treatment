"""
Test script for MaxEnt IRL reward model and PPO on MIMIC-III sepsis dataset.

Usage:

    python -m src.maxentirl.test

"""

import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/mdp")
TEST_META = DATA_DIR / "test" / "metadata.csv"

def load_reward_model():
    # Load learned weights (theta)
    with open(MODEL_DIR / "learned_reward_weights.pkl", "rb") as f:
        theta = pickle.load(f)
    
    # Load KMeans to reconstruct the feature mapping (phi)
    with open(DATA_DIR / "kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    
    centroids = kmeans.cluster_centers_
    n_f = centroids.shape[1]
    
    # Reconstruct the 752-state feature matrix used in training
    phi = np.zeros((752, n_f + 2))
    phi[:750, :n_f] = centroids
    phi[750, -2] = 1.0  # Discharge flag
    phi[751, -1] = 1.0  # Death flag
    
    # Calculate R(s) for all states
    state_rewards = (phi @ theta).flatten()
    
    # Print the top weights for sanity check
    print("Top 5 Positive Weights (Drivers of Reward):")
    indices = np.argsort(theta.flatten())[-5:][::-1]
    for i in indices:
        print(f"  Feature {i}: {theta.flatten()[i]:.6f}")
        
    return state_rewards

def evaluate_on_test_set(state_rewards):
    df = pd.read_csv(TEST_META)
    
    # Group by patient stay to calculate total reward per trajectory
    patient_rewards = []
    
    print(f"\nEvaluating on {df['icustayid'].nunique()} test trajectories...")
    
    for pid, group in df.groupby("icustayid"):
        # Sort by timestep (bloc)
        group = group.sort_values("bloc")
        
        # 1. Sum rewards for clinical states visited
        states = group['state'].values
        total_r = np.sum(state_rewards[states])
        
        # 2. Add the terminal reward based on outcome
        # 0 = survived (Discharge: 750), 1 = died (Death: 751)
        outcome = group['outcome'].values[0]
        terminal_state = 751 if outcome == 1 else 750
        total_r += state_rewards[terminal_state]
        
        patient_rewards.append({
            "icustayid": pid,
            "total_reward": total_r,
            "outcome": outcome
        })
        
    res_df = pd.DataFrame(patient_rewards)
    
    # --- Statistical Summary ---
    survived = res_df[res_df['outcome'] == 0]['total_reward']
    died = res_df[res_df['outcome'] == 1]['total_reward']
    
    print("\n" + "="*40)
    print("TEST SET REWARD SUMMARY")
    print("="*40)
    print(f"Survived (Mean Reward): {survived.mean():.4f}")
    print(f"Died     (Mean Reward): {died.mean():.4f}")
    print(f"Difference:            {survived.mean() - died.mean():.4f}")
    print("="*40)
    
    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.hist(survived, bins=50, alpha=0.5, label='Survived', color='green')
    plt.hist(died, bins=50, alpha=0.5, label='Died', color='red')
    plt.axvline(survived.mean(), color='green', linestyle='dashed', linewidth=2)
    plt.axvline(died.mean(), color='red', linestyle='dashed', linewidth=2)
    plt.title("Distribution of Accumulated Rewards (Test Set)")
    plt.xlabel("Cumulative Reward")
    plt.ylabel("Patient Count")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    rewards = load_reward_model()
    evaluate_on_test_set(rewards)