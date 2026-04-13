"""
Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) training loop implementation.

Given a set of expert trajectories (state-action sequences), MaxEnt IRL learns a reward 
function that explains the expert's behavior under the principle of maximum entropy. 
The algorithm iteratively updates the reward function parameters to maximize the 
likelihood of the expert trajectories while ensuring that the resulting policy is as 
random as possible (maximum entropy) given the constraints of matching feature expectations.

K-means cluster centroids for states (750, 1) are precomputed in precompute.py.
Actions are sorted into bins (25, 1) via Action=(Fluid Level−1)×5+(Vaso Level−1), also precomputed in precompute.py.
Transition Matrix P(s'|s,a) (752, 25, 752) is precomputed from the training data in transitions.py.

Soft Value Iteration is used instead of  State Visitation Frequencies as in the original Zeitbart Paper.

Usage:

    python -m src.maxentirl.train

"""
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration & Paths ---
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/mdp")
N_STATES = 750
N_ACTIONS = 25
GAMMA = 0.99
LEARNING_RATE = 0.01
EPOCHS = 100

class MaxEntIRL(nn.Module):
    def __init__(self, n_states, n_actions, n_features, T):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        # Learned reward weights: R = Phi * Theta
        self.theta = nn.Parameter(torch.randn(n_features, 1) * 0.01)
        # Transition matrix (fixed)
        self.register_buffer('T', T)

    def get_reward(self, phi):
        return torch.matmul(phi, self.theta).squeeze()

    def soft_value_iteration(self, phi, eps=1e-3):
        """Forward Pass: Compute the soft-optimal policy."""
        R = self.get_reward(phi)
        V = torch.zeros(self.n_states, device=self.T.device)
        
        for _ in range(500):
            # Q(s,a) = R(s) + gamma * sum_{s'} T(s'|s,a) * V(s')
            expected_v = torch.einsum('san,n->sa', self.T, V)
            Q = R.view(-1, 1) + GAMMA * expected_v
            
            new_V = torch.logsumexp(Q, dim=1)
            if torch.norm(new_V - V) < eps:
                break
            V = new_V
            
        # Policy pi(a|s) = exp(Q(s,a) - V(s))
        policy = torch.exp(Q - V.view(-1, 1))
        return policy

def compute_svf(T, policy, n_steps=50):
    """Compute State Visitation Frequency (D)."""
    n_states = T.shape[0]
    # Start with a uniform distribution (or actual start states from data)
    d = torch.ones(n_states, device=T.device) / n_states
    svf = torch.zeros(n_states, device=T.device)
    
    for _ in range(n_steps):
        svf += d
        # d_{t+1} = sum_s sum_a d_t(s) * pi(a|s) * T(s'|s,a)
        d = torch.einsum('s,sa,san->n', d, policy, T)
    return svf

def load_data():
    # Load Transitions
    with open(MODEL_DIR / "transitions.pkl", "rb") as f:
        mdp_data = pickle.load(f)
    T = torch.FloatTensor(mdp_data["T"])

    # Load KMeans for State Features (Centroids)
    with open(DATA_DIR / "kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    
    # Map clusters to features. Shape: (752, n_features + 2)
    centroids = kmeans.cluster_centers_
    n_f = centroids.shape[1]
    phi = np.zeros((752, n_f + 2)) 
    phi[:750, :n_f] = centroids
    # Add one-hot terminal indicators for Discharge (750) and Death (751)
    phi[750, -2] = 1.0 
    phi[751, -1] = 1.0 
    
    # Load Expert Metadata for Target Feature Expectations
    train_meta = pd.read_csv(DATA_DIR / "train" / "metadata.csv")
    
    return T, torch.FloatTensor(phi), train_meta

def main():
    T, phi, train_meta = load_data()
    n_states, n_actions, n_features = T.shape[0], T.shape[1], phi.shape[1]

    # Calculate Expert Feature Expectations (mu_expert)
    # The average 'health status' visited by real clinicians
    expert_states = torch.LongTensor(train_meta['state'].values)
    expert_mu = phi[expert_states].mean(dim=0)

    # Initialize Model & Optimizer
    model = MaxEntIRL(n_states, n_actions, n_features, T)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting IRL Training... Features: {n_features}")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Solve for Policy (Forward)
        policy = model.soft_value_iteration(phi)
        
        # Solve for Visitation Frequency (Backward)
        svf = compute_svf(T, policy)
        
        # Compute Learner's Feature Expectations
        mu_theta = torch.matmul(svf / svf.sum(), phi)
        
        # Loss: the difference between expert and learner expectations
        # Gradient of MaxEnt objective is (mu_expert - mu_theta)
        # We use a dot product to set up the gradient for .backward()
        loss = torch.abs(torch.dot(model.theta.flatten(), (expert_mu - mu_theta)))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

    # Save the learned rewards
    learned_theta = model.theta.detach().numpy()
    with open(MODEL_DIR / "learned_reward_weights.pkl", "wb") as f:
        pickle.dump(learned_theta, f)
    print("\nTraining Complete. Weights saved.")

if __name__ == "__main__":
    main()