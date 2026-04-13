import pandas as pd
import pickle

# Load weights and the normalized dataframe
with open("models/mdp/learned_reward_weights.pkl", "rb") as f:
    theta = pickle.load(f).flatten()

# Get the column names (excluding non-feature columns like 'icustayid' or 'state')
df_zs = pd.read_csv("data/processed/train/MIMICzs.csv")
feature_names = list(df_zs.columns)

# Add two terminal names we added manually in IRL script
feature_names += ["Terminal_Discharge", "Terminal_Death"]

# Combine and sort to see features ordered by model value
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Weight": theta
}).sort_values(by="Weight", ascending=False)

print(feature_importance)