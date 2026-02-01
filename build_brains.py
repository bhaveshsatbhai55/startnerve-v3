import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

print("🧠 STARTNERVES BRAIN BUILDER INITIATED...")

# --- BRAIN 1: SOLUBILITY (The Math Model) ---
print("1️⃣  Building Solubility Model (model.pkl)...")
# (Training on a small dummy dataset for speed - good enough for the MVP connection)
# Real training happens when you have the full dataset, but this connects the pipes.
data = pd.DataFrame({
    'MolLogP': [2.3, -1.2, 0.5, 3.1, -0.5],
    'MolWt': [180, 342, 58, 200, 150],
    'NumRotatableBonds': [2, 5, 0, 3, 1],
    'AromaticProportion': [0.5, 0.0, 0.0, 0.8, 0.2],
    'LogS': [-2.5, -0.5, 1.2, -4.0, -1.0] # Labels
})
X_sol = data[['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion']]
y_sol = data['LogS']
sol_model = LinearRegression()
sol_model.fit(X_sol, y_sol)

with open('model.pkl', 'wb') as f:
    pickle.dump(sol_model, f)
print("✅ model.pkl created.")

# --- BRAIN 2: TOXICITY (The Hunter Model) ---
print("2️⃣  Building Toxicity Model (toxicity_model.pkl)...")
# We load the one you just made if it exists, or create a placeholder if missing
try:
    with open('toxicity_model.pkl', 'rb') as f:
        print("✅ toxicity_model.pkl already exists. Skipping re-train.")
except:
    # Fallback if you accidentally deleted it
    print("⚠️  Toxicity model missing. Creating basic placeholder...")
    tox_model = RandomForestClassifier()
    # Dummy train
    tox_model.fit([[0]*2048, [1]*2048], [0, 1])
    with open('toxicity_model.pkl', 'wb') as f:
        pickle.dump(tox_model, f)
    print("✅ toxicity_model.pkl created.")

print("🚀 ALL SYSTEMS READY.")