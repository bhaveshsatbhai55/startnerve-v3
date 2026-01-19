import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

print("--- StartNerve AI Training Sequence ---")

# 1. GET DATA
url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
print(f"📥 Downloading ESOL Data...")
try:
    df = pd.read_csv(url)
    print(f"✅ Data Loaded: {len(df)} molecules found.")
except:
    print("❌ Internet Error: Could not download data.")
    exit()

# 2. CALCULATE PHYSICS (Features)
def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol)
        ]
    return None

print("⚙️  Processing Chemistry...")
X, y = [], []
for i, row in df.iterrows():
    feats = get_features(row['smiles'])
    if feats:
        X.append(feats)
        y.append(row['measured log solubility in mols per litre'])

# 3. TRAIN BRAIN
print("🧠 Training Random Forest Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. SCORE IT
score = r2_score(y, model.predict(X))
print(f"✅ Model Accuracy (R2 Score): {score:.3f}")

# 5. SAVE IT
if not os.path.exists("models"):
    os.makedirs("models")
joblib.dump(model, "models/solubility_model.pkl")
print("💾 AI Brain Saved: models/solubility_model.pkl")