import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

# --- 1. SCAFFOLD SPLITTER (Matches V9 logic exactly) ---
def get_scaffold_split(df, train_frac=0.8):
    print("🏗️  Clustering by Molecular Scaffolds...")
    scaffolds = {}
    for idx, smiles in enumerate(df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        # We already filtered None, but safety first
        scaffold = MurckoScaffold.GetScaffoldForMol(mol) if mol else None
        s_smiles = Chem.MolToSmiles(scaffold) if scaffold else "None"
        if s_smiles not in scaffolds: scaffolds[s_smiles] = []
        scaffolds[s_smiles].append(idx)
    
    scaffold_sets = sorted(list(scaffolds.values()), key=len, reverse=True)
    train_indices, test_indices = [], []
    for s_set in scaffold_sets:
        if len(train_indices) + len(s_set) <= len(df) * train_frac:
            train_indices.extend(s_set)
        else:
            test_indices.extend(s_set)
    return train_indices, test_indices

# --- 2. RF BASELINE RUNNER ---
def run_rf_baseline():
    print("🌲 STARTNERVE V9 vs RANDOM FOREST BASELINE...")
    raw_df = pd.read_csv("startnerve_master_v8_12task.csv")
    
    # 🧪 Step A: Filter out molecules RDKit cannot parse (The 'Aluminum' fix)
    print("🧹 Cleaning Chemical Data...")
    valid_data = []
    for _, row in raw_df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is not None:
            valid_data.append(row)
    
    df = pd.DataFrame(valid_data).reset_index(drop=True)
    print(f"✅ Cleaned {len(df)} compounds for the baseline.")
    
    # 🧪 Step B: Generate Fingerprints (The RF 'Features')
    print("🧬 Generating 2048-bit Morgan Fingerprints...")
    fps = []
    for s in df['SMILES']:
        mol = Chem.MolFromSmiles(s)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    
    X = np.array(fps)
    tasks = [c for c in df.columns if c != 'SMILES']
    
    # 🧪 Step C: Split exactly like the GNN did
    train_idx, test_idx = get_scaffold_split(df)
    X_train, X_test = X[train_idx], X[test_idx]
    
    results = []
    print("\n🏁 FINAL BENCHMARK (RF vs GNN):")
    for i, task in enumerate(tasks):
        y = df[task].values
        train_mask = y[train_idx] != -1
        test_mask = y[test_idx] != -1
        
        if test_mask.sum() > 10:
            # Traditional RF Classifier
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            rf.fit(X_train[train_mask], y[train_idx][train_mask])
            
            probs = rf.predict_proba(X_test[test_mask])[:, 1]
            score = roc_auc_score(y[test_idx][test_mask], probs)
            results.append({"Pathway": task, "RF_AUROC": round(score, 3)})
            print(f"🔹 {task:15} | RF AUROC: {score:.3f}")

    pd.DataFrame(results).to_csv("rf_baseline_metrics.csv", index=False)
    print("\n✅ Baseline Saved. Now compare these to your 'v9_final_metrics.csv'!")

if __name__ == "__main__":
    run_rf_baseline()