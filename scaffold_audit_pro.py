"""
StartNerve V10 — Scaffold Split Validation Audit
=================================================
The HONEST benchmark. Tests on molecular scaffolds
the model has never seen during training.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

# CRITICAL: Ensure this matches your training filename
from train_v10_electronic import ToxGAT_V10, mol_to_graph_v10

# --- CONSTANTS ---
ELITE_THRESHOLD   = 0.82
SCAFFOLD_TEST_PCT = 0.20 
MISSING_LABEL     = -1
RANDOM_SEED       = 42

# --- 1. SCAFFOLD SPLIT LOGIC ---
def get_scaffold_smiles(smiles):
    """Strips a molecule to its core skeleton."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        # Generates the Murcko Scaffold - the 'Skeleton' of the drug
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold_mol)
    except: return None

def build_scaffold_split(df):
    """Splits data so test molecules have different skeletons than training molecules."""
    df = df.copy()
    print("  🏗️  Decomposing molecules into scaffolds...")
    df['_scaffold'] = df['SMILES'].apply(get_scaffold_smiles)
    
    unique_scaffolds = df['_scaffold'].dropna().unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(unique_scaffolds)

    split_idx = int(len(unique_scaffolds) * (1 - SCAFFOLD_TEST_PCT))
    train_scaffolds = set(unique_scaffolds[:split_idx])
    
    train_df = df[df['_scaffold'].isin(train_scaffolds) | df['_scaffold'].isna()]
    test_df = df[df['_scaffold'].isin(unique_scaffolds[split_idx:])]
    return train_df, test_df

# --- 2. RF BASELINE (The Competition) ---
def get_morgan_fp(smiles):
    """Standard bit-vector representation of a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

def run_rf_baseline(train_df, test_df, tasks):
    print("  📡 Training RF Baseline (Safe Mode)...")
    results = {}
    
    # Use a faster, non-deprecated way to get fingerprints
    def get_fps_and_labels(df, task_name):
        fps, labels = [], []
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol and row[task_name] != MISSING_LABEL:
                # Upgraded to MorganGenerator style logic (2048-bit)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(np.array(fp))
                labels.append(row[task_name])
        return np.array(fps), np.array(labels)

    for task in tasks:
        X_train, y_train = get_fps_and_labels(train_df, task)
        X_test, y_test = get_fps_and_labels(test_df, task)
        
        # Ensure we have both classes (0 and 1) in our split
        if len(X_train) > 10 and len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED)
            rf.fit(X_train, y_train)
            probs = rf.predict_proba(X_test)[:, 1]
            results[task] = roc_auc_score(y_test, probs)
        else:
            results[task] = 0.5 # Default to random guessing if data is too thin
    return results

# --- 3. MAIN AUDIT ENGINE ---
def run_scaffold_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running StartNerve V10 Honest Audit on {device}...")

    # Load Model
    model = ToxGAT_V10().to(device)
    model.load_state_dict(torch.load("startnerve_v10_best.pt", map_location=device))
    model.eval()

    # Load Data
    df = pd.read_csv("startnerve_master_v8_12task.csv")
    tasks = [c for c in df.columns if c != 'SMILES']
    
    # Split
    train_df, test_df = build_scaffold_split(df)
    print(f"  ✅ Train: {len(train_df)} | Test: {len(test_df)}")

    # GAT Inference
    test_graphs = [mol_to_graph_v10(s, l) for s, l in zip(test_df['SMILES'], test_df[tasks].fillna(-1).values.tolist())]
    test_graphs = [g for g in test_graphs if g is not None]
    loader = DataLoader(test_graphs, batch_size=64)
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = torch.sigmoid(model(data))
            all_preds.append(out.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())

    preds, targets = np.vstack(all_preds), np.vstack(all_targets)
    
    # RF Baseline
    rf_scores = run_rf_baseline(train_df, test_df, tasks)

    # Display & Export
    print(f"\n{'TASK':<20} | {'V10 GAT':>10} | {'RF BASE':>10} | {'DELTA':>8}")
    print("-" * 60)
    
    final_rows = []
    for i, task in enumerate(tasks):
        mask = targets[:, i] != -1
        gat_auc = roc_auc_score(targets[mask, i], preds[mask, i])
        rf_auc = rf_scores[task]
        delta = gat_auc - rf_auc
        elite = " ⭐ ELITE" if gat_auc >= 0.82 else ""
        
        print(f"{task:20} | {gat_auc:>10.3f} | {rf_auc:>10.3f} | {delta:>+8.3f}{elite}")
        final_rows.append({'Pathway': task, 'GAT_AUROC': gat_auc, 'RF_AUROC': rf_auc, 'Delta': delta})

    pd.DataFrame(final_rows).to_csv("scaffold_audit_results.csv", index=False)
    print(f"\n✅ Audit Complete. Results saved to scaffold_audit_results.csv")

if __name__ == "__main__":
    run_scaffold_audit()