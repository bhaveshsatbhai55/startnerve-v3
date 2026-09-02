import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np

# --- 1. SCAFFOLD SPLIT LOGIC ---
def get_scaffold_split(df, train_frac=0.8):
    print("🏗️  Clustering by Molecular Scaffolds (Skeleton Check)...")
    scaffolds = {}
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
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

# --- 2. VALIDATION ENGINE ---
def run_validation():
    device = torch.device("cpu")
    print("📊 STARTNERVE V8: Generating the Validation Whitepaper...")
    
    # Import your specific V8 architecture and featurizer
    from train_v8_superbrain import ToxGAT_V8_Superbrain, mol_to_graph
    
    df = pd.read_csv("startnerve_master_v8_12task.csv")
    _, test_idx = get_scaffold_split(df)
    
    model = ToxGAT_V8_Superbrain(in_channels=162, hidden=126)
    model.load_state_dict(torch.load("startnerve_v8_superbrain.pt", map_location=device))
    model.eval()
    
    tasks = [c for c in df.columns if c != 'SMILES']
    test_graphs = []
    
    print(f"🧪 Testing on {len(test_idx)} novel chemical skeletons...")
    for idx in test_idx:
        row = df.iloc[idx]
        g = mol_to_graph(row['SMILES'], row[tasks].values.tolist())
        if g: test_graphs.append(g)
        
    loader = DataLoader(test_graphs, batch_size=64)
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data in loader:
            out = torch.sigmoid(model(data)) # Convert to 0-1 probability
            all_preds.append(out.numpy())
            all_targets.append(data.y.numpy())
            
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    results = []
    print("\n🏁 STARTNERVE V8 PERFORMANCE (AUROC):")
    for i, task in enumerate(tasks):
        mask = targets[:, i] != -1
        if mask.sum() > 10:
            score = roc_auc_score(targets[mask, i], preds[mask, i])
            results.append({"Pathway": task, "AUROC": round(score, 3)})
            # Special flagging for 'Kill-Gates'
            flag = " ⭐ [KILL-GATE]" if task in ['NR-AhR', 'NR-AR', 'SR-p53'] else ""
            print(f"🔹 {task:15} | AUROC: {score:.3f}{flag}")

    pd.DataFrame(results).to_csv("validation_metrics.csv", index=False)
    print("\n✅ VALIDATION COMPLETE. 'validation_metrics.csv' is your proof of product.")

if __name__ == "__main__":
    run_validation()