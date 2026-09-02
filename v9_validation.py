import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np

# --- 1. SCAFFOLD SPLIT ---
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

# --- 2. VALIDATION ENGINE (V9 COMPATIBLE) ---
def run_validation_v9():
    device = torch.device("cpu")
    print("📊 STARTNERVE V9: Final Performance Audit...")
    
    # Import the V9 Architecture from your boosted trainer
    from train_v9_boosted import ToxGAT_V9_Deep, atom_features
    
    # Simple internal graph converter for validation (no augmentation needed here)
    def mol_to_graph_val(smiles, labels):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
        src, dst = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [i, j]; dst += [j, i]
        return Data(x=x, edge_index=torch.tensor([src, dst], dtype=torch.long), y=torch.tensor([labels], dtype=torch.float))

    from torch_geometric.data import Data
    
    df = pd.read_csv("startnerve_master_v8_12task.csv")
    _, test_idx = get_scaffold_split(df)
    
    # Must match V9: hidden=126
    model = ToxGAT_V9_Deep(in_channels=162, hidden=126)
    model.load_state_dict(torch.load("startnerve_v9_boosted.pt", map_location=device))
    model.eval()
    
    tasks = [c for c in df.columns if c != 'SMILES']
    test_graphs = []
    
    print(f"🧪 Testing on {len(test_idx)} unseen chemical skeletons...")
    for idx in test_idx:
        row = df.iloc[idx]
        g = mol_to_graph_val(row['SMILES'], row[tasks].values.tolist())
        if g: test_graphs.append(g)
        
    loader = DataLoader(test_graphs, batch_size=64)
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data in loader:
            out = torch.sigmoid(model(data))
            all_preds.append(out.numpy())
            all_targets.append(data.y.numpy())
            
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    results = []
    print("\n🏁 STARTNERVE V9 BOOSTED RESULTS (SCAFFOLD-SPLIT):")
    for i, task in enumerate(tasks):
        mask = targets[:, i] != -1
        if mask.sum() > 10:
            score = roc_auc_score(targets[mask, i], preds[mask, i])
            results.append({"Pathway": task, "AUROC": round(score, 3)})
            star = " ⭐⭐⭐ [ELITE]" if score >= 0.80 else " 🔹 [SOLID]"
            print(f"{task:15} | AUROC: {score:.3f}{star}")

    pd.DataFrame(results).to_csv("v9_final_metrics.csv", index=False)
    print("\n✅ V9 Metrics saved to v9_final_metrics.csv.")

if __name__ == "__main__":
    run_validation_v9()