import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
import argparse

# 1. THE UTILITY TOOL
def _one_hot(value, choices: list) -> list:
    vec = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    vec[idx] = 1
    return vec

# 2. THE FEATURIZER (Locked to exactly 162 features - SYNCED TO V6)
def atom_features(atom) -> list:
    atomic_num_vec = [0] * 118
    num = atom.GetAtomicNum()
    if 1 <= num <= 118:
        atomic_num_vec[num-1] = 1
    
    features = (
        atomic_num_vec
        + _one_hot(atom.GetDegree(), list(range(0, 11)))
        + _one_hot(atom.GetImplicitValence(), list(range(0, 11)))
        + _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        + _one_hot(atom.GetTotalNumHs(), list(range(0, 5)))
        + _one_hot(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ])
        + [int(atom.GetIsAromatic())]
        + [int(atom.IsInRing())]
    )
    return features

# 3. THE BRIDGE
def mol_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None: return None
    try:
        flags = SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_PROPERTIES
        Chem.SanitizeMol(mol, flags)
    except: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# 4. THE V6 BRAIN ARCHITECTURE (Matches train_v6_universe.py)
class ToxGAT_V6(torch.nn.Module):
    def __init__(self, in_channels=162, hidden=128, heads=4):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_channels, hidden)
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads)
        self.norm = torch.nn.LayerNorm(hidden)
        self.classifier = torch.nn.Linear(hidden * 2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = F.relu(self.input_proj(x))
        h = F.relu(self.gat1(h, edge_index))
        h = self.norm(F.relu(self.gat2(h, edge_index)) + h)
        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=-1)
        return self.classifier(pooled).squeeze(-1)

# 5. THE ENGINE
def run_prediction(input_csv, output_csv, model_path):
    device = torch.device("cpu")
    print(f"🧠 Loading StartNerve V6 Universe Brain: {model_path}")
    
    model = ToxGAT_V6()
    # Loading the state dict we just saved from training
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    df = pd.read_csv(input_csv)
    results = []
    print(f"🧪 Processing {len(df)} compounds...")
    
    for index, row in df.iterrows():
        smiles = str(row['SMILES'])
        graph = mol_to_graph(smiles)
        if graph:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
            with torch.no_grad():
                # Sigmoid turns the raw output into a 0.0 to 1.0 probability
                prob = torch.sigmoid(model(graph)).item()
                
                # Result classification
                if prob > 0.5:
                    res_text = "TOXIC"
                else:
                    res_text = "SAFE"
                
                results.append({
                    "SMILES": smiles, 
                    "Probability": round(prob, 4), 
                    "Safety_Result": res_text
                })
        else:
            results.append({"SMILES": smiles, "Probability": None, "Safety_Result": "INVALID_SMILES"})
            
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✅ Success! V6 Safety Report saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_drugs.csv")
    parser.add_argument("--output", default="startnerve_v6_report.csv")
    parser.add_argument("--model", default="startnerve_v6_universe.pt")
    args = parser.parse_args()
    
    run_prediction(args.input, args.output, args.model)