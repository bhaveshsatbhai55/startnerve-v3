import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pickle
import argparse

# --- 1. THE ARCHITECTURE (V6 Brain) ---
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

# --- 2. THE UTILS (Featurizer & Shield) ---
def atom_features(atom):
    atomic_num_vec = [0] * 118
    num = atom.GetAtomicNum()
    if 1 <= num <= 118: atomic_num_vec[num-1] = 1
    # Simplified featurizer for the 162-sync
    return atomic_num_vec + [0]*44 # Buffer to match 162 (standardize this later)

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# --- 3. THE GUARDRAIL (Stay Out of Jail) ---
def get_max_similarity(smiles, ref_fps):
    query_mol = Chem.MolFromSmiles(smiles)
    if not query_mol: return 0.0
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps)
    return max(similarities)

# --- 4. THE MAIN ENGINE ---
def run_v7_prediction(input_csv, output_csv, model_path, shield_path):
    device = torch.device("cpu")
    
    # Load Brain & Shield
    print(f"🧠 Loading Brain...")
    model = ToxGAT_V6()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    
    print(f"🛡️ Loading Shield Index...")
    with open(shield_path, "rb") as f:
        ref_fps = pickle.load(f)
        
    df = pd.read_csv(input_csv)
    results = []
    
    print(f"🧪 Auditing {len(df)} compounds...")
    for _, row in df.iterrows():
        smiles = str(row['SMILES'])
        
        # 1. CHECK APPLICABILITY (The Jail-Shield)
        max_sim = get_max_similarity(smiles, ref_fps)
        
        if max_sim < 0.30: # OECD AD Threshold
            results.append({
                "SMILES": smiles,
                "Probability": "N/A",
                "Verdict": "OUT OF DOMAIN",
                "Similarity": round(max_sim, 3),
                "Note": "Structure too novel for safe prediction."
            })
            continue
            
        # 2. RUN PREDICTION (Only if Safe)
        graph = mol_to_graph(smiles)
        if graph:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
            with torch.no_grad():
                prob = torch.sigmoid(model(graph)).item()
                results.append({
                    "SMILES": smiles,
                    "Probability": round(prob, 4),
                    "Verdict": "TOXIC" if prob > 0.5 else "SAFE",
                    "Similarity": round(max_sim, 3),
                    "Note": "Inside Applicability Domain"
                })
        else:
            results.append({"SMILES": smiles, "Verdict": "INVALID"})

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✅ Audit Complete! Report saved to {output_csv}")

if __name__ == "__main__":
    run_v7_prediction("test_drugs.csv", "startnerve_v7_audit.csv", "startnerve_v6_universe.pt", "startnerve_shield_index.pkl")