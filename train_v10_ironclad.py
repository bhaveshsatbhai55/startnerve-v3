"""
StartNerve ToxGAT V10 — IRONCLAD EDITION
==========================================
Upgrades:
  1. Early Stopping: Stops training if Val Loss doesn't improve for 7 epochs.
  2. Nan-Proofing: Enhanced guards for Gasteiger charges on radical molecules.
  3. Optimized Weighting: Specifically tuned for your +21% HSE advantage.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from torch.utils.data import random_split

# --- CONSTANTS (Locked for Ecosystem Sync) ---
NODE_FEATURE_DIM = 162
HIDDEN_DIM       = 126
ATTENTION_HEADS  = 6
N_TASKS          = 12
BATCH_SIZE       = 128
EPOCHS           = 100 # Increased, but Early Stopping will likely cut it at ~45
LEARNING_RATE    = 0.001

# The 'Sales Edge' Weights
KILL_GATE_WEIGHTS = {
    'NR-AhR': 3.5,   # The Wedge
    'SR-HSE': 4.5,   # The +21% Dominator (Slightly boosted)
    'SR-p53': 4.0,   # The DNA Guardian (Electronic Vision focus)
}

HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

# --- 1. THE FEATURIZER (The 'Electronic Eye') ---
def get_v10_node_features(mol):
    # Computes electrical reactivity signals
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except: pass

    all_node_feats = []
    for atom in mol.GetAtoms():
        # [0-117] One-hot Atomic Number
        features = [0] * 118
        num = atom.GetAtomicNum()
        if 1 <= num <= 118: features[num-1] = 1
        
        # [118] Gasteiger Charge (Hardened Guard)
        charge = 0.0
        try:
            val = atom.GetProp('_GasteigerCharge')
            if val not in ['-nan', 'nan', 'inf', '-inf']:
                charge = np.clip(float(val), -2.0, 2.0)
        except: pass
        features.append(charge)

        # [119-123] Hybridization (Geometric Logic)
        h_type = atom.GetHybridization()
        features += [1 if h_type == t else 0 for t in HYBRIDIZATION_TYPES]

        # [124-128] Structural Flags
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        features.append(float(atom.GetFormalCharge()))
        features.append(float(atom.GetTotalNumHs()))
        features.append(1.0 if atom.IsInRing() else 0.0)
        features.append(float(atom.GetDegree()))

        # [129-161] Reserved for V11 (3D) and V12 (Global)
        features += [0.0] * (NODE_FEATURE_DIM - len(features))
        all_node_feats.append(features)
        
    return torch.tensor(all_node_feats, dtype=torch.float)

def mol_to_graph_v10(smiles, labels):
    mol = Chem.MolFromSmiles(smiles)
    if not mol or mol.GetNumAtoms() == 0: return None
    x = get_v10_node_features(mol)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    if not src: return None # Discard single atoms
    return Data(x=x, edge_index=torch.tensor([src, dst], dtype=torch.long), y=torch.tensor([labels], dtype=torch.float))

# --- 2. ARCHITECTURE (The V10 Deep GAT) ---
class ToxGAT_V10(torch.nn.Module):
    def __init__(self, in_channels=162, hidden=126, heads=6, n_tasks=12):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_channels, hidden)
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.2)
        self.gat3 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.2)
        self.norm = torch.nn.LayerNorm(hidden)
        self.fc_extra = torch.nn.Linear(hidden * 2, hidden)
        self.classifier = torch.nn.Linear(hidden, n_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = F.relu(self.input_proj(x))
        h = F.relu(self.gat1(h, edge_index))
        h = F.relu(self.gat2(h, edge_index))
        h = F.relu(self.gat3(h, edge_index))
        h = self.norm(h) 
        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=-1)
        return self.classifier(F.relu(self.fc_extra(pooled)))

# --- 3. THE ENGINE ---
def train_v10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training V10 IRONCLAD on {device}...")
    
    df = pd.read_csv("startnerve_master_v8_12task.csv")
    tasks = [c for c in df.columns if c != 'SMILES']
    
    graphs = []
    for _, row in df.iterrows():
        g = mol_to_graph_v10(row['SMILES'], row[tasks].fillna(-1).values.tolist())
        if g: graphs.append(g)
    
    train_size = int(0.8 * len(graphs))
    train_set, val_set = random_split(graphs, [train_size, len(graphs)-train_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = ToxGAT_V10().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Task Weighting Logic
    weights = torch.ones(N_TASKS).to(device)
    for name, w in KILL_GATE_WEIGHTS.items():
        if name in tasks: weights[tasks.index(name)] = w

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            mask = data.y != -1
            loss = (F.binary_cross_entropy_with_logits(out, data.y, reduction='none') * mask.float() * weights).sum() / mask.sum().clamp(min=1)
            loss.backward()
            optimizer.step()
        
        # Validation & Early Stopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device); out = model(data); mask = data.y != -1
                val_loss += ((F.binary_cross_entropy_with_logits(out, data.y, reduction='none') * mask.float() * weights).sum() / mask.sum().clamp(min=1)).item()
        
        avg_val = val_loss/len(val_loader)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "startnerve_v10_best.pt")
            patience_counter = 0
            marker = "⭐"
        else:
            patience_counter += 1
            marker = ""

        if epoch % 5 == 0 or marker == "⭐":
            print(f"Epoch {epoch:03d} | Val Loss: {avg_val:.4f} {marker}")
        
        if patience_counter >= 7: # Stop if no improvement for 7 epochs
            print(f"🛑 Early Stopping at Epoch {epoch}. Best Val Loss: {best_val:.4f}")
            break

    print("✅ V10 IRONCLAD BRAIN SAVED!")

if __name__ == "__main__":
    train_v10()