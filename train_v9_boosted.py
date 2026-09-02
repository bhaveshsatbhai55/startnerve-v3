import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from rdkit import Chem
import os

# --- 1. THE FEATURIZER (Hardened 162) ---
def atom_features(atom):
    features = [0] * 162 
    num = atom.GetAtomicNum()
    if 1 <= num <= 118: 
        features[num-1] = 1
    return features

def get_augmented_graphs(smiles, labels, n_aug=3):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return []
    
    graphs = []
    # We always include the Canonical version first (The 'Truth')
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    
    # Set for tracking unique SMILES to avoid redundant training
    unique_smiles = {canonical_smiles}
    
    # Generate variants
    for _ in range(n_aug * 2): # Try extra times to get enough unique ones
        if len(unique_smiles) >= n_aug: break
        
        # FIX: Using 'doRandom=True' and 'kekuleSmiles=False' to match RDKit C++ signature
        aug = Chem.MolToSmiles(mol, doRandom=True, canonical=False, kekuleSmiles=False)
        unique_smiles.add(aug)

    for s in unique_smiles:
        m = Chem.MolFromSmiles(s)
        if not m: continue
        
        x = torch.tensor([atom_features(a) for a in m.GetAtoms()], dtype=torch.float)
        src, dst = [], []
        for bond in m.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [i, j]; dst += [j, i]
        
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        y = torch.tensor([labels], dtype=torch.float)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs

# --- 2. V9 ARCHITECTURE (3-LAYER DEPTH, HIDDEN=126) ---
class ToxGAT_V9_Deep(torch.nn.Module):
    def __init__(self, in_channels=162, hidden=126, heads=6, n_tasks=12):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_channels, hidden)
        # 126 // 6 = 21 (Perfect Integer)
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads)
        self.gat3 = GATConv(hidden, hidden // heads, heads=heads) 
        self.norm = torch.nn.LayerNorm(hidden)
        self.classifier = torch.nn.Linear(hidden * 2, n_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h_init = F.relu(self.input_proj(x))
        
        h1 = F.relu(self.gat1(h_init, edge_index))
        h2 = F.relu(self.gat2(h1, edge_index))
        h3 = F.relu(self.gat3(h2, edge_index))
        
        # Deep Residual connection
        h = self.norm(h3 + h_init) 
        
        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=-1)
        return self.classifier(pooled)

# --- 3. LOSS & TRAIN ---
def v8_weighted_loss(out, target):
    mask = target != -1
    weights = torch.ones(target.shape[1]).to(out.device)
    # Aggressive weighting for the Wedge (AhR) and Kill-Gates
    weights[0], weights[2], weights[11] = 2.5, 3.5, 2.5 
    loss = F.binary_cross_entropy_with_logits(out, target, reduction='none')
    return (loss * mask.float() * weights).sum() / mask.sum()

def train_v9():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training V9 BOOSTED on {device}...")
    
    df = pd.read_csv("startnerve_master_v8_12task.csv")
    tasks = [c for c in df.columns if c != 'SMILES']
    
    graphs = []
    print("🧪 Augmenting Dataset (SMILES Randomization)...")
    for _, row in df.iterrows():
        graphs.extend(get_augmented_graphs(row['SMILES'], row[tasks].values.tolist(), n_aug=3))
    
    print(f"✅ Training Set Size: {len(graphs)} graphs.")
    loader = DataLoader(graphs, batch_size=128, shuffle=True)
    
    model = ToxGAT_V9_Deep().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(1, 41):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = v8_weighted_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(loader)
        scheduler.step(avg_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}")

    torch.save(model.state_dict(), "startnerve_v9_boosted.pt")
    print("✅ V9 BOOSTED BRAIN SAVED!")

if __name__ == "__main__":
    train_v9()