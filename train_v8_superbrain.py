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

def mol_to_graph(smiles, labels):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    raw_features = [atom_features(a) for a in mol.GetAtoms()]
    if any(len(f) != 162 for f in raw_features):
        return None
    x = torch.tensor(raw_features, dtype=torch.float)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([labels], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

# --- 2. THE ARCHITECTURE (Hidden=126 for 6 Heads Alignment) ---
class ToxGAT_V8_Superbrain(torch.nn.Module):
    def __init__(self, in_channels=162, hidden=126, heads=6, n_tasks=12):
        super().__init__()
        # hidden=126 ensures 126 // 6 = 21 (Perfect Integer)
        self.input_proj = torch.nn.Linear(in_channels, hidden)
        
        # gat1 output will be (126 // 6) * 6 = 126
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads)
        
        self.norm = torch.nn.LayerNorm(hidden)
        self.classifier = torch.nn.Linear(hidden * 2, n_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # h is now exactly 126 features
        h_init = F.relu(self.input_proj(x))
        
        # h1 and h2 will stay exactly 126 features
        h1 = F.relu(self.gat1(h_init, edge_index))
        h2 = F.relu(self.gat2(h1, edge_index))
        
        # Residual connection: 126 + 126 = Perfect Match
        h = self.norm(h2 + h_init) 
        
        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=-1)
        return self.classifier(pooled)

# --- 3. LOSS ---
def v8_weighted_loss(out, target):
    mask = target != -1
    weights = torch.ones(target.shape[1]).to(out.device)
    weights[0], weights[2], weights[11] = 2.5, 3.0, 2.0
    loss = F.binary_cross_entropy_with_logits(out, target, reduction='none')
    return (loss * mask.float() * weights).sum() / mask.sum()

# --- 4. TRAIN ---
def train_v8():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training V8 Superbrain on {device}...")
    
    df = pd.read_csv("startnerve_master_v8_12task.csv")
    tasks = [c for c in df.columns if c != 'SMILES']
    
    graphs = []
    print("🧪 Converting Universe to Molecular Graphs (Final Alignment Check)...")
    for _, row in df.iterrows():
        g = mol_to_graph(row['SMILES'], row[tasks].values.tolist())
        if g:
            if g.x.shape[1] == 162:
                graphs.append(g)
    
    print(f"✅ Cleaned {len(graphs)} graphs. Ready for 12-Pathway Profiling.")
    loader = DataLoader(graphs, batch_size=64, shuffle=True)
    
    model = ToxGAT_V8_Superbrain(in_channels=162, hidden=126).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, 51):
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
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "startnerve_v8_superbrain.pt")
    print("✅ V8 SUPERBRAIN SAVED! The engine is now perfectly balanced.")

if __name__ == "__main__":
    train_v8()