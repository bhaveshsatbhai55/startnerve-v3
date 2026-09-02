import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from rdkit import Chem
from sklearn.model_selection import train_test_split
import numpy as np

# 1. THE 162-FEATURE ENGINE (Synchronized with your successful inference)
def _one_hot(value, choices: list) -> list:
    vec = [0] * (len(choices) + 1)
    try: idx = choices.index(value)
    except: idx = len(choices)
    vec[idx] = 1
    return vec

def atom_features(atom) -> list:
    atomic_num_vec = [0] * 118
    num = atom.GetAtomicNum()
    if 1 <= num <= 118: atomic_num_vec[num-1] = 1
    
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

def smilesto_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))

# 2. LOAD THE UNIVERSE
print("🌌 Loading the StartNerve Universe...")
df = pd.read_csv("startnerve_universe_v1.csv")
data_list = []
for _, row in df.iterrows():
    g = smilesto_graph(row['SMILES'], row['label'])
    if g: data_list.append(g)

train_data, val_data = train_test_split(data_list, test_size=0.1, random_state=42)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

# 3. THE ARCHITECTURE (Locked at 162 in_channels)
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

# 4. TRAINING WITH PENALTY LOGIC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ToxGAT_V6().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Weighted Loss: Force focus on Safe drugs
pos_weight = torch.tensor([0.5]).to(device) # Reduces emphasis on toxic, focuses on safe accuracy

print("🚀 Starting V6 Training...")
for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")

# 5. SAVE THE FINAL BRAIN
torch.save(model.state_dict(), "startnerve_v6_universe.pt")
print("✅ StartNerve V6 Universe Brain is ready.")