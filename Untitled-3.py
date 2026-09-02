"""
STARTNERVE INTELLIGENCE — DISK STREAMING CONVERSION UTILITY
Converts the massive 58k pickle file into tiny, independent on-disk tensors.
"""
import os
import pickle
import torch
from torch_geometric.data import Data
from tqdm import tqdm

INPUT_PICKLE = "titan_graph_features.pkl"
OUTPUT_DIR = "titan_disk_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🛰️  Loading binary feature array matrix from: {INPUT_PICKLE}")
with open(INPUT_PICKLE, 'rb') as f:
    raw_graphs = pickle.load(f)

print(f"📦 Serializing {len(raw_graphs)} graphs directly to disk channel...")
for idx, item in enumerate(tqdm(raw_graphs, desc="Writing graphs to disk")):
    
    # Handle V1 vs V2 structural variations seamlessly
    if 'z' in item:
        x_features = torch.tensor(item['x'], dtype=torch.float)
        z_tensor   = torch.tensor(item['z'], dtype=torch.long)
    else:
        z_tensor = torch.tensor(item['x'], dtype=torch.long)
        num_nodes = z_tensor.shape[0]
        x_features = torch.zeros((num_nodes, 162), dtype=torch.float)
        for i in range(num_nodes):
            atomic_num = int(z_tensor[i].item())
            if 1 <= atomic_num <= 118:
                x_features[i, atomic_num - 1] = 1.0

    y_label = torch.tensor(item['y'], dtype=torch.float)
    if y_label.dim() == 1:
        y_label = y_label.unsqueeze(0)

    data = Data(
        x          = x_features,
        z          = z_tensor,
        pos        = torch.tensor(item['pos'], dtype=torch.float),
        edge_index = torch.tensor(item['edge_index'], dtype=torch.long),
        y          = y_label
    )
    
    # Save each graph independently on disk
    torch.save(data, os.path.join(OUTPUT_DIR, f"data_{idx}.pt"))

print(f"\n🏁 SUCCESSFULLY CONVERTED: All elements resting in '{OUTPUT_DIR}/'")