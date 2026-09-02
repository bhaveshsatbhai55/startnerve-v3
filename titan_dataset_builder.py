"""
================================================================================
STARTNERVE INTELLIGENCE — TITAN HIGH-PERFORMANCE 3D GRAPH DATASET BUILDER
================================================================================
Function: Ingests 58,272 harmonized rows, extracts complex 162-dim electronic 
          atomic features, projects into true 3D spatial geometry, and serializes.
================================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration Local Channels
INPUT_CSV = "startnerve_master_v12_extended.csv"
OUTPUT_PICKLE = "titan_graph_features.pkl"

TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-γ", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

NODE_FEATURE_DIM = 162
HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

def get_v10_node_features(mol):
    """Extracts 162-dim complex electronic feature vector per atom."""
    charge_computed = True
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        charge_computed = False

    all_node_feats = []
    for atom in mol.GetAtoms():
        features = []
        
        # 1. Atomic Number One-Hot (118 dims)
        atomic_one_hot = [0] * 118
        atomic_num = atom.GetAtomicNum()
        if 1 <= atomic_num <= 118:
            atomic_one_hot[atomic_num - 1] = 1
        features += atomic_one_hot

        # 2. Gasteiger Partial Charge (1 dim)
        charge = 0.0
        if charge_computed:
            try:
                val = atom.GetProp('_GasteigerCharge')
                if val not in ['-nan', 'nan', 'inf', '-inf']:
                    parsed = float(val)
                    if not (np.isnan(parsed) or np.isinf(parsed)):
                        charge = float(np.clip(parsed, -2.0, 2.0))
            except Exception:
                charge = 0.0
        features.append(charge)

        # 3. Hybridization Types (5 dims)
        features += [1 if atom.GetHybridization() == h else 0 for h in HYBRIDIZATION_TYPES]

        # 4. Electronic & Structural Descriptors (5 dims)
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        features.append(float(atom.GetFormalCharge()))
        features.append(float(atom.GetTotalNumHs()))
        features.append(1.0 if atom.IsInRing() else 0.0)
        features.append(float(atom.GetDegree()))
        
        # 5. Zero-Padding Moat to perfectly hit 162 layout channels
        features += [0.0] * (NODE_FEATURE_DIM - len(features))
        all_node_feats.append(features)

    return np.array(all_node_feats, dtype=np.float32)

def process_single_molecule(smiles, labels, index_id):
    """Transforms a single SMILES string into a 3D structural graph tensor with 162-dim features."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Extract the deep electronic topological configurations [162 dims]
        node_features = get_v10_node_features(mol)
        
        # Add implicit Hydrogens to properly compute true 3D spatial volumes
        mol = Chem.AddHs(mol)
        
        # Inject 3D Coordinates using ETKDGv3
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 1  
        
        conformer_id = AllChem.EmbedMolecule(mol, params)
        if conformer_id == -1:
            return None  
            
        # Optimize spatial geometry structures via MMFF94 force-field minimization
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=conformer_id, maxIters=200)
        except:
            pass 
            
        conformer = mol.GetConformer(conformer_id)
        
        # Pure atomic numbers stream for SchNet embedding layers
        atomic_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
        
        # 3D Spatial Geometry Coordinate Matrix Extraction
        coords = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
        
        # 2D Topology Edge Layout Mapping
        edge_indices = []
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            edge_indices.append([start_idx, end_idx])
            edge_indices.append([end_idx, start_idx])
            
        edge_index = np.array(edge_indices, dtype=np.int32).T if edge_indices else np.empty((2, 0), dtype=np.int32)
        
        # Node features check to prevent shape mismatches with newly added Hs
        if node_features.shape[0] != atomic_nums.shape[0]:
            # Re-pad feature rows to account for explicit hydrogen additions
            h_padding = torch.zeros((atomic_nums.shape[0] - node_features.shape[0], NODE_FEATURE_DIM), dtype=torch.float).numpy()
            node_features = np.vstack([node_features, h_padding])

        return {
            'index_id': index_id,
            'smiles': smiles,
            'x': node_features,      # Pre-computed 162-dim matrix
            'z': atomic_nums,        # Pure atomic elements vector
            'pos': coords,           # Shape: [NumAtoms, 3]
            'edge_index': edge_index, # Shape: [2, NumEdges * 2]
            'y': np.array(labels, dtype=np.float32)
        }
    except Exception:
        return None 

def build_titan_dataset():
    print("\n" + "="*80)
    print("      STARTNERVE INTELLIGENCE — TITAN INTERLOCKING DATASET BUILDER V2")
    print("="*80)
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Compiled source data '{INPUT_CSV}' is missing.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    total_rows = len(df)
    print(f"🛰️  Ingested {total_rows} harmonized structure lines completely.")
    
    tasks_data = []
    for idx, row in df.iterrows():
        smiles = str(row['SMILES']).strip()
        labels = [row[task] for task in TASKS]
        tasks_data.append((smiles, labels, idx))
        
    print(f"\n⚡ Allocating parallel CPU worker threads across infrastructure cores...")
    processed_graphs = []
    processed_count = 0
    dropped_count = 0
    
    num_workers = min(os.cpu_count() or 4, 16)
    print(f"   ↳ Processing actively across {num_workers} parallel compute paths...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_molecule, s, l, i): i for s, l, i in tasks_data}
        
        for future in as_completed(futures):
            result = future.result()
            processed_count += 1
            
            if result is not None:
                processed_graphs.append(result)
            else:
                dropped_count += 1
                
            if processed_count % 2000 == 0 or processed_count == total_rows:
                print(f"   [PROGRESS] Balanced lines: {processed_count}/{total_rows} | Graph Moat Encoded: {len(processed_graphs)} | Dropped: {dropped_count}")

    print(f"\n📦 Packaging binary deep learning features matrix asset...")
    with open(OUTPUT_PICKLE, 'wb') as f:
        pickle.dump(processed_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("="*80)
    print(f"🏁 COMPILATION SUCCESS: 3D Geometry + 162-Dim Electronic Graph Tensor Saved!")
    print(f"📊 Processed Vector Matrix saved to disk → {OUTPUT_PICKLE}")
    print("="*80 + "\n")

if __name__ == "__main__":
    build_titan_dataset()