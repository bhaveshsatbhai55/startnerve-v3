"""
================================================================================
STARTNERVE INTELLIGENCE — 3D MOLECULAR GEOMETRY GRAPH COMPILER
================================================================================
Function: Ingests the 12-task CSV master asset, generates valid physical 3D 
          conformers via RDKit, builds molecular graph topologies, and 
          compiles the dataset into a ready-to-train pickle file.
================================================================================
"""

import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

CSV_PATH = "startnerve_master_v12_extended.csv"
OUTPUT_PKL = "titan_graph_features.pkl"

TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-γ", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

def smiles_to_3d_graph(smiles, labels):
    # 1. Parse molecule and explicitly add hydrogens for true 3D space tracking
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    
    # 2. Embed 3D Conformer using ETKDGv3 parameters (Protected against hard C++ exceptions)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = 500
    
    try:
        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status < 0:
            return None  # Skip if conformer generation fails to converge
    except Exception as e:
        # Catch hard C++ Invariant Violations / Distance Geometry failures safely
        return None
        
    # 3. Optimize structure using MMFF94 force field energy minimization
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94")
    except Exception:
        pass # Fallback to base conformer coordinates if strict minimization fails
        
    conf = mol.GetConformer()
    num_nodes = mol.GetNumAtoms()
    
    # 4. Extract atomic identity numbers (z) and 3D coordinate positions (pos)
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(num_nodes)], dtype=np.float32)
    
    # 5. Extract connectivity paths (Edge Index pairs)
    src_indices, dst_indices = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Create bidirectional graph connections
        src_indices.extend([i, j])
        dst_indices.extend([j, i])
        
    edge_index = np.array([src_indices, dst_indices], dtype=np.int64)
    
    # 6. Map the 12 target toxicity labels
    y = np.array(labels, dtype=np.float32)
    
    # Format graph elements into a clean dictionary structure matching the training model
    return {
        "z": z,
        "pos": pos,
        "edge_index": edge_index,
        "y": y,
        "x": None # train_v11_titan will dynamically map this via atomic number arrays
    }

def main():
    print("\n" + "="*80)
    print("        STARTNERVE INTELLIGENCE — 3D GRAPH COMPILATION STARTED")
    print("="*80)
    
    print(f"🛰️  Reading master chemistry matrix: {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    compiled_graphs = []
    total_rows = len(df)
    
    print(f"🧬 Featurizing structures and computing 3D space vectors for {total_rows:,} entries...")
    
    for idx, row in df.iterrows():
        smiles = str(row['SMILES'])
        labels = [row[task] for task in TASKS]
        
        graph_obj = smiles_to_3d_graph(smiles, labels)
        if graph_obj is not None:
            compiled_graphs.append(graph_obj)
            
        if (idx + 1) % 5000 == 0 or (idx + 1) == total_rows:
            print(f"   ↳ Progress: {idx + 1:,} / {total_rows:,} rows completed.")
            
    print(f"🛰️  Saving compiled 3D features to binary file: {OUTPUT_PKL}...")
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(compiled_graphs, f, protocol=4)
        
    print("="*80)
    print(f"🏁 COMPILATION SUCCESS: {len(compiled_graphs):,} / {total_rows:,} graphs compiled!")
    print(f"📊 Features successfully outputted to workspace directory → {OUTPUT_PKL}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()