"""
StartNerve — Phase 2, Layer 2 Diagnostics V11.5
==================================================
Micro-logged version to isolate the CPU thread bottleneck.
Completely bypasses MMFF optimization for instant 3D coordination.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from torch_geometric.data import Data

ORGANIC_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
RISK_THRESHOLD = 0.40

# Highly responsive 5-compound diagnostic array
DIAGNOSTIC_DATABASE = [
    {"name": "Bromfenac", "smiles": "C1=CC=C(C=C1)C(=O)C2=CC=C(C=C2N)CC(=O)O", "class": "Hepatotoxicity"},
    {"name": "Benoxaprofen", "smiles": "CC(C(=O)O)c1ccc2oc3cc(Cl)ccc3c2c1", "class": "Mitochondrial Failure"},
    {"name": "Nimesulide", "smiles": "CS(=O)(=O)NC1=CC=C(C=C1)OC2=CC=CC=C2[N+](=O)[O-]", "class": "Acute Hepatic Necrosis"},
    {"name": "Bisphenol A", "smiles": "CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O", "class": "Endocrine Disruptor"},
    {"name": "Aniline", "smiles": "C1=CC=C(C=C1)N", "class": "Methemoglobinemia Inducer"}
]

def get_node_features(mol):
    charge_computed = True
    try: 
        AllChem.ComputeGasteigerCharges(mol)
    except Exception: 
        charge_computed = False
        
    all_feats = []
    for atom in mol.GetAtoms():
        features = [0]*118
        num = atom.GetAtomicNum()
        if 1 <= num <= 118: features[num - 1] = 1
        
        # Restore true electronic hotspot tracking
        charge = 0.0
        if charge_computed:
            try:
                val = atom.GetProp('_GasteigerCharge')
                if val not in ['-nan', 'nan', 'inf', '-inf']:
                    parsed = float(val)
                    if not (np.isnan(parsed) or np.isinf(parsed)): 
                        charge = float(np.clip(parsed, -2.0, 2.0))
            except Exception: 
                pass
        features.append(charge)
        
        features += [1 if atom.GetHybridization() == h else 0 for h in [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D, rdchem.HybridizationType.SP3D2]]
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        features.append(float(atom.GetFormalCharge()))
        features.append(float(atom.GetTotalNumHs()))
        features.append(1.0 if atom.IsInRing() else 0.0)
        features.append(float(atom.GetDegree()))
        features += [0.0] * (162 - len(features))
        all_feats.append(features)
    return torch.tensor(all_feats, dtype=torch.float)

def get_edge_index(mol):
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    return torch.tensor([src, dst], dtype=torch.long) if src else None

def generate_instant_3d_conformer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol_h, params)
    mol_3d = Chem.RemoveHs(mol_h)
    z = torch.tensor([a.GetAtomicNum() for a in mol_3d.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(mol_3d.GetConformer().GetPositions(), dtype=torch.float)
    return z, pos

def run_diagnostic_recall():
    print("\n[STEP 1] Initializing Script & Loading V11 Neural Network weights...")
    try:
        from train_v11_titan import StartNerve_Titan_V11
        model = StartNerve_Titan_V11(n_tasks=len(TASKS))
        model.load_state_dict(torch.load("startnerve_v11_best.pt", map_location='cpu'))
        model.eval()
        print(" -> SUCCESS: Model weights successfully loaded into CPU system memory.\n")
    except Exception as e:
        print(f" -> CRITICAL FAILURE loading weights: {e}")
        return

    print("[STEP 2] Commencing Loop Execution over Diagnostic Set...")
    compounds_caught = 0

    for idx, item in enumerate(DIAGNOSTIC_DATABASE, 1):
        name, smiles = item['name'], item['smiles']
        print(f"\n --- Processing [{idx}/5]: {name} ---")
        
        print("     -> Converting SMILES to RDKit Mol object...")
        mol = Chem.MolFromSmiles(smiles)
        
        print("     -> Building topological node and edge tensor matrices...")
        x = get_node_features(mol)
        edge_index = get_edge_index(mol)
        
        print("     -> Generating instant 3D coordinates via ETKDGv3...")
        z, pos = generate_instant_3d_conformer(smiles)
        
        print("     -> Packaging geometric PyG Data object...")
        data = Data(x=x, z=z, pos=pos, edge_index=edge_index, batch=torch.zeros(x.shape[0], dtype=torch.long))
        
        print("     -> Passing graph tensors into frozen GCN/SchNet layers...")
        with torch.no_grad():
            output = model(data)
            preds = torch.sigmoid(output).numpy()[0]
            
        max_val = np.max(preds)
        max_task_name = TASKS[np.argmax(preds)]
        
        is_caught = max_val >= RISK_THRESHOLD
        if is_caught:
            compounds_caught += 1
            status = f"🔴 CAUGHT ({max_task_name})"
        else:
            status = "🟢 MISSED"
            
        print(f" >>> RESULT FOR {name}: Max Tox Value = {max_val:.4f} | Status = {status}")

    recall_rate = (compounds_caught / len(DIAGNOSTIC_DATABASE)) * 100
    print(f"\n{'='*60}\n[STEP 3] DIAGNOSTIC RUN COMPLETE\n{'='*60}")
    print(f" Total Evaluated: {len(DIAGNOSTIC_DATABASE)}")
    print(f" Total Caught   : {compounds_caught}")
    print(f" Diagnostic Recall Rate: {recall_rate:.2f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_diagnostic_recall()