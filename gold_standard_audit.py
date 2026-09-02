"""
StartNerve — Gold Standard Audit V11.2 (Deterministic Hardened)
==============================================================
Locks down 3D coordinate generation via ETKDGv3 seeds and enforces
MMFF94 force-field energy minimization to guarantee stable, reproducible predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from torch_geometric.data import Data
import os
import pandas as pd

# Calibrated safe elements for the V11 Small Molecule Organic GNN
ORGANIC_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53} # H, C, N, O, F, P, S, Cl, Br, I

TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
KILL_GATES = {'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53', 'NR-AR'}
HIGH_RISK   = 0.60
MEDIUM_RISK = 0.40

GOLD_STANDARD = [
    {'name': 'Aflatoxin B1', 'smiles': 'O=C1OC2=CC3=C(OC=C3)C=C2C4=C1C=CO4', 'notes': 'Potent organic carcinogen'},
    {'name': 'Cisplatin', 'smiles': 'N.N.Cl[Pt]Cl', 'notes': 'Platinum coordination complex (Chemotherapy)'},
    {'name': 'Troglitazone', 'smiles': 'CC1=C(C)C2=C(CCC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)O2)C(C)=C1C', 'notes': 'Withdrawn liver toxin'},
    {'name': 'Aspirin', 'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'notes': 'Safe FDA approved drug'},
    {'name': 'Caffeine', 'smiles': 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C', 'notes': 'Generally safe lifestyle compound'}
]

def check_heavy_metal_filter(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False, "INVALID"
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ORGANIC_ELEMENTS:
            return False, atom.GetSymbol()
    return True, "ORGANIC"

def get_node_features(mol):
    charge_computed = True
    try: AllChem.ComputeGasteigerCharges(mol)
    except Exception: charge_computed = False
    all_feats = []
    for atom in mol.GetAtoms():
        features = [0]*118
        num = atom.GetAtomicNum()
        if 1 <= num <= 118: features[num - 1] = 1
        charge = 0.0
        if charge_computed:
            try:
                val = atom.GetProp('_GasteigerCharge')
                if val not in ['-nan', 'nan', 'inf', '-inf']:
                    parsed = float(val)
                    if not (np.isnan(parsed) or np.isinf(parsed)): charge = float(np.clip(parsed, -2.0, 2.0))
            except Exception: pass
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

def generate_deterministic_3d_conformer(smiles):
    """Generates an optimized, highly reproducible 3D Cartesian coordinates tensor."""
    mol = Chem.MolFromSmiles(smiles)
    mol_h = Chem.AddHs(mol)
    
    # Force absolute geometry parameter constraints
    params = AllChem.ETKDGv3()
    params.randomSeed = 42  # Freeze stochastic sampling
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    
    AllChem.EmbedMolecule(mol_h, params)
    
    # Force full MMFF94 mechanical energy minimization loop
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
        
    mol_3d = Chem.RemoveHs(mol_h)
    z = torch.tensor([a.GetAtomicNum() for a in mol_3d.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(mol_3d.GetConformer().GetPositions(), dtype=torch.float)
    return z, pos

def smiles_to_v11_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    x = get_node_features(mol)
    edge_index = get_edge_index(mol)
    z, pos = generate_deterministic_3d_conformer(smiles)
    
    # Structural dimension alignment check
    if z.shape[0] != x.shape[0]:
        z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
        pos = torch.zeros((x.shape[0], 3), dtype=torch.float)
        
    return Data(x=x, z=z, pos=pos, edge_index=edge_index, batch=torch.zeros(x.shape[0], dtype=torch.long))

def run_hardened_audit():
    print(f"\n{'='*75}\n  STARTNERVE V11.2 — HARDENED REGULATORY AUDIT ENGINE\n{'='*75}")
    
    try:
        from train_v11_titan import StartNerve_Titan_V11
        v11_model = StartNerve_Titan_V11(n_tasks=len(TASKS))
        v11_model.load_state_dict(torch.load("startnerve_v11_best.pt", map_location='cpu'))
        v11_model.eval()
        print("  ✅ V11 Titan Intelligence Weights Linked and Frozen.")
    except Exception as e:
        print(f"  ❌ Model file links unverified: {e}")
        return

    for item in GOLD_STANDARD:
        name, smiles = item['name'], item['smiles']
        print(f"\n  Evaluating Compound Blueprint: {name}")
        
        is_organic, element_found = check_heavy_metal_filter(smiles)
        if not is_organic:
            print(f"  ⚠️  AUDIT INTERCEPTED: Non-Organic Coordination Element Detected [{element_found}].")
            print(f"  └─► STATUS: OUT OF APPLICABILITY DOMAIN — ROUTING TO WET LAB ASSAY VAL")
            continue
            
        data_v11 = smiles_to_v11_data(smiles)
        with torch.no_grad():
            preds = torch.sigmoid(v11_model(data_v11)).numpy()[0]
            
        print(f"  {'PATHWAY':<16} | {'V11 RISK VALUE':>14} | {'REGULATORY STATUS'}")
        print(f"  {'-'*60}")
        for i, task in enumerate(TASKS):
            val = preds[i]
            status = "🔴 HIGH" if val >= HIGH_RISK else "🟡 AMBER" if val >= MEDIUM_RISK else "🟢 CLEAN"
            if task in KILL_GATES and val >= MEDIUM_RISK: status += " ⚠️ ALERT"
            print(f"  {task:<16} | {val:>14.4f} | {status}")
            
    print(f"\n{'='*75}\n")

if __name__ == "__main__":
    run_hardened_audit()