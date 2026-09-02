"""
StartNerve V11 — Task 1: 3D Conformer Generation
=================================================
Converts 2D SMILES → 3D Coordinate Tensors using ETKDGv3.
Prerequisite for the Titan Dual-Stream Architecture.

Output: startnerve_v11_3d_base.pt
  Each entry contains:
    - smiles        : original SMILES string
    - atomic_nums   : heavy atom atomic numbers (for SchNet z input)
    - coords        : 3D coordinates in Angstroms (for SchNet pos input)
    - labels        : 12-task toxicity labels (-1 = missing)
    - h_count       : number of hydrogens (for future H-bond features)
    - fallback      : True if 3D embedding failed, coords are zeros
"""

import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_CSV    = "startnerve_master_v8_12task.csv"
OUTPUT_PT    = "startnerve_v11_3d_base.pt"
RANDOM_SEED  = 42
MISSING_LABEL = -1
MMFF_MAX_ITER = 200


# ─────────────────────────────────────────────
# 3D CONFORMER GENERATOR
# ─────────────────────────────────────────────
def generate_3d_conformer(smiles):
    """
    Titan-Grade 3D Conformer Generator.
    Hardened against RDKit BFGSOpt Invariant Violations and RuntimeErrors.
    
    Process:
      1. SMILES -> Mol + Hydrogens
      2. ETKDGv3 Multi-threaded Embedding
      3. MMFF Physics Refinement (with Exception Handling)
      4. Heavy Atom Extraction (Fallback to zeros on failure)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, 0, False

    # Add hydrogens for accurate geometry calculation
    mol_h = Chem.AddHs(mol)
    h_count = mol_h.GetNumAtoms() - mol.GetNumAtoms()

    # ETKDGv3 — industry standard 3D embedding
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.useSmallRingTorsions = True   # Better accuracy for rings
    params.useMacrocycleTorsions = True  # Handles macrocycles
    params.numThreads = 0                # 🔥 USE ALL CPU CORES

    # --- THE HARDENING WRAPPER ---
    try:
        # Step 1: Attempt 3D Embedding
        # This is where the 'Invariant Violation' usually happens
        status = AllChem.EmbedMolecule(mol_h, params)

        if status == -1:
            raise RuntimeError("3D Embedding Failed")

        # Step 2: Physics-based energy minimization
        # MMFF is a steep energy landscape; we catch any linearSearch errors here
        try:
            # MMFFOptimizeMolecule also returns status, but we wrap in try-except for C++ crashes
            result = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=MMFF_MAX_ITER)
        except Exception:
            # If optimization fails, we still have the raw 3D embedding from Step 1
            pass

        # Step 3: Extract Coordinates
        mol_3d = Chem.RemoveHs(mol_h)
        conf = mol_3d.GetConformer()
        atomic_nums = np.array([atom.GetAtomicNum() for atom in mol_3d.GetAtoms()], dtype=np.int32)
        coords = np.array(conf.GetPositions(), dtype=np.float32)
        success = True

    except Exception as e:
        # --- FALLBACK PATH ---
        # If ANY C++ or Python error occurs, we gracefully fall back to 2D
        # This keeps the training loop alive for the GAT stream
        heavy_mol = Chem.RemoveHs(mol)
        n_atoms = heavy_mol.GetNumAtoms()
        atomic_nums = np.array([atom.GetAtomicNum() for atom in heavy_mol.GetAtoms()], dtype=np.int32)
        coords = np.zeros((n_atoms, 3), dtype=np.float32)
        success = False

    return atomic_nums, coords, h_count, success

# ─────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────
def build_v11_dataset():
    print(f"\n{'='*60}")
    print(f"  StartNerve V11 — 3D Conformer Generation")
    print(f"{'='*60}\n")

    # ── Load dataset ──
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_CSV}\n"
            f"Place your Tox21 CSV in the same directory."
        )

    df   = pd.read_csv(INPUT_CSV)
    tasks = [c for c in df.columns if c != 'SMILES']

    print(f"  Input  : {INPUT_CSV}")
    print(f"  Output : {OUTPUT_PT}")
    print(f"  Molecules : {len(df)}")
    print(f"  Tasks     : {len(tasks)}")
    print(f"\n  Starting 3D conformer generation...\n")

    # ── Generate conformers ──
    v11_data = []
    failed_smiles   = 0
    failed_3d       = 0
    fallback_used   = 0
    success_full    = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Generating"):
        smiles = row['SMILES']

        # Fill missing labels with -1
        labels = row[tasks].fillna(MISSING_LABEL).values.tolist()

        atomic_nums, coords, h_count, success = generate_3d_conformer(smiles)

        if atomic_nums is None:
            # Invalid SMILES — skip entirely
            failed_smiles += 1
            continue

        if not success:
            # 3D embedding failed — stored with fallback zeros
            # Still usable for 2D GAT stream in dual-stream model
            fallback_used += 1
        else:
            success_full += 1

        v11_data.append({
            'smiles':       smiles,
            'atomic_nums':  atomic_nums.tolist(),   # List of ints for SchNet z
            'coords':       coords.tolist(),         # List of [x,y,z] for SchNet pos
            'labels':       labels,                  # 12 task labels
            'h_count':      h_count,                 # Hydrogen count
            'n_atoms':      len(atomic_nums),        # Heavy atom count
            'fallback':     not success              # Flag for 2D-only fallback
        })

    # ── Save ──
    torch.save(v11_data, OUTPUT_PT)

    # ── Validation check ──
    loaded = torch.load(OUTPUT_PT)
    assert len(loaded) == len(v11_data), "Save/load validation failed"

    # ── Report ──
    total_stored = len(v11_data)
    fallback_pct = fallback_used / total_stored * 100 if total_stored > 0 else 0
    success_pct  = success_full  / total_stored * 100 if total_stored > 0 else 0

    print(f"\n{'='*60}")
    print(f"  3D CONFORMER GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total stored     : {total_stored:,}")
    print(f"  Full 3D success  : {success_full:,} ({success_pct:.1f}%)")
    print(f"  Fallback (2D)    : {fallback_used:,} ({fallback_pct:.1f}%)")
    print(f"  Invalid SMILES   : {failed_smiles:,}")
    print(f"\n  Saved → {OUTPUT_PT}")
    print(f"  Verified loadable ✅")

    # Sample inspection
    sample = v11_data[0]
    print(f"\n  Sample molecule : {sample['smiles']}")
    print(f"  Heavy atoms     : {sample['n_atoms']}")
    print(f"  Coord shape     : ({sample['n_atoms']}, 3)")
    print(f"  Labels          : {sample['labels']}")
    print(f"  Fallback used   : {sample['fallback']}")
    print(f"\n  V11 dataset ready for Titan training.")
    print(f"{'='*60}\n")

    return v11_data


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    build_v11_dataset()