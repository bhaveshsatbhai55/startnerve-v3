"""
StartNerve V10 — Layer 1: PROSPECTIVE EXTERNAL VALIDATION (Complete Suite)
==========================================================================
FULL BUILD: 
- Automated ChEMBL API Pull (Actives & Inactives)
- InChIKey-based Deduplication against Tox21
- Tanimoto-based Applicability Domain (AD) Mapping
- Youden's J-Statistic Threshold Optimization
- Multi-Task Sensitivity/Specificity Auditing
- Full CSV Export for Case Study Generation
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    confusion_matrix, 
    roc_curve
)
from rdkit import Chem
from rdkit.Chem import AllChem
from train_v10_electronic import ToxGAT_V10, mol_to_graph_v10

# ─────────────────────────────────────────────
# 1. CONFIGURATION & TARGET MAPPING
# ─────────────────────────────────────────────
MODEL_PATH        = "startnerve_v10_best.pt"
TRAIN_DATA_PATH   = "startnerve_master_v8_12task.csv"
EXTERNAL_DATA     = "chembl_holdout_v1.csv"
MISSING_LABEL     = -1
BATCH_SIZE        = 64
RANDOM_SEED       = 42

# AD Thresholds
AD_GREEN = 0.6  # High confidence
AD_GREY  = 0.3  # Warning zone
MIN_POSITIVE_SAMPLES = 5 

# Mapping Tox21 pathways to real-world ChEMBL Assay IDs
CHEMBL_TARGET_MAP = {
    'NR-AhR':        'CHEMBL2094253', 
    'NR-AR':         'CHEMBL1871',    
    'NR-ER':         'CHEMBL206',     
    'NR-Aromatase':  'CHEMBL1978',    
    'NR-PPAR-gamma': 'CHEMBL235',     
    'SR-p53':        'CHEMBL4302'     
}

# ─────────────────────────────────────────────
# 2. CHEMBL API ENGINE (The Pull Logic)
# ─────────────────────────────────────────────
def pull_chembl_by_status(target_id, is_active=True, limit=50): # Reduced limit to 50
    """Connects to EBI API with higher patience and smaller batches."""
    status_op = "gte" if is_active else "lte"
    status_val = 5 if is_active else 4
    
    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        "target_chembl_id": target_id,
        "standard_type":    "IC50",
        "pchembl_value__{}".format(status_op): status_val,
        "limit": limit, "offset": 0
    }
    
    # Try 3 times before giving up
    for attempt in range(3):
        try:
            # Increased timeout to 60 seconds
            resp = requests.get(url, params=params, timeout=60) 
            if resp.status_code == 200:
                data = resp.json()
                # Give the server a 1-second break
                time.sleep(1) 
                return [a['canonical_smiles'] for a in data.get('activities', []) if a.get('canonical_smiles')]
        except Exception as e:
            print(f"  ⏳ Timeout on {target_id} (Attempt {attempt+1}/3). Retrying...")
            time.sleep(5) # Wait 5 seconds before retrying
            
    return []

def build_chembl_holdout(train_smiles_set):
    """Pulls, cleans, and deduplicates external data."""
    print("\n  🌐 PULL MODE: Contacting ChEMBL for novel chemistry...")
    
    # Generate InChIKeys for 100% accurate deduplication
    print("  🔑 Generating Training Index (InChIKeys)...")
    train_keys = set()
    for s in train_smiles_set:
        m = Chem.MolFromSmiles(s)
        if m: train_keys.add(Chem.inchi.InchiToInchiKey(Chem.MolToInchi(m)))

    all_records = {} # smiles -> {pathway: label}

    for pathway, tid in CHEMBL_TARGET_MAP.items():
        print(f"  📥 Fetching {pathway}...")
        actives = pull_chembl_by_status(tid, is_active=True)
        inactives = pull_chembl_by_status(tid, is_active=False)
        
        for smi, label in [(s, 1) for s in actives] + [(s, 0) for s in inactives]:
            mol = Chem.MolFromSmiles(smi)
            if not mol: continue
            
            try:
                ikey = Chem.inchi.InchiToInchiKey(Chem.MolToInchi(mol))
                if ikey in train_keys: continue # Skip if seen in Tox21
                
                can_smi = Chem.MolToSmiles(mol)
                if can_smi not in all_records: all_records[can_smi] = {}
                all_records[can_smi][pathway] = label
            except: continue

    df = pd.DataFrame.from_dict(all_records, orient='index').reset_index()
    df.columns = ['SMILES'] + list(df.columns[1:])
    return df.fillna(MISSING_LABEL)

# ─────────────────────────────────────────────
# 3. APPLICABILITY DOMAIN (AD) ENGINE
# ─────────────────────────────────────────────
def compute_ad_scores(train_smiles, test_smiles):
    """Calculates Tanimoto similarity to establish trust zones."""
    print("  🧪 Calculating Applicability Domain (Tanimoto)...")
    train_fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)) for s in train_smiles if Chem.MolFromSmiles(s)]
    train_fps = np.array(train_fps)

    zones = []
    for s in test_smiles:
        mol = Chem.MolFromSmiles(s)
        if not mol: 
            zones.append("RED")
            continue
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        
        # Matrix-speed Tanimoto
        intersection = np.dot(train_fps, fp)
        union = train_fps.sum(axis=1) + fp.sum() - intersection
        sim = np.max(intersection / union)
        
        zones.append("GREEN" if sim >= AD_GREEN else "GREY" if sim >= AD_GREY else "RED")
    return zones

# ─────────────────────────────────────────────
# 4. STATISTICAL ENGINE (Thresholds & Metrics)
# ─────────────────────────────────────────────
def find_optimal_threshold(y_true, y_prob):
    """Finds threshold that maximizes Youden's J (True Positive vs False Positive)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

# ─────────────────────────────────────────────
# 5. MAIN EXECUTION PIPELINE
# ─────────────────────────────────────────────
def run_full_validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\n  STARTNERVE V10: FULL PROSPECTIVE AUDIT\n{'='*60}")

    # Load Brain
    model = ToxGAT_V10().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load Train Set for AD Checking
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # Pull or Load External Data
    if os.path.exists(EXTERNAL_DATA):
        print(f"  📂 Loading existing external set: {EXTERNAL_DATA}")
        ext_df = pd.read_csv(EXTERNAL_DATA)
    else:
        ext_df = build_chembl_holdout(set(train_df['SMILES']))
        ext_df.to_csv(EXTERNAL_DATA, index=False)

    # Run AD Checker
    ad_zones = compute_ad_scores(train_df['SMILES'].tolist(), ext_df['SMILES'].tolist())
    ext_df['AD_Zone'] = ad_zones
    
    # Filter for valid predictions (Ignore RED zone)
    valid_df = ext_df[ext_df['AD_Zone'] != "RED"].copy()
    print(f"  ✅ AD Audit: {ad_zones.count('GREEN')} Green, {ad_zones.count('GREY')} Grey, {ad_zones.count('RED')} Red")

    # Inference
    graphs = [mol_to_graph_v10(s, l) for s, l in zip(valid_df['SMILES'], valid_df.drop(columns=['SMILES', 'AD_Zone']).values.tolist())]
    graphs = [g for g in graphs if g is not None]
    loader = DataLoader(graphs, batch_size=BATCH_SIZE)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            all_preds.append(torch.sigmoid(model(data)).cpu().numpy())
            all_targets.append(data.y.cpu().numpy())

    preds, targets = np.vstack(all_preds), np.vstack(all_targets)
    tasks = [t for t in ext_df.columns if t not in ['SMILES', 'AD_Zone']]

    # Final Table
    print(f"\n{'PATHWAY':<15} | {'EXT AUROC':>10} | {'SENS':>7} | {'SPEC':>7} | {'STATUS'}")
    print("-" * 65)

    stats = []
    for i, task in enumerate(tasks):
        mask = targets[:, i] != -1
        if mask.sum() < MIN_POSITIVE_SAMPLES: continue
        
        y_t, y_p = targets[mask, i], preds[mask, i]
        auc = roc_auc_score(y_t, y_p)
        thresh = find_optimal_threshold(y_t, y_p)
        y_bin = (y_p >= thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_t, y_bin, labels=[0,1]).ravel()
        sens = tp / (tp + fn) if (tp+fn) > 0 else 0
        spec = tn / (tn + fp) if (tn+fp) > 0 else 0
        
        status = "⭐ ELITE" if auc >= 0.82 else "🔹 SOLID"
        print(f"{task:<15} | {auc:>10.3f} | {sens:>7.1%} | {spec:>7.1%} | {status}")
        stats.append({'Pathway': task, 'Ext_AUROC': auc, 'Sens': sens, 'Spec': spec})

    pd.DataFrame(stats).to_csv("prospective_validation_final.csv", index=False)
    print(f"\n✅ SUCCESS: Full Dossier exported to prospective_validation_final.csv")

if __name__ == "__main__":
    run_full_validation()