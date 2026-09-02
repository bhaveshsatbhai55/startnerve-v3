import pandas as pd
from rdkit import Chem
import os

def canonicalize_smiles(smiles):
    """
    Standardizes chemical strings so different notations 
    of the same molecule are merged.
    """
    try:
        if not isinstance(smiles, str) or smiles.strip() == "":
            return None
        # We allow a slightly loose parse to keep the 'exotic' molecules Claude found
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except Exception:
        return None

def harmonize_datasets(file_paths):
    """
    Merges Hansen + Tox21, resolves conflicts, and outputs 
    only the columns the GNN needs.
    """
    all_data = []
    
    for path in file_paths:
        if not os.path.exists(path):
            print(f"❌ Skipping: {path} (File not found)")
            continue
            
        print(f"📖 Loading: {path}...")
        df = pd.read_csv(path)
        
        # Ensure we only care about SMILES and ACTIVITY 
        # (This ignores the extra Tox21 task columns)
        if 'smiles' not in df.columns or 'activity' not in df.columns:
            print(f"⚠️ Error in {path}: Must have 'smiles' and 'activity' columns.")
            continue

        df['canonical_smiles'] = df['smiles'].apply(canonicalize_smiles)
        
        # Select only the two columns we need for merging
        df_subset = df[['canonical_smiles', 'activity']].dropna(subset=['canonical_smiles'])
        all_data.append(df_subset)

    if not all_data:
        print("🛑 No data loaded. Check your CSV filenames.")
        return None

    # Merge Hansen and Tox21
    combined = pd.concat(all_data)
    
    # Conflict Resolution Logic
    grouped = combined.groupby('canonical_smiles')['activity'].agg(['mean', 'count']).reset_index()
    
    # Keep only unanimous agreement (Mean 0 or 1)
    clean_agreement = grouped[(grouped['mean'] == 0) | (grouped['mean'] == 1)].copy()
    conflicts = grouped[(grouped['mean'] > 0) & (grouped['mean'] < 1)]
    
    print("\n--- StartNerve Data Audit ---")
    print(f"✅ Unique Clean Compounds: {len(clean_agreement)}")
    print(f"⚠️ Conflicting Compounds Removed: {len(conflicts)}")
    
    # Map back to final format
    final_output = clean_agreement[['canonical_smiles', 'mean']].copy()
    final_output.columns = ['smiles', 'activity']
    return final_output

if __name__ == "__main__":
    # 🚀 FILES TO MERGE
    data_files = ['hansen_ames.csv', 'ntp_tox_2026.csv']
    
    master_df = harmonize_datasets(data_files)
    
    if master_df is not None:
        master_df.to_csv('startnerve_master_train.csv', index=False)
        print("\n🏆 'startnerve_master_train.csv' created. Phase 2 Complete.")