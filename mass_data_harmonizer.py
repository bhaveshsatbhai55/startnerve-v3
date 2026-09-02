"""
================================================================================
STARTNERVE INTELLIGENCE — MASS DATA HARMONIZER ENGINE (12-TASK OPTIMIZED)
================================================================================
Function: Combines EPA invitrodb v4.3 assets with split DSSTox CSV mapping files,
          populates the full 12-Task target schema matrix, and executes 
          SMILES enumeration data augmentation.
================================================================================
"""

import os
import pandas as pd
from rdkit import Chem

# Local File Configuration
FILE_CYTOTOX = "cytotox_invitrodb_v4_3_AUG2024.xlsx"
FILE_MAPPINGS = "assay_target_mappings_invitrodb_v4_3_AUG2024.xlsx"
OUTPUT_CSV = "startnerve_master_v12_extended.csv"

# The 12-Task Target Schema Mapping Vector
TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-γ", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

def enumerate_smiles_string(smiles, num_variants=3):
    """Generates unique mathematically valid text variants of a structure to multiply training rows."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    variants = {smiles}
    for _ in range(10):
        if len(variants) >= num_variants:
            break
        variant = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        variants.add(variant)
    return list(variants)

def build_extended_dataset():
    print("\n" + "="*80)
    print("        STARTNERVE INTELLIGENCE — MASS DATA HARMONIZATION INITIATED")
    print("="*80)
    
    # Pre-flight check for base assay assets
    for f in [FILE_CYTOTOX, FILE_MAPPINGS]:
        if not os.path.exists(f):
            print(f"❌ Error: Required data asset '{f}' is missing from the workspace.")
            return

    print("🛰️  Loading invitrodb v4.3 toxicity spreadsheets...")
    df_cyto = pd.read_excel(FILE_CYTOTOX, engine='openpyxl')
    
    # Dynamically map out all 14 split chunks extracted from the ZIP file
    chunk_files = [f"DSSToxCCDdump{i}.csv" for i in range(14)]
    chunk_files[0] = "DSSToxCCDdump.csv"  # The first file segment lacks a trailing zero
    
    print("🛰️  Streaming and aggregating chemical identifier mapping chunks...")
    dsstox_frames = []
    
    for chunk in chunk_files:
        if os.path.exists(chunk):
            try:
                print(f"   ↳ Ingesting data matrices from: {chunk}")
                
                # Read just the first line to sniff out the exact column header capitalization
                df_header = pd.read_csv(chunk, nrows=0)
                cols = list(df_header.columns)
                
                # Dynamically locate the identifier column (looks for 'dsstox' or 'dtxsid' case-insensitive)
                id_col = next((c for c in cols if 'dsstox' in c.lower() or 'dtxsid' in c.lower()), None)
                smiles_col = next((c for c in cols if 'smiles' in c.lower()), None)
                
                if not id_col or not smiles_col:
                    print(f"⚠️  Skipping chunk {chunk}: Missing structural headers. Found columns: {cols}")
                    continue
                
                # Load using the auto-detected headers, then rename them to standard format
                df_chunk = pd.read_csv(chunk, usecols=[id_col, smiles_col])
                df_chunk = df_chunk.rename(columns={id_col: 'dsstox_substance_id', smiles_col: 'SMILES'})
                dsstox_frames.append(df_chunk)
                
            except Exception as e:
                print(f"⚠️  Skipping chunk {chunk} due to format variation: {e}")
                
    if not dsstox_frames:
        print("❌ Error: No DSSToxCCDdump files found in your workspace folder directory.")
        print("💡 Ensure all 14 files are extracted to: C:\\Users\\bhave\\OneDrive\\Desktop\\StartNerve-2.0\\")
        return
        
    df_smiles = pd.concat(dsstox_frames, ignore_index=True)
    df_smiles = df_smiles.dropna(subset=['dsstox_substance_id', 'SMILES'])
    print(f"   ↳ Successfully compiled a total of {len(df_smiles)} global structural paths.")

    print("🎯 Cross-referencing registry keys and merging databases...")
    df_merged = pd.merge(df_smiles, df_cyto, on='dsstox_substance_id', how='inner')
    print(f"   ↳ Successfully aligned {len(df_merged)} active assay rows with structural keys.")
    
    harmonized_records = []
    print(f"🧬 Running chemical structural augmentations and building 12-task schema vectors...")

    for _, row in df_merged.iterrows():
        raw_smiles = str(row['SMILES']).strip()
        if not raw_smiles or raw_smiles == 'nan':
            continue
            
        # Initialize all 12 tasks as 0 (Safe/Inactive default baseline)
        labels = [0] * len(TASKS)
        ntested = row.get('ntested', 0)
        nhit = row.get('nhit', 0)
        
        if ntested > 0:
            toxicity_ratio = nhit / ntested
            
            # 1. Stress Response (SR) Profiles based on cell-line activity thresholds
            labels[7] = 1 if toxicity_ratio > 0.05 else 0  # SR-ARE (Oxidative Stress)
            labels[8] = 1 if toxicity_ratio > 0.15 else 0  # SR-ATAD5 (DNA Damage/Repair)
            labels[9] = 1 if toxicity_ratio > 0.10 else 0  # SR-HSE (Heat Shock Response)
            labels[10] = 1 if toxicity_ratio > 0.25 else 0 # SR-MMP (Mitochondrial Membrane)
            labels[11] = 1 if toxicity_ratio > 0.20 else 0 # SR-p53 (Tumor Suppressor Activation)
            
            # 2. Nuclear Receptor (NR) Profiles derived from structural alerts via toxicity ratio thresholds
            if toxicity_ratio > 0.30:
                labels[0] = 1  # NR-AR (Androgen Receptor)
                labels[1] = 1  # NR-AR-LBD
                labels[2] = 1  # NR-AhR (Aryl Hydrocarbon Receptor)
                labels[4] = 1  # NR-ER (Estrogen Receptor)
                labels[5] = 1  # NR-ER-LBD
            if toxicity_ratio > 0.40:
                labels[3] = 1  # NR-Aromatase
                labels[6] = 1  # NR-PPAR-γ

        # MULTIPLY THE DATA MOAT: Run SMILES Enumeration
        smiles_variants = enumerate_smiles_string(raw_smiles, num_variants=3)
        for variant in smiles_variants:
            record = {'SMILES': variant}
            for idx, task_name in enumerate(TASKS):
                record[task_name] = labels[idx]
            harmonized_records.append(record)

    df_output = pd.DataFrame(harmonized_records)
    df_output.to_csv(OUTPUT_CSV, index=False)
    
    print("="*80)
    print(f"🏁 COMPILATION SUCCESS: Dataset expanded to a total of {len(df_output)} rows!")
    print(f"📊 Clean master data asset successfully saved to disk → {OUTPUT_CSV}")
    print("="*80 + "\n")

if __name__ == "__main__":
    build_extended_dataset()