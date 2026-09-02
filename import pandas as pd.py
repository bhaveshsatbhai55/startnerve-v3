import pandas as pd
import deepchem as dc
import os

def build_v8_master_universe():
    print("🚀 STARTNERVE V8: Initializing Master Universe Construction...")
    
    # 1. Load the Tox21 dataset (The 12-Pathway Industry Standard)
    # Tasks: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, 
    #        NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
    
    # 2. Combine all splits (Train/Valid/Test) for the full 9k+ dataset
    print("📦 Merging global datasets...")
    full_df = pd.concat([d.to_dataframe() for d in datasets])
    
    # 3. Filter for SMILES + the 12 Task Columns
    # Note: DeepChem's dataframe uses 'smiles' (lowercase)
    cols_to_keep = ['smiles'] + tasks
    df_master = full_df[cols_to_keep].copy()
    
    # 4. Handle Missing Data (The 'Honesty' Layer)
    # We fill NaNs with -1. The V8 Trainer will see -1 and know to skip it.
    df_master = df_master.fillna(-1)
    
    # 5. Export the DNA of StartNerve
    output_name = "startnerve_master_v8_12task.csv"
    df_master.to_csv(output_name, index=False)
    
    print("-" * 30)
    print(f"✅ MASTER UNIVERSE CREATED!")
    print(f"📍 File: {os.path.abspath(output_name)}")
    print(f"🧪 Compounds: {len(df_master)}")
    print(f"🧬 Pathways: {len(tasks)} (Including AhR and p53)")
    print("-" * 30)

if __name__ == "__main__":
    build_v8_master_universe()