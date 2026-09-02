import pandas as pd
import os

def pull_and_merge():
    print("🚀 Fetching Gold Datasets via Direct URL...")
    
    # Direct links to the raw CSVs from MoleculeNet/DeepChem S3
    clintox_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
    tox21_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

    try:
        # 1. Load ClinTox (FDA Approved vs Failed)
        print("📥 Downloading ClinTox...")
        df_clintox = pd.read_csv(clintox_url)
        # We want SMILES and the 'FDA_APPROVED' column (1 = Safe, 0 = Failed)
        # For our master list, let's flip it: 0 = Safe, 1 = Toxic
        df_clintox['is_toxic'] = 1 - df_clintox['FDA_APPROVED']
        clintox_final = df_clintox[['smiles', 'is_toxic']].copy()
        clintox_final.columns = ['SMILES', 'label']

        # 2. Load Tox21 (Toxic Pathways)
        print("📥 Downloading Tox21...")
        df_tox21 = pd.read_csv(tox21_url)
        # Tox21 has 12 tasks. If ANY task is 1, it's toxic.
        tasks = df_tox21.columns[:-2] # Get the 12 task columns
        df_tox21['is_toxic'] = df_tox21[tasks].max(axis=1)
        tox21_final = df_tox21[['smiles', 'is_toxic']].dropna().copy()
        tox21_final.columns = ['SMILES', 'label']

        # 3. Merge with your existing Master Data
        print("🔄 Merging into StartNerve Universe...")
        universe = pd.concat([clintox_final, tox21_final], ignore_index=True)
        
        # Remove duplicates and clean SMILES
        universe = universe.drop_duplicates(subset=['SMILES'])
        universe = universe.dropna()
        
        universe.to_csv("startnerve_universe_v1.csv", index=False)
        
        print(f"✅ Success! Universe created with {len(universe)} compounds.")
        print(f"🟢 Safe Samples: {len(universe[universe['label'] == 0])}")
        print(f"🔴 Toxic Samples: {len(universe[universe['label'] == 1])}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    pull_and_merge()