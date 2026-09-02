import pandas as pd
import os
import urllib.request

def build_v8_master_universe():
    print("🚀 STARTNERVE V8: Universal Data Acquisition...")
    
    # Direct URL to the Tox21 dataset (The 12-Pathway Industry Standard)
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    local_gz = "tox21.csv.gz"
    
    try:
        if not os.path.exists(local_gz):
            print("⏳ Downloading Tox21 Dataset directly from S3...")
            urllib.request.urlretrieve(url, local_gz)
            print("✅ Download Complete.")
        
        # Load the 12 Tasks
        tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        
        print("🧬 Processing Chemical Graph Data...")
        # Read the compressed CSV
        df = pd.read_csv(local_gz, compression='gzip')
        
        # Tox21 usually has 'smiles' and the task columns
        # We ensure we only keep SMILES and our 12 Kill-Gates
        cols_to_keep = ['smiles'] + tasks
        df_v8 = df[cols_to_keep].copy()
        
        # Rename to our standard 'SMILES'
        df_v8.rename(columns={'smiles': 'SMILES'}, inplace=True)
        
        # Handle missing data (-1 for training honesty)
        df_v8 = df_v8.fillna(-1)
        
        output_name = "startnerve_master_v8_12task.csv"
        df_v8.to_csv(output_name, index=False)
        
        print("-" * 40)
        print(f"✅ MASTER UNIVERSE CREATED!")
        print(f"📍 File: {os.path.abspath(output_name)}")
        print(f"🧪 Total Compounds: {len(df_v8)}")
        print(f"🛡️ Kill-Gates Verified: {len(tasks)} pathways ready.")
        print("-" * 40)
        
        # Clean up the zip file to save space
        if os.path.exists(local_gz):
            os.remove(local_gz)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    build_v8_master_universe()