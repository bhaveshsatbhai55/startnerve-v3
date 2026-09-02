import pandas as pd
from chembl_webresource_client.new_client import new_client
import time

def pull_chembl_training_data():
    print("\n🚀 Booting up StartNerve V2 Enterprise Data Pipeline...")
    
    # Target: Aromatase (A major toxicity endpoint used in Tox21)
    # The official ChEMBL ID for Aromatase is CHEMBL1978
    target_id = "CHEMBL1978" 
    print(f"📡 Connecting to European ChEMBL Database for target: {target_id}...")
    
    activity = new_client.activity
    
    # We are specifically asking for "IC50" values (the exact concentration that causes toxicity)
    print("📥 Downloading lab-verified datasets... (This may take 1-2 minutes)")
    start_time = time.time()
    
    res = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
    df = pd.DataFrame.from_dict(res)
    
    if not df.empty:
        # We only want the high-quality data: The SMILES string and the actual toxicity number
        clean_df = df[['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units']].dropna()
        
        filename = "StartNerve_V2_Aromatase_Training_Data.csv"
        clean_df.to_csv(filename, index=False)
        
        elapsed = round(time.time() - start_time, 1)
        print(f"\n✅ SUCCESS! Extracted {len(clean_df)} ultra-high-quality data points in {elapsed} seconds.")
        print(f"📁 Saved V2 training block as: {filename}")
        print("🎯 Next Step: We will feed these SMILES into our new Graph Neural Network.")
        print("--------------------------------------------------\n")
    else:
        print("\n⚠️ No data found or connection timed out.")

if __name__ == "__main__":
    pull_chembl_training_data()