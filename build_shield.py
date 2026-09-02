import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem

def create_reference_library(csv_path):
    print("🧬 Indexing the StartNerve Universe (9,292 compounds)...")
    df = pd.read_csv(csv_path)
    ref_fps = []
    
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Generate a 2048-bit Morgan Fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            ref_fps.append(fp)
            
    with open("startnerve_shield_index.pkl", "wb") as f:
        pickle.dump(ref_fps, f)
    
    print(f"✅ Shield Index Created! {len(ref_fps)} fingerprints saved to 'startnerve_shield_index.pkl'.")

if __name__ == "__main__":
    # Ensure your universe CSV is in the same folder
    create_reference_library("startnerve_universe_v1.csv")