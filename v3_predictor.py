import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import deepchem as dc
import numpy as np
from rdkit import Chem

def predict_toxicity(smiles_string):
    print(f"\n🔬 Analyzing Molecule: {smiles_string}")
    
    # 1. Convert to 3D Graph
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    feat = featurizer.featurize([smiles_string])
    
    if feat[0] is None:
        print("❌ Error: Invalid SMILES string. Cannot process.")
        return
        
    dataset = dc.data.NumpyDataset(X=np.array(feat, dtype=object))
    
    # 2. Wake up the V3 Brain
    model = dc.models.GraphConvModel(
        n_tasks=1, 
        mode='classification', 
        model_dir="startnerve_v3_ames_gnn"
    )
    
    # model.predict outputs probabilities. 
    # Usually shape is (1, 2) -> [Probability Safe, Probability Mutagen]
    prediction = model.predict(dataset)
    mutagen_probability = prediction[0][0][1] * 100 
    
    # 3. Apply the Enterprise Threshold Shift (Hyper-Paranoid Mode)
    # Default is 50%. We drop it to 15% to prioritize Recall.
    THRESHOLD = 15.0 
    
    print("\n" + "="*40)
    print(f"🧬 Toxicity Probability: {mutagen_probability:.2f}%")
    print("="*40)
    
    if mutagen_probability >= THRESHOLD:
        print("🚨 ICH M7 STATUS: MUTAGENIC IMPURITY DETECTED")
        print(f"⚠️ Flagged by StartNerve (Confidence > {THRESHOLD}%)")
        print("👉 Recommendation: Do NOT synthesize. Run Applicability Domain check.")
    else:
        print("✅ ICH M7 STATUS: CLEARED")
        print("👉 Recommendation: Safe to proceed to next synthesis stage.")

if __name__ == "__main__":
    # Let's test it on a known nightmare molecule: Benzene (highly toxic/mutagenic)
    test_smiles = "C1=CC=CC=C1" 
    predict_toxicity(test_smiles)