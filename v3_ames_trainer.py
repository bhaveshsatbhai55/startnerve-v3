import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas as pd
import deepchem as dc
import numpy as np
import urllib.request
import ssl

# Shut up the annoying RDKit deprecation warnings
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

def build_v3_ames_engine():
    print("\n🚀 Booting up StartNerve V3: ICH M7 Ames Mutagenicity Engine...")
    
    csv_file = "Hansen_Mutagenicity.csv"
    
    # 1. The Indestructible Data Pull 
    if not os.path.exists(csv_file):
        print("📥 Pulling data directly from TU Berlin servers...")
        url = "http://doc.ml.tu-berlin.de/toxbenchmark/Mutagenicity_N6512.csv"
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, context=ctx) as response, open(csv_file, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"🛑 Error: {e}")
            return
            
    df = pd.read_csv(csv_file)
    smiles_col = [col for col in df.columns if 'smiles' in col.lower()][0]
    class_col = [col for col in df.columns if 'activity' in col.lower() or 'class' in col.lower()][0]
    
    raw_smiles = df[smiles_col].values
    raw_labels = df[class_col].values
    
    # --- THE NEW FIX: The RDKit Acid Bath ---
    print("🧼 Scrubbing dirty chemical data and typos from the dataset...")
    clean_smiles = []
    clean_labels = []
    
    for s, l in zip(raw_smiles, raw_labels):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:  # Only keep it if it obeys the laws of chemistry
            clean_smiles.append(s)
            clean_labels.append(l)
            
    smiles = np.array(clean_smiles)
    labels = np.array(clean_labels)
    print(f"📁 Kept {len(smiles)} chemically valid Ames test records. Tossed {len(raw_smiles) - len(smiles)} corrupted ones.")
    
    # 2. V3 Featurizer: Forcing 3D Stereochemistry
    print("🧬 Translating SMILES into 3D-Aware Molecular Graphs... (This takes a moment)")
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True) 
    
    valid_X = []
    valid_indices = []
    
    for i, smile in enumerate(smiles):
        try:
            feat = featurizer.featurize([smile])
            if feat is not None and len(feat) > 0 and feat[0] is not None:
                valid_X.append(feat[0])
                valid_indices.append(i)
        except Exception:
            continue
            
    X = np.array(valid_X, dtype=object)
    y = labels[valid_indices]
    valid_smiles = smiles[valid_indices]
    
    print(f"✅ Successfully mapped {len(X)} 3D-aware molecules.")
    
    # 3. Create Dataset and Apply Scaffold Split
    dataset = dc.data.NumpyDataset(X, y, ids=valid_smiles)
    print("🔪 Applying Scaffold Split to prevent data leakage...")
    splitter = dc.splits.ScaffoldSplitter() 
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1
    )
    
    # 4. Save the Training SMILES for the Applicability Domain (AD)
    np.save("v3_training_smiles.npy", train_dataset.ids)
    print("🛡️ Applicability Domain (AD) reference map saved.")
    
    # 5. Build the V3 Neural Network
    print("\n🧠 Initializing the V3 Ames Graph Convolutional Network...")
    model = dc.models.GraphConvModel(
        n_tasks=1, 
        mode='classification', 
        dropout=0.2,
        model_dir="startnerve_v3_ames_gnn"
    )
    
    # 6. Train the AI
    print("⚙️ Training the Neural Network to hunt DNA Mutations... (This will take a few minutes)")
    model.fit(train_dataset, nb_epoch=25)
    
    # 7. Evaluate 
    print("\n📈 Evaluating V3 Brain Accuracy...")
    metric_auc = dc.metrics.Metric(dc.metrics.roc_auc_score)
    metric_recall = dc.metrics.Metric(dc.metrics.recall_score) 
    
    test_score = model.evaluate(test_dataset, [metric_auc, metric_recall])
    
    print("\n" + "="*50)
    print(f"🎯 Testing ROC-AUC: {test_score['roc_auc_score'] * 100:.2f}%")
    print(f"🚨 Testing Sensitivity (Recall): {test_score['recall_score'] * 100:.2f}%")
    print("="*50 + "\n")
    print("✅ V3 Ames Brain successfully trained and saved to the 'startnerve_v3_ames_gnn' folder!")

if __name__ == "__main__":
    build_v3_ames_engine()