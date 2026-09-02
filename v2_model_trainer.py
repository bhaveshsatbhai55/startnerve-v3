import os
# THE MAGIC SWITCH: Forces modern TensorFlow to use the legacy Keras 2 engine
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pandas as pd
import deepchem as dc
import numpy as np

def train_v2_brain():
    print("\n🚀 Booting up StartNerve V2 Graph Neural Network...")
    
    # 1. Load the ChEMBL Data
    data_file = "StartNerve_V2_Aromatase_Training_Data.csv"
    if not os.path.exists(data_file):
        print(f"⚠️ Error: Cannot find {data_file}. Did you run the extractor?")
        return
        
    print("📁 Loading 4,300+ lab-verified Aromatase records...")
    df = pd.read_csv(data_file)
    
    # 2. Convert IC50 values into binary Risk Labels (1 = Toxic/Active, 0 = Safe/Inactive)
    df['toxicity_label'] = (df['standard_value'] < 10000).astype(int)
    
    smiles = df['canonical_smiles'].values
    labels = df['toxicity_label'].values
    
    # 3. Featurize: Converting text strings into 2D Mathematical Graphs
    print("🧬 Translating SMILES into 2D Molecular Graphs... (This takes a moment)")
    featurizer = dc.feat.ConvMolFeaturizer()
    X = featurizer.featurize(smiles)
    
    valid_indices = [i for i, x in enumerate(X) if x is not None]
    X = X[valid_indices]
    y = labels[valid_indices]
    
    print(f"✅ Successfully graphed {len(X)} molecules.")
    
    # 4. Create Dataset and Split
    dataset = dc.data.NumpyDataset(X, y)
    splitter = dc.splits.RandomSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=0.8)
    
    print(f"📊 Dataset Split: {len(train_dataset)} Training | {len(test_dataset)} Testing")
    
    # 5. Build the Graph Convolutional Network
    print("\n🧠 Initializing the Graph Convolutional Network (GCN)...")
    model = dc.models.GraphConvModel(
        n_tasks=1, 
        mode='classification', 
        dropout=0.2,
        model_dir="startnerve_v2_gnn"
    )
    
    # 6. Train the AI
    print("⚙️ Training the Neural Network... (Please wait)")
    model.fit(train_dataset, nb_epoch=15)
    
    # 7. Evaluate
    print("\n📈 Evaluating V2 Brain Accuracy...")
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    
    train_score = model.evaluate(train_dataset, [metric])
    test_score = model.evaluate(test_dataset, [metric])
    
    print(f"🎯 Training Accuracy (ROC-AUC): {train_score['roc_auc_score'] * 100:.2f}%")
    print(f"🎯 Real-World Testing Accuracy: {test_score['roc_auc_score'] * 100:.2f}%")
    
    print("\n✅ V2 GNN Brain successfully trained and saved to the 'startnerve_v2_gnn' folder!")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    train_v2_brain()