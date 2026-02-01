import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("🏥 STARTNERVE PHASE 3: Initializing 'Full Body Scan' (Tox21 Lite)...")

# 1. GET THE DATA
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
print(f"⬇️  Downloading 12-Label Government Data from: {url}")

try:
    data = pd.read_csv(url)
    print(f"✅ Data Loaded! Found {len(data)} molecules.")
except:
    print("❌ Error downloading data.")
    exit()

# 2. DEFINING THE 12 LABELS
tasks = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
task_names = {
    'NR-AR': 'Androgen Receptor',
    'NR-AhR': 'AhR (Toxin Metabolism)',
    'NR-ER': 'Estrogen Receptor',
    'SR-p53': 'p53 (Cancer Risk)',
    'SR-HSE': 'Heat Shock (Stress)',
    'SR-MMP': 'Mitochondrial Energy'
}

# --- THE FIX: FILLNA -> FORCE INTEGER ---
print("🧹 Cleaning missing values and converting to Integers...")
# 1. Fill holes with 0
# 2. Force conversion to Integer (0, not 0.0) so the Classifier is happy
data[tasks] = data[tasks].fillna(0).astype(int)

# 3. CONVERT MOLECULES TO MATH
print("🧮 Processing compounds...")
X = []
y = [] 

for index, row in data.iterrows():
    try:
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            X.append(list(fp))
            # .tolist() ensures it stays clean
            y.append(row[tasks].values.tolist())
    except:
        continue

X = np.array(X)
y = np.array(y) # Now this is an array of Integers

print(f"✅ Vectors created. Training 12 separate AI Doctors on {len(X)} compounds...")

# 4. TRAIN THE MULTI-LABEL BRAIN (LITE VERSION)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# OPTIMIZED FOR SIZE (<100MB)
# n_estimators=50 (instead of 100) cuts size in half
# max_depth=20 prevents the trees from getting too complex and heavy
print("🧠 Training 'Lite' Model (Optimized for Cloud)...")
model = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# 5. TEST THE DOCTORS
print("------------------------------------------------")
print("🎯 ACCURACY REPORT (AUC Scores):")
predictions = model.predict(X_test)

# Calculate scores safely
for i, task in enumerate(tasks):
    if task in task_names:
        try:
            if len(np.unique(y_test[:, i])) > 1:
                score = roc_auc_score(y_test[:, i], predictions[:, i])
                print(f"  - {task_names[task]}: {score*100:.1f}%")
        except:
            pass
print("------------------------------------------------")

# 6. SAVE THE SUPER BRAIN
with open('tox21_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("💾 SAVED: 'tox21_model.pkl' (Size Optimized)")
print("🚀 Ready to upgrade App to v4.0")