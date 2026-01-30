import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

print("🧪 STARTNERVE PHASE 2: Initializing Toxicity Hunter...")

# 1. GET THE DATA (Official ClinTox Dataset from MoleculeNet)
# Label 1 = FDA Approved (Safe), Label 0 = Failed/Toxic
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
print(f"⬇️  Downloading Ground Truth Data from: {url}")

try:
    data = pd.read_csv(url)
    print(f"✅ Data Loaded! Found {len(data)} molecules.")
except Exception as e:
    print("❌ Error downloading data. Check internet connection.")
    exit()

# 2. CLEAN THE DATA
# We use 'FDA_APPROVED' column (1 = Safe, 0 = Toxic)
df = data[['smiles', 'FDA_APPROVED']].dropna()

# 3. FEATURE ENGINEERING (Turn Molecules into Math)
print("🧮 Calculating Molecular Fingerprints...")

X = []
y = []
valid_count = 0

for index, row in df.iterrows():
    try:
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            # Create a list of 2048 numbers representing the molecule
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            X.append(list(fp))
            y.append(row['FDA_APPROVED'])
            valid_count += 1
    except:
        continue

print(f"✅ Converted {valid_count} molecules into vectors.")

# 4. TRAIN THE BRAIN (Random Forest)
print("🧠 Training Toxicity Safety Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 'balanced' class_weight is crucial because Safe drugs are rare in this list
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 5. TEST THE BRAIN
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("------------------------------------------------")
print(f"🎯 TOXICITY HUNTER ACCURACY: {accuracy * 100:.2f}%")
print("------------------------------------------------")
print(f"Safe Drugs Identified Correctly: {cm[1][1]}")
print(f"Toxic Drugs Identified Correctly: {cm[0][0]}")
print("------------------------------------------------")

# 6. SAVE THE BRAIN
filename = 'toxicity_model.pkl'
with open(filename, 'wb') as f:
    pickle.dump(model, f)

print(f"💾 SAVED: {filename}")
print("🚀 Ready to deploy to GitHub.")