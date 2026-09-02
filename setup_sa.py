import os
import urllib.request

base_dir = os.path.dirname(os.path.abspath(__file__))

# Official RDKit SA Scorer source URLs
FPSCORES_URL = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz"
SASCORER_PY_URL = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py"

print("📥 Fetching RDKit SA Scorer assets...")

try:
    sascorer_path = os.path.join(base_dir, "sascorer.py")
    fpscores_path = os.path.join(base_dir, "fpscores.pkl.gz")

    if not os.path.exists(sascorer_path):
        urllib.request.urlretrieve(SASCORER_PY_URL, sascorer_path)
        print("  ✓ Downloaded sascorer.py")
    else:
        print("  ✓ sascorer.py already present")
        
    if not os.path.exists(fpscores_path):
        urllib.request.urlretrieve(FPSCORES_URL, fpscores_path)
        print("  ✓ Downloaded fpscores.pkl.gz")
    else:
        print("  ✓ fpscores.pkl.gz already present")
        
    print("\n✅ RDKit SA Scorer assets configured successfully!")
except Exception as e:
    print(f"\n❌ Failed to download SA assets: {e}")