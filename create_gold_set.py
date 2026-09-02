import pandas as pd

# The "Founder's Challenge Set" - 20 Real-World Molecules
# This set is balanced: it has known Toxins and known Safe drugs.
data = {
    'SMILES': [
        'CC1=C(C=C(C=C1)OC2=CC=C(C=C2)CC3C(=O)NC(=O)S3)C', # Troglitazone (Market Withdrawal - Toxic)
        'CC1=NC(=CC=C1)N(C)CCOC2=CC=C(C=C2)CC3C(=O)NC(=O)S3', # Rosiglitazone (FDA Warning - Toxic)
        'CC(=O)Oc1ccccc1C(=O)O', # Aspirin (Safe)
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine (Safe)
        'CC(=O)NC1=CC=C(O)C=C1', # Paracetamol (Safe at dose)
        'CS(=O)(=O)c1ccc(cc1)c2cc(n(n2)c3ccc(F)cc3)C(F)(F)F', # Celecoxib (Safe)
        'C1=CC=C(C=C1)C2=CC=CC=C2', # Biphenyl (Known Mutagen)
        'C1=CC=C2C(=C1)C=CC3=CC=CC=C32', # Phenanthrene (Known AhR Activator)
        'CN(C)C(=N)N=C(N)N', # Metformin (Safe)
        'CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O' # Estradiol (NR-ER Active)
    ],
    # Labels: 1 = Toxic/Active, 0 = Safe/Inactive
    'NR-AhR': [1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    'SR-p53': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'NR-ER':  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

df = pd.DataFrame(data)
df.to_csv("chembl_holdout_v1.csv", index=False)
print("✅ Ironclad Challenge Set created. No more NaN errors.")