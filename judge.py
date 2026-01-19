import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

print("\n---------------------------------------")
print("StartNerve Module 1: THE JUDGE")
print("---------------------------------------")

# 1. INPUT: Test Caffeine
molecule_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
mol = Chem.MolFromSmiles(molecule_smiles)

# 2. CALCULATION
mol_weight = Descriptors.MolWt(mol)
log_p = Descriptors.MolLogP(mol)
h_donors = Descriptors.NumHDonors(mol)
h_acceptors = Descriptors.NumHAcceptors(mol)

# 3. JUDGMENT
print(f"Analyzing Molecule: CAFFEINE")
print(f"1. Molecular Weight: {mol_weight:.2f} (Limit: <500)")
print(f"2. LogP (Solubility): {log_p:.2f}    (Limit: <5)")
print(f"3. H-Bond Donors:    {h_donors}       (Limit: <5)")
print(f"4. H-Bond Acceptors: {h_acceptors}       (Limit: <10)")

if mol_weight < 500 and log_p < 5 and h_donors < 5 and h_acceptors < 10:
    print("RESULT: PASSED. Drug-Like.")
else:
    print("RESULT: FAILED.")
print("---------------------------------------\n")