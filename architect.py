import os
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

print("\n---------------------------------------")
print("StartNerve Module 2: THE ARCHITECT")
print("---------------------------------------")

# Ensure Results folder exists
if not os.path.exists("Results"):
    os.makedirs("Results")

# 1. GENERATE
functional_groups = ["C", "O", "N", "C(=O)O", "F"]
generated_mols = []
names = []

print(f"Generating variants based on Benzene...")

for group in functional_groups:
    new_smiles = "c1ccccc1" + group 
    mol = Chem.MolFromSmiles(new_smiles)
    
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        if mw < 500 and logp < 5:
            generated_mols.append(mol)
            names.append(f"+{group}")
            print(f"  > CREATED: Benzene-{group} | PASSED")

# 2. SAVE to Results folder
output_file = "Results/StartNerve_Library.png"
img = Draw.MolsToGridImage(generated_mols, molsPerRow=3, subImgSize=(300,300), legends=names)
img.save(output_file)

print(f"\nSUCCESS: Library saved to {output_file}")
print("---------------------------------------\n")