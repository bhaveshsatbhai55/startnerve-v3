from rdkit import Chem
from rdkit.Chem import Draw

def generate_2d_molecule(smiles, filename="molecule.png"):
    # Convert SMILES string to an RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        # Draw the molecule and save it with a white background
        Draw.MolToFile(mol, filename, size=(400, 400), imageType="png")
        print(f"✅ Success! 2D Molecule saved as {filename}")
    else:
        print("⚠️ Error: Invalid SMILES string. Could not generate image.")

# ==========================================
# TEST THE VISUALIZER
# ==========================================
if __name__ == "__main__":
    # This is the SMILES string for Aspirin as a test
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    generate_2d_molecule(test_smiles)