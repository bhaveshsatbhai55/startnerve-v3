from rdkit import Chem
from rdkit.Chem import AllChem

cisplatin = "N.N.Cl[Pt]Cl"
mol = Chem.MolFromSmiles(cisplatin)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
print(f"\nFingerprint bit sum for Cisplatin: {sum(fp)}")