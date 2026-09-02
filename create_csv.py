import csv

# Sample chemical SMILES (ranging from simple solvents to complex drug molecules)
sample_molecules = [
    {"name": "Troglitazone (Antidiabetic)", "smiles": "CC1=C(C)C2=C(CCC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)O2)C(C)=C1C"},
    {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    {"name": "Benzene", "smiles": "C1=CC=CC=C1"},
    {"name": "Paracetamol", "smiles": "CC(=O)NC1=CC=C(O)C=C1"},
    {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
    {"name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
    {"name": "Ethanol", "smiles": "CCO"}
]

# Write to test_batch.csv
with open('test_batch.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Header row required by Titan V11 batch parser
    writer.writerow(['smiles', 'compound_name'])
    
    for item in sample_molecules:
        writer.writerow([item['smiles'], item['name']])

print("✅ 'test_batch.csv' successfully created in your project directory!")