"""
================================================================================
STARTNERVE INTELLECTUAL PROPERTY — PHASE 3 BACKEND CORE ENGINEERING
================================================================================
Module: data_curator.py
Function: High-Throughput Dataset Ingestion, Element Gating, and Scaffold Profiling
================================================================================
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

# Suppress annoying background RDKit warning messages during mass iteration runs
RDLogger.DisableLog('rdApp.*')

class StartNerveDataCurator:
    def __init__(self):
        # Strict element boundary set defined in our master blueprint
        self.VALID_ORGANIC_SET = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53} # C, H, N, O, P, S, F, Cl, Br, I
        
    def validate_and_profile_compound(self, smiles: str) -> dict:
        """
        Executes strict computational chemistry gating loops on an incoming structure string.
        """
        metrics = {
            "valid": False, 
            "reason": "Unknown", 
            "num_atoms": 0, 
            "scaffold_smiles": "", 
            "has_rings": False
        }
        
        if not isinstance(smiles, str) or not smiles.strip():
            metrics["reason"] = "Empty or non-string input"
            return metrics
            
        # Parse SMILES string into a formal RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            metrics["reason"] = "SMILES parsing failure / valence corruption"
            return metrics
            
        # 🛡️ Layer 1: Element Domain Gating Guardrail
        atomic_numbers = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
        if not atomic_numbers.issubset(self.VALID_ORGANIC_SET):
            metrics["reason"] = "Out-Of-Domain Element detected (Heavy Metal/Rare Earth)"
            return metrics
            
        # 🧬 Layer 2: Bemis-Murcko Scaffold Extraction
        try:
            metrics["num_atoms"] = mol.GetNumAtoms()
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            
            if scaffold_mol is not None and scaffold_mol.GetNumAtoms() > 0:
                metrics["scaffold_smiles"] = Chem.MolToSmiles(scaffold_mol)
                metrics["has_rings"] = True
            else:
                metrics["scaffold_smiles"] = "Acyclic Core"
                metrics["has_rings"] = False
                
            metrics["valid"] = True
            metrics["reason"] = "PASSED"
        except Exception as e:
            metrics["reason"] = f"Scaffold extraction exception: {str(e)}"
            
        return metrics

    def process_incoming_batch(self, input_dataframe: pd.DataFrame, smiles_column: str, label_column: str = None) -> pd.DataFrame:
        """
        Processes high-throughput molecule matrices, discarding redundant or out-of-domain targets.
        """
        print(f"\n[STARTNERVE DATA ENGINE] Initiating filtration array across {len(input_dataframe)} raw rows...")
        
        cleaned_records = []
        rejected_counts = {"malformed": 0, "metals": 0}
        unique_scaffolds = set()
        
        for idx, row in input_dataframe.iterrows():
            smiles = row[smiles_column]
            profile = self.validate_and_profile_compound(smiles)
            
            if not profile["valid"]:
                if "Element" in profile["reason"]:
                    rejected_counts["metals"] += 1
                else:
                    rejected_counts["malformed"] += 1
                continue
                
            # Accumulate records that clear the chemical safety thresholds
            record = {
                "SMILES": smiles,
                "Num_Atoms": profile["num_atoms"],
                "Bemis_Murcko_Scaffold": profile["scaffold_smiles"],
                "Is_Cyclic": profile["has_rings"]
            }
            
            # Map labels if they are passed in from training sets
            if label_column and label_column in row:
                record["Target_Label"] = row[label_column]
                
            cleaned_records.append(record)
            if profile["has_rings"]:
                unique_scaffolds.add(profile["scaffold_smiles"])
                
        output_df = pd.DataFrame(cleaned_records)
        
        # Output technical telemetry analysis directly to terminal window
        print(f"{'='*65}")
        print("                 DATASET CURATION METRIC TRAIL")
        print(f"{'='*65}")
        print(f"  📥 Total Raw Chemical Entries Ingested   : {len(input_dataframe)}")
        print(f"  ✅ Total Stabilized Records Approved    : {len(output_df)}")
        print(f"  ❌ Intercepted Malformed/Valence Drops  : {rejected_counts['malformed']}")
        print(f"  🛡️  Intercepted Out-of-Domain Heavy Metals: {rejected_counts['metals']}")
        print(f"  🧬 Unique Bemis-Murcko Topologies Charted: {len(unique_scaffolds)}")
        print(f"{'='*65}\n")
        
        return output_df

if __name__ == "__main__":
    # Simulated incoming uncurated chemical block (e.g., scraping output or raw database dump)
    # Includes standard drugs, a structural metal complex (cisplatin), and a broken chemical string
    raw_mock_data = pd.DataFrame({
        "chemical_structure": [
            "CC1=C(C(=C(C(=C1O)C)C)CC2CC(=O)NC(=O)S2)C", # Troglitazone (Valid)
            "CC(=O)OC1=CC=CC=C1C(=O)O",                # Aspirin (Valid)
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",            # Caffeine (Valid)
            "N.N.[Cl-].[Cl-].[Pt+2]",                  # Cisplatin (Heavy Metal - Should Be Blocked)
            "CC1=CC=C(C=C1)C(C)C2=CC=CC=C2INVALID!!",  # Broken SMILES (Should Be Blocked)
            "CC1==C==C1"                               # Invalid Carbon Valence (Should Be Blocked)
        ],
        "hazard_flag": [1, 0, 0, 1, 0, 0]
    })
    
    curator = StartNerveDataCurator()
    curated_matrix = curator.process_incoming_batch(raw_mock_data, smiles_column="chemical_structure", label_column="hazard_flag")
    
    # Save the physical output file cleanly into your workspace
    output_file = "StartNerve_Curated_Expansion_Set.csv"
    curated_matrix.to_csv(output_file, index=False)
    print(f"[🚀 SUCCESS] Cleaned data matrix written to storage location: '{output_file}'")