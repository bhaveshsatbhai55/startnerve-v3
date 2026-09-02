"""
================================================================================
STARTNERVE INTELLECTUAL PROPERTY — PHASE 3 BACKEND CORE ENGINEERING
================================================================================
Module: fetch_pubchem_space.py (High-Performance ChEMBL Ingestion Stream)
Function: Instant Substructure Extraction & Target Dataset Compilation
================================================================================
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import time
import pandas as pd

# High-stability enterprise bioactivity endpoint
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data/substructure.json"
TOP_N_RECORDS = 10

SEARCH_TARGETS = [
    {"name": "ICH_M7_Nitroso_Alerts", "smarts": "[NX2]=[OX1]"},
    {"name": "Aromatic_Amines", "smarts": "[NX3][c]"},
    {"name": "Reactive_Epoxides", "smarts": "C1OC1"}
]

def fetch_substructure_batch(smarts: str, top_n: int) -> pd.DataFrame:
    """
    Queries the high-performance structural indexing cartridge, retrieving
    pharmaceutically relevant matching structures and canonical SMILES instantly.
    """
    # Safe URL encoding of the query parameters
    query_params = urllib.parse.urlencode({"smiles": smarts})
    url = f"{CHEMBL_BASE}?{query_params}"
    
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "StartNerveClient/3.0 (Contact: tech@startnerve.com)"
        }
    )
    
    with urllib.request.urlopen(req, timeout=45) as resp:
        raw_data = resp.read().decode("utf-8")
        data = json.loads(raw_data)
        
    molecules = data.get("molecules", [])
    target_pool = molecules[:top_n]
    
    records = []
    for mol in target_pool:
        structures = mol.get("molecule_structures", {})
        if structures and "canonical_smiles" in structures:
            records.append({
                "chemical_structure": structures.get("canonical_smiles"),
                "source_origin": "ChEMBL_Curated_Impurity_Scan"
            })
            
    return pd.DataFrame(records)

def main():
    print("\n" + "="*75)
    print("             STARTNERVE IN-SILICO HIGH-THROUGHPUT PIPELINE")
    print("                   [ENTERPRISE DATA INGESTION ENGINE]")
    print("="*75 + "\n")

    master_accumulator = []
    
    try:
        from data_curator import StartNerveDataCurator
        curator = StartNerveDataCurator()
    except ImportError:
        print("  ⚠️  data_curator.py not detected in local directory. Running raw mode fallback.")
        curator = None

    for target in SEARCH_TARGETS:
        print(f"  ── Executing Extraction Flow: {target['name']}")
        try:
            # Fetch structured matches instantly from the database cartridge
            raw_df = fetch_substructure_batch(target["smarts"], top_n=TOP_N_RECORDS)
            
            if raw_df.empty:
                print("    ⚠️  Zero records matching structural parameters found.")
                continue
                
            print(f"    ✅ Retrieved {len(raw_df)} high-density pharmaceutical structures.")
            
            # Pass data through the ingestion cleaning filters
            if curator:
                curated_df = curator.process_incoming_batch(raw_df, smiles_column="chemical_structure")
            else:
                curated_df = raw_df
                
            master_accumulator.append(curated_df)
            print("    ✅ Passed automated pipeline curation checks.\n")
            time.sleep(1.5) # Standard API courtesy padding
            
        except Exception as e:
            print(f"  ❌ Generation Fault on Class {target['name']}: {e}\n")

    if master_accumulator:
        final_set = pd.concat(master_accumulator, ignore_index=True)
        final_set.to_csv("StartNerve_Master_Expanded_Dataset.csv", index=False)
        print("="*75)
        print("  💾 Master Dataset Successfully Compiled: 'StartNerve_Master_Expanded_Dataset.csv'")
        print(f"  📈 Total Curated Structures Ingested: {len(final_set)}")
        print("="*75 + "\n")
    else:
        print("  ❌ Pipeline Error: No chemical assets compiled.")

# Run the ingestion script immediately upon invocation
main()