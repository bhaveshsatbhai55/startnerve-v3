import pandas as pd
import numpy as np
import requests
import os
from rdkit import Chem
from rdkit.Chem import SanitizeFlags

# ── Strategy 1: Direct CSV download (bypasses DeepChem loader entirely) ──────

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

def fetch_tox21_raw(cache_path="tox21_raw.csv.gz"):
    """Download Tox21 CSV directly — zero RDKit involvement at load time."""
    if not os.path.exists(cache_path):
        print("Downloading Tox21 CSV...")
        r = requests.get(TOX21_URL, stream=True)
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    return pd.read_csv(cache_path)


# ── Strategy 2: Partial sanitization (keeps exotic-valence molecules) ────────

def safe_mol_from_smiles(smiles: str):
    """
    Parse SMILES with sanitization flags that skip valence checks.
    Returns a valid Mol object even for Al with 6 bonds, or None on
    truly unparseable strings.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        # Run everything EXCEPT the valence check
        san_flags = (
            SanitizeFlags.SANITIZE_ALL
            ^ SanitizeFlags.SANITIZE_PROPERTIES   # skips valence enforcement
        )
        Chem.SanitizeMol(mol, san_flags)
        return mol
    except Exception:
        return None


# ── Strategy 3: Robust label extraction from ragged rows ─────────────────────

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

def build_master_csv(output_path="ntp_tox_2026.csv"):
    df_raw = fetch_tox21_raw()

    # Normalise column names (source CSV uses exact task names as columns)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    records = []
    skipped_parse = 0
    skipped_no_labels = 0

    for _, row in df_raw.iterrows():
        smiles = str(row.get("smiles", "")).strip()
        if not smiles:
            continue

        mol = safe_mol_from_smiles(smiles)

        # Collect per-task labels; NaN → -1 sentinel so dtype stays int8
        task_labels = {}
        for task in TOX21_TASKS:
            raw_val = row.get(task, np.nan)
            try:
                val = float(raw_val)
                task_labels[task] = 0 if np.isnan(val) else int(val)
            except (ValueError, TypeError):
                task_labels[task] = -1   # missing / malformed

        # Derive a single aggregate label (1 if active in ANY assay)
        defined = [v for v in task_labels.values() if v >= 0]
        if not defined:
            skipped_no_labels += 1
            continue

        agg_activity = int(any(v == 1 for v in defined))

        records.append({
            "smiles":        smiles,
            "rdkit_valid":   mol is not None,   # audit flag — don't silently drop
            "activity":      agg_activity,
            **task_labels,                        # one column per assay
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_path, index=False)

    print(f"Saved {len(df_out):,} rows  →  {output_path}")
    print(f"  RDKit-valid molecules : {df_out['rdkit_valid'].sum():,}")
    print(f"  Exotic-valence kept   : {(~df_out['rdkit_valid']).sum():,}")
    print(f"  Skipped (no labels)   : {skipped_no_labels}")
    print(f"  Active (any assay)    : {df_out['activity'].sum():,}")
    return df_out


if __name__ == "__main__":
    df = build_master_csv()
    print(df.head())