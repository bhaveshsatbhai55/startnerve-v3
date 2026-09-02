import os
import pandas as pd

file1 = "cytotox_invitrodb_v4_3_AUG2024.xlsx"
file2 = "assay_target_mappings_invitrodb_v4_3_AUG2024.xlsx"

print("📊 Reading Cytotoxicity Sheet Headers...")
try:
    df1 = pd.read_excel(file1, nrows=3, engine='openpyxl')
    print("Columns found:", df1.columns.tolist())
    print("\nFirst row sample values:\n", df1.iloc[0].to_dict())
except Exception as e:
    print(f"❌ Error reading {file1}: {str(e)}")

print("\n🎯 Reading Assay Target Mappings Headers...")
try:
    df2 = pd.read_excel(file2, nrows=3, engine='openpyxl')
    print("Columns found:", df2.columns.tolist())
except Exception as e:
    print(f"❌ Error reading {file2}: {str(e)}")
    