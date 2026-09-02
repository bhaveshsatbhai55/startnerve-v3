import pandas as pd
df_maps = pd.read_excel("assay_target_mappings_invitrodb_v4_3_AUG2024.xlsx", nrows=5)
print(df_maps.columns)
print(df_maps.head(2))