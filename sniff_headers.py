import pandas as pd
df_cyto = pd.read_excel("cytotox_invitrodb_v4_3_AUG2024.xlsx", nrows=5)
print(df_cyto.columns)
print(df_cyto.head(2))