import streamlit as st

st.set_page_config(page_title="Methodology - StartNerve", page_icon="📘")

st.title("📘 Scientific Methodology")
st.markdown("Transparency is the core of scientific advancement. Here is how StartNerve works.")

st.header("1. Data Sources")
st.write("""
We utilize the **Tox21 (Toxicology in the 21st Century)** dataset, a collaboration between:
* **NIH** (National Institutes of Health)
* **EPA** (Environmental Protection Agency)
* **FDA** (Food and Drug Administration)

The dataset contains over **8,000 environmental chemicals and drugs** tested against 12 specific biological targets.
""")

st.header("2. The AI Architecture")
st.write("StartNerve employs a **Multi-Label Random Forest Classifier** with the following specifications:")
st.code("""
Model: Scikit-Learn RandomForestClassifier
Estimators (Trees): 50
Max Depth: 20
Features: 2048-bit Morgan Fingerprints (ECFP4)
Processing: RDKit (Cheminformatics)
""", language="text")

st.header("3. Validation Metrics")
st.success("""
**Current Model Performance:**
* **Accuracy:** ~94% (Binary Toxicity)
* **AUC Scores:** 60-75% across specific endpoints.
* **Speed:** <500ms per molecule.
""")

st.info("⚠️ **Disclaimer:** StartNerve is an In-Silico estimation tool. All predictions should be verified with In-Vitro (wet lab) testing.")