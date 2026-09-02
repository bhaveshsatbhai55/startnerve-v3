"""
================================================================================
STARTNERVE INTELLIGENCE — INTERACTIVE DEMO DASHBOARD
================================================================================
Module: ui.py (Local Pitch Demo UI for Titan V11 Engine)
Run via: streamlit run ui.py
================================================================================
"""

import streamlit as st
import requests
import json
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import py3Dmol
from stmol import showmol

# Page Configuration
st.set_page_config(
    page_title="StartNerve Intelligence | Titan V11 Demo",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS Styling
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; background-color: #238636; color: white; font-weight: bold; border-radius: 6px; }
    .status-clear { color: #2ea043; font-weight: bold; font-size: 18px; }
    .status-intercept { color: #f85149; font-weight: bold; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric/500/hexagon.png", width=60)
st.sidebar.title("TITAN V11 CORE")
st.sidebar.caption("GATv2 + SchNet Parallel Inference Engine")
st.sidebar.markdown("---")

api_url = st.sidebar.text_input("Flask Backend API", "http://127.0.0.1:5000/api/audit")
pdf_api_url = st.sidebar.text_input("PDF Export API", "http://127.0.0.1:5000/api/export/pdf")

# Sample Presets for Live Pitching
st.sidebar.markdown("### 🧪 Quick Presets")
preset = st.sidebar.selectbox(
    "Load Real-World Cases:",
    ["Custom Input", "Troglitazone (Withdrawn Drug)", "Aspirin (Compliant)", "Testosterone Derivative"]
)

preset_smiles = {
    "Custom Input": "",
    "Troglitazone (Withdrawn Drug)": "CC1=C(C2=C(C(=C1O)C)OCCC2(C)C)CC3C(=O)NC(=O)S3",
    "Aspirin (Compliant)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Testosterone Derivative": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"
}

default_value = preset_smiles[preset] if preset != "Custom Input" else "CC1=C(C2=C(C(=C1O)C)OCCC2(C)C)CC3C(=O)NC(=O)S3"

st.title("⬡ STARTNERVE INTELLIGENCE")
st.caption("Predictive Cheminformatics & Regulatory Safety Audit Engine")
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Molecule Topology Input")
    smiles_input = st.text_area("SMILES Sequence Matrix:", value=default_value, height=100)
    
    run_btn = st.button("RUN TITAN V11 INFERENCE AUDIT")

    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input.strip())
        if mol:
            st.markdown("**2D Structural Formula:**")
            img = Draw.MolToImage(mol, size=(400, 200))
            st.image(img, use_column_width=True)
            
            st.markdown("**3D Conformer Geometry:**")
            try:
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
                AllChem.MMFFOptimizeMolecule(mol_3d)
                mblock = Chem.MolToMolBlock(mol_3d)
                
                xyzview = py3Dmol.view(width=400, height=250)
                xyzview.addModel(mblock, 'mol')
                xyzview.setStyle({'stick': {}})
                xyzview.zoomTo()
                showmol(xyzview, height=250, width=400)
            except Exception:
                st.info("3D Conformer rendering skipped.")

with col_right:
    st.subheader("2. Regulatory Risk Matrix")
    
    if run_btn and smiles_input:
        with st.spinner("Executing GATv2 Topology + SchNet 3D Spatial Forward Pass..."):
            try:
                response = requests.post(api_url, json={"smiles_input": smiles_input.strip()})
                if response.status_code == 200:
                    data = response.json()
                    res = data["results"][0]
                    
                    if "error" in res:
                        st.error(f"Execution Error: {res['error']}")
                    else:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Molecular Weight", f"{res['mw']} g/mol")
                        m2.metric("LogP (Lipophilicity)", res['logp'])
                        m3.metric("Max Risk Score", f"{res['risk_score']:.4f}")
                        
                        st.markdown("### Audit Verdict")
                        if res['risk_score'] > 0.4100:
                            st.markdown(f"<p class='status-intercept'>⚠️ {res['verdict']}</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p class='status-clear'>✅ {res['verdict']}</p>", unsafe_allow_html=True)
                        
                        st.markdown("### 12-Pathway Safety Profile")
                        tasks = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                                 "NR-PPAR-γ", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
                        scores = res["scores"]
                        
                        for i in range(0, 12, 2):
                            c1, c2 = st.columns(2)
                            p1_val = scores[i]
                            p2_val = scores[i+1]
                            
                            c1.progress(float(p1_val), text=f"{tasks[i]}: {p1_val:.4f}")
                            c2.progress(float(p2_val), text=f"{tasks[i+1]}: {p2_val:.4f}")
                        
                        # PDF Download Section
                        st.markdown("---")
                        if st.button("GENERATE PDF COMPLIANCE REPORT"):
                            pdf_resp = requests.post(pdf_api_url, json={"results": [res]})
                            if pdf_resp.status_code == 200:
                                pdf_data = pdf_resp.json()
                                st.success("PDF Generated Successfully!")
                                st.markdown(f"[📥 Download Official PDF Ledger]({pdf_data['download_url']})")
                            else:
                                st.error("PDF generation failed.")
                else:
                    st.error("Failed to connect to Flask API server.")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")