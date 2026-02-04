import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, Draw
from stmol import showmol
import py3Dmol
import sys
import os

# --- IMPORT PDF GENERATOR FROM PARENT FOLDER ---
# This trick allows us to import 'utils.py' which is one folder up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import generate_pdf

# ---------------------------------------------------------
# 🎨 PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="StartNerve Bio-Engine Pro", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #4F46E5; font-weight: 800; margin-bottom: 0;}
    .sub-header {font-size: 1rem; color: #6B7280; margin-bottom: 2rem;}
    .metric-box {text-align: center; padding: 15px; background: #F3F4F6; border-radius: 8px; margin: 5px; color: #111827 !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .card-safe {background-color: #ECFDF5; border: 1px solid #10B981; padding: 10px; border-radius: 8px; color: #065F46; font-weight: bold; font-size: 0.9rem;}
    .card-danger {background-color: #FEF2F2; border: 1px solid #EF4444; padding: 10px; border-radius: 8px; color: #991B1B; font-weight: bold; font-size: 0.9rem;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 LOAD RESOURCES
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Load the model from the parent directory (StartNerve-2.0/tox21_model.pkl)
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tox21_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        return None

tox_model = load_model()

# FDA REFERENCE DRUGS (For Similarity Check)
FDA_DRUGS = {
    "Aspirin (Pain)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen (Pain)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Paracetamol (Fever)": "CC(=O)NC1=CC=C(O)C=C1",
    "Testosterone (Hormone)": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "Caffeine (Stimulant)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Amoxicillin (Antibiotic)": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
    "Benzene (Toxic Carcinogen)": "C1=CC=CC=C1"
}

TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
TASK_DESC = {
    'NR-AR': 'Androgen Receptor', 'NR-AhR': 'AhR (Metabolic)', 'NR-ER': 'Estrogen Receptor',
    'SR-p53': 'p53 (Cancer Risk)', 'SR-HSE': 'Heat Shock Stress', 'SR-MMP': 'Mitochondrial Energy'
}

# ---------------------------------------------------------
# 🧬 HELPER FUNCTIONS
# ---------------------------------------------------------
def make_3d_view(mol):
    """Generates an interactive 3D object"""
    mol = Chem.AddHs(mol) # Add hydrogens for 3D realism
    AllChem.EmbedMolecule(mol, AllChem.ETKDG()) # Calculate 3D coordinates
    mol_block = Chem.MolToMolBlock(mol)
    
    view = py3Dmol.view(width=400, height=300)
    view.addModel(mol_block, 'mol')
    view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}}) # Stick & Ball model
    view.zoomTo()
    view.setBackgroundColor('#FFFFFF')
    return view

def find_similarity(target_mol):
    """Compare input drug against FDA database"""
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
    best_match = "None"
    highest_sim = 0.0
    
    for name, smiles in FDA_DRUGS.items():
        ref_mol = Chem.MolFromSmiles(smiles)
        if ref_mol:
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(target_fp, ref_fp)
            if sim > highest_sim:
                highest_sim = sim
                best_match = name
    return best_match, highest_sim

def predict_single(smiles, model):
    """Run model on one molecule (Used for Batch Mode)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None, None
        fp = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))])
        preds = model.predict(fp)[0]
        return mol, preds
    except:
        return None, None

# ---------------------------------------------------------
# 🖥️ DASHBOARD UI
# ---------------------------------------------------------
st.markdown('<div class="main-header">StartNerve Bio-Engine <span style="font-size:1rem; vertical-align:middle; background:#4F46E5; color:white; padding:2px 8px; border-radius:10px;">PRO</span></div>', unsafe_allow_html=True)
st.write("Advanced In-Silico Toxicology & Market Intelligence Engine")

if tox_model is None:
    st.error("⚠️ CRITICAL: 'tox21_model.pkl' missing. Please ensure the model file is in the main directory.")
else:
    tab1, tab2 = st.tabs(["⚗️ Digital Lab (3D)", "📂 Batch Pipeline (CSV)"])

    # =========================================================
    # TAB 1: SINGLE MOLECULE LAB (3D + PDF)
    # =========================================================
    with tab1:
        col_input, col_3d, col_results = st.columns([1.5, 2, 2])
        
        with col_input:
            st.subheader("1. Design")
            smiles_input = st.text_area("Input SMILES:", value="CC(=O)OC1=CC=CC=C1C(=O)O", height=100)
            run_btn = st.button("Run Simulation 🧬", type="primary")
            
            if run_btn:
                mol = Chem.MolFromSmiles(smiles_input)
                if mol:
                    # ANALYSIS
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    match_name, match_score = find_similarity(mol)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Drug Intelligence")
                    st.markdown(f"""
                    <div class="metric-box">
                        <b>Molecular Weight:</b> {mw:.1f}<br>
                        <b>Lipophilicity (LogP):</b> {logp:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-box" style="background: #E0E7FF; color: #3730A3 !important;">
                        <b>Closest FDA Match:</b><br>
                        {match_name}<br>
                        (Similarity: {match_score*100:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Invalid Structure.")

        with col_3d:
            st.subheader("2. 3D Structure")
            if run_btn and mol:
                try:
                    view = make_3d_view(mol)
                    showmol(view, height=300, width=400)
                    st.caption("Interactive: Click & Drag to Rotate")
                except:
                    st.warning("3D Rendering requires a valid structure.")

        with col_results:
            st.subheader("3. Toxicity Profile")
            if run_btn and mol:
                fp = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))])
                preds = tox_model.predict(fp)[0]
                
                grid = st.columns(2)
                shown = 0
                for i, task in enumerate(TASKS):
                    if task in TASK_DESC:
                        col = grid[shown % 2]
                        risk = preds[i]
                        label = TASK_DESC[task]
                        with col:
                            if risk == 0:
                                st.markdown(f'<div class="card-safe">✅ {label}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="card-danger">⚠️ {label}</div>', unsafe_allow_html=True)
                            st.write("")
                        shown += 1
                
                # --- PDF DOWNLOAD SECTION ---
                st.markdown("---")
                st.subheader("4. Official Report")
                # Generate PDF in memory
                pdf_data = generate_pdf(smiles_input, mw, logp, preds, TASKS, TASK_DESC, match_name, match_score)
                
                st.download_button(
                    label="📥 Download Certificate of Analysis (PDF)",
                    data=pdf_data,
                    file_name="StartNerve_Analysis_Report.pdf",
                    mime="application/pdf",
                    type="secondary"
                )

    # =========================================================
    # TAB 2: BATCH PIPELINE (CSV)
    # =========================================================
    with tab2:
        st.write("### 📂 High-Throughput Screening (HTS)")
        st.write("Upload a CSV file containing a column named `smiles`. The AI will screen all candidates simultaneously.")
        
        uploaded_file = st.file_uploader("Upload CSV (Column 'smiles')", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            # Normalize column names to lowercase to be safe
            df.columns = [c.lower() for c in df.columns]
            
            if 'smiles' in df.columns:
                st.success(f"Pipeline Active: Processing {len(df)} candidates...")
                
                # Run predictions
                results = []
                progress_bar = st.progress(0)
                
                for index, row in df.iterrows():
                    mol_batch, preds_batch = predict_single(row['smiles'], tox_model)
                    
                    if preds_batch is not None:
                        # Convert 0/1 to Safe/Toxic for the report
                        row_data = {'SMILES': row['smiles']}
                        
                        # Add Basic Properties
                        try:
                            row_data['MW'] = Descriptors.MolWt(mol_batch)
                            row_data['LogP'] = Descriptors.MolLogP(mol_batch)
                        except:
                            pass

                        # Add Toxicity Flags
                        for i, task in enumerate(TASKS):
                            if task in TASK_DESC:
                                row_data[TASK_DESC[task]] = "RISK" if preds_batch[i] == 1 else "Safe"
                        
                        results.append(row_data)
                    
                    # Update progress
                    progress_bar.progress((index + 1) / len(df))
                
                # Show Result Table
                res_df = pd.DataFrame(results)
                st.dataframe(res_df)
                
                # DOWNLOAD BUTTON
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Pipeline Report (CSV)",
                    data=csv,
                    file_name="StartNerve_HTS_Results.csv",
                    mime="text/csv",
                    type="primary"
                )
            else:
                st.error("CSV Error: File must contain a column named 'smiles'.")