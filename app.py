import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw

# ---------------------------------------------------------
# 🎨 PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="StartNerve Bio-Engine", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #4F46E5; font-weight: 800; margin-bottom: 0;}
    .sub-header {font-size: 1rem; color: #6B7280; margin-bottom: 2rem;}
    .card-safe {background-color: #ECFDF5; border: 1px solid #10B981; padding: 15px; border-radius: 8px; color: #065F46; font-weight: bold;}
    .card-danger {background-color: #FEF2F2; border: 1px solid #EF4444; padding: 15px; border-radius: 8px; color: #991B1B; font-weight: bold;}
    .metric-box {text-align: center; padding: 10px; background: #F3F4F6; border-radius: 5px; margin: 5px;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 LOAD THE SUPER BRAIN (Tox21)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # We look for the NEW model first
        with open('tox21_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        return None

tox_model = load_model()

# The 12 Labels the model predicts (In the exact order of training)
TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
# Human Readable descriptions
TASK_DESC = {
    'NR-AR': 'Androgen Receptor (Hormones)',
    'NR-AhR': 'AhR (Metabolic Toxicity)',
    'NR-ER': 'Estrogen Receptor (Fertility)',
    'SR-p53': 'p53 (Cancer/Tumor Risk)',
    'SR-HSE': 'Heat Shock (Cell Stress)',
    'SR-MMP': 'Mitochondrial Energy Loss'
}

# ---------------------------------------------------------
# 🖥️ DASHBOARD UI
# ---------------------------------------------------------
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.markdown('<div class="main-header">StartNerve Bio-Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced In-Silico Toxicology & Solubility Screening</div>', unsafe_allow_html=True)

# INPUT SECTION
with st.sidebar:
    st.header("⚗️ Input Molecule")
    smiles_input = st.text_area("Paste SMILES Code:", value="CC(=O)OC1=CC=CC=C1C(=O)O", height=100)
    analyze_btn = st.button("Run Full Body Scan 🚀", type="primary")
    st.markdown("---")
    st.caption("Powered by Tox21 Government Data")

if analyze_btn:
    if not smiles_input:
        st.warning("Please enter a SMILES code.")
    elif tox_model is None:
        st.error("⚠️ CRITICAL: 'tox21_model.pkl' not found. Please run train_tox21.py locally and git push.")
    else:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            # 1. MOLECULE VISUALIZATION
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(Draw.MolToImage(mol), caption="Chemical Structure", use_column_width=True)
                
                # Basic Properties
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                st.markdown(f"""
                <div class="metric-box">
                    <b>Weight:</b> {mw:.1f} g/mol<br>
                    <b>Lipophilicity (LogP):</b> {logp:.2f}
                </div>
                """, unsafe_allow_html=True)

            # 2. RUN THE AI DIAGNOSIS
            with col2:
                st.subheader("🏥 Toxicology Report")
                
                # Prepare Math Vector
                fp = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))])
                
                # Get Predictions (The model returns 0 or 1 for all 12 tasks)
                preds = tox_model.predict(fp)[0]
                
                # Display Results Grid
                results_cols = st.columns(2)
                
                # Iterate through key health indicators
                shown_count = 0
                for i, task in enumerate(TASKS):
                    if task in TASK_DESC:
                        # Which column to put it in
                        target_col = results_cols[shown_count % 2]
                        
                        risk = preds[i] # 0 = Safe, 1 = Toxic
                        label = TASK_DESC[task]
                        
                        with target_col:
                            if risk == 0:
                                st.markdown(f'<div class="card-safe">✅ {label}<br><small>No Interaction Detected</small></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="card-danger">⚠️ {label}<br><small>POSSIBLE RISK DETECTED</small></div>', unsafe_allow_html=True)
                            st.write("") # Spacer
                        
                        shown_count += 1
                
        else:
            st.error("Invalid SMILES. Please check your chemical code.")