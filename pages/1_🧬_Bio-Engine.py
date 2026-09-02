import os
# THE MAGIC SWITCH: Forces the web server to use the stable Keras 2 engine
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from stmol import showmol
import py3Dmol
import sys

# --- IMPORT PDF GENERATOR FROM PARENT FOLDER ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dossier_engine import generate_dossier
from molecule_maker import generate_2d_molecule

# ---------------------------------------------------------
# 🎨 PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="StartNerve Bio-Engine Hybrid", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #4F46E5; font-weight: 800; margin-bottom: 0;}
    .metric-box {text-align: center; padding: 15px; background: #F3F4F6; border-radius: 8px; margin: 5px; color: #111827 !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .card-safe {background-color: #ECFDF5; border: 1px solid #10B981; padding: 10px; border-radius: 8px; color: #065F46; font-weight: bold; font-size: 0.9rem;}
    .card-danger {background-color: #FEF2F2; border: 1px solid #EF4444; padding: 10px; border-radius: 8px; color: #991B1B; font-weight: bold; font-size: 0.9rem;}
    .gnn-badge {background: #000; color: #00FF41; padding: 4px 8px; border-radius: 4px; font-family: monospace; font-size: 0.8rem; margin-bottom:10px; display:inline-block;}
    .rf-badge {background: #374151; color: #60A5FA; padding: 4px 8px; border-radius: 4px; font-family: monospace; font-size: 0.8rem; margin-bottom:10px; display:inline-block;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 LOAD BOTH BRAINS (V1 & V2)
# ---------------------------------------------------------
@st.cache_resource
def load_v1_brain():
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tox21_model.pkl')
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except: return None

@st.cache_resource
def load_v2_brain():
    try:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'startnerve_v2_gnn')
        model = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir=model_dir)
        model.restore()
        return model
    except: return None

v1_model = load_v1_brain()
v2_model = load_v2_brain()

# FDA REFERENCE DRUGS
FDA_DRUGS = {
    "Aspirin (Pain)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen (Pain)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Fluoxetine (Antidepressant)": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
    "Ondansetron (Antiemetic)": "CC1=CC=C2C(=C1)NC3=C2C(=O)CC(C3)CN4CCNCC4"
}

TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
TASK_DESC = {'NR-AR': 'Androgen Receptor', 'NR-AhR': 'AhR (Metabolic)', 'NR-ER': 'Estrogen Receptor', 'SR-p53': 'p53 (Cancer Risk)', 'SR-HSE': 'Heat Shock Stress', 'SR-MMP': 'Mitochondrial Energy'}

# ---------------------------------------------------------
# 🧬 HELPER FUNCTIONS
# ---------------------------------------------------------
def make_3d_view(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    mol_block = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=400, height=300)
    view.addModel(mol_block, 'mol')
    view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
    view.zoomTo()
    view.setBackgroundColor('#FFFFFF')
    return view

def find_similarity(target_mol):
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
    best_match, highest_sim = "Unknown Framework", 0.0
    for name, smiles in FDA_DRUGS.items():
        ref_mol = Chem.MolFromSmiles(smiles)
        if ref_mol:
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(target_fp, ref_fp)
            if sim > highest_sim:
                highest_sim, best_match = sim, name
    # Fix the mislabeling bug: If it's not a strong match, don't label it as an analog!
    if highest_sim < 0.4:
        return "Novel API Structure", highest_sim
    return best_match, highest_sim

# ---------------------------------------------------------
# 🖥️ DASHBOARD UI
# ---------------------------------------------------------
st.markdown('<div class="main-header">StartNerve Bio-Engine <span style="font-size:1rem; vertical-align:middle; background:#4F46E5; color:white; padding:2px 8px; border-radius:10px;">HYBRID AI</span></div>', unsafe_allow_html=True)
st.write("Comprehensive In-Silico Toxicology Pipeline: Combining Deep Learning & Broad Spectrum Screening.")

tab1, tab2, tab3 = st.tabs(["⚗️ Digital Lab (Audit)", "📖 How It Works (For Clients)", "📂 Batch Pipeline"])

# =========================================================
# TAB 1: DIGITAL LAB
# =========================================================
with tab1:
    st.sidebar.header("🧪 Demo Chemicals (Quick Load)")
    st.sidebar.write("Select a molecule to demonstrate the workflow to clients:")
    
    example_molecules = {
        "Custom Input (Paste your own)": "",
        "Ondansetron (Safe Target)": "CC1=CC=C2C(=C1)NC3=C2C(=O)CC(C3)CN4CCNCC4",
        "Aspirin (Safe Target)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Letrozole (Toxic/Aromatase Attacker)": "N#Cc1ccc(cc1)C(c2ccc(C#N)cc2)n3cncn3",
        "DDT (Toxic Pesticide)": "ClC1=CC=C(C(C2=CC=C(Cl)C=C2)C(Cl)(Cl)Cl)C=C1"
    }
    
    selected_example = st.sidebar.selectbox("Choose a Compound:", list(example_molecules.keys()))
    default_val = example_molecules[selected_example]

    col_input, col_3d, col_results = st.columns([1.5, 2, 2])
    
    with col_input:
        st.subheader("1. Design")
        smiles_input = st.text_area("Input SMILES Code:", value=default_val, height=100)
        run_btn = st.button("Run Full Hybrid Audit 🧬", type="primary")
        
        if run_btn:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                match_name, match_score = find_similarity(mol)
                
                st.markdown("---")
                st.markdown("### 📊 Chemical Intelligence")
                st.markdown(f"""
                <div class="metric-box">
                    <b>Molecular Weight:</b> {mw:.1f}<br>
                    <b>Lipophilicity (LogP):</b> {logp:.2f}
                </div>
                <div class="metric-box" style="background: #E0E7FF; color: #3730A3 !important;">
                    <b>Structural Analysis:</b><br>{match_name} ({match_score*100:.1f}% Similarity)
                </div>
                """, unsafe_allow_html=True)

    with col_3d:
        st.subheader("2. 3D Structure")
        if run_btn and mol:
            view = make_3d_view(mol)
            showmol(view, height=300, width=400)

    with col_results:
        st.subheader("3. StartNerve Toxicology Profile")
        if run_btn and mol and v1_model and v2_model:
            tox_results_dict = {}
            total_fails = 0
            
            # === V2 GRAPH NEURAL NETWORK (DEEPCHEM) ===
            st.markdown('<span class="gnn-badge">V2 GNN ENGINE (Targeted)</span>', unsafe_allow_html=True)
            featurizer = dc.feat.ConvMolFeaturizer()
            X = featurizer.featurize([smiles_input])
            
            if len(X) > 0 and X[0] is not None:
                dataset = dc.data.NumpyDataset(X)
                preds_v2 = v2_model.predict(dataset) 
                toxic_prob = preds_v2[0][0][1] * 100 
                
                if toxic_prob > 50:
                    st.markdown(f'<div class="card-danger">⚠️ Aromatase Endocrine Risk: {toxic_prob:.1f}%</div>', unsafe_allow_html=True)
                    # FIX: Inject exact percentage into the PDF dictionary
                    tox_results_dict["Aromatase Endocrine Disruption (GNN)"] = f"FAIL ({toxic_prob:.1f}% Binding Probability)"
                    total_fails += 2
                else:
                    st.markdown(f'<div class="card-safe">✅ Aromatase Endocrine Risk: {toxic_prob:.1f}%</div>', unsafe_allow_html=True)
                    # FIX: Inject exact percentage into the PDF dictionary
                    tox_results_dict["Aromatase Endocrine Disruption (GNN)"] = f"PASS ({toxic_prob:.1f}% Binding Probability)"

            # === V1 RANDOM FOREST (BROAD SPECTRUM) ===
            st.markdown('<span class="rf-badge">V1 RANDOM FOREST (Broad Spectrum)</span>', unsafe_allow_html=True)
            fp = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))])
            preds_v1 = v1_model.predict(fp)[0]
            
            for i, task in enumerate(TASKS):
                if task in TASK_DESC:
                    label = TASK_DESC[task]
                    if preds_v1[i] == 1:
                        st.markdown(f'<div class="card-danger">⚠️ {label}</div>', unsafe_allow_html=True)
                        tox_results_dict[f"{label} (V1)"] = "FAIL (HIGH RISK)"
                        total_fails += 1
                    else:
                        st.markdown(f'<div class="card-safe">✅ {label}</div>', unsafe_allow_html=True)
                        tox_results_dict[f"{label} (V1)"] = "PASS"

            # === PDF DOSSIER GENERATION ===
            st.markdown("---")
            st.subheader("4. Generate Official Dossier")
            
            # Dynamic Risk Score Calculation
            risk_score = min(12 + (total_fails * 28), 98)
            target_dossier_name = f"{match_name} Analog" if "Novel" not in match_name else "Custom/Novel API Framework"

            generate_2d_molecule(smiles_input, "molecule.png")
            pdf_filename = "StartNerve_Hybrid_Regulatory_Audit.pdf"
            generate_dossier(target_dossier_name, risk_score, tox_results_dict, pdf_filename)
            
            if os.path.exists(pdf_filename):
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download ₹1.5L Pre-Manufacturing Dossier",
                        data=pdf_file,
                        file_name=f"StartNerve_Audit.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

# =========================================================
# TAB 2: HOW IT WORKS (FOR PITCHING)
# =========================================================
with tab2:
    st.header("📖 The StartNerve Workflow")
    st.write("Use this page to explain the value of In-Silico screening to clients and investors.")
    
    st.markdown("""
    ### The Problem with Traditional "Wet Labs"
    Normally, when a factory wants to synthesize a new drug or alter a chemical route, they have to physically mix the chemicals in a lab, wait weeks, and test it on cells or animals. If it fails FDA toxicity standards (like ICH M7), they lose months of time and millions of rupees.
    
    ### The StartNerve "Dry Lab" Solution
    We digitize the entire process. 
    1. **Data Ingestion:** We take the exact chemical "recipe" (the SMILES code).
    2. **Graph Translation:** Our AI converts that code into a 2D mathematical map of atoms and bonds.
    3. **Dual-Engine Screening:** * *Engine 1 (Broad Spectrum):* Checks 12 different human stress-response pathways instantly.
        * *Engine 2 (Deep Learning):* Uses a Graph Neural Network trained on thousands of European lab results to predict deep endocrine disruption.
    4. **The Audit:** We generate a compliant PDF dossier, allowing the factory to fix chemical impurities *before* they ever turn on the machines.V2 GRAPH NEURAL NETWORK
    """)

# =========================================================
# TAB 3: BATCH PIPELINE
# =========================================================
with tab3:
    st.write("### 📂 High-Throughput Screening")
    st.write("Upload a CSV file of SMILES strings to process hundreds of target APIs simultaneously.")
    # (Batch logic can be fully integrated here later)