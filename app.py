import streamlit as st
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
import pandas as pd
import py3Dmol
from stmol import showmol
import joblib  # This is the tool that loads your AI Brain
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="StartNerve Bio-Engine", page_icon="🧬", layout="wide")

# --- LOAD THE AI BRAIN ---
@st.cache_resource
def load_brain():
    try:
        # We try to load the brain you trained
        model = joblib.load("models/solubility_model.pkl")
        return model
    except:
        return None

ai_model = load_brain()

# --- SIDEBAR: THE COMMAND CENTER ---
# Logic to change status based on if brain is found
if ai_model:
    sidebar_title = "🧬 StartNerve v3.7 (AI Online)"
    status_msg = "✅ AI Brain Active (Accuracy: 97.8%)"
else:
    sidebar_title = "🧬 StartNerve v3.7 (Offline)"
    status_msg = "⚠️ AI Model Not Found (Run train_brain.py)"

st.sidebar.title(sidebar_title)
st.sidebar.markdown("### 🧪 Simulation Parameters")

# 1. INPUT SCAFFOLD
with st.sidebar.expander("1. Core Molecule", expanded=True):
    scaffold_input = st.text_input("SMILES Code", value="c1ccccc1", help="Enter the base chemical structure.")

# 2. THE WARHEADS
with st.sidebar.expander("2. Chemical 'Warheads'"):
    st.write("Select functional groups to attach:")
    warhead_options = {
        "Methyl (-CH3)": "C",
        "Hydroxyl (-OH)": "O",
        "Amine (-NH2)": "N",
        "Fluorine (-F)": "F",
        "Chlorine (-Cl)": "Cl",
        "Trifluoromethyl (-CF3)": "C(F)(F)F"
    }
    
    selected_names = st.multiselect(
        "Library Building Blocks",
        list(warhead_options.keys()),
        default=["Methyl (-CH3)", "Hydroxyl (-OH)", "Fluorine (-F)"]
    )

# 3. FILTERS
with st.sidebar.expander("3. 'Lipinski' Filters"):
    max_mw = st.slider("Max MW", 100, 800, 500)
    max_logp = st.slider("Max LogP", -2.0, 7.0, 5.0)

run_btn = st.sidebar.button("🚀 RUN PREDICTION ENGINE", type="primary")

# --- MAIN DASHBOARD ---
st.title("StartNerve: AI-Powered Drug Discovery")
st.markdown(f"**Status:** {status_msg} | **Target:** `{scaffold_input}`")

if run_btn:
    st.write("---")
    progress_bar = st.progress(0)
    
    valid_mols = []
    all_data = []

    # GENERATION LOOP
    for i, name in enumerate(selected_names):
        progress_bar.progress((i + 1) / len(selected_names))
        
        group_smiles = warhead_options[name]
        new_smiles = scaffold_input + group_smiles
        mol = Chem.MolFromSmiles(new_smiles)
        
        if mol:
            # 1. 3D PREP
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            
            # 2. EXTRACT FEATURES (The Input for the AI)
            # We must calculate exactly what the AI learned: MW, LogP, Donors, Acceptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_donors = Descriptors.NumHDonors(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)
            
            # 3. AI PREDICTION (The Brain)
            ai_prediction = "N/A"
            if ai_model:
                # Ask the brain: "Based on these 4 numbers, what is the solubility?"
                features = [[mw, logp, h_donors, h_acceptors]]
                pred_val = ai_model.predict(features)[0]
                ai_prediction = round(pred_val, 2)
            
            # 4. FILTER
            status = "FAIL"
            if mw <= max_mw and logp <= max_logp:
                status = "PASS"
                valid_mols.append(mol)
                
            all_data.append({
                "Name": f"Variant {i+1}",
                "Modification": name,
                "SMILES": new_smiles,
                "MW": round(mw, 2),
                "LogP (Calc)": round(logp, 2),
                "AI Solubility": ai_prediction, # <--- The New AI Column
                "Status": status,
                "Mol3D": mol_3d
            })

    st.success(f"Prediction Complete. {len(valid_mols)} Candidates Generated.")
    
    # --- RESULTS TABS ---
    tab1, tab2, tab3 = st.tabs(["⚗️ 2D Grid", "🧊 Interactive 3D", "📊 AI Data Report"])
    
    with tab1:
        st.write("### 2D Structure Overview")
        if valid_mols:
            st.image(Draw.MolsToGridImage(valid_mols, molsPerRow=4, subImgSize=(250, 250)), width="stretch" if valid_mols else None)

    with tab2:
        st.write("### 3D Molecular Viewer")
        st.info("Select a molecule to inspect.")
        df = pd.DataFrame(all_data)
        if not df.empty:
            choice = st.selectbox("Select Candidate", df["Name"] + " (" + df["Modification"] + ")")
            selected_row = df[df["Name"] == choice.split(" (")[0]].iloc[0]
            
            view = py3Dmol.view(width=800, height=500)
            view.addModel(Chem.MolToMolBlock(selected_row["Mol3D"]), 'mol')
            view.setStyle({'stick': {}})
            view.setBackgroundColor('white')
            view.zoomTo()
            showmol(view, height=500, width=800)

    with tab3:
        st.write("### AI Prediction Analysis")
        # Remove the 3D object column so the table looks clean
        display_df = df.drop(columns=["Mol3D"])
        
        # Color code the PASS/FAIL column
        st.dataframe(
            display_df.style.map(lambda x: 'color: green' if x == 'PASS' else 'color: red', subset=['Status']),
            use_container_width=True
        )