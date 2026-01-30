import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# ---------------------------------------------------------
# 🎨 PAGE CONFIGURATION (Modern UI)
# ---------------------------------------------------------
st.set_page_config(page_title="StartNerve Phase 2", page_icon="🧬", layout="centered")

# Custom CSS for the "Pro" look
st.markdown("""
    <style>
    .main-header {font-size: 3rem; color: #4F46E5; text-align: center; font-weight: 800;}
    .sub-header {font-size: 1.2rem; color: #6B7280; text-align: center; margin-bottom: 2rem;}
    .card {padding: 1.5rem; border-radius: 10px; background-color: #f3f4f6; margin-bottom: 1rem; border-left: 5px solid #4F46E5;}
    .safe {background-color: #D1FAE5; color: #065F46; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;}
    .toxic {background-color: #FEE2E2; color: #991B1B; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 LOAD THE BRAINS (Solubility + Toxicity)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    # Load Brain 1: Solubility
    try:
        with open('model.pkl', 'rb') as f:
            sol_model = pickle.load(f)
    except:
        sol_model = None
    
    # Load Brain 2: Toxicity Hunter (NEW)
    try:
        with open('toxicity_model.pkl', 'rb') as f:
            tox_model = pickle.load(f)
    except:
        tox_model = None
        
    return sol_model, tox_model

solubility_model, toxicity_model = load_models()

# ---------------------------------------------------------
# 🧪 HELPER FUNCTIONS (The Science)
# ---------------------------------------------------------
def generate_descriptors(smiles):
    """Generates the 4 inputs needed for Solubility (LogS)"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    ap = Descriptors.MolLogP(mol) / Descriptors.MolWt(mol) # Simplified aromatic proxy
    
    return np.array([[logp, mw, rb, ap]])

def generate_fingerprint(smiles):
    """Generates the 2048-bit vector for Toxicity Hunter"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    return np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))])

# ---------------------------------------------------------
# 🖥️ THE INTERFACE
# ---------------------------------------------------------
st.markdown('<div class="main-header">StartNerve AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Digital Drug Discovery • Solubility & Toxicity Screening</div>', unsafe_allow_html=True)

# Input Section
smiles_input = st.text_input("Enter Molecule SMILES Code", placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)")

if st.button("🚀 Analyze Molecule"):
    if not smiles_input:
        st.warning("Please enter a SMILES string first.")
    else:
        # Check if Models are Loaded
        if solubility_model is None or toxicity_model is None:
            st.error("⚠️ System Error: Brains not found. Please ensure 'model.pkl' and 'toxicity_model.pkl' are in the folder.")
        else:
            # 1. VISUALIZE
            mol = Chem.MolFromSmiles(smiles_input)
            if not mol:
                st.error("❌ Invalid Chemical Structure. Please check the SMILES code.")
            else:
                st.image(Chem.Draw.MolToImage(mol), caption="Molecular Structure", width=300)

                # 2. RUN BRAIN 1 (Solubility)
                desc_vec = generate_descriptors(smiles_input)
                sol_pred = solubility_model.predict(desc_vec)[0]
                
                # 3. RUN BRAIN 2 (Toxicity)
                fp_vec = generate_fingerprint(smiles_input)
                tox_prob = toxicity_model.predict_proba(fp_vec)[0] # Returns [Prob_Toxic, Prob_Safe]
                
                # Decision Logic
                # If Safe probability > 50%, it's safe.
                is_safe = tox_prob[1] > 0.5 
                safety_score = tox_prob[1] * 100

                # 4. REPORT CARD
                st.write("---")
                st.subheader("🔍 Analysis Report")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**💧 Solubility Prediction**")
                    st.info(f"LogS: {sol_pred:.2f}")
                    if sol_pred > -4:
                        st.write("✅ **Highly Soluble** (Good for pills)")
                    else:
                        st.write("⚠️ **Low Solubility** (Might need injection)")
                        
                with col2:
                    st.markdown("**💀 Toxicity Hunter**")
                    if is_safe:
                        st.markdown(f'<div class="safe">✅ LIKELY SAFE ({safety_score:.1f}%)</div>', unsafe_allow_html=True)
                        st.write("Passed in-silico toxicity screen.")
                    else:
                        st.markdown(f'<div class="toxic">⚠️ POTENTIAL TOXICITY</div>', unsafe_allow_html=True)
                        st.write("Structural alerts detected.")

                st.write("---")
                st.caption("Disclaimer: StartNerve AI is a research tool. Predictions are based on Tox21 & ClinTox datasets. Always verify in a wet lab.")

# Footer
st.markdown("<br><hr><center>Built by StartNerve Technologies • Pune, India</center>", unsafe_allow_html=True)