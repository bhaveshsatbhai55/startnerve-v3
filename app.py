import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw # <--- ADDED 'Draw' HERE

# ---------------------------------------------------------
# 🎨 PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="StartNerve Phase 2", page_icon="🧬", layout="centered")

st.markdown("""
    <style>
    .main-header {font-size: 3rem; color: #4F46E5; text-align: center; font-weight: 800;}
    .sub-header {font-size: 1.2rem; color: #6B7280; text-align: center; margin-bottom: 2rem;}
    .safe {background-color: #D1FAE5; color: #065F46; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;}
    .toxic {background-color: #FEE2E2; color: #991B1B; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 LOAD BRAINS
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as f:
            m1 = pickle.load(f)
    except:
        m1 = None
    
    try:
        with open('toxicity_model.pkl', 'rb') as f:
            m2 = pickle.load(f)
    except:
        m2 = None
    return m1, m2

solubility_model, toxicity_model = load_models()

# ---------------------------------------------------------
# 🖥️ UI & LOGIC
# ---------------------------------------------------------
st.markdown('<div class="main-header">StartNerve AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Digital Drug Discovery • Solubility & Toxicity Screening</div>', unsafe_allow_html=True)

# Check if brains exist
if solubility_model is None or toxicity_model is None:
    st.error("⚠️ SYSTEM ERROR: Brain files missing. Please run build_brains.py locally and git push -f.")
else:
    smiles_input = st.text_input("Enter Molecule SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")

    if st.button("Analyze Molecule"):
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            # FIX: Now Draw is imported, so this works!
            st.image(Draw.MolToImage(mol), width=300, caption="Chemical Structure")
            
            # Prediction 1: Solubility
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            rb = Descriptors.NumRotatableBonds(mol)
            ap = logp / mw if mw > 0 else 0
            sol_pred = solubility_model.predict([[logp, mw, rb, ap]])[0]
            
            # Prediction 2: Toxicity
            fp = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))])
            tox_prob = toxicity_model.predict_proba(fp)[0][1] # Probability of being safe
            
            # Display
            c1, c2 = st.columns(2)
            c1.metric("Solubility (LogS)", f"{sol_pred:.2f}")
            
            with c2:
                if tox_prob > 0.5:
                    st.markdown(f'<div class="safe">✅ LIKELY SAFE ({tox_prob*100:.1f}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="toxic">⚠️ TOXICITY ALERT</div>', unsafe_allow_html=True)
        else:
            st.error("Invalid SMILES code.")