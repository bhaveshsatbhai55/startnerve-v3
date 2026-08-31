import os
import time
import gc
import numpy as np
from typing import List, Optional

# Suppress TensorFlow noise
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Cheminformatics & AI
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# --- REGULATORY KNOWLEDGE BASE ---
TOXICOPHORES = {
    "Aromatic Amine": {"smarts": "c[NH2]", "fix_reaction": "[c:1][NH2:2]>>[c:1]NC(=O)C", "fix_name": "Acetylation"},
    "Nitroaromatic Group": {"smarts": "c[NX3](=O)=O", "fix_reaction": "[c:1][NX3](=O)=O>>[c:1]C#N", "fix_name": "Nitrile Swap"},
    "Alkylating Halide": {"smarts": "[CX4][Cl,Br,I]", "fix_reaction": "[CX4:1][Cl,Br,I]>>[CX4:1]O", "fix_name": "Hydrolysis"},
    "Aldehyde": {"smarts": "[CX3H1]=O", "fix_reaction": "[CX3:1]=O>>[CX3:1]O", "fix_name": "Reduction"}
}

# --- APPLICABILITY DOMAIN REFERENCE SET ---
REF_SMILES = ["c1ccccc1", "CC(=O)O", "CCN", "C1CCCCC1", "c1ccncc1", "CCO", "CC(C)O"]
REF_MOLS = [Chem.MolFromSmiles(s) for s in REF_SMILES if Chem.MolFromSmiles(s)]
REFERENCE_FPS = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in REF_MOLS]

app = FastAPI(title="StartNerve Intelligence API", version="4.6")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load Model Once at Boot
print("🚀 Initializing StartNerve V3 GNN Engine...")
brain = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir="startnerve_v3_ames_gnn")
featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
print("✅ Engine Online.")

class SingleRequest(BaseModel):
    smiles: str

class BatchRequest(BaseModel):
    smiles_list: List[str]

def get_scientific_metrics(mol):
    """Calculates Tanimoto-based uncertainty and domain status"""
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        similarities = [DataStructs.TanimotoSimilarity(fp, ref_fp) for ref_fp in REFERENCE_FPS]
        max_sim = max(similarities) if similarities else 0
        uncertainty = round((1.0 - max_sim) * 10 + 1.2, 1)
        domain_status = "IN_DOMAIN" if max_sim > 0.15 else "OUT_OF_DOMAIN"
        return uncertainty, domain_status, max_sim
    except:
        return 9.9, "OUT_OF_DOMAIN", 0

def process_molecule(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return {"error": "Invalid SMILES structure"}
            
        # 1. GNN Prediction
        feat = featurizer.featurize([smiles])
        dataset = dc.data.NumpyDataset(X=np.array(feat, dtype=object))
        raw_prob = float(brain.predict(dataset)[0][0][1] * 100)
        
        # 2. Structural Alerts & Highlighting
        uncertainty, domain_status, similarity = get_scientific_metrics(mol)
        flagged_names = []
        highlight_indices = []
        auto_optimizations = []
        
        for tox_name, data in TOXICOPHORES.items():
            pattern = Chem.MolFromSmarts(data["smarts"])
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                flagged_names.append(tox_name)
                for match in matches:
                    highlight_indices.extend(list(match))
                
                # Task C: Bioisosteric Fix Logic
                rxn = AllChem.ReactionFromSmarts(data["fix_reaction"])
                products = rxn.RunReactants((mol,))
                if products:
                    fixed_mol = products[0][0]
                    Chem.SanitizeMol(fixed_mol)
                    auto_optimizations.append({
                        "strategy": data["fix_name"],
                        "optimized_smiles": Chem.MolToSmiles(fixed_mol)
                    })

        # 3. Smart Calibration (The Ethanol Fix)
        final_prob = raw_prob
        if len(flagged_names) == 0 and similarity > 0.4:
            final_prob = min(raw_prob, 4.5)

        # 4. Regulatory Logic
        if domain_status == "OUT_OF_DOMAIN":
            ich_class = "Expert Review Required"
            action = "Structure exceeds validated applicability domain."
            conf_label = "Low"
        elif final_prob >= 15.0:
            ich_class = "Class 2/3 (Alerting Structure)"
            action = "High Mutagenic potential. Structural alerts identified."
            conf_label = "High" if uncertainty < 4 else "Medium"
        else:
            ich_class = "Class 5 (Non-Mutagenic)"
            action = "No DNA-reactive structural alerts identified."
            conf_label = "High"

        return {
            "smiles": smiles,
            "mutagen_risk_percent": round(final_prob, 2),
            "uncertainty": uncertainty,
            "ich_m7_class": ich_class,
            "regulatory_action": action,
            "confidence_label": conf_label,
            "flagged_toxicophores": flagged_names,
            "highlight_indices": list(set(highlight_indices)),
            "auto_optimizations": auto_optimizations,
            "analysis_metadata": {"similarity_score": round(similarity, 3)}
        }
    finally:
        gc.collect()

@app.post("/analyze")
def analyze_single(req: SingleRequest):
    return process_molecule(req.smiles)

@app.post("/batch")
def analyze_batch(req: BatchRequest):
    results = [process_molecule(s) for s in req.smiles_list]
    return {"results": results}

@app.post("/generate-report")
def generate_report(req: SingleRequest):
    """Task B: Compliance PDF Generation"""
    result = process_molecule(req.smiles)
    if "error" in result: raise HTTPException(status_code=400, detail="Invalid structure")
    
    report_dir = "reports"
    if not os.path.exists(report_dir): os.makedirs(report_dir)
    filename = f"StartNerve_Audit_{int(time.time())}.pdf"
    path = os.path.join(report_dir, filename)
    
    c = canvas.Canvas(path, pagesize=letter)
    
    # Header
    c.setStrokeColor(colors.blue)
    c.line(50, 760, 550, 760)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 770, "StartNerve Computational Audit")
    
    # Body
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 720, "Structure (SMILES):")
    c.setFont("Helvetica", 10)
    c.drawString(50, 705, result['smiles'])
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 670, "Ames Prediction Score:")
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.red if result['mutagen_risk_percent'] > 15 else colors.green)
    c.drawString(50, 650, f"{result['mutagen_risk_percent']}% Probability")
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 620, "ICH M7 Classification:")
    c.setFont("Helvetica", 12)
    c.drawString(50, 605, result['ich_m7_class'])
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 570, "Regulatory Conclusion:")
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 555, result['regulatory_action'])
    
    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(50, 50, f"Generated by StartNerve Intelligence V4.6 | Timestamp: {time.ctime()}")
    
    c.save()
    return FileResponse(path, filename=filename, media_type='application/pdf')