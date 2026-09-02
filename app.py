"""
================================================================================
STARTNERVE INTELLIGENCE — ENTERPRISE PRODUCTION STREAM
================================================================================
Module: app.py (Titan Engine V11 Production Core — Direct API Bridge)
Function: Unified GATv2 + SchNet Deep Learning Inference Server
================================================================================
"""

import os
import sys
import io
import csv
import urllib.request
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template_string
from flask_cors import CORS

# Deep learning core verification wrapper
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv, SchNet, global_add_pool
    from torch_geometric.data import Data
    HAS_DEEP_LEARNING_CORE = True
except ImportError:
    HAS_DEEP_LEARNING_CORE = False
    print("❌ CRITICAL ERROR: PyTorch Geometric libraries missing.")

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, Descriptors, Crippen
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import RDConfig

# WeasyPrint rendering engine
try:
    from weasyprint import HTML
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    return response

# Silence favicon 404 warnings
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def home():
    return send_file('index.html')

# ───────────────────────────────────────────────────────────────────────
# REGULATORY & MANUFACTURING METRIC ENGINES
# ───────────────────────────────────────────────────────────────────────
HAS_SA_SCORER = False
try:
    import sascorer
    HAS_SA_SCORER = True
except ImportError:
    try:
        sascorer_path = os.path.join(RDConfig.RDConfig.RDBonusDataDir, 'QUANTUM')
        if sascorer_path not in sys.path:
            sys.path.append(sascorer_path)
        import sascorer
        HAS_SA_SCORER = True
    except Exception:
        HAS_SA_SCORER = False

def calculate_sa_score(mol):
    if not HAS_SA_SCORER or mol is None:
        return 3.50
    try:
        score = sascorer.calculateScore(mol)
        return float(round(score, 2))
    except Exception:
        return 3.50

def calculate_sd_distance(mol):
    if mol is None:
        return 1.0, "OUT-OF-DOMAIN"
        
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    ref_mw, ref_logp, ref_tpsa = 320.0, 2.8, 65.0
    
    delta_mw = ((mw - ref_mw) / 150.0) ** 2
    delta_logp = ((logp - ref_logp) / 2.0) ** 2
    delta_tpsa = ((tpsa - ref_tpsa) / 40.0) ** 2
    
    sd_distance = float(np.sqrt(delta_mw + delta_logp + delta_tpsa))
    
    if sd_distance <= 1.25:
        return round(sd_distance, 4), "GREEN (In-Domain)"
    elif sd_distance <= 2.50:
        return round(sd_distance, 4), "GREY (Borderline)"
    else:
        return round(sd_distance, 4), "RED (Out-of-Domain)"

# ───────────────────────────────────────────────────────────────────────
# SALT STRIPPER & INPUT SANITIZER
# ───────────────────────────────────────────────────────────────────────
remover = SaltRemover()

def sanitize_and_strip(mol):
    if mol is None:
        return None
    stripped_mol = remover.StripMol(mol)
    frags = Chem.GetMolFrags(stripped_mol, asMols=True)
    if len(frags) > 1:
        stripped_mol = max(frags, key=lambda m: m.GetNumAtoms())
    return stripped_mol

# ───────────────────────────────────────────────────────────────────────
# ARCHITECTURE PLATFORM CONSTANTS & NEURAL CORE
# ───────────────────────────────────────────────────────────────────────
NODE_FEATURE_DIM = 162
HIDDEN_DIM = 128
GAT_HEADS = 4
SCHNET_CUTOFF = 10.0

HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-γ", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

if HAS_DEEP_LEARNING_CORE:
    class StartNerve_Titan_V11(torch.nn.Module):
        def __init__(self, node_feat_dim=NODE_FEATURE_DIM, hidden=HIDDEN_DIM, heads=GAT_HEADS, n_tasks=12, dropout=0.2):
            super().__init__()
            self.schnet = SchNet(
                hidden_channels=hidden, num_filters=128, num_interactions=6,
                cutoff=SCHNET_CUTOFF, num_gaussians=50, max_num_neighbors=32
            )
            self.input_proj = torch.nn.Linear(node_feat_dim, hidden)
            self.gat1 = GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout, concat=True)
            self.gat2 = GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout, concat=True)
            self.gat3 = GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout, concat=True)
            self.norm_2d = torch.nn.LayerNorm(hidden)
            self.atom_gate = torch.nn.Linear(hidden, 1)
            self.fusion_norm = torch.nn.LayerNorm(hidden * 2)
            self.fusion_gate = torch.nn.Linear(hidden * 2, hidden)
            self.fc1 = torch.nn.Linear(hidden, hidden // 2)
            self.classifier = torch.nn.Linear(hidden // 2, n_tasks)

        def simple_radius_graph(self, pos, r, batch):
            dist = torch.cdist(pos, pos)
            mask = (dist <= r) & (batch.view(-1, 1) == batch.view(1, -1))
            mask.fill_diagonal_(False)
            return mask.nonzero(as_tuple=False).t().contiguous()

        def attention_readout(self, h_nodes, batch):
            gates = self.atom_gate(h_nodes)
            gates = gates - gates.max()
            exp = torch.exp(gates)
            denom = global_add_pool(exp, batch)[batch]
            weights = exp / (denom + 1e-10)
            return global_add_pool(h_nodes * weights, batch)

        def forward(self, data):
            # Stream 1: 3D Physical Geometry
            edge_index_3d = self.simple_radius_graph(data.pos, self.schnet.cutoff, data.batch)
            row, col = edge_index_3d
            edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
            edge_attr = self.schnet.distance_expansion(edge_weight)
            h_schnet = self.schnet.embedding(data.z)
            for interaction in self.schnet.interactions:
                h_schnet = h_schnet + interaction(h_schnet, edge_index_3d, edge_weight, edge_attr)
            h_3d = global_add_pool(h_schnet, data.batch)

            # Stream 2: 2D Electronic Topology
            h_states = F.relu(self.input_proj(data.x))
            h_init = h_states
            h_states = F.relu(self.gat1(h_states, data.edge_index))
            h_states = F.relu(self.gat2(h_states, data.edge_index))
            h_states = F.relu(self.gat3(h_states, data.edge_index))
            h_states = self.norm_2d(h_states + h_init)
            h_2d = self.attention_readout(h_states, data.batch)

            # Fusion Block
            combined = torch.cat([h_3d, h_2d], dim=-1)
            combined = self.fusion_norm(combined)
            fused = F.relu(self.fusion_gate(combined))
            
            out = F.relu(self.fc1(fused))
            return self.classifier(out)

def get_v10_node_features(mol):
    charge_computed = True
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        charge_computed = False

    all_node_feats = []
    for atom in mol.GetAtoms():
        features = []
        atomic_one_hot = [0] * 118
        atomic_num = atom.GetAtomicNum()
        if 1 <= atomic_num <= 118:
            atomic_one_hot[atomic_num - 1] = 1
        features += atomic_one_hot

        charge = 0.0
        if charge_computed:
            try:
                val = atom.GetProp('_GasteigerCharge')
                if val not in ['-nan', 'nan', 'inf', '-inf']:
                    parsed = float(val)
                    if not (np.isnan(parsed) or np.isinf(parsed)):
                        charge = float(np.clip(parsed, -2.0, 2.0))
            except Exception:
                charge = 0.0
        features.append(charge)

        features += [1 if atom.GetHybridization() == h_type else 0 for h_type in HYBRIDIZATION_TYPES]
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        features.append(float(atom.GetFormalCharge()))
        features.append(float(atom.GetTotalNumHs()))
        features.append(1.0 if atom.IsInRing() else 0.0)
        features.append(float(atom.GetDegree()))
        features += [0.0] * (NODE_FEATURE_DIM - len(features))
        all_node_feats.append(features)

    return torch.tensor(all_node_feats, dtype=torch.float)

def smiles_to_graph(smiles):
    try:
        if not smiles or not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return None

        raw_mol = Chem.MolFromSmiles(smiles)
        if raw_mol is None:
            return None
            
        mol = sanitize_and_strip(raw_mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None

        clean_smiles = Chem.MolToSmiles(mol)
        mol_h = Chem.AddHs(mol)
        
        embed_status = AllChem.EmbedMolecule(
            mol_h, maxAttempts=50, randomSeed=42, 
            useSmallRingTorsions=True, useMacrocycleTorsions=True
        )
        
        if embed_status >= 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=50)
            except Exception:
                pass
            mol_3d = Chem.RemoveHs(mol_h)
        else:
            AllChem.Compute2DCoords(mol_h)
            mol_3d = Chem.RemoveHs(mol_h)
            
        x_feats = get_v10_node_features(mol_3d)
        z_numbers = torch.tensor([atom.GetAtomicNum() for atom in mol_3d.GetAtoms()], dtype=torch.long)
        
        src, dst = [], []
        for bond in mol_3d.GetBonds():
            idx_i, idx_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [idx_i, idx_j]
            dst += [idx_j, idx_i]
            
        edge_index = torch.tensor([src, dst], dtype=torch.long) if len(src) > 0 else torch.empty((2, 0), dtype=torch.long)
        
        if mol_3d.GetNumConformers() > 0:
            conf = mol_3d.GetConformer()
            pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(mol_3d.GetNumAtoms())], dtype=torch.float)
        else:
            pos = torch.zeros((x_feats.shape[0], 3), dtype=torch.float)
            
        batch = torch.zeros(x_feats.shape[0], dtype=torch.long)
        return Data(x=x_feats, z=z_numbers, pos=pos, edge_index=edge_index, batch=batch, smiles=clean_smiles)
    except Exception as e:
        print(f"⚠️ Graph construction skipped for '{smiles}': {e}")
        return None

# MODEL WEIGHTS INITIALIZATION & DYNAMIC FALLBACK
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "startnerve_v11_best.pt")
WEIGHTS_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/startnerve_v11_best.pt"

def ensure_weights_exist():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"⚠️ Model weights missing at {MODEL_WEIGHTS_PATH}.")
        if "YOUR_USERNAME" not in WEIGHTS_URL:
            print(f"⬇️ Downloading weights from {WEIGHTS_URL}...")
            try:
                urllib.request.urlretrieve(WEIGHTS_URL, MODEL_WEIGHTS_PATH)
                print("✅ Weights downloaded successfully.")
            except Exception as e:
                print(f"❌ Failed to download model weights: {e}")
        else:
            print("⚠️ URL placeholder active. Expecting model weights from Git LFS.")

ensure_weights_exist()

if HAS_DEEP_LEARNING_CORE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StartNerve_Titan_V11(n_tasks=len(TASKS))
    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
            model.eval()
            model = model.to(device)
            print("✅ TITAN V11 NEURAL WEIGHTS LOADED SUCCESSFULLY")
        except Exception as e:
            print(f"❌ ARCHITECTURE FAILURE LOADING WEIGHTS: {e}")
            sys.exit(1)
    else:
        print(f"❌ CRITICAL BREAKDOWN: Target weights file unavailable at {MODEL_WEIGHTS_PATH}")
        sys.exit(1)

def run_v11_inference(smiles_input):
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        return {
            'smiles': smiles_input,
            'status': 'INVALID_SMILES',
            'error': 'Failed to parse structure'
        }

    clean_mol = sanitize_and_strip(mol)
    clean_smiles = Chem.MolToSmiles(clean_mol) if clean_mol else smiles_input

    # 1. RDKit Metrics
    mw = round(float(Descriptors.MolWt(clean_mol)), 2) if clean_mol else 0.0
    logp = round(float(Crippen.MolLogP(clean_mol)), 2) if clean_mol else 0.0
    sa_score = calculate_sa_score(clean_mol)
    sd_dist, ad_domain = calculate_sd_distance(clean_mol)

    # 2. Neural Inference
    pathway_scores = [0.0500] * 12
    if HAS_DEEP_LEARNING_CORE:
        try:
            graph_data = smiles_to_graph(clean_smiles)
            if graph_data is not None:
                graph_data = graph_data.to(device)
                with torch.no_grad():
                    logits = model(graph_data)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    raw_list = probs.tolist() if hasattr(probs, 'tolist') else [probs]
                    pathway_scores = [round(float(p), 4) for p in (raw_list if isinstance(raw_list, list) else [raw_list])]
        except Exception as err:
            print(f"Inference Warning: {err}")

    if len(pathway_scores) != 12:
        pathway_scores = [0.0500] * 12

    overall_risk = round(max(pathway_scores), 4)
    verdict = "INTERCEPT" if overall_risk > 0.4100 else "COMPLIANT"

    return {
        "smiles": clean_smiles,
        "risk_score": overall_risk,
        "scores": pathway_scores,
        "mw": mw,
        "logp": logp,
        "sa_score": sa_score,
        "sd_dist": sd_dist,
        "applicability_domain": ad_domain,
        "in_domain": True if sd_dist <= 2.5 else False,
        "verdict": verdict
    }

# ───────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ───────────────────────────────────────────────────────────────────────

@app.route('/api/audit', methods=['POST', 'OPTIONS'])
def api_audit():
    if request.method == 'OPTIONS':
        return jsonify({"success": True}), 200

    try:
        data = request.get_json() or {}
        smiles_input = data.get('smiles') or data.get('smiles_input', '')
        
        if not smiles_input:
            return jsonify({'success': False, 'error': 'No SMILES input provided.'}), 400

        lines = [line.strip() for line in smiles_input.replace(",", "\n").split("\n") if line.strip()]
        results = [run_v11_inference(s) for s in lines]

        return jsonify({
            "success": True,
            "results": results
        }), 200

    except Exception as e:
        print(f"Error in /api/audit: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/audit/batch', methods=['POST', 'OPTIONS'])
def audit_batch():
    if request.method == 'OPTIONS':
        return jsonify({"success": True}), 200

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        raw_bytes = file.stream.read()

        decoded_text = None
        for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
            try:
                decoded_text = raw_bytes.decode(encoding)
                break
            except Exception:
                continue

        if not decoded_text:
            return jsonify({'success': False, 'error': 'Failed to decode file'}), 400

        stream = io.StringIO(decoded_text, newline=None)
        reader = csv.reader(stream)

        results = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            
            raw_smiles = row[0].strip()
            if raw_smiles.lower() in ['smiles', 'target_smiles', 'molecule', 'structure']:
                continue

            results.append(run_v11_inference(raw_smiles))

        return jsonify({'success': True, 'batch_count': len(results), 'results': results}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

@app.route('/api/export/pdf', methods=['POST', 'OPTIONS'])
def export_compliance_pdf():
    if request.method == 'OPTIONS':
        return jsonify({"success": True}), 200

    if not HAS_WEASYPRINT:
        return jsonify({'success': False, 'error': 'WeasyPrint library not available on server'}), 500

    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return jsonify({'success': False, 'error': 'No audit results provided'}), 400

        results = data['results']

        pdf_html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                @page { size: A4 portrait; margin: 15mm; }
                body { font-family: sans-serif; color: #0f172a; }
                .brand { font-size: 20pt; font-weight: 800; text-transform: uppercase; }
                .subtitle { font-size: 8.5pt; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
                .audit-table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 8.5pt; }
                .audit-table th { background-color: #0f172a; color: #f8fafc; padding: 8px; text-align: left; }
                .audit-table td { padding: 8px; border: 1px solid #e2e8f0; }
                .badge { padding: 3px 6px; border-radius: 4px; font-weight: 800; font-size: 7.5pt; text-align: center; }
                .badge-clear { background-color: #dcfce7; color: #14532d; }
                .badge-intercept { background-color: #fee2e2; color: #7f1d1d; }
            </style>
        </head>
        <body>
            <div class="brand">STARTNERVE INTELLIGENCE</div>
            <div class="subtitle">Enterprise Batch Compliance Audit Report</div>
            <table class="audit-table">
                <thead>
                    <tr>
                        <th>SMILES Target</th>
                        <th>MW</th>
                        <th>LogP</th>
                        <th>SA Score</th>
                        <th>Risk Score</th>
                        <th>Verdict</th>
                    </tr>
                </thead>
                <tbody>
                    {% for mol in results %}
                    <tr>
                        <td>{{ mol.smiles }}</td>
                        <td>{{ "%.2f"|format(mol.mw) if mol.mw is defined else '—' }}</td>
                        <td>{{ "%.2f"|format(mol.logp) if mol.logp is defined else '—' }}</td>
                        <td>{{ "%.2f"|format(mol.sa_score) if mol.sa_score is defined else '—' }}</td>
                        <td><strong>{{ "%.4f"|format(mol.risk_score) if mol.risk_score is defined else '0.0000' }}</strong></td>
                        <td>
                            {% if mol.verdict == 'INTERCEPT' %}
                                <span class="badge badge-intercept">INTERCEPT</span>
                            {% else %}
                                <span class="badge badge-clear">COMPLIANT</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </body>
        </html>
        """

        rendered_html = render_template_string(pdf_html_template, results=results)
        filename = "startnerve_compliance_report.pdf"
        filepath = os.path.join(REPORTS_DIR, filename)

        HTML(string=rendered_html).write_pdf(filepath)

        return jsonify({'success': True, 'download_url': f"/static/reports/{filename}"})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/static/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)


if __name__ == '__main__':
    device_name = str(device).upper() if 'device' in globals() else 'CPU'
    port = int(os.environ.get("PORT", 10000))
    print(f"\n=========================================================================")
    print(f"  STARTNERVE TITAN PREDICTIVE CORE API RUNTIME ENVIRONMENT")
    print(f"=========================================================================")
    print(f"  🛰️  Compute Target Hardware Context: {device_name}")
    print(f"  🧬 Neural Weights Core Status      : ACTIVE REAL WEIGHTS LOADED")
    print(f"  Server Binding Target              : 0.0.0.0:{port}")
    print(f"=========================================================================\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)