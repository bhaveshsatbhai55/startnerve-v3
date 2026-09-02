"""
predict_tox.py — StartNerve Safety Report Generator
====================================================
ToxGAT Inference Script for Pharmaceutical Manufacturers
Model: 3-layer GAT with Residual Connections and LayerNorm
Trained on: 7,830 harmonized compounds (startnerve_v5_gat)

Usage:
    python predict_tox.py --input compounds.csv --output report.csv
    python predict_tox.py --input compounds.csv --output report.csv --xai --xai_dir attention_maps/
    python predict_tox.py --input compounds.csv --model_dir custom_model_dir/ --mc_dropout 30

Requirements:
    pip install torch torch-geometric rdkit-pypi pandas numpy matplotlib tqdm
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
#  Logging Setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("predict_tox.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  RDKit Imports (graceful failure)
# ─────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    log.warning("RDKit not found. Molecular visualizations will be disabled.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    log.warning("Matplotlib not found. Attention map images will be disabled.")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1: Safe Molecule Utilities  (mirrors training script logic)
# ─────────────────────────────────────────────────────────────────────────────

# Atom feature vocabulary — must match training exactly
ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg",
    "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl",
    "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn",
    "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
    "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]
ATOM_TYPE_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}

HYBRIDIZATION_TYPES = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
CHIRALITY_TYPES    = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]

NUM_ATOM_FEATURES = len(ATOM_TYPES) + 6 + len(HYBRIDIZATION_TYPES) + len(CHIRALITY_TYPES) + 4
# breakdown: one-hot atom (44) + degree(6) + hybridization(7) + chirality(4) + [H_count, charge, radical, aromatic] (4)


def safe_mol(smiles: str) -> Optional[object]:
    """
    Safely parse a SMILES string. Mirrors training-time logic:
    - Handles exotic/extended atoms gracefully
    - Returns None instead of crashing on invalid SMILES
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for molecule parsing.")
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Sanitize — catch aromatic perception errors for exotic systems
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                return None
        return mol
    except Exception:
        return None


def atom_features(atom) -> List[float]:
    """
    Encode a single RDKit atom into a fixed-length feature vector.
    Identical to training-time featurisation — do NOT modify.
    """
    symbol = atom.GetSymbol()
    atom_one_hot = [0] * len(ATOM_TYPES)
    idx = ATOM_TYPE_IDX.get(symbol, ATOM_TYPE_IDX["Unknown"])
    atom_one_hot[idx] = 1

    degree = min(atom.GetDegree(), 5)
    degree_one_hot = [0] * 6
    degree_one_hot[degree] = 1

    hyb = str(atom.GetHybridization()).split(".")[-1]
    hyb_one_hot = [0] * len(HYBRIDIZATION_TYPES)
    hyb_idx = HYBRIDIZATION_TYPES.index(hyb) if hyb in HYBRIDIZATION_TYPES else len(HYBRIDIZATION_TYPES) - 1
    hyb_one_hot[hyb_idx] = 1

    chi = str(atom.GetChiralTag()).split(".")[-1]
    chi_one_hot = [0] * len(CHIRALITY_TYPES)
    chi_idx = CHIRALITY_TYPES.index(chi) if chi in CHIRALITY_TYPES else len(CHIRALITY_TYPES) - 1
    chi_one_hot[chi_idx] = 1

    misc = [
        atom.GetTotalNumHs() / 8.0,          # H count (normalized)
        atom.GetFormalCharge() / 4.0,         # formal charge (normalized)
        atom.GetNumRadicalElectrons() / 4.0,  # radical electrons
        float(atom.GetIsAromatic()),           # aromaticity flag
    ]

    return atom_one_hot + degree_one_hot + hyb_one_hot + chi_one_hot + misc


def mol_to_graph(mol, smiles: str = "") -> Optional[Data]:
    """
    Convert an RDKit molecule to a PyTorch Geometric Data object.
    Mirrors training-time graph construction — must remain identical.
    """
    if mol is None:
        return None

    try:
        atoms = mol.GetAtoms()
        x = torch.tensor([atom_features(a) for a in atoms], dtype=torch.float)

        edges_src, edges_dst = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges_src += [i, j]
            edges_dst += [j, i]

        if len(edges_src) == 0:
            # Single-atom molecule — add self-loop
            edges_src = [0]
            edges_dst = [0]

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, smiles=smiles, num_nodes=x.size(0))

    except Exception as e:
        log.debug(f"mol_to_graph failed for '{smiles}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2: ToxGAT Model Architecture  (must match saved best_model.pt)
# ─────────────────────────────────────────────────────────────────────────────

class ToxGAT(nn.Module):
    """
    3-layer Graph Attention Network with:
      - Residual connections (per-layer skip)
      - LayerNorm after each GAT layer
      - MC-Dropout for uncertainty quantification
      - Attention weight return for XAI
    """

    def __init__(
        self,
        in_channels: int = NUM_ATOM_FEATURES,
        hidden_channels: int = 128,
        out_channels: int = 1,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout_rate = dropout

        # Layer 1: in_channels → hidden_channels * heads
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.proj1 = nn.Linear(in_channels, hidden_channels * heads)  # residual projection

        # Layer 2: hidden_channels * heads → hidden_channels * heads
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)
        # residual: same shape, no projection needed

        # Layer 3: hidden_channels * heads → hidden_channels  (single head for clean pooling)
        self.gat3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False)
        self.norm3 = nn.LayerNorm(hidden_channels)
        self.proj3 = nn.Linear(hidden_channels * heads, hidden_channels)  # residual projection

        # Classifier head
        self.fc1 = nn.Linear(hidden_channels, 64)
        self.fc2 = nn.Linear(64, out_channels)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List]]:

        attention_weights_all = []

        # ── GAT Layer 1 ──
        residual = self.proj1(x)
        if return_attention_weights:
            out1, (ei1, aw1) = self.gat1(x, edge_index, return_attention_weights=True)
            attention_weights_all.append((ei1, aw1))
        else:
            out1 = self.gat1(x, edge_index)
        out1 = F.elu(self.norm1(out1 + residual))
        out1 = self.dropout(out1)

        # ── GAT Layer 2 ──
        residual = out1
        if return_attention_weights:
            out2, (ei2, aw2) = self.gat2(out1, edge_index, return_attention_weights=True)
            attention_weights_all.append((ei2, aw2))
        else:
            out2 = self.gat2(out1, edge_index)
        out2 = F.elu(self.norm2(out2 + residual))
        out2 = self.dropout(out2)

        # ── GAT Layer 3 ──
        residual = self.proj3(out2)
        if return_attention_weights:
            out3, (ei3, aw3) = self.gat3(out2, edge_index, return_attention_weights=True)
            attention_weights_all.append((ei3, aw3))
        else:
            out3 = self.gat3(out2, edge_index)
        out3 = F.elu(self.norm3(out3 + residual))
        out3 = self.dropout(out3)

        # ── Global Mean Pooling ──
        pooled = global_mean_pool(out3, batch)

        # ── Classifier ──
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        logits = self.fc2(out)

        if return_attention_weights:
            return logits, attention_weights_all
        return logits, None


def load_model(model_dir: str, device: torch.device) -> ToxGAT:
    """
    Load the saved ToxGAT model from best_model.pt.
    Tries to read hyperparameters from a companion config if available.
    """
    model_path = Path(model_dir) / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Support both bare state_dict saves and full checkpoint dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        hparams    = checkpoint.get("hparams", {})
        log.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}  |  Val AUC: {checkpoint.get('val_auc', 'N/A')}")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        hparams    = checkpoint.get("hparams", {})
    else:
        # Raw state dict
        state_dict = checkpoint
        hparams    = {}

    model = ToxGAT(
        in_channels     = hparams.get("in_channels",      NUM_ATOM_FEATURES),
        hidden_channels = hparams.get("hidden_channels",  128),
        out_channels    = hparams.get("out_channels",     1),
        heads           = hparams.get("heads",            4),
        dropout         = hparams.get("dropout",          0.2),
    )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    log.info(f"✓ Model loaded from {model_path}  ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3: Uncertainty Quantification — MC Dropout
# ─────────────────────────────────────────────────────────────────────────────

def enable_mc_dropout(model: nn.Module):
    """Force all Dropout layers into train mode for MC-Dropout inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(
    model: ToxGAT,
    data_list: List[Data],
    device: torch.device,
    n_passes: int = 30,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run N stochastic forward passes with dropout enabled.
    Returns mean and std of sigmoid probabilities across passes.
    """
    model.eval()
    enable_mc_dropout(model)

    all_probs = []  # shape: (n_passes, n_compounds)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for _ in range(n_passes):
            pass_probs = []
            for batch in loader:
                batch = batch.to(device)
                logits, _ = model(batch.x, batch.edge_index, batch.batch, return_attention_weights=False)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                pass_probs.extend(probs.tolist())
            all_probs.append(pass_probs)

    all_probs = np.array(all_probs)   # (n_passes, n_compounds)
    mean_prob = all_probs.mean(axis=0)
    std_prob  = all_probs.std(axis=0)
    return mean_prob, std_prob


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4: Explainable AI — Attention-Weight Structural Alerts
# ─────────────────────────────────────────────────────────────────────────────

def get_attention_weights_single(
    model: ToxGAT,
    graph: Data,
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Run a single deterministic forward pass and retrieve per-atom attention scores
    from the final GAT layer (averaged across heads).

    Returns: np.ndarray of shape (n_atoms,) — atom-level importance scores.
    """
    model.eval()
    graph = graph.to(device)

    # Add fake batch vector (single graph)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        _, attn_list = model(
            graph.x, graph.edge_index, batch, return_attention_weights=True
        )

    if attn_list is None or len(attn_list) == 0:
        return None

    # Use last layer attention weights: (ei, aw) where aw shape = (n_edges, heads)
    edge_index, attn_weights = attn_list[-1]
    attn_weights = attn_weights.cpu().numpy()  # (n_edges, heads)
    attn_mean    = attn_weights.mean(axis=1)   # mean over heads

    # Aggregate attention per destination atom (receiving node)
    n_atoms     = graph.num_nodes
    atom_scores = np.zeros(n_atoms)
    dst_nodes   = edge_index[1].cpu().numpy()

    np.add.at(atom_scores, dst_nodes, attn_mean)

    # Normalize to [0, 1]
    if atom_scores.max() > 0:
        atom_scores /= atom_scores.max()

    return atom_scores


def draw_attention_map(
    smiles: str,
    atom_scores: np.ndarray,
    output_path: str,
    title: str = "",
) -> bool:
    """
    Render a 2D molecule image with atoms coloured by attention weight.
    Green = low importance, Red = high importance (toxic alert atoms).
    Saves a PNG to output_path. Returns True on success.
    """
    if not (RDKIT_AVAILABLE and MATPLOTLIB_AVAILABLE):
        return False

    mol = safe_mol(smiles)
    if mol is None:
        return False

    try:
        AllChem.Compute2DCoords(mol)

        # Map scores → RGB colours (green → yellow → red)
        cmap   = cm.get_cmap("RdYlGn_r")
        norm   = Normalize(vmin=0, vmax=1)
        colors = {i: cmap(norm(float(s)))[:3] for i, s in enumerate(atom_scores)}
        radii  = {i: 0.3 + 0.4 * float(s) for i, s in enumerate(atom_scores)}

        drawer = rdMolDraw2D.MolDraw2DSVG(600, 400)
        drawer.drawOptions().addStereoAnnotation = False
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=list(range(mol.GetNumAtoms())),
            highlightAtomColors=colors,
            highlightAtomRadii=radii,
        )
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()

        # Convert SVG → PNG via matplotlib
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        ax.axis("off")

        # Embed SVG as text note (fallback) — for full PNG rendering use cairosvg if available
        try:
            import cairosvg
            png_bytes = cairosvg.svg2png(bytestring=svg_text.encode())
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(png_bytes))
            ax.imshow(img)
        except Exception:
            # Fallback: save raw SVG instead
            svg_path = output_path.replace(".png", ".svg")
            with open(svg_path, "w") as f:
                f.write(svg_text)
            plt.close(fig)
            return True

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Attention Weight (Toxicity Contribution)", fontsize=9)

        if title:
            ax.set_title(title, fontsize=10, pad=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True

    except Exception as e:
        log.debug(f"draw_attention_map failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5: Safety Classification & Structural Alert Logic
# ─────────────────────────────────────────────────────────────────────────────

# Uncertainty boundaries (regulatory-grade thresholds)
UNCERTAINTY_LOW_BOUND  = 0.35   # below → clearly Safe
UNCERTAINTY_HIGH_BOUND = 0.65   # above → clearly Toxic
HIGH_UNCERTAINTY_DELTA = 0.10   # std threshold for "High Uncertainty" flag

# Known toxicophore SMARTS patterns for Structural Alert flag
STRUCTURAL_ALERT_SMARTS: Dict[str, str] = {
    "Michael acceptor"         : "[C,c]=[C,c]-[C,c]=[O,N,S]",
    "Epoxide"                  : "C1OC1",
    "Aldehyde"                 : "[CX3H1](=O)[#6]",
    "Nitro group"              : "[$([N+](=O)[O-]),$([n+](=O)[o-])]",
    "Halogen on aromatic"      : "[c][F,Cl,Br,I]",
    "Aniline"                  : "[NH2]c1ccccc1",
    "Quinone"                  : "O=C1C=CC(=O)C=C1",
    "Acrylate"                 : "C=CC(=O)[OH,OR]",
    "Acyl halide"              : "[C](=O)[F,Cl,Br,I]",
    "Peroxide"                 : "[OX2][OX2]",
    "Hydrazine"                : "[NX3][NX3]",
    "Thiol"                    : "[#16X2H]",
    "Imine"                    : "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    "Alkyl halide (reactive)"  : "[CX4][F,Cl,Br,I]",
    "Diazo"                    : "[N+]#[N-]",
}

_compiled_alerts = None

def _get_compiled_alerts():
    global _compiled_alerts
    if _compiled_alerts is None and RDKIT_AVAILABLE:
        _compiled_alerts = {}
        for name, smarts in STRUCTURAL_ALERT_SMARTS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                _compiled_alerts[name] = patt
    return _compiled_alerts or {}


def check_structural_alerts(smiles: str) -> Tuple[bool, List[str]]:
    """Return (has_alert, list_of_matched_alert_names)."""
    if not RDKIT_AVAILABLE:
        return False, []
    mol = safe_mol(smiles)
    if mol is None:
        return False, []
    matched = []
    for name, patt in _get_compiled_alerts().items():
        try:
            if mol.HasSubstructMatch(patt):
                matched.append(name)
        except Exception:
            pass
    return len(matched) > 0, matched


def classify(prob: float, std: float) -> str:
    """Assign Safety Category based on probability and uncertainty."""
    if std >= HIGH_UNCERTAINTY_DELTA and UNCERTAINTY_LOW_BOUND <= prob <= UNCERTAINTY_HIGH_BOUND:
        return "Uncertain"
    if prob < UNCERTAINTY_LOW_BOUND:
        return "Safe"
    if prob > UNCERTAINTY_HIGH_BOUND:
        return "Toxic"
    return "Uncertain"


def confidence_score(prob: float, std: float) -> float:
    """
    Regulatory confidence score in [0, 1]:
      - distance from 0.5 scaled to [0,1], penalised by std
    """
    distance = abs(prob - 0.5) * 2.0        # 0 at boundary, 1 at extremes
    penalty  = min(std * 5.0, 1.0)          # std = 0.2 → full penalty
    return round(max(0.0, distance - penalty), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6: Batch Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    input_csv    : str,
    output_csv   : str,
    model_dir    : str  = "startnerve_v5_gat",
    smiles_col   : str  = "SMILES",
    mc_passes    : int  = 30,
    batch_size   : int  = 64,
    xai          : bool = False,
    xai_dir      : str  = "attention_maps",
    device_str   : str  = "auto",
):
    start_time = datetime.now()
    log.info("=" * 70)
    log.info("  StartNerve ToxGAT — Inference Pipeline")
    log.info(f"  Input  : {input_csv}")
    log.info(f"  Output : {output_csv}")
    log.info(f"  MC Dropout passes : {mc_passes}")
    log.info(f"  XAI attention maps: {xai}")
    log.info("=" * 70)

    # ── Device ──────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info(f"Device: {device}")

    # ── Load Model ───────────────────────────────────────────────────────────
    model = load_model(model_dir, device)

    # ── Load Input ──────────────────────────────────────────────────────────
    df_in = pd.read_csv(input_csv)
    if smiles_col not in df_in.columns:
        candidates = [c for c in df_in.columns if "smiles" in c.lower() or "smi" in c.lower()]
        if candidates:
            smiles_col = candidates[0]
            log.warning(f"Column '{smiles_col}' not found; using '{smiles_col}' instead.")
        else:
            raise ValueError(
                f"Could not find SMILES column in {input_csv}. "
                f"Available columns: {list(df_in.columns)}"
            )

    smiles_list = df_in[smiles_col].astype(str).tolist()
    compound_ids = (
        df_in["compound_id"].tolist() if "compound_id" in df_in.columns
        else [f"CPD-{i+1:05d}" for i in range(len(smiles_list))]
    )
    log.info(f"Loaded {len(smiles_list)} SMILES strings.")

    # ── Featurise ────────────────────────────────────────────────────────────
    log.info("Featurising molecules...")
    graphs, valid_idx, invalid_smiles = [], [], []

    for i, smi in enumerate(smiles_list):
        mol = safe_mol(smi)
        if mol is None:
            log.debug(f"  [SKIP] Invalid SMILES at row {i}: {smi!r}")
            invalid_smiles.append(i)
            continue
        g = mol_to_graph(mol, smiles=smi)
        if g is None:
            invalid_smiles.append(i)
            continue
        graphs.append(g)
        valid_idx.append(i)

    log.info(f"Valid: {len(graphs)} | Invalid/Skipped: {len(invalid_smiles)}")

    # ── MC-Dropout Inference ─────────────────────────────────────────────────
    log.info(f"Running MC-Dropout inference ({mc_passes} passes)...")
    mean_probs, std_probs = mc_dropout_predict(
        model, graphs, device, n_passes=mc_passes, batch_size=batch_size
    )

    # ── XAI: Attention Maps ──────────────────────────────────────────────────
    attn_paths = [""] * len(graphs)
    if xai:
        if not (RDKIT_AVAILABLE and MATPLOTLIB_AVAILABLE):
            log.warning("XAI requested but RDKit or Matplotlib unavailable. Skipping.")
        else:
            Path(xai_dir).mkdir(parents=True, exist_ok=True)
            log.info(f"Generating attention maps → {xai_dir}/")
            for j, (graph, smi, prob) in enumerate(
                zip(graphs, [smiles_list[i] for i in valid_idx], mean_probs)
            ):
                atom_scores = get_attention_weights_single(model, graph, device)
                if atom_scores is None:
                    continue
                cid   = compound_ids[valid_idx[j]]
                fname = f"{cid}_{prob:.3f}.png".replace("/", "_").replace("\\", "_")
                fpath = str(Path(xai_dir) / fname)
                cat   = classify(prob, std_probs[j])
                title = f"{cid} | P(Toxic)={prob:.3f} | {cat}"
                ok    = draw_attention_map(smi, atom_scores, fpath, title=title)
                if ok:
                    attn_paths[j] = fpath

    # ── Structural Alerts ────────────────────────────────────────────────────
    log.info("Checking structural alerts (SMARTS toxicophores)...")
    alert_flags, alert_details = [], []
    for i in valid_idx:
        has_alert, matched = check_structural_alerts(smiles_list[i])
        alert_flags.append(has_alert)
        alert_details.append("; ".join(matched) if matched else "None")

    # ── Assemble Report ──────────────────────────────────────────────────────
    log.info("Assembling StartNerve Safety Report...")
    rows = []
    for j, idx in enumerate(valid_idx):
        prob = float(mean_probs[j])
        std  = float(std_probs[j])
        cat  = classify(prob, std)
        conf = confidence_score(prob, std)
        rows.append({
            "Compound_ID"            : compound_ids[idx],
            "SMILES"                 : smiles_list[idx],
            "Toxicity_Probability"   : round(prob, 6),
            "Prediction_StdDev"      : round(std, 6),
            "Safety_Category"        : cat,
            "Confidence_Score"       : conf,
            "High_Uncertainty_Flag"  : "YES" if cat == "Uncertain" else "NO",
            "Structural_Alert_Flag"  : "YES" if alert_flags[j] else "NO",
            "Structural_Alert_Detail": alert_details[j],
            "Attention_Map_Path"     : attn_paths[j] if xai else "N/A",
        })

    # Rows for invalid SMILES
    for idx in invalid_smiles:
        rows_entry = {
            "Compound_ID"            : compound_ids[idx],
            "SMILES"                 : smiles_list[idx],
            "Toxicity_Probability"   : "PARSE_ERROR",
            "Prediction_StdDev"      : "PARSE_ERROR",
            "Safety_Category"        : "INVALID",
            "Confidence_Score"       : "N/A",
            "High_Uncertainty_Flag"  : "N/A",
            "Structural_Alert_Flag"  : "N/A",
            "Structural_Alert_Detail": "N/A",
            "Attention_Map_Path"     : "N/A",
        }
        rows.append(rows_entry)

    # Sort: INVALID last, then Toxic > Uncertain > Safe
    sort_order = {"Toxic": 0, "Uncertain": 1, "Safe": 2, "INVALID": 3}
    rows.sort(key=lambda r: sort_order.get(r["Safety_Category"], 9))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    valid_rows = df_out[df_out["Safety_Category"] != "INVALID"]
    n_toxic    = (valid_rows["Safety_Category"] == "Toxic").sum()
    n_safe     = (valid_rows["Safety_Category"] == "Safe").sum()
    n_uncert   = (valid_rows["Safety_Category"] == "Uncertain").sum()
    n_alert    = (valid_rows["Structural_Alert_Flag"] == "YES").sum()

    log.info("")
    log.info("══════════════════════════════════════════════════════════════════════")
    log.info("  StartNerve Safety Report — SUMMARY")
    log.info("══════════════════════════════════════════════════════════════════════")
    log.info(f"  Total compounds    : {len(smiles_list)}")
    log.info(f"  Successfully parsed: {len(valid_idx)}")
    log.info(f"  Parse failures     : {len(invalid_smiles)}")
    log.info(f"  ── Predictions ────────────────────")
    log.info(f"  🟢 Safe            : {n_safe}")
    log.info(f"  🔴 Toxic           : {n_toxic}")
    log.info(f"  🟡 Uncertain       : {n_uncert}")
    log.info(f"  ⚠️  Structural Alerts: {n_alert}")
    log.info(f"  ── Performance ────────────────────")
    log.info(f"  Wall time          : {elapsed:.1f}s  ({elapsed/max(len(smiles_list),1)*1000:.1f} ms/compound)")
    log.info(f"  Report saved to    : {output_csv}")
    if xai:
        log.info(f"  Attention maps     : {xai_dir}/")
    log.info("══════════════════════════════════════════════════════════════════════")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7: CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="StartNerve ToxGAT Inference — Pharmaceutical Safety Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic batch prediction
  python predict_tox.py --input compounds.csv --output safety_report.csv

  # With XAI attention maps
  python predict_tox.py --input compounds.csv --output safety_report.csv \\
      --xai --xai_dir ./attention_maps

  # Custom model directory, GPU, more MC passes
  python predict_tox.py --input compounds.csv --output safety_report.csv \\
      --model_dir /path/to/startnerve_v5_gat --mc_dropout 50 --device cuda

  # Specify non-default SMILES column name
  python predict_tox.py --input compounds.csv --output safety_report.csv \\
      --smiles_col canonical_smiles
        """,
    )
    parser.add_argument("--input",      required=True,  help="Input CSV file with SMILES column")
    parser.add_argument("--output",     required=True,  help="Output CSV path for Safety Report")
    parser.add_argument("--model_dir",  default="startnerve_v5_gat", help="Directory containing best_model.pt")
    parser.add_argument("--smiles_col", default="SMILES", help="Name of the SMILES column in input CSV")
    parser.add_argument("--mc_dropout", default=30, type=int, help="Number of MC-Dropout forward passes (default: 30)")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for inference (default: 64)")
    parser.add_argument("--device",     default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--xai",        action="store_true", help="Generate attention-weight atom maps (XAI)")
    parser.add_argument("--xai_dir",    default="attention_maps", help="Directory to save attention map images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        input_csv  = args.input,
        output_csv = args.output,
        model_dir  = args.model_dir,
        smiles_col = args.smiles_col,
        mc_passes  = args.mc_dropout,
        batch_size = args.batch_size,
        xai        = args.xai,
        xai_dir    = args.xai_dir,
        device_str = args.device,
    )