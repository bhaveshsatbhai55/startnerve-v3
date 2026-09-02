# ============================================================
#  StarterNerve GNN v5 — PyTorch Geometric, No DGL, No TF
#  Windows-safe | Exotic-valence-safe | No deprecated kwargs
#
#  Tested against:
#    torch          >= 2.1
#    torch-geometric >= 2.4
#    rdkit          >= 2023.09
#    scikit-learn   >= 1.3
#    pandas         >= 2.0
#
#  Install:
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#    pip install torch-geometric
#    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
#    pip install rdkit scikit-learn pandas numpy
# ============================================================

import os
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from rdkit import Chem, RDLogger
from rdkit.Chem import SanitizeFlags
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ── Silence RDKit valence warnings (handled manually below) ──────────────────
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("StartNerve")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — ATOM & BOND FEATURES
# ══════════════════════════════════════════════════════════════════════════════

_ATOMIC_NUMS       = list(range(1, 119))
_DEGREES           = list(range(0, 11))
_IMPLICIT_VALENCES = list(range(0, 11))
_FORMAL_CHARGES    = [-2, -1, 0, 1, 2]
_NUM_HS            = list(range(0, 5))
_HYBRIDIZATIONS    = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(value, choices: list) -> list:
    """One-hot encode value; unknown values map to a trailing 'other' bucket."""
    enc = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    enc[idx] = 1
    return enc


def _atom_features(atom) -> list:
    return (
        _one_hot(atom.GetAtomicNum(),         _ATOMIC_NUMS)
        + _one_hot(atom.GetDegree(),          _DEGREES)
        + _one_hot(atom.GetImplicitValence(),  _IMPLICIT_VALENCES)
        + _one_hot(atom.GetFormalCharge(),     _FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(),       _NUM_HS)
        + _one_hot(atom.GetHybridization(),    _HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic())]
    )


def _bond_features(bond) -> list:
    return (
        _one_hot(bond.GetBondType(), _BOND_TYPES)
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
    )


# Pre-compute dims so the model head is always consistent
ATOM_FEAT_DIM = (
    len(_ATOMIC_NUMS)        + 1   # 119
    + len(_DEGREES)          + 1   # 12
    + len(_IMPLICIT_VALENCES)+ 1   # 12
    + len(_FORMAL_CHARGES)   + 1   #  6
    + len(_NUM_HS)           + 1   #  6
    + len(_HYBRIDIZATIONS)   + 1   #  6
    + 1                            # is_aromatic
)  # total = 163

BOND_FEAT_DIM = len(_BOND_TYPES) + 1 + 2   # 4 types + other + conjugated + ring = 7


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SAFE FEATURIZATION (exotic-valence-proof)
# ══════════════════════════════════════════════════════════════════════════════

def safe_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Parse SMILES without enforcing RDKit valence rules.
    Aluminium-6 and other non-standard atoms pass through cleanly.
    Returns None only if the SMILES string itself is unparseable.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        # Run all sanitization EXCEPT valence checking
        flags = SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_PROPERTIES
        Chem.SanitizeMol(mol, flags)
        return mol
    except Exception:
        return None


def mol_to_graph(smiles: str, label: float) -> Optional[Data]:
    """
    Convert a SMILES string + scalar label → PyG Data object.
    Returns None silently on any failure; the dataset loop skips it.
    """
    mol = safe_mol(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # ── Node features ─────────────────────────────────────────────────────
    try:
        x = torch.tensor(
            [_atom_features(a) for a in mol.GetAtoms()],
            dtype=torch.float,
        )
    except Exception:
        return None

    # ── Edge index + edge features (undirected: both directions) ──────────
    src, dst, edge_attrs = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        src += [i, j]
        dst += [j, i]
        edge_attrs += [bf, bf]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Single-atom molecule — valid graph with no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ToxDataset(Dataset):
    """
    In-memory PyG Dataset.  Builds all graphs up-front so DataLoader
    never touches SMILES strings or RDKit again during training.
    """

    def __init__(self, records: list):
        super().__init__()
        self._graphs = []
        skipped = 0

        for r in records:
            g = mol_to_graph(str(r["smiles"]).strip(), float(r["label"]))
            if g is not None:
                self._graphs.append(g)
            else:
                skipped += 1

        log.info(
            f"Dataset built: {len(self._graphs):,} graphs  |  "
            f"{skipped} skipped (unparseable SMILES)"
        )

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ToxGAT(torch.nn.Module):
    """
    3-layer Graph Attention Network for binary toxicity prediction.

    Design choices that avoid every known compatibility trap:
      • Pure PyTorch Geometric — no DGL, no TensorFlow, no Keras
      • LayerNorm instead of BatchNorm — no 'fused' kwarg anywhere
      • Residual skip connections — stable training on small datasets
      • Concat(mean_pool, sum_pool) readout — richer graph representation
    """

    def __init__(
        self,
        in_channels: int   = ATOM_FEAT_DIM,
        hidden:      int   = 128,
        heads:       int   = 4,
        n_tasks:     int   = 1,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.dropout_p = dropout

        # Input projection → hidden dim
        self.input_proj = torch.nn.Linear(in_channels, hidden)

        # GAT layers
        # Layers 1 & 2: multi-head → concat → project back to hidden
        self.gat1    = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.gat2    = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
        # Layer 3: single head → hidden (for clean pooling)
        self.gat3    = GATConv(hidden, hidden, heads=1, dropout=dropout)

        # LayerNorm — compatible with every PyTorch version, no fused= arg
        self.norm1   = torch.nn.LayerNorm(hidden)
        self.norm2   = torch.nn.LayerNorm(hidden)
        self.norm3   = torch.nn.LayerNorm(hidden)

        # Readout MLP
        # Input dim is hidden*2 because we concat mean_pool + sum_pool
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, n_tasks),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = F.relu(self.input_proj(x))

        # GAT layer 1 with residual
        x = self.norm1(F.relu(self.gat1(x, edge_index)) + x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # GAT layer 2 with residual
        x = self.norm2(F.relu(self.gat2(x, edge_index)) + x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # GAT layer 3
        x = self.norm3(F.relu(self.gat3(x, edge_index)))

        # Global pooling: concat mean + sum for richer graph-level repr
        x = torch.cat(
            [global_mean_pool(x, batch), global_add_pool(x, batch)],
            dim=-1,
        )

        return self.classifier(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — TRAIN / EVALUATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss   = F.binary_cross_entropy_with_logits(logits, batch.y)
        loss.backward()
        # Gradient clipping — prevents exploding gradients on exotic molecules
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        probs = torch.sigmoid(model(batch)).cpu().numpy()
        all_preds.extend(probs.tolist())
        all_labels.extend(batch.y.cpu().numpy().tolist())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    # Guard against splits with only one class (can happen in tiny val sets)
    try:
        auroc = roc_auc_score(labels, preds)
    except ValueError:
        auroc = float("nan")

    acc = ((preds >= 0.5).astype(int) == labels.astype(int)).mean()
    return {"auroc": auroc, "acc": acc}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(
    csv_path:   str   = "startnerve_master_train.csv",
    smiles_col: str   = "smiles",
    label_col:  str   = "activity",
    model_dir:  str   = "startnerve_v5_gat",
    epochs:     int   = 60,
    batch_size: int   = 64,
    lr:         float = 1e-3,
    hidden:     int   = 128,
    heads:      int   = 4,
    dropout:    float = 0.3,
    patience:   int   = 10,
    seed:       int   = 42,
):
    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in [smiles_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    df = df[[smiles_col, label_col]].dropna().reset_index(drop=True)
    records = [{"smiles": row[smiles_col], "label": row[label_col]}
               for _, row in df.iterrows()]
    log.info(f"Loaded {len(records):,} rows from {csv_path}")

    # ── Build graph dataset ───────────────────────────────────────────────────
    dataset = ToxDataset(records)

    if len(dataset) < 10:
        raise RuntimeError("Too few valid molecules to train. Check your CSV.")

    # ── Stratified 80 / 10 / 10 split ────────────────────────────────────────
    all_labels = [int(g.y.item()) for g in dataset]
    idx = list(range(len(dataset)))

    tr_idx, tmp_idx = train_test_split(
        idx, test_size=0.2, stratify=all_labels, random_state=seed
    )
    tmp_labels = [all_labels[i] for i in tmp_idx]
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.5, stratify=tmp_labels, random_state=seed
    )

    train_set = [dataset[i] for i in tr_idx]
    val_set   = [dataset[i] for i in va_idx]
    test_set  = [dataset[i] for i in te_idx]

    log.info(
        f"Split → train {len(train_set):,} "
        f"/ val {len(val_set):,} "
        f"/ test {len(test_set):,}"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    # ── Model, optimizer, scheduler ───────────────────────────────────────────
    model = ToxGAT(
        in_channels=ATOM_FEAT_DIM,
        hidden=hidden,
        heads=heads,
        n_tasks=1,
        dropout=dropout,
    ).to(device)

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # verbose= removed — deprecated in PyTorch 2.2, removed in 2.4+
    # LR changes are logged manually in the loop below
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # ── Training loop with early stopping ─────────────────────────────────────
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "best_model.pt")

    best_auroc = 0.0
    wait       = 0
    history    = []

    log.info("─" * 65)
    log.info("Starting training...")
    log.info("─" * 65)

    for epoch in range(1, epochs + 1):

        tr_loss = train_one_epoch(model, train_loader, optimizer, device)

        val_metrics = evaluate(model, val_loader, device)
        val_auroc   = val_metrics["auroc"]
        val_acc     = val_metrics["acc"]

        # Track LR changes manually (replaces removed verbose= kwarg)
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auroc)
        new_lr = optimizer.param_groups[0]["lr"]

        lr_tag = f"  ↓ LR {prev_lr:.1e}→{new_lr:.1e}" if new_lr < prev_lr else ""

        log.info(
            f"Epoch {epoch:3d}/{epochs}  |  "
            f"Loss {tr_loss:.4f}  |  "
            f"Val AUROC {val_auroc:.4f}  |  "
            f"Val Acc {val_acc:.4f}"
            + lr_tag
        )

        history.append({
            "epoch":      epoch,
            "train_loss": tr_loss,
            "val_auroc":  val_auroc,
            "val_acc":    val_acc,
            "lr":         new_lr,
        })

        # Checkpoint on improvement
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            wait = 0
            torch.save(model.state_dict(), checkpoint_path)
            log.info(f"  ✓ Best model saved (AUROC {best_auroc:.4f})")
        else:
            wait += 1
            if wait >= patience:
                log.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # ── Final test evaluation ─────────────────────────────────────────────────
    log.info("─" * 65)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    log.info(f"Test AUROC : {test_metrics['auroc']:.4f}")
    log.info(f"Test Acc   : {test_metrics['acc']:.4f}")
    log.info("─" * 65)

    # ── Save training log ─────────────────────────────────────────────────────
    log_path = os.path.join(model_dir, "training_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False)
    log.info(f"Training log → {log_path}")
    log.info(f"Checkpoint   → {checkpoint_path}")

    return model, test_metrics


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(
        csv_path   = "startnerve_master_train.csv",
        smiles_col = "smiles",
        label_col  = "activity",
        model_dir  = "startnerve_v5_gat",
        epochs     = 60,
        batch_size = 64,
        lr         = 1e-3,
        hidden     = 128,
        heads      = 4,
        dropout    = 0.3,
        patience   = 10,
        seed       = 42,
    )