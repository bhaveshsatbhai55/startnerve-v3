"""
================================================================================
StartNerve Intelligence - Titan V11 Training Pipeline (ASCII-Safe Edition)
Optimized for: Windows 11, CPU-only, 57,855 3D molecular graph dataset
Fix: Removed non-ASCII terminal log characters to prevent UnicodeEncodeError
================================================================================
"""

import os
import sys
import time
import math
import pickle
import logging
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# -----------------------------------------------------------------------------
# GLOBAL HYPERPARAMETERS
# -----------------------------------------------------------------------------
PKL_PATH         = "titan_graph_features.pkl"   
CACHE_DIR        = Path("titan_cache_v11")       
BATCH_SIZE       = 16                            
NUM_EPOCHS       = 1
LR               = 3e-4
WEIGHT_DECAY     = 1e-5
SCHNET_CUTOFF    = 4.5          # Angstroms - tight local shell, CPU-safe
SCHNET_FILTERS   = 64           
SCHNET_HIDDEN    = 128          
GAT_HIDDEN       = 128          
GAT_HEADS        = 4
FUSION_DIM       = 256
NUM_TASKS        = 12           
MISSING_LABEL    = -1           
TRAIN_FRAC       = 0.8
VAL_FRAC         = 0.2
SEED             = 42
LOG_INTERVAL     = 20           

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s : %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("titan_v11_training.log", mode="w"),
    ],
)
log = logging.getLogger("TitanV11")

torch.manual_seed(SEED)

# -----------------------------------------------------------------------------
# STEP 1 - LAZY ON-DISK CACHE BUILDER
# -----------------------------------------------------------------------------

def _pkl_fingerprint(pkl_path: str) -> str:
    h = hashlib.md5()
    with open(pkl_path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()[:12]


def build_cache_if_needed(pkl_path: str, cache_dir: Path) -> int:
    fingerprint_file = cache_dir / ".fingerprint"
    current_fp = _pkl_fingerprint(pkl_path)

    if cache_dir.exists() and fingerprint_file.exists():
        stored_fp = fingerprint_file.read_text().strip()
        if stored_fp == current_fp:
            count = len(list(cache_dir.glob("*.pt")))
            if count > 0:
                log.info(f"Cache valid : {count:,} graph files found in '{cache_dir}'. Skipping rebuild.")
                return count

    log.info(f"Building on-disk cache from '{pkl_path}' to '{cache_dir}' ...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(pkl_path, "rb") as f:
        raw_list = pickle.load(f)

    log.info(f"Pickle loaded : {len(raw_list):,} raw entries  ({time.time()-t0:.1f}s)")

    valid = 0
    skipped = 0
    t1 = time.time()

    for i, entry in enumerate(raw_list):
        data_obj = _entry_to_pyg(entry, i)
        if data_obj is None:
            skipped += 1
            continue
        torch.save(data_obj, cache_dir / f"{valid:07d}.pt")
        valid += 1

        if valid % 5000 == 0:
            elapsed = time.time() - t1
            rate = valid / elapsed
            log.info(f"  Cached {valid:,} / ~{len(raw_list):,}  ({rate:.0f} graphs/s)")

    fingerprint_file.write_text(current_fp)
    log.info(f"Cache complete : {valid:,} valid graphs written, {skipped} skipped  ({time.time()-t1:.1f}s)")
    return valid

def _entry_to_pyg(entry, idx: int):
    try:
        # 1. Adapt dynamically to either dict or object formats
        if isinstance(entry, dict):
            raw_x      = entry.get("x", entry.get("x_features", None))
            raw_z      = entry.get("z", None)
            pos        = entry.get("pos", entry.get("positions", None))
            edge_index = entry.get("edge_index", entry.get("edges", None))
            edge_attr  = entry.get("edge_attr", None)
            
            # Dynamic lookup for targets across common label dictionary keys
            y = entry.get("y", None)
            if y is None:
                # Fallback: find any dictionary entry containing your target arrays or search keys
                y = [entry[k] for k in entry if k in TASKS or k == 'labels']
                if not y and 'SMILES' not in entry:
                    # Alternative fallback: get values that match the length of our target vector
                    vals = [v for v in entry.values() if isinstance(v, (list, torch.Tensor)) and len(v) == NUM_TASKS]
                    y = vals[0] if vals else None
        else:
            raw_x      = getattr(entry, "x", None)
            raw_z      = getattr(entry, "z", None)
            pos        = getattr(entry, "pos", None)
            edge_index = getattr(entry, "edge_index", None)
            edge_attr  = getattr(entry, "edge_attr", None)
            y          = getattr(entry, "y", None)

        # 2. Strict spatial and graph connectivity check
        if pos is None or edge_index is None:
            return None

        # 3. Reconstruct or map node identity vectors
        if raw_x is not None:
            x = _to_float_tensor(raw_x)
            z = _to_long_tensor(raw_z) if raw_z is not None else torch.zeros(x.shape[0], dtype=torch.long)
        elif raw_z is not None:
            z = _to_long_tensor(raw_z)
            num_nodes = z.shape[0]
            x = torch.zeros((num_nodes, 162), dtype=torch.float)
            for i in range(num_nodes):
                atomic_num = int(z[i].item())
                if 1 <= atomic_num <= 118:
                    x[i, atomic_num - 1] = 1.0
        else:
            # Fallback for bare coordinates
            pos_tensor = _to_float_tensor(pos)
            num_nodes = pos_tensor.shape[0]
            z = torch.zeros(num_nodes, dtype=torch.long)
            x = torch.zeros((num_nodes, 162), dtype=torch.float)

        # 4. Standardize tensor formatting
        pos        = _to_float_tensor(pos)          
        edge_index = _to_long_tensor(edge_index)    
        edge_attr  = _to_float_tensor(edge_attr) if edge_attr is not None else None

        # 5. Force align the 12-task target vector array structure
        if y is not None:
            y = _to_float_tensor(y)
        else:
            # Absolute fallback baseline array if labels are structurally detached
            y = torch.zeros(NUM_TASKS, dtype=torch.float)

        if y.numel() < NUM_TASKS:
            pad = torch.full((NUM_TASKS - y.numel(),), float(MISSING_LABEL))
            y = torch.cat([y.view(-1), pad])
        y = y[:NUM_TASKS].view(1, NUM_TASKS)

        data = Data(
            x=x,
            z=z,
            pos=pos,
            edge_index=edge_index,
            y=y,
        )
        if edge_attr is not None:
            data.edge_attr = edge_attr

        return data
    except Exception as e:
        return None


def _to_float_tensor(v):
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.float()
    import numpy as np
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v).float()
    return torch.tensor(v, dtype=torch.float)


def _to_long_tensor(v):
    if isinstance(v, torch.Tensor):
        return v.long()
    import numpy as np
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v).long()
    return torch.tensor(v, dtype=torch.long)

# -----------------------------------------------------------------------------
# STEP 2 - STREAMING DATASET
# -----------------------------------------------------------------------------

class TitanDiskDataset(Dataset):
    def __init__(self, indices, cache_dir: Path):
        super().__init__()
        self.custom_indices   = indices       
        self.cache_dir = cache_dir

    def len(self):
        return len(self.custom_indices)

    def get(self, idx):
        global_idx = self.custom_indices[idx]
        path = self.cache_dir / f"{global_idx:07d}.pt"
        return torch.load(path, weights_only=False)

# -----------------------------------------------------------------------------
# STEP 3 - SCHNET 3D STREAM (CPU-optimized)
# -----------------------------------------------------------------------------

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=SCHNET_CUTOFF, num_gaussians=SCHNET_FILTERS):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * dist.pow(2))


class SchNetInteraction(nn.Module):
    def __init__(self, hidden_dim, num_filters):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_filters, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, edge_weight):
        src, dst = edge_index
        W   = self.filter_net(edge_weight)          
        msg = h[src] * W                            
        agg = torch.zeros_like(h).scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return h + self.update_net(agg)


class SchNetEncoder(nn.Module):
    def __init__(self, hidden_dim=SCHNET_HIDDEN, num_interactions=3,
                 num_gaussians=SCHNET_FILTERS, cutoff=SCHNET_CUTOFF):
        super().__init__()
        self.cutoff    = cutoff
        self.embedding = nn.Embedding(119, hidden_dim)
        self.smearing  = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim, num_gaussians)
            for _ in range(num_interactions)
        ])
        self.out_norm  = nn.LayerNorm(hidden_dim)

    def _build_radius_graph(self, pos, batch):
        src_list, dst_list, dist_list = [], [], []
        num_graphs = batch.max().item() + 1

        for g in range(num_graphs):
            mask  = (batch == g).nonzero(as_tuple=False).view(-1)
            p     = pos[mask]                          
            n     = p.shape[0]
            if n < 2:
                continue
            d = torch.cdist(p, p)                      
            i_idx, j_idx = torch.where((d < self.cutoff) & (d > 0.0))
            if i_idx.numel() == 0:
                continue
            global_i = mask[i_idx]
            global_j = mask[j_idx]
            src_list.append(global_i)
            dst_list.append(global_j)
            dist_list.append(d[i_idx, j_idx])

        if not src_list:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos.device)
            dists      = torch.zeros((0,),   dtype=torch.float, device=pos.device)
            return edge_index, dists

        src   = torch.cat(src_list)
        dst   = torch.cat(dst_list)
        dists = torch.cat(dist_list)
        return torch.stack([src, dst], dim=0), dists

    def forward(self, z, pos, batch):
        edge_index, dist = self._build_radius_graph(pos, batch)
        h = self.embedding(z)                          

        if edge_index.shape[1] > 0:
            rbf = self.smearing(dist)                  
            for layer in self.interactions:
                h = layer(h, edge_index, rbf)

        h = self.out_norm(h)
        return global_mean_pool(h, batch)              

# -----------------------------------------------------------------------------
# STEP 4 - GATv2 2D STREAM
# -----------------------------------------------------------------------------

class GATv2Encoder(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=GAT_HIDDEN, heads=GAT_HEADS,
                 num_layers=3, edge_dim=None):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = hidden_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=0.1,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            if edge_attr is not None:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = norm(h)
            h = F.silu(h)

        return global_mean_pool(h, batch)              

# -----------------------------------------------------------------------------
# STEP 5 - TITAN V11: DUAL-STREAM FUSION MODEL
# -----------------------------------------------------------------------------

class TitanV11(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim=None,
                 schnet_hidden=SCHNET_HIDDEN, gat_hidden=GAT_HIDDEN,
                 fusion_dim=FUSION_DIM, num_tasks=NUM_TASKS):
        super().__init__()

        self.schnet = SchNetEncoder(
            hidden_dim=schnet_hidden,
        )

        self.gat = GATv2Encoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=gat_hidden,
            heads=GAT_HEADS,
            edge_dim=edge_feat_dim,
        )

        combined_dim = schnet_hidden + gat_hidden

        self.gate_proj  = nn.Linear(combined_dim, fusion_dim)
        self.value_proj = nn.Linear(combined_dim, fusion_dim)
        self.norm       = nn.LayerNorm(fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, num_tasks),
        )

    def forward(self, data):
        x          = data.x
        z          = data.z
        pos        = data.pos
        edge_index = data.edge_index
        edge_attr  = getattr(data, "edge_attr", None)
        batch      = data.batch

        h3d = self.schnet(z, pos, batch)              
        h2d = self.gat(x, edge_index, edge_attr, batch)  

        h = torch.cat([h3d, h2d], dim=-1)             

        gate  = torch.sigmoid(self.gate_proj(h))
        value = self.value_proj(h)
        h     = self.norm(gate * value)               

        return self.classifier(h)                      

# -----------------------------------------------------------------------------
# STEP 6 - DYNAMIC BATCH-WEIGHTED LOSS
# -----------------------------------------------------------------------------

def compute_weighted_bce_loss(logits, targets):
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_tasks = 0

    for t in range(targets.shape[1]):
        t_logits  = logits[:, t]                       
        t_targets = targets[:, t]                      

        mask = (t_targets != MISSING_LABEL)
        if mask.sum() == 0:
            continue

        valid_logits  = t_logits[mask]
        valid_targets = t_targets[mask]

        n_pos = (valid_targets == 1).float().sum().clamp(min=1.0)
        n_neg = (valid_targets == 0).float().sum().clamp(min=1.0)
        beta  = n_neg / n_pos                          

        pos_weight = torch.tensor([beta], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(
            valid_logits, valid_targets, pos_weight=pos_weight
        )
        total_loss  = total_loss + loss
        valid_tasks += 1

    if valid_tasks == 0:
        return total_loss
    return total_loss / valid_tasks

# -----------------------------------------------------------------------------
# STEP 7 - TRAIN / VALIDATION LOOPS
# -----------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, phase="train"):
    is_train = (phase == "train")
    model.train(is_train)

    total_loss  = 0.0
    total_steps = 0
    t_start     = time.time()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            logits = model(batch)
            targets = batch.y.view(-1, NUM_TASKS)

            loss = compute_weighted_bce_loss(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss  += loss.item()
            total_steps += 1

            if is_train and (step + 1) % LOG_INTERVAL == 0:
                avg = total_loss / total_steps
                elapsed = time.time() - t_start
                log.info(
                    f"  [{phase}] step {step+1:>5}/{len(loader)}  "
                    f"loss={avg:.4f}  elapsed={elapsed:.1f}s"
                )

    avg_loss = total_loss / max(total_steps, 1)
    return avg_loss

# -----------------------------------------------------------------------------
# STEP 8 - FEATURE DIMENSION PROBE
# -----------------------------------------------------------------------------

def probe_feature_dims(cache_dir: Path, total: int):
    for i in range(min(total, 200)):
        p = cache_dir / f"{i:07d}.pt"
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        node_dim = d.x.shape[1] if d.x.dim() == 2 else None
        edge_dim = d.edge_attr.shape[1] if (
            hasattr(d, "edge_attr") and d.edge_attr is not None
            and d.edge_attr.dim() == 2
        ) else None
        if node_dim is not None:
            return node_dim, edge_dim
    raise RuntimeError("Could not find a valid graph in cache to probe feature dims.")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    log.info("=" * 72)
    log.info("StartNerve Intelligence - Titan V11 Training Pipeline")
    log.info("=" * 72)
    log.info(f"PyTorch {torch.__version__} | PyG {torch_geometric.__version__}")

    device = torch.device("cpu")
    log.info(f"Compute device: {device}")

    if not Path(PKL_PATH).exists():
        log.error(f"Source pickle not found: '{PKL_PATH}'")
        sys.exit(1)

    total_graphs = build_cache_if_needed(PKL_PATH, CACHE_DIR)
    log.info(f"Total valid graphs: {total_graphs:,}")

    rng     = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(total_graphs, generator=rng).tolist()
    n_train = int(total_graphs * TRAIN_FRAC)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]
    log.info(f"Split - Train: {len(train_idx):,}  |  Val: {len(val_idx):,}")
    
    train_ds = TitanDiskDataset(train_idx, CACHE_DIR)
    val_ds   = TitanDiskDataset(val_idx,   CACHE_DIR)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    node_feat_dim, edge_feat_dim = probe_feature_dims(CACHE_DIR, total_graphs)
    log.info(f"Node feature dim: {node_feat_dim}  |  Edge feature dim: {edge_feat_dim}")

    model = TitanV11(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    ckpt_dir = Path("titan_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_val_loss = math.inf

    log.info("-" * 72)
    log.info("Beginning training ...")
    log.info("-" * 72)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_t = time.time()
        log.info(f"\n[Epoch {epoch}/{NUM_EPOCHS}]  lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss = run_epoch(model, train_loader, optimizer, device, phase="train")
        val_loss   = run_epoch(model, val_loader,   optimizer, device, phase="val")

        scheduler.step()
        elapsed = time.time() - epoch_t

        log.info(
            f"  [SUMMARY] Train loss: {train_loss:.4f}  |  "
            f"Val loss: {val_loss:.4f}  |  "
            f"Time: {elapsed:.1f}s"
        )

        ckpt = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss":   val_loss,
        }
        torch.save(ckpt, ckpt_dir / "latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, ckpt_dir / "best.pt")
            log.info(f"  [NEW BEST] New best val loss: {best_val_loss:.4f} - saved to checkpoints/best.pt")

    log.info("=" * 72)
    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info("=" * 72)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main() 