"""
================================================================================
STARTNERVE INTELLIGENCE — TITAN V11 MASTER MODEL EVALUATION SUITE (FIXED)
================================================================================
Function: Streams validation tensor assets from disk, computes multi-task
          ROC-AUC, PR-AUC, and F1-scores, and identifies pathway vulnerabilities.
================================================================================
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

# Re-import definitions directly with fixed matching variable names
from train_v11_titan import TitanV11, TitanDiskDataset, CACHE_DIR, SEED

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = Path("titan_checkpoints/best.pt")
BATCH_SIZE       = 16
MISSING_LABEL    = -1.0

TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

def run_model_inference():
    device = torch.device("cpu")
    print("\n" + "="*80)
    print("        STARTNERVE INTELLIGENCE — TITAN V11 FORENSIC MODEL EVALUATION")
    print("="*80)
    print(f"Target Checkpoint Asset : {MODEL_CHECKPOINT}")
    
    if not MODEL_CHECKPOINT.exists():
        print(f"Error: Model parameter weights file '{MODEL_CHECKPOINT}' not found.")
        return

    # 1. Recover the exact validation index split using the deterministic seed
    total_graphs = len(list(CACHE_DIR.glob("*.pt")))
    if total_graphs == 0:
        print(f"Error: Cache directory '{CACHE_DIR}' is empty.")
        return

    rng = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(total_graphs, generator=rng).tolist()
    n_train = int(total_graphs * 0.8)
    val_idx = indices[n_train:]
    
    print(f"Mapped {len(val_idx):,} unseen verification graph profiles from disk arrays.")

    # 2. Instantiate the dynamic disk streaming loader
    val_ds = TitanDiskDataset(val_idx, CACHE_DIR)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Re-probe or define standard feature dimensionality channel bounds
    sample_file = next(CACHE_DIR.glob("*.pt"), None)
    if sample_file is None:
        print("Error: No cached tensor profiles found to probe feature dimensions.")
        return
        
    sample_data = torch.load(sample_file, weights_only=False)
    node_feat_dim = sample_data.x.shape[1]
    edge_feat_dim = sample_data.edge_attr.shape[1] if (hasattr(sample_data, "edge_attr") and sample_data.edge_attr is not None) else None

    # 4. Reconstruct the neural network and apply the optimized parameter states
    model = TitanV11(node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    
    # Handle both raw state dicts and nested checkpoint dictionary outputs smoothly
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Run inference streaming
    all_logits = []
    all_targets = []
    
    print("Running validation datasets completely through GNN streams...")
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch)
            targets = batch.y.view(-1, len(TASKS))
            
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Stack the batch matrices cleanly into global numpy matrices
    y_pred_probs = 1 / (1 + np.exp(-np.vstack(all_logits))) # Apply Sigmoid function to raw logits
    y_true = np.vstack(all_targets)

    # 6. Execute per-task evaluation calculus loops
    metrics_report = []

    print("\n" + "-"*80)
    print(f"  {'PATHWAY':<15} | {'VALID SAMPLES':<15} | {'ROC-AUC':<10} | {'PR-AUC':<10} | {'F1-SCORE':<10}")
    print("-"*80)

    for idx, task_name in enumerate(TASKS):
        task_true = y_true[:, idx]
        task_pred = y_pred_probs[:, idx]

        # Filter out the missing-label annotations (-1) completely
        valid_mask = task_true != MISSING_LABEL
        true_filtered = task_true[valid_mask]
        pred_filtered = task_pred[valid_mask]

        num_valid = int(np.sum(valid_mask))
        num_pos = int(np.sum(true_filtered == 1.0))
        num_neg = int(np.sum(true_filtered == 0.0))

        # Check for class presence to prevent metric singularities
        if num_pos == 0 or num_neg == 0 or num_valid < 10:
            print(f"  {task_name:<15} | {num_valid:<15} | {'[Omitted: Insufficient Label Diversity]':<35}")
            continue

        # Compute standard binary classifications
        roc_auc = roc_auc_score(true_filtered, pred_filtered)
        
        precision, recall, _ = precision_recall_curve(true_filtered, pred_filtered)
        pr_auc = auc(recall, precision)

        # Apply a default 0.5 binary classifier decision boundary to calculate the macro F1 metric
        binary_predictions = (pred_filtered >= 0.5).astype(float)
        f1 = f1_score(true_filtered, binary_predictions, zero_division=0)

        print(f"  {task_name:<15} | {num_valid:<15,} | {roc_auc:<10.4f} | {pr_auc:<10.4f} | {f1:<10.4f}")

        metrics_report.append({
            "Pathway": task_name,
            "Valid Samples": num_valid,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "F1-Score": f1
        })

    # Save summary report to local storage
    df_report = pd.DataFrame(metrics_report)
    df_report.to_csv("titan_v11_evaluation_report.csv", index=False)
    
    print("-"*80)
    print("COMPILATION SUCCESS: PERFORMANCE EVALUATION SUITE COMPLETE")
    print("Process matrix layer compiled to -> titan_v11_evaluation_report.csv")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_model_inference()