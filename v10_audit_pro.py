import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

from train_v10_ironclad import ToxGAT_V10, mol_to_graph_v10

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "startnerve_v10_best.pt"
DATA_PATH    = "startnerve_master_v8_12task.csv"
TEST_FRAC    = 0.2
RANDOM_SEED  = 42
MC_PASSES    = 30        # number of forward passes for MC Dropout uncertainty
ELITE_THRESH = 0.82      # AUROC threshold for ⭐ ELITE tag
BATCH_SIZE   = 64
# ─────────────────────────────────────────────────────────────────────────────


def mc_dropout_predict(model, loader, device, n_passes=MC_PASSES):
    """
    Run inference with dropout ACTIVE (MC Dropout mode).
    Returns mean prediction and epistemic uncertainty (std) per molecule per task.
    """
    model.train()   # keeps dropout layers active
    all_pass_preds = []

    with torch.no_grad():
        for _ in range(n_passes):
            batch_preds = []
            for data in loader:
                data = data.to(device)
                out = torch.sigmoid(model(data))
                batch_preds.append(out.cpu().numpy())
            all_pass_preds.append(np.vstack(batch_preds))

    stacked = np.stack(all_pass_preds, axis=0)   # shape: (n_passes, n_mols, n_tasks)
    mean_preds = stacked.mean(axis=0)             # shape: (n_mols, n_tasks)
    uncertainty = stacked.std(axis=0)             # epistemic uncertainty
    return mean_preds, uncertainty


def deterministic_predict(model, loader, device):
    """Standard eval-mode inference (no dropout)."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = torch.sigmoid(model(data))
            all_preds.append(out.cpu().numpy())

    return np.vstack(all_preds), None


def run_v10_audit():
    device = torch.device("cpu")
    print("=" * 55)
    print("  STARTNERVE V10 — IRONCLAD PERFORMANCE AUDIT")
    print("=" * 55)

    # ── Load model ────────────────────────────────────────────────────────────
    model = ToxGAT_V10().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"\n✅ Weights loaded from: {MODEL_PATH}")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    tasks = [c for c in df.columns if c != 'SMILES']
    print(f"✅ Dataset loaded: {len(df)} molecules | {len(tasks)} tasks")

    # ── Holdout split ─────────────────────────────────────────────────────────
    test_df = df.sample(frac=TEST_FRAC, random_state=RANDOM_SEED)
    print(f"✅ Test holdout: {len(test_df)} molecules ({int(TEST_FRAC*100)}% | seed={RANDOM_SEED})")

    # ── Build graphs ──────────────────────────────────────────────────────────
    test_graphs = []
    skipped = 0
    for _, row in test_df.iterrows():
        g = mol_to_graph_v10(row['SMILES'], row[tasks].fillna(-1).values.tolist())
        if g:
            test_graphs.append(g)
        else:
            skipped += 1

    print(f"✅ Graphs built: {len(test_graphs)} valid | {skipped} skipped (invalid SMILES)")

    loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # ── Collect targets ───────────────────────────────────────────────────────
    all_targets = []
    for data in loader:
        all_targets.append(data.y.numpy())
    targets = np.vstack(all_targets)    # shape: (n_mols, n_tasks)

    # ── Run both inference modes ───────────────────────────────────────────────
    print(f"\n🔬 Running deterministic inference...")
    det_preds, _ = deterministic_predict(model, loader, device)

    print(f"🔬 Running MC Dropout inference ({MC_PASSES} passes)...")
    mc_preds, mc_uncertainty = mc_dropout_predict(model, loader, device, MC_PASSES)

    # ── Per-pathway AUROC ─────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  {'TASK':<20} {'DET AUROC':>10} {'MC AUROC':>10} {'MEAN UNC':>10}")
    print(f"{'─'*55}")

    det_scores, mc_scores = [], []

    for i, task in enumerate(tasks):
        mask = targets[:, i] != -1
        n_valid = mask.sum()

        if n_valid < 10:
            print(f"  {task:<20} {'SKIP (n<10)':>10}")
            continue

        det_auc = roc_auc_score(targets[mask, i], det_preds[mask, i])
        mc_auc  = roc_auc_score(targets[mask, i], mc_preds[mask, i])
        mean_unc = mc_uncertainty[mask, i].mean()

        det_scores.append(det_auc)
        mc_scores.append(mc_auc)

        elite_tag = " ⭐ ELITE" if mc_auc >= ELITE_THRESH else ""
        print(f"  {task:<20} {det_auc:>10.3f} {mc_auc:>10.3f} {mean_unc:>10.4f}{elite_tag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'─'*55}")
    if det_scores:
        print(f"  {'MEAN (all tasks)':<20} {np.mean(det_scores):>10.3f} {np.mean(mc_scores):>10.3f}")
        print(f"  {'MIN / MAX':<20} {min(det_scores):.3f}/{max(det_scores):.3f} {min(mc_scores):.3f}/{max(mc_scores):.3f}")
        elite_count = sum(1 for s in mc_scores if s >= ELITE_THRESH)
        print(f"\n  ⭐ ELITE pathways (AUROC ≥ {ELITE_THRESH}): {elite_count}/{len(mc_scores)}")
        print(f"\n  ONE-LINE PITCH NUMBER:")
        print(f"  → StartNerve V10 achieves mean AUROC of {np.mean(mc_scores):.3f} across {len(mc_scores)} Tox21 pathways")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_v10_audit()