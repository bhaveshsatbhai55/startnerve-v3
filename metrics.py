"""
StartNerve Intelligence — Phase 3, Task 1
======================================================
Institutional Evaluation Metrics Patch: Validating True 
Sensitivity, Precision, and False Negative Rates.
"""

import numpy as np

def calculate_enterprise_metrics(y_true, y_pred_scores, threshold=0.40):
    """
    Computes professional clinical screening metrics from raw model scores.
    """
    # Convert continuous model probabilities to binary predictions based on safety threshold
    y_pred_binary = (y_pred_scores >= threshold).astype(int)
    y_true = np.array(y_true)
    
    # Calculate Confusion Matrix elements
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    # Calculate Institutional Metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Recall / TPR
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # TNR
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0    # PPV
    false_negative_rate = FN / (TP + FN) if (TP + FN) > 0 else 0.0 # FNR (The Risk Metric)
    
    print(f"\n{'='*60}")
    print("        STARTNERVE V11 INSTITUTIONAL PERFORMANCE PATCH")
    print(f"{'='*60}")
    print(f"  [CONFUSION MATRIX]")
    print(f"    True Positives (Correctly Flagged Hazard) : {TP}")
    print(f"    True Negatives (Correctly Cleared Safe)   : {TN}")
    print(f"    False Positives (False Alarm Raised)      : {FP}")
    print(f"    False Negatives (Missed Hazard - CRITICAL): {FN}")
    print(f"  {'-'*50}")
    print(f"  [METRIC ANALYSIS] (Threshold: {threshold:.2f})")
    print(f"    🚀 Sensitivity (True Positive Rate) : {sensitivity * 100:.2f}%")
    print(f"    🛡️  Specificity (True Negative Rate) : {specificity * 100:.2f}%")
    print(f"    🎯 Precision (Positive Predictive)   : {precision * 100:.2f}%")
    print(f"    ⚠️  FALSE NEGATIVE RATE (Risk Slip)  : {false_negative_rate * 100:.2f}%")
    print(f"{'='*60}\n")
    
    if false_negative_rate > 0.20:
        print("  [CTO ADVISORY]: False Negative Rate is above the 20% institutional barrier.")
        print("                  Consider lowering the RISK_THRESHOLD to tighten guardrails.\n")
    else:
        print("  [VALIDATION PASS]: System risk slip profile is within enterprise tolerances.\n")

if __name__ == "__main__":
    # Simulated validation run matching your Gold Standard / Layer 2 local outcomes
    # 1 = Truly Toxic Compound, 0 = Confirmed Safe OTC Compound
    ground_truth = [1, 1, 1, 1, 1, 0, 0] # e.g., Bromfenac, Benoxaprofen, BPA, Aniline, Nimesulide, Aspirin, Caffeine
    model_scores = np.array([0.4011, 0.7052, 0.8807, 0.5767, 0.1399, 0.0210, 0.0150])
    
    calculate_enterprise_metrics(ground_truth, model_scores, threshold=0.40)