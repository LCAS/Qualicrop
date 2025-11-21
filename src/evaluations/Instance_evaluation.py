import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score,
    precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)


#################    METRIC CALCULATION ##############################

def compute_metrics(all_scores, all_labels, title_suffix=""):
    thresholds = np.linspace(0.005, all_scores.max(), 500)
    f1s, precisions, recalls = [], [], []

    for t in thresholds:
        preds = (all_scores > t).astype(int)
        precisions.append(precision_score(all_labels, preds, zero_division=0))
        recalls.append(recall_score(all_labels, preds, zero_division=0))
        f1s.append(f1_score(all_labels, preds, zero_division=0))
        
    f1s = np.array(f1s)
    best_idx = np.nanargmax(f1s)
    best_threshold = thresholds[best_idx]

    preds_best = (all_scores > best_threshold).astype(int)
    best_precision = precision_score(all_labels, preds_best, zero_division=0)
    best_recall = recall_score(all_labels, preds_best, zero_division=0)
    best_f1 = f1_score(all_labels, preds_best, zero_division=0)
    best_mcc = matthews_corrcoef(all_labels, preds_best)
    best_kappa = cohen_kappa_score(all_labels, preds_best)

    # PR Curve
    precisions_curve, recalls_curve, _ = precision_recall_curve(all_labels, all_scores)
    pr_auc = auc(recalls_curve, precisions_curve)

    print("\n RESULTS", title_suffix)
    print("-------------------------------------")
    print(f"Best Threshold : {best_threshold:.4f}")
    print(f"Precision      : {best_precision:.4f}")
    print(f"Recall         : {best_recall:.4f}")
    print(f"F1-score       : {best_f1:.4f}")
    print(f"MCC            : {best_mcc:.4f}")
    print(f"Kappa          : {best_kappa:.4f}")
    print(f"PR_AUC         : {pr_auc:.4f}")
    print("\n")

    # --- Plot PR Curve ---
    plt.figure(figsize=(8,6))
    plt.plot(recalls_curve, precisions_curve, label=f'PR Curve (AUC={pr_auc:.4f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve {title_suffix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, preds_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalous"])
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("Groundtruth labels")
    ax.set_yticklabels(["Normal", "Anomalous"], rotation=90, va='center')
    plt.title(f"Confusion Matrix (Threshold={best_threshold:.4f})")
    plt.show()



################### RATIO-BASED EVALUATION (.mat) ###########################

def load_maps(file_path, method):
    with h5py.File(file_path, 'r') as f:
        return np.array(f[method])


def compute_anomaly_ratios(maps):
    ratios = []
    for i in range(maps.shape[0]):
        img = maps[i]
        #valid_mask = img > 0  # to ignore background
        valid_mask = img >= 0
        if np.sum(valid_mask) == 0:
            ratios.append(0)
        else:
            anomalous = np.sum(img[valid_mask] == 2)
            ratios.append(anomalous / np.sum(valid_mask))
    return np.array(ratios)



#####################   RATIO-BASED SETTINGS(.mat)   ###################################
## Editable 
def run_ratio_mode():
    file_normal = '/workspace/src/class_maps_norm.mat'
    file_anomalous = '/workspace/src/class_maps_anom.mat'
    method = 'knn_maps'   # options: 'gt_maps', 'knn_maps', 'rf_maps', 'svm_rbf_maps'

    print(f"\nRunning ratio-based evaluation using method: {method}")

    normal_maps = load_maps(file_normal, method)
    anomalous_maps = load_maps(file_anomalous, method)

    normal_ratios = compute_anomaly_ratios(normal_maps)
    anomalous_ratios = compute_anomaly_ratios(anomalous_maps)

    all_scores = np.concatenate([normal_ratios, anomalous_ratios])
    all_labels = np.array([0] * len(normal_ratios) + [1] * len(anomalous_ratios))

    compute_metrics(all_scores, all_labels, title_suffix="(Instance-Ratio Mode)")



###################     SCORE-BASED SETTINGS (.npz)  ##########################
## Editable 
def run_score_mode():
    path = "/workspace/Qualicrop_report/scores/dataset2/trans_scores.npz"

    print("\nRunning score-based evaluation:", path)

    data = np.load(path)
    all_scores = data["scores"]
    all_labels = data["labels"]

    compute_metrics(all_scores, all_labels, title_suffix="(Instance level Score-Mode)")



#######################    MAIN EXECUTION  ########################################

# Choose the mode:
# MODE = "ratio"   # uses .mat and ratio method
# MODE = "score"   # uses .npz and score method

MODE = "ratio"   

if __name__ == "__main__":
    if MODE == "ratio":
        run_ratio_mode()
    elif MODE == "score":
        run_score_mode()
    else:
        raise ValueError("MODE must be 'ratio' or 'score'")
