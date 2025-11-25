import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
)




#################    METRIC CALCULATION ##############################

def compute_metrics(y_true, y_pred, class_mode="anomaly"):
   
    #y_scores = y_pred.astype(float)

    # Metrics
    if class_mode == "anomaly":
        print("computing for anomaly")
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    elif class_mode == "macro":
        print("computing for macro")
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1        = f1_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        raise ValueError("class_mode must be 'anomaly' or 'macro'")

    mcc  = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    print("\n========================= RESULTS", class_mode, "=========================")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"Kappa     : {kappa:.4f}")
    print("==========================================================")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_rel = cm.astype(float) / cm.sum() * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rel,
                                  display_labels=["Normal", "Anomalous"])
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, values_format=".3f")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("Groundtruth labels")
    ax.set_yticklabels(["Normal", "Anomalous"], rotation=90, va='center')
    plt.title("Normalised Confusion Matrix (%)")
    plt.show()




#################   Mask based ##############################

def load_pred_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)


def load_groundtruth(gt_path):
    mapping = {0:0, 30:0, 60:0, 90:1, 120:1, 150:1, 210:1}
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(gt_path)
    binary = np.zeros_like(gt, dtype=np.uint8)
    for k, v in mapping.items():
        binary[gt == k] = v
    return binary


def evaluate_masks(pred_dir, gt_dir):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])
    
    all_labels, all_preds = [], []
    
    for p, g in zip(pred_files, gt_files):
        pred_mask = load_pred_mask(os.path.join(pred_dir, p))
        gt_mask = load_groundtruth(os.path.join(gt_dir, g))  # mapped already
        
       
        orig_gt = cv2.imread(os.path.join(gt_dir, g), cv2.IMREAD_GRAYSCALE)
        valid_mask = orig_gt != 0 # Ignore BG
        #valid_mask = orig_gt >= 0 
        
        all_labels.append(gt_mask[valid_mask].flatten())
        all_preds.append(pred_mask[valid_mask].flatten())
    
    return np.concatenate(all_labels), np.concatenate(all_preds)


#################   .Mat Based ##############################


def load_maps(file_path, method):
    with h5py.File(file_path, "r") as f:
        return np.array(f[method])


def convert_to_binary(gt_map, pred_map):
    valid = gt_map != 0 # ignore background
    gt_bin = np.zeros_like(gt_map, dtype=np.uint8)
    gt_bin[gt_map == 1] = 0
    gt_bin[gt_map == 2] = 1
    pred_bin = np.zeros_like(pred_map, dtype=np.uint8)
    pred_bin[pred_map == 1] = 0
    pred_bin[pred_map == 2] = 1
    y_true = gt_bin[valid].flatten()
    y_pred = pred_bin[valid].flatten()
    return y_true, y_pred





###################    DATA LOADING MODES  #####################################
# Editable 
def run_mat_mode():
    mat_file="/workspace/src/class_maps2.mat"
    
    gt_key="gt_maps" # Ground Truth
    pred_method="knn_maps" #'knn_maps', 'rf_maps', 'svm_rbf_maps'

    print("Loading maps from .mat file...")
    gt_maps = load_maps(mat_file, gt_key)
    pred_maps = load_maps(mat_file, pred_method)
    y_true, y_pred = convert_to_binary(gt_maps, pred_maps)
    return y_true, y_pred


def run_masks_mode():
    #Editable
    gt_dir="/workspace/src/work_masking/testing2/masks"         #GT
    pred_dir="/workspace/Qualicrop_report/mask_gen/testing2/trans_dataset2"
    
    
    print("Loading predicted masks & groundtruth images...")
    y_true, y_pred = evaluate_masks(pred_dir, gt_dir)
    return y_true, y_pred

#######################    MAIN EXECUTION  ########################################
# MODE = mat, mask
# TYPE = anomaly, macro

MODE = "mask"  
TYPE = "anomaly" 

if __name__ == "__main__":
    if MODE == "mat":
       y_true, y_pred = run_mat_mode()
    elif MODE == "mask":
        y_true, y_pred = run_masks_mode()
    else:
        raise ValueError("MODE must be 'mat' or 'mask")
    print("Computing metrics...")
    compute_metrics(y_true, y_pred, TYPE)
    
