import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


def load_maps(file_path, method):
    with h5py.File(file_path, 'r') as f:
        return np.array(f[method])

def compute_anomaly_ratios(maps):
    ratios = []
    for i in range(maps.shape[0]):
        img = maps[i]
        valid_mask = img >= 0  
        if np.sum(valid_mask) == 0:
            ratios.append(0)
        else:
            anomalous = np.sum(img[valid_mask] == 2)
            ratios.append(anomalous / np.sum(valid_mask))
    return np.array(ratios)

def plot_pr_curves(models_methods, file_normal, file_anom):
    plt.figure(figsize=(8,6))
    
    for model_name, method in models_methods.items():
      
        normal_maps = load_maps(file_normal, method)
        anomalous_maps = load_maps(file_anom, method)
        
       
        normal_ratios = compute_anomaly_ratios(normal_maps)
        anomalous_ratios = compute_anomaly_ratios(anomalous_maps)
        
        
        all_scores = np.concatenate([normal_ratios, anomalous_ratios])
        all_labels = np.array([0]*len(normal_ratios) + [1]*len(anomalous_ratios))
        
        
        precisions, recalls, _ = precision_recall_curve(all_labels, all_scores)
        pr_auc = auc(recalls, precisions)
        plt.plot(recalls, precisions, label=f'{model_name} (AUC={pr_auc:.4f})')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Ratio-Based, .mat)")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

################### MAIN EXECUTION ############

if __name__ == "__main__":

    file_normal = "/workspace/src/class_maps_norm.mat"
    file_anom = "/workspace/src/class_maps_anom.mat"

    models_methods = {
        "KNN": "knn_maps",
        "RF": "rf_maps",
        "SVM-RBF": "svm_rbf_maps"
        # (CNN, Transformer ). just add a ,
    }

    plot_pr_curves(models_methods, file_normal, file_anom)
