import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from fitting_bands import HybridBandSelectionModel
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Config  ======

# Uncomment for Validation Set
#normal_dir = "/workspace/src/new_set/val"
#anomalous_dir = "/workspace/src/benchmarking_rgb/val"

normal_dir = "/workspace/src/new_set/test2"
anomalous_dir = "/workspace/src/benchmarking_rgb/masked2"
patch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== Custom Dataset Class ======
class CustomSpectralDataset(Dataset):
    def __init__(self, image, patch_size, min, max):
        self.image = image
        self.patch_size = patch_size

        # Compute padded dimensions
        self.height, self.width, _ = image.shape
        self.padded_height = ((self.height - 1) // patch_size + 1) * patch_size
        self.padded_width = ((self.width - 1) // patch_size + 1) * patch_size

        # Pad the image
        pad_h = self.padded_height - self.height
        pad_w = self.padded_width - self.width
        self.padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')  # Shape: (padded_height, padded_width, num_bands)

    def __len__(self):
        return (self.padded_height // self.patch_size) * (self.padded_width // self.patch_size)

    def __getitem__(self, idx):
        # Compute patch coordinates
        i = (idx // (self.padded_width // self.patch_size)) * self.patch_size
        j = (idx % (self.padded_width // self.patch_size)) * self.patch_size

        # Extract the patch
        patch = self.padded_image[i:i+self.patch_size, j:j+self.patch_size]  # Shape: (patch_size, patch_size, num_bands)

        # Convert to tensor and permute dimensions
        patch = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1)  # Shape: (num_bands, patch_size, patch_size)
        return patch

# ====== Load Selected Bands, Scores and Model ======
selected_band_indices = np.load("test_bands.npy")
selected_band_scores = np.load("test_scores.npy")
sorted_indices = np.argsort(selected_band_indices)
sorted_band_indices = selected_band_indices[sorted_indices]
sorted_band_scores = selected_band_scores[sorted_indices]

model = HybridBandSelectionModel(num_bands=400, latent_dim=8, num_heads=2, num_layers=2, num_selected_bands=18).to(device)
model.load_state_dict(torch.load("enhanced_test_less_data.pth"))
model.eval()

# ====== Fuction to Compute the MAE (reconstruction loss)  ======
def compute_mae_loss(image_path):
    """Compute MAE loss for a single image"""
    image = np.load(image_path)[:, :, 28:] 
    image = image[:, :, :400]
    
    dataset = CustomSpectralDataset(image, patch_size, min=0, max=1)
    reconstructed_image = torch.zeros((image.shape[2], dataset.padded_height, dataset.padded_width), 
                                   dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            patch = dataset[idx].unsqueeze(0).to(device)
            original_patch = patch.clone()
            
            selected_bands = original_patch[:, sorted_band_indices, :, :]
            selected_bands = selected_bands #* torch.tensor(sorted_band_scores, 
                                                       # dtype=torch.float32).to(device)[None, :, None, None]
            
            encoded = model.lightweight_encoder(
                selected_bands.permute(0, 2, 3, 1).reshape(-1, selected_bands.shape[1]))
            reconstructed_patch = model.decoder(encoded)
            reconstructed_patch = reconstructed_patch.reshape(1, patch_size, patch_size, -1).permute(0, 3, 1, 2)
            
            i = (idx // (dataset.padded_width // patch_size)) * patch_size
            j = (idx % (dataset.padded_width // patch_size)) * patch_size
            reconstructed_image[:, i:i+patch_size, j:j+patch_size] = reconstructed_patch.squeeze(0)
    
    reconstructed_image = reconstructed_image[:, :image.shape[0], :image.shape[1]]
    reconstructed = reconstructed_image.cpu().numpy().transpose(1, 2, 0)
    
    return np.mean(np.abs(image - reconstructed))

# ====== Computing MAE, plotting and printing result ======
normal_mae = []
anomalous_mae = []

print("Processing normal images...")
for filename in tqdm(sorted(os.listdir(normal_dir))):
    if filename.endswith(".npy"):
        try:
            mae = compute_mae_loss(os.path.join(normal_dir, filename))
            normal_mae.append(mae)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("\nProcessing anomalous images...")
for filename in tqdm(sorted(os.listdir(anomalous_dir))):
    if filename.endswith(".npy"):
        try:
            mae = compute_mae_loss(os.path.join(anomalous_dir, filename))
            anomalous_mae.append(mae)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# Create index arrays
normal_indices = np.arange(len(normal_mae))
anomalous_indices = np.arange(len(anomalous_mae))

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(normal_indices, normal_mae, c='green', alpha=0.6, label='Normal')
plt.scatter(anomalous_indices, anomalous_mae, c='red', alpha=0.6, label='Anomalous')
plt.xlabel('Sample Index')
plt.ylabel('MAE Loss')
plt.title('Reconstruction MAE Loss by Sample Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print statistics
print("\nNormal images stats:")
print(f"MAE mean: {np.mean(normal_mae):.4f}")
print("\nAnomalous images stats:")
print(f"MAE mean: {np.mean(anomalous_mae):.4f}")



# ====== Metric Calculations ======

# 1. F1 Curve (applies to Validation Set Only but leaving in test for visualisation)

def plot_f1_curve(normal_scores, anomalous_scores):
 

    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomalous_scores))])
    y_scores = np.concatenate([normal_scores, anomalous_scores])
    
    # Calculate F1 across thresholds
    thresholds = np.linspace(min(y_scores), max(y_scores), 100)
    f1_scores = [f1_score(y_true, y_scores > t) for t in thresholds]
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    max_f1 = f1_scores[optimal_idx]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Main F1 curve
    plt.plot(thresholds, f1_scores, label=f'F1 Score (Max = {max_f1:.4f})',linewidth=2, color='#1f77b4')
    # Optimal threshold marker
    plt.scatter(optimal_threshold, max_f1, color='red', s=100,zorder=5,label=f'Optimal Threshold = {optimal_threshold:.4f}')
    # Threshold line
    plt.axvline(optimal_threshold, color='red', linestyle=':', alpha=0.7)

    # Formatting
    plt.xlabel('Reconstruction Loss', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs. Reconstruction Loss', fontsize=14, pad=20)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return optimal_threshold, max_f1




# 2.ROC and ROC_AUC
def plot_roc_auc(normal_scores, anomalous_scores):

    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomalous_scores))])
    y_scores = np.concatenate([normal_scores, anomalous_scores])
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    

    plt.figure(figsize=(8, 6))
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    # Diagonal reference line
    #plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return roc_auc



# 3. PR and PR_AUC
def plot_pr_curve(normal_scores, anomalous_scores):
    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomalous_scores))])
    y_scores = np.concatenate([normal_scores, anomalous_scores])
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,label=f'PR Curve (AUC = {pr_auc:.4f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()




# 4. Evaluation at the pre-determind threshold (f1, Confusion Matrix)
def evaluate_at_threshold(normal_scores, anomalous_scores, threshold):

    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomalous_scores))])
    y_scores = np.concatenate([normal_scores, anomalous_scores])
    y_pred = (y_scores > threshold).astype(int)
    
    # Compute F1 score
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score at threshold {threshold:.4f}: {f1:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['Normal', 'Anomalous'], yticklabels=['Normal', 'Anomalous'])
    plt.title(f'Confusion Matrix (Threshold = {threshold:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return f1, cm

# ====== Implemention ======

# For Validation Only
optimal_thresh, best_f1 = plot_f1_curve(normal_mae, anomalous_mae) 
print(f"\nOptimal Threshold: {optimal_thresh:.4f}")
print(f"Maximum F1 Score: {best_f1:.4f}")
#####################################################################

# Testing 
auc_score = plot_roc_auc(normal_mae, anomalous_mae)
print(f"ROC AUC Score: {auc_score:.4f}")
plot_pr_curve(normal_mae, anomalous_mae)
optimal_threshold = 0.0165   # Hard code this with the obtained optimal threshold using the validation set and not testing set
f1, cm = evaluate_at_threshold(normal_mae, anomalous_mae, optimal_threshold)