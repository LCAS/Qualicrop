import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
from fitting_bands import HybridBandSelectionModel


torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

patch_size = 4

# ====== Custom Dataset Class ======
class CustomSpectralDataset(Dataset):
    #def __init__(self, image, patch_size, mean, std):
    def __init__(self, image, patch_size, min, max):
        self.image = image
        self.patch_size = patch_size
        #self.mean = mean
        #self.std = std

        self.min = min
        self.max = max

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
selected_band_scores = np.load("top_10_scores.npy")
sorted_indices = np.argsort(selected_band_indices)
sorted_band_indices = selected_band_indices[sorted_indices]
sorted_band_scores = selected_band_scores[sorted_indices]
model = HybridBandSelectionModel(num_bands=400, latent_dim=8, num_heads=2, num_layers=2, num_selected_bands=18).to(device)
model.load_state_dict(torch.load("enhanced_test_less_data.pth"))
model.eval()

# ====== Function to process Images through Model ======
def process_image(image_path, patch_size=patch_size):
    """Process a single image and return results"""
    image = np.load(image_path)[:, :, 28:]  
    image = image[:, :, :400]
    
    dataset = CustomSpectralDataset(image, patch_size, min=0, max=1)  # Dummy min/max
    
    # Initialize reconstructed image tensor
    reconstructed_image = torch.zeros((image.shape[2], dataset.padded_height, dataset.padded_width), 
                                   dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            patch = dataset[idx].unsqueeze(0).to(device)
            
            # Get the original patch for reconstruction
            original_patch = patch.clone()
            
            selected_bands = original_patch[:, sorted_band_indices, :, :]
            selected_bands = selected_bands #* torch.tensor(sorted_band_scores, 
                                                        # dtype=torch.float32).to(device)[None, :, None, None]
            
            # Process through model
            encoded = model.lightweight_encoder(selected_bands.permute(0, 2, 3, 1).reshape(-1, selected_bands.shape[1]))
            reconstructed_patch = model.decoder(encoded)
            reconstructed_patch = reconstructed_patch.reshape(1, patch_size, patch_size, -1).permute(0, 3, 1, 2)
            
            # Place in output image
            i = (idx // (dataset.padded_width // patch_size)) * patch_size
            j = (idx % (dataset.padded_width // patch_size)) * patch_size
            reconstructed_image[:, i:i+patch_size, j:j+patch_size] = reconstructed_patch.squeeze(0)
    
    # Remove padding and transpose
    reconstructed_image = reconstructed_image[:, :image.shape[0], :image.shape[1]]
    reconstructed_image_unnormalized = reconstructed_image.cpu().numpy()
    reconstructed_image_unnormalized = np.transpose(reconstructed_image_unnormalized, (1, 2, 0))
    
    return image, reconstructed_image_unnormalized

# ====== Function to Plot process Images (Spectral comparison, Error Highlight, Error Map) ======
def plot_results(image, reconstructed, image_path, error_threshold=0.03):

    fig = plt.figure(figsize=(24, 6))
    ax2 = plt.subplot(1, 3, 1)
    original_spectral = image.mean(axis=(0, 1))
    reconstructed_spectral = reconstructed.mean(axis=(0, 1))
    ax2.plot(original_spectral, label="Original", color='blue', linewidth=2)
    ax2.plot(reconstructed_spectral, label="Reconstructed", color='red', linewidth=2)
    ax2.set_xlabel("Band Index")
    ax2.set_ylabel("Spectral Response")
    ax2.set_title("Spectral Response Comparison")
    ax2.legend()
    ax2.grid(True)
    
    ax3 = plt.subplot(1, 3, 2)
    error = np.abs(image - reconstructed).mean(axis=2)
    
    print("Reconstruction error = ", np.mean(error))
    print("Max error = ", np.max(error))
    error_mask = error > error_threshold
    red_overlay = np.zeros((image.shape[0], image.shape[1], 4))
    red_overlay[error_mask] = [1, 0, 0, 0.5]
    
    ax3.imshow(image[:, :, 0], cmap='gray')
    ax3.imshow(red_overlay)
    ax3.set_title(f"Error Highlights\n(Threshold = {error_threshold})")
    ax3.axis('off')
    
    # Error color map plot
    ax4 = plt.subplot(1, 3, 3)
    error_map = ax4.imshow(error, cmap='hot')  
    plt.colorbar(error_map, ax=ax4, label='Error magnitude')
    ax4.set_title("Pixel-wise Error Map")
    ax4.axis('off')
    
    plt.suptitle(os.path.basename(image_path))
    plt.tight_layout()
    plt.show()

# ====== Visualisation Implementation ======
directory_path = "/workspace/src/benchmarking_rgb/masked2"
error_threshold =  0.035 #Adjust as needed for the error highlight

for filename in sorted(os.listdir(directory_path)):
    if filename.endswith(".npy"):
        image_path = os.path.join(directory_path, filename)
        print(f"Processing {filename}...")
        
        try:
            image, reconstructed = process_image(image_path)
            plot_results(image, reconstructed, image_path, error_threshold)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")