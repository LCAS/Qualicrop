import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import mode
import random


torch.backends.cuda.enable_mem_efficient_sdp(False)

# Randomization
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#  GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ====== Model Architecture  ======
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_bands, latent_dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_bands))

        nn.init.xavier_uniform_(self.positional_encoding)
    def forward(self, x):
        return x + self.positional_encoding

class HybridBandSelectionModel(nn.Module):
    def __init__(self, num_bands, latent_dim, num_heads, num_layers, num_selected_bands):
        super(HybridBandSelectionModel, self).__init__()
        self.num_selected_bands = num_selected_bands

        self.band_importance = nn.Parameter(torch.zeros(num_bands))
       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_bands, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=0.15
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Initialize Transformer weights
        self._init_transformer_weights()

        # Positional Encoding 
        self.positional_encoding = LearnablePositionalEncoding(num_bands, num_bands)

        # Lightweight Encoder
        self.lightweight_encoder = nn.Sequential(
            nn.Linear(num_selected_bands, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.Sigmoid()
            
        )
        
        # Initialize Lightweight Encoder weights
        self._init_linear_weights(self.lightweight_encoder)

        # Lightweight Encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_bands),
            nn.Sigmoid()
        
        )
        
        # Initialize Decoder weights
        self._init_linear_weights(self.decoder)

    def _init_transformer_weights(self):
        """Initialize Transformer weights with Xavier/Glorot initialization."""
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_linear_weights(self, sequential_model):
        """Initialize Linear layer weights with Xavier/Glorot initialization."""
        for layer in sequential_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    
    def forward(self, x, x_original):
        batch_size, num_bands, height, width = x.shape

        # Flatten spatial dimensions
        x = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, num_bands)  # Shape: (batch_size * height * width, num_bands)
        
        # Transformer Encoder (with positional encoding)
        x = self.positional_encoding(x.unsqueeze(0)).squeeze(0)  # Add positional encoding
        encoded_bands = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)  # Shape: (batch_size * height * width, num_bands)

        attention_scores = encoded_bands * torch.sigmoid(self.band_importance)

        # Select top-k bands based on attention scores
        top_k_scores, top_k_indices = torch.topk(attention_scores, k=self.num_selected_bands, dim=1)  # Shape: (batch_size * height * width, num_selected_bands)

        # Sort the selected bands by their indices
        sorted_indices = torch.argsort(top_k_indices, dim=1)  # Shape: (batch_size * height * width, num_selected_bands)
        sorted_top_k_indices = torch.gather(top_k_indices, dim=1, index=sorted_indices)  # Shape: (batch_size * height * width, num_selected_bands)
        sorted_top_k_scores = torch.gather(top_k_scores, dim=1, index=sorted_indices)  # Shape: (batch_size * height * width, num_selected_bands)

        # Crop selected bands from the original input
        selected_bands = torch.gather(x_original.permute(0, 2, 3, 1).reshape(batch_size * height * width, num_bands), dim=1, index=sorted_top_k_indices)  # Shape: (batch_size * height * width, num_selected_bands)

        # Lightweight Encoder 
        encoded_selected_bands = self.lightweight_encoder(selected_bands)  # Shape: (batch_size * height * width, latent_dim)

        # Decoder (reconstruct full spectrum)
        reconstructed = self.decoder(encoded_selected_bands)  # Shape: (batch_size * height * width, num_bands)

        # Reshape output back to original spatial dimensions
        reconstructed = reconstructed.reshape(batch_size, height, width, num_bands).permute(0, 3, 1, 2)  # Shape: (batch_size, num_bands, height, width)

        return reconstructed, sorted_top_k_indices, sorted_top_k_scores




# ====== Loss Function  ======
def loss_function(reconstructed, target):
    reconstruction_loss = F.l1_loss(reconstructed, target)
    total_loss = reconstruction_loss 
    return total_loss

class SpectralDataset(Dataset):
    def __init__(self, file_paths, patch_size, start_band=28):
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.start_band = start_band  

        # Compute the maximum height and width in the dataset
        self.max_height = max([np.load(fp).shape[0] for fp in file_paths])
        self.max_width = max([np.load(fp).shape[1] for fp in file_paths])

        # Ensure the padded size is divisible by patch_size
        self.padded_height = ((self.max_height - 1) // patch_size + 1) * patch_size
        self.padded_width = ((self.max_width - 1) // patch_size + 1) * patch_size 

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load .npy file
        image = np.load(self.file_paths[idx])  # Shape: (height, width, num_bands)

        # Use only bands from the 50th band onwards
        image = image[:, :, self.start_band:]  # Shape: (height, width, num_bands - start_band)
        image = image[:, :, :400]

    
        # Pad the image to the common size
        h, w, _ = image.shape
        pad_h = self.padded_height - h
        pad_w = self.padded_width - w
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')  # Shape: (padded_height, padded_width, num_bands - start_band)

        # Extract patches
        patches = []
        for i in range(0, self.padded_height, self.patch_size):
            for j in range(0, self.padded_width, self.patch_size):
                patch = padded_image[i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        patches = np.stack(patches)  # Shape: (num_patches, patch_size, patch_size, num_bands - start_band)

        # Convert to tensor
        patches = torch.tensor(patches, dtype=torch.float32).permute(0, 3, 1, 2)  # Shape: (num_patches, num_bands - start_band, patch_size, patch_size)
        return patches

def custom_collate_fn(batch):
    # Concatenate all patches into a single tensor
    patches = torch.cat(batch, dim=0)  # Shape: (total_patches, num_bands - start_band, patch_size, patch_size)
    return patches

# ====== Parameters =====

num_bands = 400  # Number of spectral bands
latent_dim =8   # Latent dimension (Lightweigth autoencoder)
num_heads = 8  # Number of attention heads
num_layers = 2  #  Number of transformer layers
num_selected_bands = 18  # Number of selected bands
patch_size = 4  # Patch size (similar to batch during training)
batch_size = 1  # Batch size (batch of whole images)

if __name__ == '__main__':

    # ====== Initialisation and loading =====

    data_dir = "/workspace/src/new_set/norm2"
    file_paths = glob.glob(f"{data_dir}/*.npy")
    dataset = SpectralDataset(file_paths, patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    model = HybridBandSelectionModel(num_bands=num_bands, latent_dim=latent_dim, num_heads=num_bands, num_layers=num_layers, num_selected_bands=num_selected_bands).to(device)

    model.load_state_dict(torch.load("enhanced_test_less_data.pth"))
    model.eval()

    # Initialize lists to store band indices and scores
    all_band_indices = []
    all_band_scores = []

    # ====== Fitting Bands =====

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)  # Move batch to GPU
            _, band_indices, band_scores = model(batch, batch)  # Get band indices and scores

            # Store band indices and scores
            all_band_indices.append(band_indices.cpu().numpy())
            all_band_scores.append(band_scores.cpu().numpy())

    # Concatenate results across all batches
    all_band_indices = np.concatenate(all_band_indices, axis=0)  # Shape: (total_patches, num_selected_bands)
    all_band_scores = np.concatenate(all_band_scores, axis=0)  # Shape: (total_patches, num_selected_bands)

    # # Compute the top 10 bands across the entire dataset
    unique_band_indices, counts = np.unique(all_band_indices, return_counts=True)
    top_bands = unique_band_indices[np.argsort(-counts)][:num_selected_bands]  # Top 10 most frequently selected bands

    most_frequent = True # Other Metric is Score 

    if most_frequent:
        print('most frequent')
        top_bands = unique_band_indices[np.argsort(-counts)][:num_selected_bands]
        top_scores = []
        
        # For each of the top 10 bands, find their scores
        for band in top_bands:
            mask = (all_band_indices == band)
            
            scores = all_band_scores[mask]
            
            avg_score = scores.mean()
            top_scores.append(avg_score)
        
        top_scores = np.array(top_scores)

    else:
        
        print('highest score')
        band_scores = []
        for band in unique_band_indices:
            mask = (all_band_indices == band)
            scores = all_band_scores[mask]
            avg_score = scores.mean()
            band_scores.append(avg_score)
        
        sorted_indices = np.argsort(-np.array(band_scores))
        top_bands = unique_band_indices[sorted_indices][:num_selected_bands]
        top_scores = np.array(band_scores)[sorted_indices][:num_selected_bands]


    # ====== Saving Bands and Scores =====

    #np.save("test_bands.npy", top_bands)
    #np.save("test_scores.npy", top_scores)

    print("Top 10 bands:", top_bands)
    print("Corresponding scores:", top_scores)


    # ====== Plotting of Band Distributions  =====


    all_band_indices = np.concatenate(all_band_indices, axis=0)  # Shape: (total_patches * num_selected_bands,)
    all_band_scores = np.concatenate(all_band_scores, axis=0)  # Shape: (total_patches * num_selected_bands,)

    # 1. Scores
    band_scores_dict = {band: [] for band in range(num_bands)}

    for band, score in zip(all_band_indices, all_band_scores):
        band_scores_dict[band].append(score)

    band_avg_scores = {band: np.mean(scores) for band, scores in band_scores_dict.items()}

    bands = list(band_avg_scores.keys())
    avg_scores = list(band_avg_scores.values())


    plt.figure(figsize=(15, 6))
    plt.bar(bands, avg_scores, color='blue', label="All Bands")
    plt.bar(top_bands, [band_avg_scores[band] for band in top_bands], color='red', label="Top 10 Bands")
    plt.xlabel("Band Index")
    plt.ylabel("Average Score")
    plt.title("Band Scores with Top 10 Bands Highlighted")
    plt.legend()
    plt.show()

    # 2. Frequency
    band_frequencies = {band: counts[i] for i, band in enumerate(unique_band_indices)}
    bands_freq = list(band_frequencies.keys())
    frequencies = list(band_frequencies.values())
    plt.figure(figsize=(15, 6))
    plt.bar(bands_freq, frequencies, color='green', label="Frequency of Selection")
    plt.bar(top_bands, [band_frequencies[band] for band in top_bands], color='orange', label="Top 10 Bands by Frequency")
    plt.xlabel("Band Index")
    plt.ylabel("Frequency of Selection")
    plt.title("Frequency of Band Selection with Top 10 Bands Highlighted")
    plt.legend()
    plt.show()

    # ====== Plotting a Random Patch for evaluation  =====

    random_patch_idx = random.randint(0, len(dataset) - 1)
    random_patch = dataset[random_patch_idx]
    random_patch = random_patch[25].unsqueeze(0).to(device) 
    with torch.no_grad():
        reconstructed_patch, _, _ = model(random_patch, random_patch) 
    random_patch = random_patch.squeeze(0).cpu().numpy()  # Shape: (num_bands, patch_size, patch_size)
    reconstructed_patch = reconstructed_patch.squeeze(0).cpu().numpy()  # Shape: (num_bands, patch_size, patch_size)

    # Select the first band
    first_band_random = random_patch[0, :, :]  # Shape: (patch_size, patch_size)
    first_band_reconstructed = reconstructed_patch[0, :, :]  # Shape: (patch_size, patch_size)

    # Determine the global min and max across both arrays
    global_min = min(np.min(first_band_random), np.min(first_band_reconstructed))
    global_max = max(np.max(first_band_random), np.max(first_band_reconstructed))

    # Create the figure
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    img1 = plt.imshow(first_band_random, cmap='viridis', vmin=global_min, vmax=global_max)
    plt.title("First Band of Random Patch (Before Unnormalization)")
    plt.xlabel("Width")
    plt.ylabel("Height")

    # Second subplot - reconstructed patch
    plt.subplot(1, 2, 2)
    img2 = plt.imshow(first_band_reconstructed, cmap='viridis', vmin=global_min, vmax=global_max)
    plt.title("First Band of Reconstructed Patch (Before Unnormalization)")
    plt.xlabel("Width")
    plt.ylabel("Height")

    # Add a single colorbar on the right side
    plt.tight_layout()
    cbar = plt.colorbar(img2, ax=plt.gcf().get_axes(), shrink=0.6, pad=0.02)
    cbar.set_label("Intensity")
    plt.show()



    random_patch_unnormalized = random_patch 
    reconstructed_patch_unnormalized = reconstructed_patch 

    original_spectral_response = random_patch_unnormalized.mean(axis=(1, 2))  # Shape: (num_bands,)
    reconstructed_spectral_response = reconstructed_patch_unnormalized.mean(axis=(1, 2))  # Shape: (num_bands,)
    plt.figure(figsize=(12, 6))
    plt.plot(original_spectral_response, label="Original Spectral Response", color='blue', linestyle='-', linewidth=2)
    plt.plot(reconstructed_spectral_response, label="Reconstructed Spectral Response", color='red', linestyle='-', linewidth=2)
    plt.xlabel("Band Index")
    plt.ylabel("Spectral Response")
    plt.title("Original vs Reconstructed Spectral Response for a Random Patch")
    plt.legend()
    plt.grid(True)
    plt.show()


