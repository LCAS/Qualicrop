import numpy as np
import os
from sklearn.preprocessing import normalize
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, opening

########################### Use thresholds obtained from SAM_ calibrator ####################################################

class BatchSAMProcessor:
    def __init__(self, input_folder, output_folder, ref1_path, ref2_path, shadow_path, 
                 shadow_threshold=0.139, tomato_threshold=0.20):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.ref1_path = ref1_path
        self.ref2_path = ref2_path
        self.shadow_path = shadow_path
        self.shadow_threshold = shadow_threshold
        self.threshold = tomato_threshold
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Load references once
        self.ref1 = np.load(ref1_path)
        self.ref2 = np.load(ref2_path)
        self.shadow_ref = np.load(shadow_path)
    
    def spectral_angle(self, spectrum, reference):
        """Calculate spectral angle between spectrum and reference"""
        return np.arccos(np.dot(spectrum, reference) / 
               (np.linalg.norm(spectrum) * np.linalg.norm(reference)))
    
    def process_image(self, hsi_data):
        """Process a single HSI image and return the masked result"""
        # Normalize data and references
        hsi_norm = normalize(hsi_data.reshape(-1, hsi_data.shape[2])).reshape(hsi_data.shape)
        shadow_norm = normalize(self.shadow_ref.reshape(1, -1))[0]
        ref1_norm = normalize(self.ref1.reshape(1, -1))[0]
        ref2_norm = normalize(self.ref2.reshape(1, -1))[0]
        
        # Stage 1: Calculate SAM for shadow/background
        shadow_sam = np.zeros((hsi_data.shape[0], hsi_data.shape[1]))
        for i in range(hsi_data.shape[0]):
            for j in range(hsi_data.shape[1]):
                shadow_sam[i,j] = self.spectral_angle(hsi_norm[i,j,:], shadow_norm)
        
        # Create shadow mask (pixels that are NOT shadow/background)
        shadow_mask = (shadow_sam > self.shadow_threshold).astype(np.uint8)
        
        # Stage 2: Calculate SAM for tomato/border only on non-shadow pixels
        sam1 = np.zeros((hsi_data.shape[0], hsi_data.shape[1]))
        sam2 = np.zeros_like(sam1)
        
        for i in range(hsi_data.shape[0]):
            for j in range(hsi_data.shape[1]):
                if shadow_mask[i,j]:  # Only process non-shadow pixels
                    sam1[i,j] = self.spectral_angle(hsi_norm[i,j,:], ref1_norm)
                    sam2[i,j] = self.spectral_angle(hsi_norm[i,j,:], ref2_norm)
                else:
                    sam1[i,j] = np.pi  # Maximum possible angle for shadow pixels
                    sam2[i,j] = np.pi
        
        # Take minimum angle (best match)
        sam_result = np.minimum(sam1, sam2)
        
        # Generate mask
        binary_mask = ((sam_result <= self.threshold) & shadow_mask).astype(np.uint8)
        
        # Three-stage morphological processing
        filled_mask = binary_fill_holes(binary_mask).astype(np.uint8) * 255
        padded = np.pad(filled_mask, pad_width=50, mode='constant', constant_values=0)
        kernel = np.ones((6,6), np.uint8)
        
        # Stage 1 processing
        refined_mask = closing(padded, kernel)
        refined_mask = refined_mask[50:-50, 50:-50].astype(np.uint8) * 255
        
        # Stage 2 processing
        filled_mask = binary_fill_holes(refined_mask).astype(np.uint8) * 255
        padded = np.pad(filled_mask, pad_width=50, mode='constant', constant_values=0)
        refined_mask = closing(padded, kernel)
        refined_mask = refined_mask[50:-50, 50:-50].astype(np.uint8) * 255
        
        # Stage 3 processing
        filled_mask = binary_fill_holes(refined_mask).astype(np.uint8) * 255
        padded = np.pad(filled_mask, pad_width=50, mode='constant', constant_values=0)
        refined_mask = closing(padded, kernel)
        refined_mask = refined_mask[50:-50, 50:-50].astype(np.uint8)
        
        # Apply mask to original HSI data
        masked_hsi = hsi_data.copy()
        mask_3d = np.repeat(refined_mask[:, :, np.newaxis], hsi_data.shape[2], axis=2)
        masked_hsi[mask_3d == 0] = 0  # Set background to 0
        
        return masked_hsi
    
    def process_folder(self):
        """Process all .npy files in the input folder"""
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.npy'):
                try:
                    # Load HSI image
                    hsi_path = os.path.join(self.input_folder, filename)
                    hsi_data = np.load(hsi_path)
                    
                    # Process image
                    masked_hsi = self.process_image(hsi_data)
                    
                    # Save result
                    output_path = os.path.join(self.output_folder, filename)
                    np.save(output_path, masked_hsi)
                    print(f"Processed and saved: {filename}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Configuration - modify these paths as needed
    input_folder = "/workspace/src/Session1/cropped"  # Folder containing .npy HSI images
    output_folder = "/workspace/src/new_set/anom"    # Where masked images will be saved
    ref1_path = "/workspace/morph/reference_border.npy"
    ref2_path = "/workspace/morph/reference_pure_tomato.npy"
    shadow_path = "/workspace/morph/reference_shadow_background.npy"
    
    # Process all images
    processor = BatchSAMProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
        ref1_path=ref1_path,
        ref2_path=ref2_path,
        shadow_path=shadow_path,
        shadow_threshold=0.137,
        tomato_threshold=0.168
    )
    processor.process_folder()
    print("Processing complete!")