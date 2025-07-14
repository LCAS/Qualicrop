import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from matplotlib.widgets import Slider
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, opening
from scipy.ndimage import binary_fill_holes, binary_dilation

#################### Calibrate your SAM using this Script and use your thresholds in the SAM_processing script #########################################


class SAMProcessor:
    def __init__(self, hsi_path, ref1_path, ref2_path, shadow_path, specular_path):
        # Load data
        self.hsi_data = np.load(hsi_path)  # Shape: (height, width, bands)
        self.ref1 = np.load(ref1_path)     # First reference spectrum (border)
        self.ref2 = np.load(ref2_path)     # Second reference spectrum (pure tomato)
        self.shadow_ref = np.load(shadow_path)  # Shadow/background reference
        self.specular_ref = np.load(specular_path)  # Specular reflection reference
        
        # Initialize parameters
        self.shadow_threshold = 0.163  # Threshold for shadow/background
        self.threshold = 0.166#0.17#0.20          # Threshold for tomato/border
        self.specular_threshold = 0.0#0.136# Threshold for specular reflection
        
        # Setup figure
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        plt.subplots_adjust(bottom=0.3, hspace=0.3, wspace=0.3)
        
        # Add threshold sliders
        ax_shadow_slider = plt.axes([0.3, 0.2, 0.4, 0.03])
        self.shadow_slider = Slider(ax_shadow_slider, 'Shadow Threshold', 0, 1, valinit=self.shadow_threshold)
        self.shadow_slider.on_changed(self.update_shadow_threshold)
        
        ax_slider = plt.axes([0.3, 0.15, 0.4, 0.03])
        self.slider = Slider(ax_slider, 'Tomato Threshold', 0, 1, valinit=self.threshold)
        self.slider.on_changed(self.update_threshold)
        
        ax_specular_slider = plt.axes([0.3, 0.1, 0.4, 0.03])
        self.specular_slider = Slider(ax_specular_slider, 'Specular Threshold', 0, 1, valinit=self.specular_threshold)
        self.specular_slider.on_changed(self.update_specular_threshold)
        
        # Initial processing
        self.process_sam()
        self.update_masks()
        self.update_display()
        
    def spectral_angle(self, spectrum, reference):
        """Calculate spectral angle between spectrum and reference"""
        return np.arccos(np.dot(spectrum, reference) / 
               (np.linalg.norm(spectrum) * np.linalg.norm(reference)))
    
    def process_sam(self):
        """Process SAM in stages: shadow/background, tomato/border, then specular"""
        # Normalize data and references
        hsi_norm = normalize(self.hsi_data.reshape(-1, self.hsi_data.shape[2])).reshape(self.hsi_data.shape)
        shadow_norm = normalize(self.shadow_ref.reshape(1, -1))[0]
        ref1_norm = normalize(self.ref1.reshape(1, -1))[0]
        ref2_norm = normalize(self.ref2.reshape(1, -1))[0]
        specular_norm = normalize(self.specular_ref.reshape(1, -1))[0]
        
        # Stage 1: Calculate SAM for shadow/background
        shadow_sam = np.zeros((self.hsi_data.shape[0], self.hsi_data.shape[1]))
        for i in range(self.hsi_data.shape[0]):
            for j in range(self.hsi_data.shape[1]):
                shadow_sam[i,j] = self.spectral_angle(hsi_norm[i,j,:], shadow_norm)
        
        # Create shadow mask (pixels that are NOT shadow/background)
        self.shadow_mask = (shadow_sam > self.shadow_threshold).astype(np.uint8)
        
        # Stage 2: Calculate SAM for tomato/border only on non-shadow pixels
        sam1 = np.zeros((self.hsi_data.shape[0], self.hsi_data.shape[1]))
        sam2 = np.zeros_like(sam1)
        
        for i in range(self.hsi_data.shape[0]):
            for j in range(self.hsi_data.shape[1]):
                if self.shadow_mask[i,j]:  # Only process non-shadow pixels
                    sam1[i,j] = self.spectral_angle(hsi_norm[i,j,:], ref1_norm)
                    sam2[i,j] = self.spectral_angle(hsi_norm[i,j,:], ref2_norm)
                else:
                    sam1[i,j] = np.pi  # Maximum possible angle for shadow pixels
                    sam2[i,j] = np.pi
        
        # Take minimum angle (best match)
        self.sam_result = np.minimum(sam1, sam2)
        
        # Stage 3: Calculate SAM for specular reflection on all pixels
        self.specular_sam = np.zeros((self.hsi_data.shape[0], self.hsi_data.shape[1]))
        for i in range(self.hsi_data.shape[0]):
            for j in range(self.hsi_data.shape[1]):
                self.specular_sam[i,j] = self.spectral_angle(hsi_norm[i,j,:], specular_norm)
    
    def update_masks(self):
        """Generate all mask versions"""
        # Basic thresholded mask (combined with shadow mask)
        self.binary_mask = ((self.sam_result <= self.threshold) & (self.shadow_mask)).astype(np.uint8)
        
        
        
        # Filled mask
        self.filled_mask = binary_fill_holes(self.binary_mask).astype(np.uint8) * 255
        
        self.padded = np.pad(self.filled_mask, pad_width=50, mode='constant', constant_values=0)
        
        # Refined mask (filled + morphological cleaning)
        kernel = np.ones((10,10), np.uint8)
        self.refined_mask = closing(self.padded, kernel)
        self.refined_mask = self.refined_mask[50:-50, 50:-50].astype(np.uint8) * 255

        # Stage 2 processing
        self.filled_mask = binary_fill_holes(self.refined_mask).astype(np.uint8) * 255
        self.padded = np.pad(self.filled_mask, pad_width=50, mode='constant', constant_values=0)
        kernel = np.ones((6,6), np.uint8)
        self.refined_mask = closing(self.padded, kernel)
        self.refined_mask = self.refined_mask[50:-50, 50:-50].astype(np.uint8) * 255

        # Stage 3 processing
        self.filled_mask = binary_fill_holes(self.refined_mask).astype(np.uint8) * 255
        self.padded = np.pad(self.filled_mask, pad_width=50, mode='constant', constant_values=0)
        kernel = np.ones((6,6), np.uint8)
        self.refined_mask = closing(self.padded, kernel)
        self.refined_mask = self.refined_mask[50:-50, 50:-50].astype(np.uint8) * 255

        # Remove specular reflections from the mask
        specular_mask = (self.specular_sam <= self.specular_threshold)
        self.refined_mask[specular_mask] = 0
    
    def update_shadow_threshold(self, val):
        self.shadow_threshold = val
        self.process_sam()  # Need to reprocess from beginning
        self.update_masks()
        self.update_display()
    
    def update_threshold(self, val):
        self.threshold = val
        self.update_masks()
        self.update_display()
        
    def update_specular_threshold(self, val):
        self.specular_threshold = val
        self.update_masks()
        self.update_display()
    
    def update_display(self):
        """Update the display with current results"""
        # Show RGB composite for reference
        rgb_bands = [221, 50, 72]  # Adjust based on your HSI bands
        rgb_img = self.hsi_data[:, :, rgb_bands]
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        
        self.ax1.clear()
        self.ax1.imshow(rgb_img)
        self.ax1.set_title('RGB Composite')
        self.ax1.axis('off')
        
        # Show SAM result (only non-shadow areas)
        self.ax2.clear()
        sam_display = self.ax2.imshow(np.where(self.shadow_mask, self.sam_result, np.nan), 
                                     cmap='jet', vmin=0, vmax=0.5)
        self.ax2.set_title('SAM Angle Map (non-shadow)')
        self.ax2.axis('off')
        self.fig.colorbar(sam_display, ax=self.ax2, fraction=0.046)
        
        # Show basic binary mask
        self.ax3.clear()
        self.ax3.imshow(self.binary_mask * 255, cmap='gray')
        self.ax3.set_title(f'Binary Mask (Threshold: {self.threshold:.2f})')
        self.ax3.axis('off')
        
        # Show filled and refined mask
        self.ax4.clear()
        self.ax4.imshow(self.refined_mask, cmap='gray')
        self.ax4.set_title(f'Final Mask (Specular Thresh: {self.specular_threshold:.2f})')
        self.ax4.axis('off')
        
        plt.draw()
    
    def show(self):
        plt.show()


# Usage
if __name__ == "__main__":
    # Load your data and references
    processor = SAMProcessor(
        hsi_path="/workspace/src/benchmarking_rgb/benchmark_HSI/s1_anorm14_bbox_1.npy",
        ref1_path="/workspace/morph/reference_pure_tomato.npy",
        ref2_path="/workspace/morph/reference_pure_tomato.npy",
        shadow_path="/workspace/morph/reference_shadow_background.npy",
        specular_path="/workspace/morph/reference_specular.npy"
    )
    processor.show()