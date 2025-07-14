import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

############################## select your refrence to use for SAM later (select 3 pixels of the region) ####################################

class SpectralLibraryBuilder:
    def __init__(self, hsi_path):
        self.hsi_data = np.load(hsi_path)  # Shape: (height, width, bands)
        self.current_band = 30  # Middle band for display
        self.selected_pixels = []
        self.references = {}
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)
        self.im = self.ax.imshow(self.hsi_data[:, :, self.current_band], cmap='gray')
        self.ax.set_title(f"Band {self.current_band} - Click 3 pixels for reference")
        
        # Add buttons
        ax_save = plt.axes([0.3, 0.05, 0.2, 0.075])
        ax_clear = plt.axes([0.55, 0.05, 0.2, 0.075])
        self.btn_save = Button(ax_save, 'Save Reference')
        self.btn_clear = Button(ax_clear, 'Clear Selection')
        self.btn_save.on_clicked(self.save_reference)
        self.btn_clear.on_clicked(self.clear_selection)
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        if len(self.selected_pixels) < 3:
            self.selected_pixels.append((y, x))
            self.ax.plot(x, y, 'ro', markersize=5)
            self.fig.canvas.draw()
            
            if len(self.selected_pixels) == 3:
                self.ax.set_title("3 pixels selected! Enter reference name in terminal")
                self.fig.canvas.draw()

    def save_reference(self, event):
        if len(self.selected_pixels) != 3:
            print("Please select exactly 3 pixels first!")
            return
            
        # Extract and average spectra
        spectra = []
        for y, x in self.selected_pixels:
            spectra.append(self.hsi_data[y, x, :])
        avg_spectrum = np.mean(spectra, axis=0)
        
        # Get reference name
        ref_name = input("Enter reference name (e.g., 'ripe_tomato'): ")
        self.references[ref_name] = avg_spectrum
        np.save(f"reference_{ref_name}.npy", avg_spectrum)
        print(f"Saved {ref_name} with shape {avg_spectrum.shape}")
        
        self.clear_selection(None)
        self.ax.set_title(f"Band {self.current_band} - New reference saved!")
        self.fig.canvas.draw()

    def clear_selection(self, event):
        self.selected_pixels = []
        [artist.remove() for artist in self.ax.lines]  # Remove red dots
        self.ax.set_title(f"Band {self.current_band} - Click 3 pixels")
        self.fig.canvas.draw()

    def show(self):
        plt.show()

# Usage
if __name__ == "__main__":
    builder = SpectralLibraryBuilder("/workspace/src/Session1/cropped/s1_anorm11_bbox_1.npy")  # Replace with your .npy path
    builder.show()