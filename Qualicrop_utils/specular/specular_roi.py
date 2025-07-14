import os
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
from matplotlib import cm

######## GUI for specular region selection ############################


class SpectralROIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Image ROI Annotation Tool")
        self.root.geometry("1200x900")  # Increased window size
        
        # Image data
        self.image_data = None
        self.current_file = None
        self.files = []
        self.current_dir = ""
        self.tk_image = None
        self.image_path = None
        
        # ROI data
        self.rois = []
        self.all_rois = {}  # Dictionary to store ROIs for all images
        self.current_roi = None
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        
        # Fixed band to display
        self.fixed_band = 35
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main container frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control frame at top
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # First row of controls
        row1 = tk.Frame(control_frame)
        row1.pack(fill=tk.X)
        
        self.load_btn = tk.Button(row1, text="Load Directory", command=self.load_directory)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.image_label = tk.Label(row1, text="Image:")
        self.image_label.pack(side=tk.LEFT, padx=5)
        
        self.image_selector = ttk.Combobox(row1, state="readonly", width=40)
        self.image_selector.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.image_selector.bind("<<ComboboxSelected>>", self.load_selected_image)
        
        # Second row of controls
        row2 = tk.Frame(control_frame)
        row2.pack(fill=tk.X, pady=5)
        
        self.prev_btn = tk.Button(row2, text="<< Prev", command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = tk.Button(row2, text="Next >>", command=self.next_image)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.undo_btn = tk.Button(row2, text="Undo", command=self.undo_last_roi)
        self.undo_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(row2, text="Clear ROIs", command=self.clear_rois)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(row2, text="Save ROIs", command=self.save_rois)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label at bottom of control frame
        self.status_label = tk.Label(control_frame, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=5)
        
        # Canvas for image display
        self.canvas_frame = tk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
    def load_directory(self):
        dir_path = filedialog.askdirectory(title="Select Directory with .npy files")
        if dir_path:
            self.current_dir = dir_path
            self.files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
            
            if not self.files:
                messagebox.showwarning("No Images", "No .npy files found in selected directory")
                return
                
            self.image_selector['values'] = self.files
            self.image_selector.current(0)
            self.load_selected_image()
            self.update_status(f"Loaded {len(self.files)} images")
    
    def load_selected_image(self, event=None):
        if not self.files:
            return
            
        idx = self.image_selector.current()
        self.current_file = self.files[idx]
        file_path = os.path.join(self.current_dir, self.current_file)
        
        try:
            self.image_data = np.load(file_path)
            self.image_path = file_path
            
            # Load any existing ROIs for this image
            self.rois = self.all_rois.get(self.current_file, [])
            
            self.update_display()
            self.update_status(f"Loaded: {self.current_file}")
        except Exception as e:
            self.update_status(f"Error loading {self.current_file}: {str(e)}")
            messagebox.showerror("Error", f"Failed to load {self.current_file}:\n{str(e)}")
    
    def prev_image(self):
        if not self.files:
            return
            
        # Save current ROIs before switching
        if self.current_file:
            self.all_rois[self.current_file] = self.rois
            
        current_idx = self.image_selector.current()
        if current_idx > 0:
            self.image_selector.current(current_idx - 1)
            self.load_selected_image()
    
    def next_image(self):
        if not self.files:
            return
            
        # Save current ROIs before switching
        if self.current_file:
            self.all_rois[self.current_file] = self.rois
            
        current_idx = self.image_selector.current()
        if current_idx < len(self.files) - 1:
            self.image_selector.current(current_idx + 1)
            self.load_selected_image()
    
    def update_display(self):
        if self.image_data is not None:
            try:
                # Use fixed band 35 (or the last band if the image has fewer bands)
                band_idx = min(self.fixed_band, self.image_data.shape[2] - 1) if len(self.image_data.shape) == 3 else 0
                
                if len(self.image_data.shape) == 3:
                    band_data = self.image_data[:, :, band_idx]
                else:
                    band_data = self.image_data
                    
                # Normalize and apply colormap
                normalized = (band_data - band_data.min()) / (band_data.max() - band_data.min())
                colored = (cm.viridis(normalized) * 255).astype(np.uint8)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(colored)
                
                # Draw ROIs
                draw = ImageDraw.Draw(pil_image)
                for roi in self.rois:
                    x0, y0, x1, y1 = roi
                    draw.ellipse([x0, y0, x1, y1], outline='red', width=2)
                
                # Resize if too large for canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if pil_image.width > canvas_width or pil_image.height > canvas_height:
                    ratio = min(canvas_width/pil_image.width, canvas_height/pil_image.height)
                    new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to Tkinter PhotoImage
                self.tk_image = ImageTk.PhotoImage(pil_image)
                
                # Display on canvas
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
                self.canvas.config(width=pil_image.width, height=pil_image.height)
                
            except Exception as e:
                self.update_status(f"Error displaying image: {str(e)}")
    
    def on_press(self, event):
        if self.image_data is not None:
            self.drawing = True
            self.start_x = event.x
            self.start_y = event.y
            self.current_roi = [event.x, event.y, event.x, event.y]
    
    def on_drag(self, event):
        if self.drawing and self.image_data is not None:
            self.current_roi[2] = event.x
            self.current_roi[3] = event.y
            self.update_display()
            
            # Draw temporary ROI
            self.canvas.delete("temp_roi")
            self.canvas.create_oval(
                self.current_roi[0], self.current_roi[1],
                self.current_roi[2], self.current_roi[3],
                outline='red', width=2, tags="temp_roi"
            )
    
    def on_release(self, event):
        if self.drawing and self.image_data is not None:
            self.drawing = False
            radius = ((self.current_roi[2] - self.current_roi[0])**2 + 
                     (self.current_roi[3] - self.current_roi[1])**2)**0.5
            
            if radius > 5:  # Minimum radius threshold
                # Convert to circle (center + radius)
                center_x = (self.current_roi[0] + self.current_roi[2]) / 2
                center_y = (self.current_roi[1] + self.current_roi[3]) / 2
                radius = radius / 2
                
                # Store as [x0, y0, x1, y1] for easier drawing
                self.rois.append([
                    center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius
                ])
            
            self.current_roi = None
            self.update_display()
            self.update_status(f"{len(self.rois)} ROIs drawn")
    
    def undo_last_roi(self):
        if self.rois:
            self.rois.pop()
            self.update_display()
            self.update_status(f"Undo last ROI - {len(self.rois)} remaining")
    
    def clear_rois(self):
        self.rois = []
        self.update_display()
        self.update_status("Cleared all ROIs")
    
    def save_rois(self):
        if not self.current_file:
            messagebox.showwarning("No Image", "No image loaded")
            return
            
        # Save current ROIs before exporting
        if self.current_file:
            self.all_rois[self.current_file] = self.rois
            
        if not self.all_rois:
            messagebox.showwarning("No ROIs", "No ROIs to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save ROIs",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        
        if file_path:
            # Convert all ROIs to serializable format
            output_data = {}
            for img_file, rois in self.all_rois.items():
                rois_data = []
                for roi in rois:
                    x0, y0, x1, y1 = roi
                    center_x = (x0 + x1) / 2
                    center_y = (y0 + y1) / 2
                    radius = (x1 - x0) / 2
                    
                    rois_data.append({
                        'center': {'x': center_x, 'y': center_y},
                        'radius': radius
                    })
                
                output_data[img_file] = {
                    'rois': rois_data,
                    'band': self.fixed_band
                }
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(output_data, f, indent=4)
                self.update_status(f"All ROIs saved to {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "All ROIs saved successfully")
            except Exception as e:
                self.update_status(f"Error saving: {str(e)}")
                messagebox.showerror("Error", f"Failed to save ROIs:\n{str(e)}")
    
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectralROIApp(root)
    root.mainloop()