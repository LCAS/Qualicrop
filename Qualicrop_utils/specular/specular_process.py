import os
import json
import numpy as np

####################################### Use JSON to remove speularity ########################################################

# Paths
json_path = '/workspace/src/new_set/specular_test.json'
npy_input_dir = '/workspace/src/new_set/test'
npy_output_dir = '/workspace/src/new_set/sup'

# Create output directory if it doesn't exist
os.makedirs(npy_output_dir, exist_ok=True)

# Load JSON
with open(json_path, 'r') as f:
    roi_data = json.load(f)

# Function to apply circular ROI mask
def apply_roi_mask(image, rois):
    H, W = image.shape[:2]
    Y, X = np.ogrid[:H, :W]
    mask = np.zeros((H, W), dtype=bool)

    for roi in rois:
        cx = roi["center"]["x"]
        cy = roi["center"]["y"]
        r = roi["radius"]
        distance_squared = (X - cx) ** 2 + (Y - cy) ** 2
        mask |= distance_squared <= r ** 2  # Union of all circles

    # Set pixels inside the mask to 0 for all channels
    if image.ndim == 3:
        image[mask] = 0
    else:
        image[mask] = 0

    return image

# Process each image
for filename, data in roi_data.items():
    npy_path = os.path.join(npy_input_dir, filename)
    if not os.path.exists(npy_path):
        print(f"Warning: File {filename} not found. Skipping.")
        continue

    image = np.load(npy_path)
    rois = data["rois"]

    processed_image = apply_roi_mask(image, rois)
    
    output_path = os.path.join(npy_output_dir, filename)
    np.save(output_path, processed_image)
    print(f"Saved processed image to {output_path}")
