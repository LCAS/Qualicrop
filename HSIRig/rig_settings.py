"""
File: rig_settings.py
Author: Robert Stevenson

Stores shared data for cross class/file saving and accessing for UI and Command control/retrieval/display
"""

# Rig  Status Dictonary
RIG_STATUS = {
    "homed": False, # is the rig homed?
    "bed_pos" : 0.0, # current bed position
    "cam_pos" : 0.0, # current camera positon
}

# Rig speed in mm/min
RIG_SPEED = 1000.0
# Rig bed start and end position
RIG_BED_START = 80.0
RIG_BED_END = 600.0

# Rig camera height position
RIG_CAM_HEIGHT = 0.0
RIG_CAM_HEIGHT_OFFSET = 80.0 # This is the distance from surface of bed (in `mm` and the camera lense when camera is at 0 (at the lowest positions)

# =========================
# ==== READ-ONLY VALUES====
# =========================

RIG_TRAVEL_SPEED_READ_ONLY = 3000
RIG_TIMEOUT_READ_ONLY = 95 # timeout in seconds
RIG_WHITE_CAL_POS_READ_ONLY = 250.0 # TODO: Check Value
RIG_BLACK_CAL_POS_READ_ONLY = 260.0 # TODO: Check Value

# =========================
# ==== Camera / UI defaults
# =========================

# Shutter: "Open" or "Close"
CAMERA_SHUTTER = "Open"
# Frame rate and exposure defaults
CAMERA_FRAME_RATE = 60
CAMERA_EXPOSURE = 50.0
# Binning and modes
CAMERA_SPECTRAL_BIN = 1
CAMERA_SPATIAL_BIN = 1
CAMERA_CAPTURE_MODE = 1
CAMERA_LINE_COUNT = 0

# Output folder for scan results
OUTPUT_FOLDER = "output/"
