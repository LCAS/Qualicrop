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

# =========================
# ==== READ-ONLY VALUES====
# =========================

RIG_TRAVEL_SPEED_READ_ONLY = 6000
RIG_TIMEOUT_READ_ONLY = 95 # timeout in seconds
RIG_WHITE_CAL_POS_READ_ONLY = 0.0 # TODO: Set Value
RIG_BLACK_CAL_POS_READ_ONLY = 0.0 # TODO: Set Value
