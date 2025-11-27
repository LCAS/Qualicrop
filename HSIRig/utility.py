import os
import yaml
import rig_settings

from datetime import datetime
PREFIX_NAME="QualiCrop"

def get_dataset_name():
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp_str

# ============================
# ==== YAML SETTINGS LOAD/SAVE====
# ============================

# include camera keys here so they are persisted to the same settings.yaml
DEFAULT_KEYS = [
    "RIG_SPEED",
    "RIG_BED_START",
    "RIG_BED_END",
    "RIG_CAM_HEIGHT",
    "RIG_WHITE_CAL_POS_READ_ONLY",
    "RIG_BLACK_CAL_POS_READ_ONLY",
    "RIG_TRAVEL_SPEED_READ_ONLY",
    "RIG_TIMEOUT_READ_ONLY",

    # Camera / acquisition settings
    "CAMERA_SHUTTER",        # "Open" / "Close"
    "CAMERA_FRAME_RATE",     # int or float
    "CAMERA_EXPOSURE",       # int or float
    "CAMERA_SPECTRAL_BIN",   # 1,2,4 ...
    "CAMERA_SPATIAL_BIN",    # 1,2,4,8 ...
    "CAMERA_CAPTURE_MODE",   # 1,2,4 ...
    "CAMERA_LINE_COUNT",     # int
    "OUTPUT_FOLDER",         # path string for output
]

def _default_settings():
    out = {}
    for k in DEFAULT_KEYS:
        out[k] = getattr(rig_settings, k, None)
    return out

def load_settings(yaml_path=None):
    """
    Load settings from YAML into rig_settings module and return dict.
    If file does not exist, create it with defaults from rig_settings.
    """
    if yaml_path is None:
        base = os.path.dirname(__file__)
        yaml_path = os.path.normpath(os.path.join(base, ".", "config", "settings.yaml"))

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    if not os.path.exists(yaml_path):
        # write defaults
        defaults = _default_settings()
        with open(yaml_path, "w") as f:
            yaml.safe_dump(defaults, f)
        # apply defaults to rig_settings
        for k, v in defaults.items():
            setattr(rig_settings, k, v)
        return defaults

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    # ensure missing keys are filled from defaults
    for k in DEFAULT_KEYS:
        if k not in data:
            data[k] = getattr(rig_settings, k, None)

    # apply loaded values into rig_settings
    for k, val in data.items():
        try:
            # basic type coercion for numeric values where appropriate
            if k in ("RIG_SPEED", "RIG_BED_START", "RIG_BED_END", "RIG_CAM_HEIGHT",
                     "RIG_WHITE_CAL_POS_READ_ONLY", "RIG_BLACK_CAL_POS_READ_ONLY",
                     "RIG_TRAVEL_SPEED_READ_ONLY", "RIG_TIMEOUT_READ_ONLY",
                     "CAMERA_FRAME_RATE", "CAMERA_EXPOSURE"):
                if val is not None and str(val) != "":
                    setattr(rig_settings, k, float(val))
                else:
                    setattr(rig_settings, k, val)
            elif k in ("CAMERA_SPECTRAL_BIN", "CAMERA_SPATIAL_BIN", "CAMERA_CAPTURE_MODE", "CAMERA_LINE_COUNT"):
                if val is not None and str(val) != "":
                    setattr(rig_settings, k, int(val))
                else:
                    setattr(rig_settings, k, val)
            else:
                setattr(rig_settings, k, val)
        except Exception:
            setattr(rig_settings, k, val)

    return data

def save_settings(settings_dict, yaml_path=None):
    """
    Save a dict of settings to YAML and apply to rig_settings module.
    This merges with existing YAML so keys that are not provided are preserved.
    """
    if yaml_path is None:
        base = os.path.dirname(__file__)
        yaml_path = os.path.normpath(os.path.join(base, ".", "config", "settings.yaml"))

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    # read existing file if present
    existing = {}
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r") as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            existing = {}

    # update existing with provided settings
    merged = existing.copy()
    for k, v in settings_dict.items():
        merged[k] = v
        # always set attribute on rig_settings module (creates attribute if missing)
        try:
            setattr(rig_settings, k, v)
        except Exception:
            pass

    # also ensure defaults for any DEFAULT_KEYS not present in merged
    for k in DEFAULT_KEYS:
        if k not in merged:
            merged[k] = getattr(rig_settings, k, None)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(merged, f)