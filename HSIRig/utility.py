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

DEFAULT_KEYS = [
    "RIG_SPEED",
    "RIG_BED_START",
    "RIG_BED_END",
    "RIG_CAM_HEIGHT",
    "RIG_WHITE_CAL_POS_READ_ONLY",
    "RIG_BLACK_CAL_POS_READ_ONLY",
    "RIG_TRAVEL_SPEED_READ_ONLY",
    "RIG_TIMEOUT_READ_ONLY",
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
        # apply defaults to rig_settings (no change, but for completeness)
        for k, v in defaults.items():
            setattr(rig_settings, k, v)
        return defaults

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    # apply loaded values into rig_settings if keys exist
    for k in DEFAULT_KEYS:
        if k in data and data[k] is not None:
            try:
                # convert numeric-like values to float where appropriate
                setattr(rig_settings, k, float(data[k]) if isinstance(data[k], (int, float, str)) and str(data[k]) != "" else data[k])
            except Exception:
                setattr(rig_settings, k, data[k])
        else:
            # ensure missing keys are populated with rig_settings defaults
            data.setdefault(k, getattr(rig_settings, k, None))

    return data

def save_settings(settings_dict, yaml_path=None):
    """
    Save a dict of settings to YAML and apply to rig_settings module.
    """
    if yaml_path is None:
        base = os.path.dirname(__file__)
        yaml_path = os.path.normpath(os.path.join(base, ".", "config", "settings.yaml"))

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    # apply to rig_settings then dump
    for k, v in settings_dict.items():
        if hasattr(rig_settings, k):
            setattr(rig_settings, k, v)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(settings_dict, f)