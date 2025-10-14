
"""
spec_prefs.py — persistence helpers for SpecSensor/Sensor using json_store.

Stores:
  - last used profile
  - recent profiles
  - per-profile feature values (e.g., exposure, gain, etc.)
  - named presets (arbitrary feature->value dicts)

Default file: ./config/spec_prefs.json  (override via set_store_path)

Example:
    from spec_prefs import (
        set_store_path, set_last_profile, get_last_profile,
        remember_profile, save_feature_value, apply_profile_settings,
        save_preset, load_preset, apply_preset
    )

    set_store_path("config/spec_prefs.json")
    set_last_profile("Lumo XYZ")
    remember_profile("Lumo XYZ")
    save_feature_value("Lumo XYZ", "Camera.ExposureTime", 12.5)

    # After opening sensor for that profile:
    err_applied = apply_profile_settings(sensor, profile="Lumo XYZ")

    # Presets:
    save_preset("low_light", {"Camera.Gain": 6, "Camera.ExposureTime": 20.0})
    preset = load_preset("low_light")
    apply_preset(sensor, preset)
"""

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
from json_store import JSONStore

_STORE_PATH = os.path.join("config", "spec_prefs.json")
_store = JSONStore(_STORE_PATH)

# --------------------------- configuration ----------------------------

def set_store_path(path: str) -> None:
    "Change where preferences are saved. Creates folders on first write."
    global _STORE_PATH, _store
    _STORE_PATH = path
    _store = JSONStore(_STORE_PATH)

def get_store_path() -> str:
    return _STORE_PATH

# --------------------------- keys / layout ----------------------------

# {
#   "last_profile": "Lumo XYZ",
#   "recent_profiles": ["Lumo XYZ", "Lumo ABC"],
#   "profiles": {
#       "Lumo XYZ": {
#           "features": {
#               "Camera.ExposureTime": 12.5,
#               "Camera.Gain": 3
#           }
#       }
#   },
#   "presets": {
#       "low_light": {"Camera.Gain": 6, "Camera.ExposureTime": 20.0}
#   }
# }

# --------------------------- basic profile prefs ----------------------

def set_last_profile(profile: str) -> None:
    _store.set("last_profile", profile)

def get_last_profile(default: Optional[str] = None) -> Optional[str]:
    return _store.get("last_profile", default)

def remember_profile(profile: str, max_recent: int = 8) -> None:
    recents: List[str] = _store.get("recent_profiles", [])
    if profile in recents:
        recents.remove(profile)
    recents.insert(0, profile)
    _store.set("recent_profiles", recents[:max_recent])

def get_recent_profiles() -> List[str]:
    return _store.get("recent_profiles", [])

# --------------------------- per-feature storage ----------------------

def save_feature_value(profile: str, feature: str, value: Any) -> None:
    key = f'profiles.{profile}.features.{feature}'
    _store.set(key, value)

def load_feature_value(profile: str, feature: str, default: Any = None) -> Any:
    key = f'profiles.{profile}.features.{feature}'
    return _store.get(key, default)

def list_saved_features(profile: str) -> Dict[str, Any]:
    return _store.get(f'profiles.{profile}.features', {})

def clear_profile(profile: str) -> None:
    # Remove profile subtree (no-op if missing)
    profiles = _store.get('profiles', {})
    if profile in profiles:
        del profiles[profile]
        _store.set('profiles', profiles)

# --------------------------- presets ----------------------------------

def save_preset(name: str, features: Dict[str, Any]) -> None:
    presets = _store.get("presets", {})
    presets[name] = dict(features)
    _store.set("presets", presets)

def load_preset(name: str) -> Dict[str, Any]:
    return _store.get(f"presets.{name}", {})

def delete_preset(name: str) -> None:
    presets = _store.get("presets", {})
    if name in presets:
        del presets[name]
        _store.set("presets", presets)

def list_presets() -> List[str]:
    return sorted(list(_store.get("presets", {}).keys()))

# --------------------------- application helpers ----------------------

def _attempt_set(sensor, feature: str, value: Any) -> int:
    """
    Try to set a feature using your Sensor wrapper's type-specific setters.
    Strategy:
      bool   -> setbool
      int    -> setint
      float  -> setfloat
      str    -> try setenumstring, fall back to setstring
    Returns: SDK error code (0 success, else error from DLL). If feature is
    not writable/implemented, returns the underlying error code.
    """
    # Fast path: check writable/implemented if available
    if hasattr(sensor, "iswritable"):
        writable, _ = sensor.iswritable(feature)
        if not writable and hasattr(sensor, "isimplemented"):
            implemented, _ = sensor.isimplemented(feature)
            if not implemented:
                return -1  # consistent "not implemented"
    try:
        if isinstance(value, bool) and hasattr(sensor, "setbool"):
            return int(sensor.setbool(feature, value))
        if isinstance(value, int) and hasattr(sensor, "setint"):
            return int(sensor.setint(feature, value))
        if isinstance(value, float) and hasattr(sensor, "setfloat"):
            return int(sensor.setfloat(feature, value))
        if isinstance(value, str):
            # enums are encoded as strings in our store; prefer enum setter
            if hasattr(sensor, "setenumstring"):
                err = int(sensor.setenumstring(feature, value))
                if err == 0:
                    return 0
            if hasattr(sensor, "setstring"):
                return int(sensor.setstring(feature, value))
    except Exception:
        # Fall-through to generic failure
        pass
    # If we reach here, try a generic 'command' with the value as string? Not appropriate.
    return -2

def apply_profile_settings(sensor, profile: Optional[str] = None, on_error: str = "ignore") -> int:
    """
    Apply all saved feature values for a profile to the given `sensor`.
    on_error: 'ignore' | 'stop' — if 'stop', stops on first nonzero error.
    Returns the number of successfully-applied features.
    """
    if profile is None:
        profile = get_last_profile(None)
    if profile is None:
        return 0

    features = list_saved_features(profile)
    applied = 0
    for feat, val in features.items():
        err = _attempt_set(sensor, feat, val)
        if err == 0:
            applied += 1
        elif on_error == "stop":
            break
    return applied

def apply_preset(sensor, preset: Dict[str, Any], on_error: str = "ignore") -> int:
    """
    Apply a preset (feature->value mapping) to `sensor`. Returns count applied.
    """
    applied = 0
    for feat, val in preset.items():
        err = _attempt_set(sensor, feat, val)
        if err == 0:
            applied += 1
        elif on_error == "stop":
            break
    return applied

# --------------------------- convenience APIs -------------------------

def remember_and_save(sensor, profile: str, feature_values: Dict[str, Any]) -> int:
    """
    Record last profile, add to recents, save all features for that profile,
    and apply them immediately. Returns number applied.
    """
    set_last_profile(profile)
    remember_profile(profile)
    for k, v in feature_values.items():
        save_feature_value(profile, k, v)
    return apply_profile_settings(sensor, profile=profile)
