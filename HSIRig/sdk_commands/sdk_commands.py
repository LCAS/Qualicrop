
"""
si_sdk.py — Single-file ctypes wrappers for all SI_* SDK calls.
See module docstring for usage. Set DLL once with set_dll(dll), then pass handles explicitly.
"""
import ctypes
from typing import Tuple

_DLL = None

def set_dll(dll) -> None:
    global _DLL
    _DLL = dll

def _dll():
    if _DLL is None:
        raise RuntimeError("DLL not set. Call set_dll(ctypes.WinDLL(path)) first.")
    return _DLL

def _wbuf(n: int):
    return ctypes.create_unicode_buffer(n)

# ---- lifecycle -----------------------------------------------------
def si_load(license_path: str = "") -> int:
    dll = _dll()
    try:
        func = dll.SI_Load
        func.argtypes = [ctypes.c_wchar_p]
    except AttributeError:
        func = dll.SI_Load
        func.argtypes = []
    func.restype = ctypes.c_int
    return int(func(license_path)) if func.argtypes else int(func())

def si_unload() -> int:
    dll = _dll()
    func = dll.SI_Unload
    func.argtypes = []
    func.restype = ctypes.c_int
    return int(func())

def si_open(index: int) -> Tuple[int, ctypes.c_void_p]:
    dll = _dll()
    func = dll.SI_Open
    func.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    func.restype = ctypes.c_int
    handle = ctypes.c_void_p(0)
    err = int(func(ctypes.c_int(index), ctypes.byref(handle)))
    return err, handle

def si_close(handle) -> int:
    dll = _dll()
    func = dll.SI_Close
    func.argtypes = [ctypes.c_void_p]
    func.restype = ctypes.c_int
    return int(func(handle))

# ---- capability ----------------------------------------------------
def si_is_implemented(handle, feature: str):
    dll = _dll()
    val = ctypes.c_bool()
    func = dll.SI_IsImplemented
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_bool)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(val)))
    return val.value, err

def si_is_read_only(handle, feature: str):
    dll = _dll()
    val = ctypes.c_bool()
    func = dll.SI_IsReadOnly
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_bool)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(val)))
    return val.value, err

def si_is_writable(handle, feature: str):
    dll = _dll()
    val = ctypes.c_bool()
    func = dll.SI_IsWritable
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_bool)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(val)))
    return val.value, err

def si_is_readable(handle, feature: str):
    dll = _dll()
    val = ctypes.c_bool()
    func = dll.SI_IsReadable
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_bool)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(val)))
    return val.value, err

# ---- feature callbacks --------------------------------------------
def si_register_feature_callback(handle, feature: str, callback, context) -> int:
    dll = _dll()
    func = dll.SI_RegisterFeatureCallback
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p]
    func.restype = ctypes.c_int
    return int(func(handle, feature, callback, context))

def si_unregister_feature_callback(handle, feature: str, callback) -> int:
    dll = _dll()
    func = dll.SI_UnregisterFeatureCallback
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p]
    func.restype = ctypes.c_int
    return int(func(handle, feature, callback))

# ---- int -----------------------------------------------------------
def si_set_int(handle, feature: str, value: int) -> int:
    dll = _dll()
    func = dll.SI_SetInt
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_longlong]
    func.restype = ctypes.c_int
    return int(func(handle, feature, ctypes.c_longlong(value)))

def si_get_int(handle, feature: str):
    dll = _dll()
    out = ctypes.c_longlong()
    func = dll.SI_GetInt
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_longlong)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(out)))
    return out.value, err

def si_get_int_max(handle, feature: str):
    dll = _dll()
    out = ctypes.c_longlong()
    func = dll.SI_GetIntMax
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_longlong)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(out)))
    return out.value, err

def si_get_int_min(handle, feature: str):
    dll = _dll()
    out = ctypes.c_longlong()
    func = dll.SI_GetIntMin
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_longlong)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(out)))
    return out.value, err

# ---- float ---------------------------------------------------------
def si_set_float(handle, feature: str, value: float) -> int:
    dll = _dll()
    func = dll.SI_SetFloat
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_double]
    func.restype = ctypes.c_int
    return int(func(handle, feature, ctypes.c_double(value)))

def si_get_float(handle, feature: str):
    dll = _dll()
    out = ctypes.c_double()
    func = dll.SI_GetFloat
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_double)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(out)))
    return out.value, err

def si_get_float_max(handle, feature: str):
    dll = _dll()
    out = ctypes.c_double()
    func = dll.SI_GetFloatMax
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_double)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(out)))
    return out.value, err

def si_get_float_min(handle, feature: str):
    dll = _dll()
    out = ctypes.c_double()
    func = dll.SI_GetFloatMin
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_double)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(out)))
    return out.value, err

# ---- bool ----------------------------------------------------------
def si_set_bool(handle, feature: str, value: bool) -> int:
    dll = _dll()
    func = dll.SI_SetBool
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_bool]
    func.restype = ctypes.c_int
    return int(func(handle, feature, ctypes.c_bool(bool(value))))

def si_get_bool(handle, feature: str):
    dll = _dll()
    out = ctypes.c_bool()
    func = dll.SI_GetBool
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_bool)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(out)))
    return out.value, err

# ---- command -------------------------------------------------------
def si_command(handle, feature: str) -> int:
    dll = _dll()
    func = dll.SI_Command
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    func.restype = ctypes.c_int
    return int(func(handle, feature))

# ---- string --------------------------------------------------------
def si_set_string(handle, feature: str, value: str) -> int:
    dll = _dll()
    func = dll.SI_SetString
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p]
    func.restype = ctypes.c_int
    return int(func(handle, feature, value))

def si_get_string(handle, feature: str, max_len: int = 255):
    dll = _dll()
    buf = _wbuf(max_len)
    func = dll.SI_GetString
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, buf, max_len))
    return buf.value, err

def si_get_string_max_length(handle, feature: str):
    dll = _dll()
    out = ctypes.c_int()
    func = dll.SI_GetStringMaxLength
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(out)))
    return out.value, err

def si_get_enum_string_max_length(handle, feature: str, index: int):
    dll = _dll()
    idx = ctypes.c_int(index)
    out = ctypes.c_int()
    func = dll.SI_GetEnumStringMaxLength
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, idx, ctypes.byref(out)))
    return out.value, err

# ---- enum ----------------------------------------------------------
def si_set_enum_index(handle, feature: str, value: int) -> int:
    dll = _dll()
    func = dll.SI_SetEnumIndex
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    func.restype = ctypes.c_int
    return int(func(handle, feature, int(value)))

def si_set_enum_index_by_string(handle, feature: str, string: str) -> int:
    dll = _dll()
    func = dll.SI_SetEnumIndexByString
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p]
    func.restype = ctypes.c_int
    return int(func(handle, feature, string))

def si_set_enum_string(handle, feature: str, string: str) -> int:
    dll = _dll()
    func = dll.SI_SetEnumString
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p]
    func.restype = ctypes.c_int
    return int(func(handle, feature, string))

def si_get_enum_index(handle, feature: str):
    dll = _dll()
    out = ctypes.c_int()
    func = dll.SI_GetEnumIndex
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(out)))
    return out.value, err

def si_get_enum_count(handle, feature: str):
    dll = _dll()
    out = ctypes.c_int()
    func = dll.SI_GetEnumCount
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, ctypes.byref(out)))
    return out.value, err

def si_get_enum_string_by_index(handle, feature: str, index: int, max_len: int = 255):
    dll = _dll()
    idx = ctypes.c_int(index)
    buf = _wbuf(max_len)
    func = dll.SI_GetEnumStringByIndex
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, idx, buf, max_len))
    return buf.value, err

def si_is_enum_index_available(handle, feature: str, index: int):
    dll = _dll()
    idx = ctypes.c_int(index)
    out = ctypes.c_bool()
    func = dll.SI_IsEnumIndexAvailable
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(ctypes.c_bool)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, idx, ctypes.byref(out)))
    return out.value, err

def si_is_enum_index_implemented(handle, feature: str, index: int):
    dll = _dll()
    idx = ctypes.c_int(index)
    out = ctypes.c_bool()
    func = dll.SI_IsEnumIndexImplemented
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(ctypes.c_bool)]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, idx, ctypes.byref(out)))
    return out.value, err

# ---- data path -----------------------------------------------------
def si_wait(handle, buffer_address, frame_size_ptr, frame_number_ptr, timeout: int = -1) -> int:
    dll = _dll()
    func = dll.SI_Wait
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                     ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong),
                     ctypes.c_longlong]
    func.restype = ctypes.c_int
    return int(func(handle, buffer_address, frame_size_ptr, frame_number_ptr, ctypes.c_longlong(timeout)))

def si_create_buffer(handle, buffer_size: int):
    dll = _dll()
    out = ctypes.c_void_p()
    func = dll.SI_CreateBuffer
    func.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.POINTER(ctypes.c_void_p)]
    func.restype = ctypes.c_int
    err = int(func(handle, ctypes.c_longlong(buffer_size), ctypes.byref(out)))
    return out.value, err

def si_dispose_buffer(handle, buffer_address) -> int:
    dll = _dll()
    func = dll.SI_DisposeBuffer
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    func.restype = ctypes.c_int
    return int(func(handle, buffer_address))

def si_register_data_callback(handle, callback, context) -> int:
    dll = _dll()
    func = dll.SI_RegisterDataCallback
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    func.restype = ctypes.c_int
    return int(func(handle, callback, context))

def si_unregister_data_callback(handle) -> int:
    dll = _dll()
    func = dll.SI_UnregisterDataCallback
    func.argtypes = [ctypes.c_void_p]
    func.restype = ctypes.c_int
    return int(func(handle))

# ---- introspection / errors ---------------------------------------
def si_get_feature_type(handle, feature: str, max_len: int = 64):
    dll = _dll()
    buf = _wbuf(max_len)
    func = dll.SI_GetFeatureType
    func.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int]
    func.restype = ctypes.c_int
    err = int(func(handle, feature, buf, max_len))
    return buf.value, err

def si_get_error_string(err_code: int) -> str:
    dll = _dll()
    func = dll.SI_GetErrorString
    func.argtypes = [ctypes.c_int]
    func.restype = ctypes.c_wchar_p
    return func(int(err_code))
