# envi_io.py
from __future__ import annotations
import os
import numpy as np
from typing import Optional, Tuple, Dict, Any
import glob
from typing import Optional

_ENVI_DTYPE_TO_CODE = {
    np.dtype(np.uint8): 1,
    np.dtype(np.int16): 2,
    np.dtype(np.int32): 3,
    np.dtype(np.float32): 4,
    np.dtype(np.float64): 5,
    np.dtype(np.complex64): 6,
    np.dtype(np.complex128): 9,
    np.dtype(np.uint16): 12,
    np.dtype(np.uint32): 13,
    np.dtype(np.int64): 14,
    np.dtype(np.uint64): 15,
}

_CODE_TO_ENVI_DTYPE = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.complex64,
    9: np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def _format_envi_list(values) -> str:
    """
    ENVI uses { v1, v2, v3 } list format, can span multiple lines.
    """
    vals = [str(v) for v in values]
    # wrap lines roughly, ENVI is permissive
    chunks = []
    line = []
    length = 0
    for v in vals:
        if length + len(v) + 2 > 75 and line:
            chunks.append(", ".join(line))
            line = [v]
            length = len(v)
        else:
            line.append(v)
            length += len(v) + 2
    if line:
        chunks.append(", ".join(line))
    if len(chunks) == 1:
        return "{ " + chunks[0] + " }"
    return "{\n  " + ",\n  ".join(chunks) + "\n}"


def write_envi_bil(base_path: str,
                   cube: np.ndarray,
                   wavelength: Optional[np.ndarray] = None,
                   *,
                   interleave: str = "bil",
                   byte_order: int = 0,
                   map_info: Optional[str] = None,
                   description: str = "HSI capture",
                   extra_header: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Write ENVI data + header.

    cube shape:
      - Expected: (lines, samples, bands)
    interleave:
      - "bil" (this function writes BIL only; argument kept for clarity)
    base_path:
      - can be 'folder/name' with or without extension
      - writes:
          <base_path>.bil  (data)
          <base_path>.hdr  (header)

    Returns: (data_path, hdr_path)
    """
    if interleave.lower() != "bil":
        raise ValueError("This writer currently supports only BIL interleave.")

    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError("cube must be 3D (lines, samples, bands).")

    lines, samples, bands = cube.shape

    dt = cube.dtype
    if dt not in _ENVI_DTYPE_TO_CODE:
        # fall back to uint16 if unsure (common for sensors)
        cube = cube.astype(np.uint16, copy=False)
        dt = cube.dtype

    envi_code = _ENVI_DTYPE_TO_CODE[dt]

    # output paths
    root, ext = os.path.splitext(base_path)
    if ext.lower() in (".hdr", ".bil", ".dat", ".img"):
        base = root
    else:
        base = base_path

    data_path = base + ".bil"
    hdr_path = base + ".hdr"

    # BIL layout: for each line -> write all bands, each band is "samples" pixels
    # Our cube is (lines, samples, bands). For BIL, per line, bands are contiguous blocks.
    bil = np.transpose(cube, (0, 2, 1))  # (lines, bands, samples)

    # Write binary (native endianness; ENVI byte order indicates endianness)
    # byte_order=0 means little-endian, 1 means big-endian.
    # Most PCs are little-endian; if your dtype is native, this is fine.
    bil.tofile(data_path)

    hdr_lines = []
    hdr_lines.append("ENVI")
    hdr_lines.append(f"description = {{{description}}}")
    hdr_lines.append(f"samples = {samples}")
    hdr_lines.append(f"lines   = {lines}")
    hdr_lines.append(f"bands   = {bands}")
    hdr_lines.append("header offset = 0")
    hdr_lines.append("file type = ENVI Standard")
    hdr_lines.append(f"data type = {envi_code}")
    hdr_lines.append(f"interleave = bil")
    hdr_lines.append(f"byte order = {int(byte_order)}")

    # Optional wavelength
    if wavelength is not None:
        w = np.asarray(wavelength).ravel()
        if w.size != bands:
            # don't silently write wrong metadata
            raise ValueError(f"wavelength length ({w.size}) must match bands ({bands}).")
        hdr_lines.append("wavelength units = Nanometers")
        hdr_lines.append("wavelength = " + _format_envi_list([f"{x:.6f}" for x in w]))

    if map_info is not None:
        hdr_lines.append(f"map info = {{{map_info}}}")

    if extra_header:
        for k, v in extra_header.items():
            # allow strings or numbers; lists should be passed already formatted if needed
            if isinstance(v, (list, tuple, np.ndarray)):
                hdr_lines.append(f"{k} = " + _format_envi_list(v))
            else:
                hdr_lines.append(f"{k} = {v}")

    with open(hdr_path, "w", encoding="utf-8") as f:
        f.write("\n".join(hdr_lines) + "\n")

    return data_path, hdr_path


def read_envi(hdr_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Minimal ENVI reader for BIL/BSQ/BIP. Returns (cube, header_dict).
    cube returned as (lines, samples, bands).
    """
    hdr = _parse_envi_header(hdr_path)

    interleave = str(hdr.get("interleave", "bil")).lower()
    samples = int(hdr["samples"])
    lines = int(hdr["lines"])
    bands = int(hdr["bands"])
    dtype_code = int(hdr["data type"])
    dtype = np.dtype(_CODE_TO_ENVI_DTYPE[dtype_code])

    # data file path: ENVI convention = same root with .bil/.img/.dat
    base, _ = os.path.splitext(hdr_path)
    candidates = [base + ext for ext in (".bil", ".img", ".dat")]
    data_path = None
    for c in candidates:
        if os.path.exists(c):
            data_path = c
            break
    if data_path is None:
        raise FileNotFoundError("Could not find ENVI data file next to header (.bil/.img/.dat).")

    raw = np.fromfile(data_path, dtype=dtype)
    expected = lines * samples * bands
    if raw.size != expected:
        raise ValueError(f"Data size mismatch. expected={expected}, got={raw.size}")

    if interleave == "bil":
        arr = raw.reshape(lines, bands, samples)
        cube = np.transpose(arr, (0, 2, 1))  # (lines, samples, bands)
    elif interleave == "bsq":
        arr = raw.reshape(bands, lines, samples)
        cube = np.transpose(arr, (1, 2, 0))
    elif interleave == "bip":
        cube = raw.reshape(lines, samples, bands)
    else:
        raise ValueError(f"Unsupported interleave: {interleave}")

    return cube, hdr


def _parse_envi_header(hdr_path: str) -> Dict[str, Any]:
    """
    Simple ENVI .hdr parser that understands:
      key = value
      key = { ... } multi-line lists/strings
    """
    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # remove initial ENVI line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].upper() == "ENVI":
        lines = lines[1:]

    hdr: Dict[str, Any] = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        if "=" not in ln:
            i += 1
            continue
        key, val = [x.strip() for x in ln.split("=", 1)]

        # handle { ... } possibly multi-line
        if val.startswith("{") and not val.endswith("}"):
            parts = [val]
            i += 1
            while i < len(lines):
                parts.append(lines[i])
                if lines[i].endswith("}"):
                    break
                i += 1
            val = "\n".join(parts)

        # strip braces
        if val.startswith("{") and val.endswith("}"):
            inner = val[1:-1].strip()
            # try parse comma-separated numeric list
            if "," in inner or "\n" in inner:
                tokens = [t.strip() for t in inner.replace("\n", " ").split(",") if t.strip()]
                nums = []
                ok = True
                for t in tokens:
                    try:
                        nums.append(float(t))
                    except Exception:
                        ok = False
                        break
                hdr[key.lower()] = np.array(nums, dtype=float) if ok else inner
            else:
                hdr[key.lower()] = inner
        else:
            # scalar
            v = val
            # attempt int/float
            try:
                if "." in v or "e" in v.lower():
                    hdr[key.lower()] = float(v)
                else:
                    hdr[key.lower()] = int(v)
            except Exception:
                hdr[key.lower()] = v

        i += 1

    return hdr

def find_latest_envi_prefix(folder: str, prefix: str) -> Optional[str]:
    """
    Finds latest ENVI header with name like: <prefix>*.hdr
    Returns hdr_path or None.
    """
    pattern = os.path.join(folder, f"{prefix}*.hdr")
    hits = glob.glob(pattern)
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p))
    return hits[-1]


def load_envi_cube_if_exists(folder: str, prefix: str):
    """
    Loads latest ENVI cube for prefix. Returns (cube, hdr, hdr_path) or (None, None, None).
    cube is (lines, samples, bands)
    """
    hdr_path = find_latest_envi_prefix(folder, prefix)
    if hdr_path is None:
        return None, None, None
    cube, hdr = read_envi(hdr_path)
    return cube, hdr, hdr_path
