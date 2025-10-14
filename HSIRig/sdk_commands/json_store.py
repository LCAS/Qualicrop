
"""
json_store.py — tiny helper to persist/read values in a JSON file.

Features
- Atomic writes (write to temp then os.replace)
- Creates the file and parent folders on first save
- Dot-path access to nested keys: e.g., "camera.exposure.time_ms"
- Convenience methods: get/set/delete/exists, and load/save
"""
from __future__ import annotations
import json, os, tempfile
from typing import Any, Dict, Iterable, List, Optional, Union

JSONDict = Dict[str, Any]
KeyPath = Union[str, Iterable[Union[str, int]]]

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _to_list(path: KeyPath) -> List[Union[str, int]]:
    if isinstance(path, str):
        # support escaping dots with backslash: "a.b\.c" -> ["a", "b.c"]
        parts: List[Union[str, int]] = []
        buf = []
        esc = False
        for ch in path:
            if esc:
                buf.append(ch); esc = False
            elif ch == "\\":
                esc = True
            elif ch == ".":
                parts.append("".join(buf)); buf = []
            else:
                buf.append(ch)
        parts.append("".join(buf))
        # Convert numeric parts to ints for list indexing
        norm: List[Union[str, int]] = []
        for p in parts:
            if isinstance(p, str) and p.isdigit():
                try:
                    norm.append(int(p)); continue
                except ValueError:
                    pass
            norm.append(p)
        return norm
    else:
        return list(path)

def _walk_create(d: JSONDict, path: List[Union[str, int]]) -> Any:
    cur = d
    for i, key in enumerate(path[:-1]):
        nxt = path[i + 1]
        if isinstance(key, int):
            if not isinstance(cur, list):
                raise TypeError(f"Expected list at segment {key}, found {type(cur).__name__}")
            while len(cur) <= key:
                cur.append({} if not isinstance(nxt, int) else [])
            cur = cur[key]
        else:
            if key not in cur or not isinstance(cur[key], (dict, list)):
                cur[key] = {} if not isinstance(nxt, int) else []
            cur = cur[key]
    return cur

def _walk_get(d: Any, path: List[Union[str, int]], default: Any) -> Any:
    cur = d
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list) or key >= len(cur):
                return default
            cur = cur[key]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
    return cur

def atomic_write(path: str, data: Union[str, bytes], mode: str = "w", encoding: Optional[str] = "utf-8") -> None:
    _ensure_dir(path)
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, mode, encoding=encoding) as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

def load(path: str) -> JSONDict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            obj = {}
    if not isinstance(obj, dict):
        return {"_": obj}
    return obj

def save(path: str, data: JSONDict) -> None:
    atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False))

def get(path: str, key: KeyPath, default: Any = None) -> Any:
    doc = load(path)
    return _walk_get(doc, _to_list(key), default)

def exists(path: str, key: KeyPath) -> bool:
    sentinel = object()
    return get(path, key, sentinel) is not sentinel

def set(path: str, key: KeyPath, value: Any) -> JSONDict:
    doc = load(path)
    kp = _to_list(key)
    if not kp:
        if isinstance(value, dict):
            doc = value
        else:
            raise ValueError("Top-level set requires a dict value")
    else:
        parent = _walk_create(doc, kp)
        last = kp[-1]
        if isinstance(last, int):
            if not isinstance(parent, list):
                raise TypeError(f"Expected list at final segment, found {type(parent).__name__}")
            while len(parent) <= last:
                parent.append(None)
            parent[last] = value
        else:
            if not isinstance(parent, dict):
                raise TypeError(f"Expected dict at final segment, found {type(parent).__name__}")
            parent[last] = value
    save(path, doc)
    return doc

def delete(path: str, key: KeyPath) -> JSONDict:
    doc = load(path)
    kp = _to_list(key)
    if not kp:
        return doc
    parent = _walk_get(doc, kp[:-1], None)
    if parent is None:
        return doc
    last = kp[-1]
    if isinstance(last, int):
        if isinstance(parent, list) and 0 <= last < len(parent):
            parent.pop(last)
    else:
        if isinstance(parent, dict):
            parent.pop(last, None)
    save(path, doc)
    return doc

class JSONStore:
    def __init__(self, path: str) -> None:
        self.path = path
    def load(self) -> JSONDict:
        return load(self.path)
    def save(self, data: JSONDict) -> None:
        save(self.path, data)
    def get(self, key: KeyPath, default: Any = None) -> Any:
        return get(self.path, key, default)
    def set(self, key: KeyPath, value: Any) -> JSONDict:
        return set(self.path, key, value)
    def exists(self, key: KeyPath) -> bool:
        return exists(self.path, key)
    def delete(self, key: KeyPath) -> JSONDict:
        return delete(self.path, key)
