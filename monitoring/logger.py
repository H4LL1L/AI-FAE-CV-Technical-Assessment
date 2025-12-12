from __future__ import annotations

import atexit
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class GpuMonitor:
    """
    Minimal GPU monitor using pynvml if available.
    Returns None on CPU-only systems.
    """

    _nvml_lock = threading.Lock()
    _nvml_refcount = 0
    _nvml_inited = False

    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = gpu_index
        self._closed = False
        try:
            import pynvml

            self._pynvml = pynvml
            with self._nvml_lock:
                if not self._nvml_inited:
                    pynvml.nvmlInit()
                    self._nvml_inited = True
                self._nvml_refcount += 1
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                # If handle acquisition fails, undo init refcount.
                with self._nvml_lock:
                    self._nvml_refcount = max(0, self._nvml_refcount - 1)
                    if self._nvml_refcount == 0 and self._nvml_inited:
                        try:
                            pynvml.nvmlShutdown()
                        except Exception:
                            pass
                        self._nvml_inited = False
                self._handle = None
            atexit.register(self.close)
        except Exception:
            self._pynvml = None
            self._handle = None

    def close(self) -> None:
        """
        Best-effort NVML cleanup. Safe to call multiple times.
        """
        if self._pynvml is None or self._closed:
            return
        try:
            with self._nvml_lock:
                self._nvml_refcount = max(0, self._nvml_refcount - 1)
                if self._nvml_refcount == 0 and self._nvml_inited:
                    self._pynvml.nvmlShutdown()
                    self._nvml_inited = False
        except Exception:
            pass
        finally:
            self._closed = True
            self._pynvml = None
            self._handle = None

    def read(self) -> Optional[Dict[str, Any]]:
        if self._pynvml is None or self._handle is None:
            return None
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            name = self._pynvml.nvmlDeviceGetName(self._handle).decode("utf-8")
            return {
                "name": name,
                "gpu_util_percent": util.gpu,
                "mem_util_percent": round(mem.used / mem.total * 100, 2),
            }
        except Exception:
            return None


class JsonlLogger:
    """Append metrics as JSON lines."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, data: Dict[str, Any]) -> None:
        line = json.dumps(data, default=_json_default, ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8")


def _json_default(obj: object) -> object:
    """
    JSON serializer fallback for common non-JSON types (numpy scalars/arrays, Path, bytes).
    """
    try:
        import numpy as np  # local import: optional dependency

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<{type(obj).__name__}:{len(obj)} bytes>"

    return repr(obj)


