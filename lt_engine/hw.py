"""Pick the engine's backends from the hardware, automatically: nobody should have to configure this.

  mlx    Apple Silicon Mac                  -> MLX models (mlx-whisper, mlx-audio, mlx-lm)
  cuda   NVIDIA GPU (Windows / Linux / WSL) -> PyTorch models on the GPU
  cpu    anything else                      -> small models on the CPU (slower; captions first, voice second)
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def detect() -> str:
    forced = os.environ.get("LIVE_TRANSLATE_BACKEND")
    if forced in ("mlx", "cuda", "cpu"):
        return forced
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    smi = nvidia_smi()
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            if out:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def nvidia_smi():
    """nvidia-smi, also where WSL keeps it (/usr/lib/wsl/lib is not on a service's PATH)."""
    import os
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for p in ("/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi"):
        if os.path.exists(p):
            return p
    return None


def gpu_memory_gb() -> float:
    try:
        out = subprocess.run([nvidia_smi() or "nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[0]) / 1024 if out else 0.0
    except Exception:
        return 0.0


def requirements_file(backend: str) -> str:
    return {"mlx": "requirements/mac.txt", "cuda": "requirements/cuda.txt", "cpu": "requirements/cpu.txt"}[backend]
