"""Protect the machine: stop translating before it runs out of memory or disk."""
from __future__ import annotations

import re
import sys


_SWAP0 = None


def _swap_mb():
    import subprocess
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=2).stdout
        return float(re.search(r"used = ([\d.]+)M", out).group(1))
    except Exception:
        return None


def overloaded():
    """Stop before the machine thrashes: free disk under 2 GB, macOS reporting CRITICAL memory pressure, or swap
    grown by more than 4 GB since translation started (swap left over from earlier work does not count)."""
    import os
    import shutil
    import subprocess
    global _SWAP0
    if shutil.disk_usage(os.path.expanduser("~")).free < 2e9:     # statvfs does not exist on Windows
        return "free disk below 2 GB"
    if sys.platform != "darwin":                                   # Windows / Linux: almost no memory left
        try:
            import psutil
            if psutil.virtual_memory().available < 0.6e9:
                return "memory is almost exhausted"
        except ImportError:
            pass
    if sys.platform == "darwin":
        try:
            level = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True,
                                       text=True, timeout=2).stdout.strip() or 1)
            if level >= 4:
                return "memory pressure is critical"
        except Exception:
            pass
        used = _swap_mb()
        if used is not None:
            if _SWAP0 is None:
                _SWAP0 = used
            elif used - _SWAP0 > 4096:
                return f"swap grew by {(used - _SWAP0) / 1024:.1f} GB"
    return None
