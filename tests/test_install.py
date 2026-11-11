"""Which PyTorch index the one-click setup uses on each machine."""
import os
import platform

from live_translate import install


def _on(monkeypatch, osname, machine):
    monkeypatch.setattr(os, "name", osname)
    monkeypatch.setattr(platform, "machine", lambda: machine)


def test_windows_arm_nvidia_uses_nvidia_index(monkeypatch):
    _on(monkeypatch, "nt", "ARM64")                       # RTX Spark on Windows 11
    assert install._torch_args("cuda") == ["--extra-index-url", install.NV_WIN_ARM]


def test_windows_arm_cpu_uses_pytorch_cpu(monkeypatch):
    _on(monkeypatch, "nt", "ARM64")
    assert install._torch_args("cpu") == ["--index-url", "https://download.pytorch.org/whl/cpu"]


def test_linux_arm_cuda_and_windows_x86_cuda_unchanged(monkeypatch):
    _on(monkeypatch, "posix", "aarch64")                  # DGX Spark / RTX Spark under Linux
    assert install._torch_args("cuda") == ["--index-url", "https://download.pytorch.org/whl/cu130"]
    _on(monkeypatch, "nt", "AMD64")
    assert install._torch_args("cuda") == ["--index-url", "https://download.pytorch.org/whl/cu128"]


def test_windows_arm_asks_uv_for_native_arm_python(monkeypatch):
    _on(monkeypatch, "nt", "ARM64")
    assert install._python_request("3.12") == "cpython-3.12-windows-aarch64-none"
    _on(monkeypatch, "nt", "AMD64")
    assert install._python_request("3.12") == "3.12"
    _on(monkeypatch, "posix", "aarch64")
    assert install._python_request("3.14") == "3.14"


def test_soxr_stand_in_resamples_like_soxr():
    import numpy as np
    from live_translate.compat import soxr
    x = np.sin(np.linspace(0, 200 * np.pi, 24000)).astype(np.float32)   # 1 s at 24 kHz
    y = soxr.resample(x, 24000, 16000)
    assert y.dtype == np.float32 and abs(len(y) - 16000) <= 1
    stereo = np.stack([x, x], axis=1)
    assert soxr.resample(stereo, 24000, 16000).shape == (16000, 2)
