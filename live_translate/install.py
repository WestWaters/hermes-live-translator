"""One-click setup: build the engine's environments for this hardware and download its models."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from .paths import HERE, MODELS, data_dir


KOKORO = "onnx-community/Kokoro-82M-v1.0-ONNX"           # the voices lt_engine.voices.KokoroVoice uses
KOKORO_VOICES = ["af_heart", "am_michael", "ef_dora", "em_alex", "ff_siwis", "hf_alpha", "hm_omega", "if_sara",
                 "im_nicola", "jf_alpha", "jm_kumo", "pf_dora", "pm_alex", "zf_xiaobei", "zm_yunjian"]

def setup(log=print, download_models=True) -> bool:
    """Create the engine's environment and fetch its models (about 6 GB, once)."""
    from .session import SESSION
    st = SESSION.state
    if st is not None and st.can("setting_up"):
        st.to("setting_up")
    try:
        ok = _setup(log, download_models)
    except Exception as e:
        if st is not None:
            st.to("error", f"setup failed: {e}")
        raise
    if st is not None and st.session == "setting_up":
        st.to("idle" if ok else "error", "" if ok else "this machine is not supported yet")
    return ok

# Voice environments for Windows / Linux: each voice model pins its own library versions (see lt_engine/worker.py).
TORCH, TORCHAUDIO = "torch>=2.8,<3", "torchaudio>=2.8,<3"
TRANSFORMERS, ACCELERATE = "transformers>=4.57,<6", "accelerate>=1.0,<2"
_AUDIO = ["soundfile>=0.13,<0.15", "numpy>=2.0,<3"]
VOICE_ENVS = {
    "qwen-tts": ("3.12", ["qwen-tts==0.1.1"] + _AUDIO, ["Qwen/Qwen3-TTS-12Hz-1.7B-Base"]),   # CPU: 0.6B
    "chatterbox": ("3.14", ["chatterbox-tts==0.1.7"] + _AUDIO, ["ResembleAI/chatterbox"]),
    "omnivoice": ("3.12", ["omnivoice==0.2.1"] + _AUDIO, ["k2-fsa/OmniVoice"]),
}

# Windows on Arm with an NVIDIA GPU (RTX Spark): PyTorch's own index has CPU-only win_arm64 wheels; NVIDIA publishes
# the CUDA builds (torch, torchaudio, torchvision only -- their dependencies still come from PyPI).
NV_WIN_ARM = "https://pypi.nvidia.com/nvtorch_oot/"


def _torch_index(backend):
    import platform as _p
    if backend != "cuda":
        return "https://download.pytorch.org/whl/cpu"
    return "https://download.pytorch.org/whl/cu130" if _p.machine().lower() in ("aarch64", "arm64") else \
        "https://download.pytorch.org/whl/cu128"


def _win_arm():
    import platform as _p
    return os.name == "nt" and _p.machine().lower() in ("arm64", "aarch64")


def _torch_args(backend):
    """Index arguments for torch/torchaudio. NVIDIA's index is an *extra* index (uv prefers it over PyPI for the
    packages it has; pip prefers the +cu134 local version), so torch's own dependencies still resolve from PyPI."""
    import platform as _p
    if backend == "cuda" and os.name == "nt" and _p.machine().lower() in ("arm64", "aarch64"):
        return ["--extra-index-url", NV_WIN_ARM]
    return ["--index-url", _torch_index(backend)]

def _py(env_dir):
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def _python_request(python):
    """uv's Python request. On Windows on Arm uv may pick an x86-64 build (run under emulation), which cannot load
    NVIDIA's win_arm64 CUDA wheels -- ask for the native Arm64 build explicitly."""
    import platform as _p
    if os.name == "nt" and _p.machine().lower() in ("arm64", "aarch64"):
        return f"cpython-{python}-windows-aarch64-none"
    return python


def _make_env(uv, env_dir, python, log):
    if not _py(env_dir).exists():
        log(f"creating {env_dir.name} (Python {python})")
        if uv:
            subprocess.run([uv, "venv", "--python", _python_request(python), str(env_dir)], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)

def _pip(uv, env_dir, args):
    if uv:
        subprocess.run([uv, "pip", "install", "--python", str(_py(env_dir))] + args, check=True)
    else:
        subprocess.run([str(_py(env_dir)), "-m", "pip", "install"] + args, check=True)

def _fetch(env_dir, repos, log):
    """Download models into the Hugging Face cache. "repo#file" fetches one file (a GGUF repo holds every
    quantization -- only the one the engine uses is downloaded)."""
    for item in repos:
        repo, _, filename = item.partition("#")
        log(f"downloading {repo}" + (f" ({filename})" if filename else ""))
        code = (f"from huggingface_hub import hf_hub_download; hf_hub_download({repo!r}, {filename!r})" if filename
                else f"from huggingface_hub import snapshot_download; snapshot_download({repo!r})")
        subprocess.run([str(_py(env_dir)), "-c", code], check=True)


UV = "uv>=0.9,<0.11"        # declared in plugin.yaml (external_dependencies); fetched here only if missing


def find_uv():
    """uv on PATH, or where its installers put it (a background service's PATH often lacks ~/.local/bin)."""
    exe = "uv.exe" if os.name == "nt" else "uv"
    home = os.path.expanduser("~")
    cands = [shutil.which("uv"), os.path.join(home, ".local", "bin", exe), os.path.join(home, ".cargo", "bin", exe),
             str(data_dir() / "tools" / "bin" / exe)]
    return next((c for c in cands if c and os.path.exists(c)), None)


def _local_uv(log):
    """One-click setup without uv on PATH: install uv from PyPI into the plugin's own data folder (nothing is
    written to the user's Python or PATH)."""
    tools = data_dir() / "tools"
    exe = tools / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    if not exe.exists():
        log("installing uv into the plugin's data folder (it builds the engine environments)")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--target", str(tools), UV],
                           capture_output=True, text=True)
        if r.returncode:
            log("could not install uv with this Python's pip: " + (r.stderr or r.stdout).strip()[-300:])
    return str(exe) if exe.exists() else None


def _setup(log, download_models):
    import sys as _s
    if str(HERE) not in _s.path:
        _s.path.insert(0, str(HERE))
    from lt_engine.hw import detect, requirements_file
    backend = detect()
    uv = find_uv() or _local_uv(log)
    if not uv:
        log("uv is not installed and could not be fetched: install it (pip install uv, or "
            "https://docs.astral.sh/uv/) and run setup again; continuing with Python's own venv and pip")
    venv = data_dir() / "venv"
    log(f"this machine: {backend} ({'Apple Silicon / MLX' if backend == 'mlx' else 'NVIDIA GPU' if backend == 'cuda' else 'CPU only'})")
    _make_env(uv, venv, "3.12", log)
    if backend != "mlx":
        _pip(uv, venv, [TORCH] + _torch_args(backend))
    log(f"installing the engine ({requirements_file(backend)})")
    _pip(uv, venv, ["-r", str(HERE / requirements_file(backend))])
    torch_mt = False
    if backend != "mlx":                                   # prebuilt llama.cpp wheels: no compiler needed
        wheels = "https://abetlen.github.io/llama-cpp-python/whl/" + ("cu125" if backend == "cuda" else "cpu")
        try:
            # a prebuilt wheel only (uv and pip spell this the same way): never a long compile during setup
            _pip(uv, venv, ["llama-cpp-python>=0.3.30,<0.4", "--extra-index-url", wheels,
                            "--only-binary", "llama-cpp-python"])
        except subprocess.CalledProcessError:
            if backend == "cuda":                       # ARM64 (DGX / RTX Spark): the same model on PyTorch
                log("no prebuilt llama.cpp for this machine: translating with transformers on the GPU instead")
                _pip(uv, venv, [TRANSFORMERS, ACCELERATE])
                torch_mt = True
            else:
                log("no prebuilt llama.cpp for this machine: translation will use Hermes's own model, a local "
                    "server (Ollama, LM Studio) or a subscription -- pick one under Models")
    if download_models:
        _fetch(venv, ["csukuangfj/speaker-embedding-models#3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"], log)
        _fetch(venv, [r for r, _ in MODELS] if backend == "mlx" else
               [("openai/whisper-large-v3-turbo" if _win_arm() else "mobiuslabsgmbh/faster-whisper-large-v3-turbo")
                if backend == "cuda" else "Systran/faster-whisper-small",
                "Qwen/Qwen3.5-4B" if torch_mt else "unsloth/Qwen3.5-4B-GGUF#Qwen3.5-4B-Q4_K_M.gguf"], log)
        if backend == "cpu":                                # Kokoro: the CPU's natural voices (8 languages)
            _fetch(venv, [f"{KOKORO}#onnx/model_quantized.onnx"] + [f"{KOKORO}#voices/{v}.bin" for v in KOKORO_VOICES],
                   log)
    if backend != "mlx":
        envs = data_dir() / "envs"
        for name, (python, pkgs, repos) in VOICE_ENVS.items():
            d = envs / name
            _make_env(uv, d, python, log)
            _pip(uv, d, [TORCH, TORCHAUDIO] + _torch_args(backend))
            if _win_arm():                                 # soxr has no win_arm64 wheel: use the scipy stand-in
                over = d / "overrides.txt"
                over.write_text("soxr; sys_platform == 'never'\n")
                _pip(uv, d, pkgs + ["scipy>=1.11,<2", "--override", str(over)])
                site = d / "Lib" / "site-packages"
                shutil.copy(HERE / "live_translate" / "compat" / "soxr.py", site / "soxr.py")
            else:
                _pip(uv, d, pkgs)
            if download_models:
                if backend == "cpu" and name == "qwen-tts":
                    repos = ["Qwen/Qwen3-TTS-12Hz-0.6B-Base"]
                _fetch(d, repos, log)
    (venv / ".ready").write_text(f"{backend} {time.strftime('%Y-%m-%d %H:%M')}")
    log("live-translate is ready: `hermes translate start --to <language>` or /translate <language>")
    return True
