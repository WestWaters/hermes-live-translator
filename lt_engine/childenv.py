"""The environment child processes get: just what Python, audio, the GPU and Hugging Face need, plus the one key the
selected service needs. Never a copy of the parent's environment (it can hold every key the user has)."""
import os

KEEP = {
    # processes, paths and temp
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TEMP", "TMP",
    # Windows: the OS, the user profile, and what platform.machine() reads
    "SYSTEMROOT", "SystemRoot", "WINDIR", "SYSTEMDRIVE", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    "PROGRAMFILES", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
    # locale and text
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "PYTHONUTF8", "PYTHONIOENCODING",
    # audio on Linux (PulseAudio / PipeWire) and GPU selection
    "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "PULSE_SERVER", "PIPEWIRE_REMOTE", "CUDA_VISIBLE_DEVICES",
    # network plumbing for the user's own subscriptions behind a proxy / custom CA
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy", "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    # where Hermes and the plugin keep their data
    "HERMES_HOME",
}
PREFIXES = ("HF_", "HUGGINGFACE_HUB_", "LIVE_TRANSLATE_", "LT_", "OMP_", "MKL_")
SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def minimal_env(extra=None, keys=None, base=None):
    """`extra`: plain settings to add (PYTHONPATH, ...). `keys`: the secrets this child needs, and only those."""
    base = os.environ if base is None else base
    env = {}
    for k, v in base.items():
        secret = k.upper().endswith(SECRET_SUFFIXES)
        if (k in KEEP or k.startswith(PREFIXES)) and not secret:
            env[k] = v
    env.update({k: v for k, v in (extra or {}).items() if v is not None})
    env.update({k: v for k, v in (keys or {}).items() if v})
    return env
