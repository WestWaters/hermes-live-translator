import os


def device():
    import torch
    if os.environ.get("LIVE_TRANSLATE_DEVICE"):
        return os.environ["LIVE_TRANSLATE_DEVICE"]
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def dtype(dev):
    import torch
    return torch.bfloat16 if dev.startswith("cuda") else torch.float32


def limit_threads():
    """LIVE_TRANSLATE_THREADS caps CPU threads (leave room for games, calls and everything else)."""
    n = os.environ.get("LIVE_TRANSLATE_THREADS")
    if n:
        import torch
        torch.set_num_threads(int(n))
