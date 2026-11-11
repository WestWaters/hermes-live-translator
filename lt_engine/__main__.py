"""python -m lt_engine --to es [--from zh] (--mic | --file talk.wav) [--out-device "BlackHole 2ch"] [--record out.wav]

Events go to stdout as JSON lines (heard / caption / latency / spoken / summary / stopped); logs go to stderr.
SIGINT or SIGTERM stops cleanly: phrases already heard are still translated and spoken."""
import argparse
import json
import signal
import sys
import threading


def list_devices():
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append({"index": i, "name": d["name"], "inputs": d["max_input_channels"],
                    "outputs": d["max_output_channels"]})
    print(json.dumps({"event": "devices", "devices": out}, ensure_ascii=False))


def _cuda_libs():
    """faster-whisper (CTranslate2) needs cuBLAS / cuDNN: point it at the pip-installed NVIDIA and torch libraries."""
    import os
    import site
    roots = [p for p in site.getsitepackages() if os.path.isdir(p)]
    dirs = []
    for r in roots:
        for sub in ("torch/lib", "nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cublas/lib", "nvidia/cudnn/lib"):
            d = os.path.join(r, sub)
            if os.path.isdir(d):
                dirs.append(d)
    for d in dirs:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])


def main(argv=None):
    _cuda_libs()
    ap = argparse.ArgumentParser(prog="lt_engine", description="Live speech-to-speech translation")
    ap.add_argument("--to", help="listener's language (code or name)")
    ap.add_argument("--from", dest="src", help="speaker's language (default: detect)")
    ap.add_argument("--mic", action="store_true", help="translate live audio from an input device")
    ap.add_argument("--file", help="translate an audio file, played at real speed")
    ap.add_argument("--in-device", help="input device name or index (default microphone, or e.g. 'BlackHole 2ch' "
                                        "to translate what a meeting app plays)")
    ap.add_argument("--out-device", help="output device name or index (e.g. 'BlackHole 2ch' as a virtual mic)")
    ap.add_argument("--two-way", action="store_true",
                    help="also translate YOUR speech into the other side's language (--from), e.g. into a meeting")
    ap.add_argument("--my-mic", help="with --two-way: your microphone (default input device)")
    ap.add_argument("--my-file", help="with --two-way: a file standing in for your microphone (testing)")
    ap.add_argument("--to-them", help="with --two-way: where your translated voice goes (e.g. 'BlackHole 2ch')")
    ap.add_argument("--no-play", action="store_true", help="do not play the translation")
    ap.add_argument("--record", help="write a timeline recording: original (quiet) + translation as heard")
    ap.add_argument("--fast", action="store_true", help="with --file: do not wait for real time")
    ap.add_argument("--no-clone", action="store_true", help="do not clone the speaker's voice")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--volume", type=float, default=1.0, help="translation volume, 0-1.5 (live: send {\"volume\": v} on stdin)")
    ap.add_argument("--translator", dest="mt_model",
                    help="'mlx:<repo>' (default Qwen3.5-4B) or 'openai:<base url>#<model>' (Ollama, LM Studio, "
                         "llama-server, or a cloud API)")
    ap.add_argument("--recognizer", dest="asr_route",
                    help="'auto' (Qwen3-ASR for its 30 languages, Whisper for the rest), 'whisper', or a route like "
                         "'*=mlx-audio:<repo>' (Nemotron, Parakeet, ...) or '*=openai:<base url>#<model>'")
    ap.add_argument("--voices", dest="tts_model",
                    help="voice preset ('commercial', the default = the speaker's own voice in 23 languages, models free for any "
                         "use; 'best' adds natural voices in 600+ (OmniVoice, non-commercial)) or a route like 'zh,ja=mlx-audio:<repo>; *=elevenlabs:eleven_flash_v2_5'")
    a = ap.parse_args(argv)
    if a.list_devices:
        list_devices()
        return 0
    if not a.to or not (a.mic or a.file):
        ap.error("--to and one of --mic / --file are required")
    from .pipeline import FileSource, MicSource, Pipeline
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", None)):   # SIGBREAK: Windows stop
        if sig is not None:
            signal.signal(sig, lambda *_: stop.set())
    dev = lambda d: int(d) if d is not None and str(d).isdigit() else d  # noqa: E731
    p = Pipeline(a.to, a.src, mt_model=a.mt_model, tts_model=a.tts_model, clone=not a.no_clone,
                 asr_route=a.asr_route, volume=a.volume)
    if a.file:
        source = FileSource(a.file, realtime=not a.fast)
        original = source.audio
    else:
        source = MicSource(dev(a.in_device))
        original = None
    back = None
    if a.two_way:
        back = FileSource(a.my_file, realtime=not a.fast) if a.my_file else MicSource(dev(a.my_mic))
    p.run(source, sink_play=not a.no_play, out_device=dev(a.out_device), record=a.record, original=original,
          stop=stop, back_source=back, back_out_device=dev(a.to_them))
    return 0


if __name__ == "__main__":
    sys.exit(main())
