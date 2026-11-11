import numpy as np
import pytest

from lt_engine import voices


@pytest.mark.parametrize("backend,first", [("mlx", "mlx-audio:mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"),
                                          ("cuda", "env:qwen-tts"),
                                          ("cpu", "kokoro")])
def test_presets_follow_the_hardware(monkeypatch, backend, first):
    monkeypatch.setenv("LIVE_TRANSLATE_BACKEND", backend)
    r = voices.Router("best")
    assert r.entries[0][1] == first
    assert "*" in r.entries[-1][0]                       # every language has a voice


def test_routes_pick_the_first_matching_voice(monkeypatch):
    r = voices.Router("ja,zh=command:echo {text}; *=system")
    assert isinstance(r.voice("ja"), voices.CommandVoice)
    assert isinstance(r.voice("sw"), voices.SystemVoice)


def test_mlx_audio_spec_options():
    v = voices.make("mlx-audio:mlx-community/OmniVoice-bf16?clone=0&steps=16")
    assert v.repo == "mlx-community/OmniVoice-bf16" and v.clone is False and v.steps == 16


@pytest.mark.parametrize("f0,gender", [(115, "male"), (225, "female")])
def test_speaker_gender_from_pitch(f0, gender):
    t = np.arange(16000 * 2) / 16000
    tone = 0.3 * np.sign(np.sin(2 * np.pi * f0 * t)).astype(np.float32)   # buzzy, voice-like
    assert voices.speaker_gender(tone) == gender


def test_new_speaker_needs_two_phrases():
    import numpy as np
    from lt_engine.speakers import SpeakerTracker
    t = SpeakerTracker.__new__(SpeakerTracker)
    t.people, t.counts, t._last, t._pending = [], [], 0, None
    a, b = np.zeros(8, np.float32), np.zeros(8, np.float32)
    a[0], b[1] = 1, 1
    t.ex = object()
    seq = iter([a, b, a, b, b])
    t._embed = lambda audio: next(seq)
    two_s = np.zeros(32000, np.float32)
    assert [t.identify(two_s) for _ in range(5)] == [0, 0, 0, 0, 1]     # b once is noise; b twice is someone
    t.people, t.counts, t._last, t._pending = [], [], 0, None
    seq = iter([a, b])
    assert [t.identify(np.zeros(16000 * 4, np.float32)) for _ in range(2)] == [0, 1]   # one long phrase is enough


def test_setup_downloads_every_kokoro_voice_the_engine_uses():
    from live_translate import install
    from lt_engine.voices import KokoroVoice
    assert install.KOKORO == KokoroVoice.REPO
    assert set(install.KOKORO_VOICES) == {v for pair in KokoroVoice.VOICES.values() for v in pair}


def test_cpu_preset_speaks_kokoro_languages_with_kokoro():
    from lt_engine.voices import PRESETS
    entries = dict(part.split("=", 1)[::-1] for part in PRESETS["best-cpu"].split("; "))
    assert set(entries["kokoro"].split(",")) == {"en", "es", "fr", "hi", "it", "pt", "ja", "zh"}


def test_finds_nvidia_smi_where_wsl_keeps_it(monkeypatch):
    import os
    import shutil
    from lt_engine import hw
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(os.path, "exists", lambda p: p == "/usr/lib/wsl/lib/nvidia-smi")
    assert hw.nvidia_smi() == "/usr/lib/wsl/lib/nvidia-smi"


def test_setup_finds_uv_where_its_installer_puts_it(monkeypatch, tmp_path):
    import os
    import shutil
    from live_translate import install
    uv = tmp_path / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    uv.parent.mkdir(parents=True)
    uv.write_text("")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
    monkeypatch.setattr(install, "data_dir", lambda: tmp_path / "data")
    assert install.find_uv() == str(uv)


@pytest.mark.parametrize("backend", ["mlx", "cuda", "cpu"])
def test_the_default_voices_are_free_for_any_use(monkeypatch, backend):
    monkeypatch.setenv("LIVE_TRANSLATE_BACKEND", backend)
    route = " ".join(spec for _, spec in voices.Router().entries).lower()
    assert "omnivoice" not in route                      # CC-BY-NC: only in the opt-in "best" preset
