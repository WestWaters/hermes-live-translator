"""Model pinning (setup and runtime load the same commit) and the UTF-8 guard. No optional dependencies: Hugging Face
is faked."""
import re
import sys
import types
from pathlib import Path

import pytest

from lt_engine import pins

ROOT = Path(__file__).resolve().parent.parent


def test_every_pin_is_a_full_commit():
    assert pins.REVISIONS and all(re.fullmatch(r"[0-9a-f]{40}", v) for v in pins.REVISIONS.values())


def test_every_model_the_code_names_is_pinned():
    """Any "org/name" model id written in the engine or setup code must be in REVISIONS."""
    known_orgs = ("mlx-community", "Qwen", "openai", "Systran", "dropbox-dash", "unsloth", "ResembleAI", "k2-fsa",
                  "onnx-community", "csukuangfj")
    named = set()
    for f in list((ROOT / "lt_engine").rglob("*.py")) + list((ROOT / "live_translate").rglob("*.py")):
        if f.name == "pins.py":
            continue
        for m in re.finditer(r"""["':= ]((?:%s)/[A-Za-z0-9._-]+)""" % "|".join(known_orgs), f.read_text(encoding="utf-8")):
            named.add(m.group(1).rstrip(".,"))
    missing = sorted(n for n in named if n not in pins.REVISIONS)
    assert not missing, missing


@pytest.fixture
def hub(monkeypatch):
    calls, cached = [], {}
    def snapshot_download(repo, revision=None, local_files_only=False):
        calls.append((repo, revision, local_files_only))
        if local_files_only and (repo, revision) not in cached:
            raise FileNotFoundError(repo)
        return f"/cache/{repo}@{revision}"
    def hf_hub_download(repo, filename, revision=None, local_files_only=False):
        calls.append((repo, filename, revision, local_files_only))
        if local_files_only and (repo, revision) not in cached:
            raise FileNotFoundError(repo)
        return f"/cache/{repo}@{revision}/{filename}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        snapshot_download=snapshot_download, hf_hub_download=hf_hub_download))
    return calls, cached


def test_the_pinned_commit_loads_from_the_cache_without_network(hub):
    calls, cached = hub
    repo = "Qwen/Qwen3.5-4B"
    cached[(repo, pins.REVISIONS[repo])] = True
    assert pins.local(repo) == f"/cache/{repo}@{pins.REVISIONS[repo]}"
    assert calls == [(repo, pins.REVISIONS[repo], True)]


def test_a_moved_repo_uses_the_copy_cached_under_its_old_name(hub):
    calls, cached = hub
    new = "dropbox-dash/faster-whisper-large-v3-turbo"
    cached[(pins.ALIASES[new], pins.REVISIONS[new])] = True
    assert pins.local(new).startswith(f"/cache/{pins.ALIASES[new]}@")


def test_not_cached_downloads_the_pinned_commit_once(hub):
    calls, cached = hub
    repo = "csukuangfj/speaker-embedding-models"
    p = pins.local(repo, "model.onnx")
    assert p.endswith(f"@{pins.REVISIONS[repo]}/model.onnx") and calls[-1][-1] is False


def test_the_users_own_model_loads_as_given(hub):
    assert pins.local("someone/their-own-model") == "someone/their-own-model"


def test_text_io_always_names_its_encoding():
    """Windows' default encoding is not UTF-8: every text read/write and text pipe must say encoding="utf-8"."""
    bad = []
    for f in list((ROOT / "lt_engine").rglob("*.py")) + list((ROOT / "live_translate").rglob("*.py")):
        src = f.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            if re.search(r"\.read_text\(\)|\.write_text\([^)]*\)\s*$", line) and "encoding" not in line:
                bad.append(f"{f.name}:{i}")
            if re.search(r"(?<![.\w])open\([^)]*[\"'](a|w|r)[\"']\)", line) and "encoding" not in line:
                bad.append(f"{f.name}:{i}")
    assert not bad, bad
