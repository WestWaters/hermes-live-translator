"""Every model this plugin downloads, pinned to the exact Hugging Face commit it was tested with. Setup downloads those
commits and the engine loads them from the local cache by the same commit, so setup and runtime always agree, a
repo changing upstream cannot change what runs, and starting a translation needs no network. A model the user picks
themselves (a repo id that is not listed here) loads as they gave it."""
REVISIONS = {
    "mlx-community/Qwen3-ASR-0.6B-8bit": "89e96d92ba34aca20b3e29fb10cc284097d1219f",
    "mlx-community/whisper-large-v3-turbo": "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    "mlx-community/Qwen3.5-4B-MLX-4bit": "32f3e8ecf65426fc3306969496342d504bfa13f3",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit": "e7dd0585652209fa0d7783659aad4e8a324de11c",
    "mlx-community/chatterbox-4bit": "f1d7b9696e1b6242e64eb8c4a823b6d1a50425a8",
    "mlx-community/S3TokenizerV2": "e0c9886f0e1c35ae85b1f27277416fb19fc72bec",
    "mlx-community/OmniVoice-bf16": "8fb0b754cad788aaefec690cd55c207e8a628f85",
    "unsloth/Qwen3.5-4B-GGUF": "e87f176479d0855a907a41277aca2f8ee7a09523",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "openai/whisper-large-v3-turbo": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    "dropbox-dash/faster-whisper-large-v3-turbo": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    "Systran/faster-whisper-small": "536b0662742c02347bc0e980a01041f333bce120",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "fd4b254389122332181a7c3db7f27e918eec64e3",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "5d83992436eae1d760afd27aff78a71d676296fc",
    "ResembleAI/chatterbox": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
    "k2-fsa/OmniVoice": "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
    "onnx-community/Kokoro-82M-v1.0-ONNX": "1939ad2a8e416c0acfeecc08a694d14ef25f2231",
    "csukuangfj/speaker-embedding-models": "0743f301363dec56491a490f6d6cbc9d67f9a3bf",
}
# Loaded by name inside a library (mlx-audio's Chatterbox fetches its tokenizer itself): setup also fetches the
# library's default, so it is cached and nothing is downloaded while translating.
LIBRARY_RESOLVED = {"mlx-community/S3TokenizerV2"}
# Repos that moved: a copy cached under the old name is still used (no re-download).
ALIASES = {"dropbox-dash/faster-whisper-large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}


def local(repo, filename=None):
    """Local path of `repo` (or one `filename` in it) at its pinned commit: from the cache when it is there,
    downloaded once otherwise. Unlisted repos load as given (the user's own choice)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    rev = REVISIONS.get(repo)

    def get(r, offline):
        if filename:
            return hf_hub_download(r, filename, revision=rev, local_files_only=offline)
        return snapshot_download(r, revision=rev, local_files_only=offline)

    if rev is None:
        return hf_hub_download(repo, filename) if filename else repo
    for r in (repo, ALIASES.get(repo)):
        if r:
            try:
                return get(r, True)
            except Exception:                                  # noqa: BLE001 -- not cached at this commit yet
                pass
    return get(repo, False)
