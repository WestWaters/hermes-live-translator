# Changelog

All notable changes to this plugin are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-11

### Added
- Real-time speech-to-speech translation: ~100 languages heard, 201 translated, the speaker's own voice in 23 and
  a natural voice in 600+.
- One-way, two-way (conversation) and call modes; live captions in the app, as an OBS browser source and as OBS
  closed captions viewers can switch on.
- Real rooms: echo cancellation (WebRTC AEC3), background voices ignored, several speakers each with their own
  language and voice, language switches followed, recogniser hallucinations dropped.
- Automatic hardware detection (Apple Silicon, NVIDIA, CPU) and one-click setup; on CPU-only machines faster
  recognition (Whisper small) and natural Kokoro voices.
- Model picker that finds built-in models, local servers (Ollama, LM Studio, llama-server, vLLM), subscriptions and
  *My Hermes model* (through Hermes's plugin LLM access; no key leaves Hermes).
- Desktop app (Tauri) for macOS, Windows and Linux on x64 and ARM64: tray icon that glows while listening, nine
  Hermes skins plus *Match my Hermes*, seven more themes.
- Agent skill, `/translate`, `hermes translate`.

### Security
- Keys only through Hermes's secret store; starting from chat apps or scheduled jobs refused unless allowed; the
  engine stops with the session that started it.
