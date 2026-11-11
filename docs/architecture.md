# Architecture

Live Translate has three parts that talk over simple, inspectable interfaces.

| Part | Runs in | Job |
|---|---|---|
| `live_translate/` | Hermes's own Python | agent tools, `/translate`, `hermes translate`, setup, model detection, the local server, the session's state machine |
| `lt_engine/` | its own environment (`plugin-data/live-translate/venv`) | audio in/out, recognition, translation, voices — the real-time work |
| `app/` + `tray/` | a web view (desktop app or browser) | the window and tray icon; drives the session through the local server |

## The session (live_translate/)

`state.py` makes the session explicit: `not_set_up → setting_up → idle → starting → live → stopping → idle`, with
`error` reachable from any running state and retryable. While live, four facets describe the moment:
`activity` (waiting / hearing / translating / speaking), `language`, `voice` (stock / cloned) and `pace`
(on time / catching up). The agent's tools, `/translate`, the window and the tray icon all read this one model.

`session.py` starts the engine as a child process and reads its JSON-lines events (`heard`, `caption`, `latency`,
`voice`, `pace`, `stopped`). Live commands go the other way on the engine's stdin (`{"volume": 0.8}`). *Stop*
ends the engine in about a second.

`server.py` serves the window (`/app`), the OBS caption page (`/`) and a small API on `127.0.0.1` only. Changing
requests need the per-run token written into `/app` and a same-origin (or the desktop app's) `Origin`.

`detect.py` lists what the user can translate with: built-in models for the detected hardware, running local
servers (Ollama, LM Studio, llama-server, vLLM), subscriptions whose keys Hermes holds, and Hermes's own model.
`keys_for()` hands the engine only the key its chosen services need.

## The engine (lt_engine/)

```text
audio.py      FileSource (any audio/video via ffmpeg) · MicSource · Sink (speakers or a virtual mic, live gain)
asr.py        Recognizer: identifies the language by two votes (Whisper + Qwen3-ASR, >= 3 s of speech), re-checks it
              on every phrase, then transcribes with the fastest engine that knows it; LocalAgreement-2 commits
              long speech early, never retracts
mt.py         translators: MLX, llama.cpp (GGUF) or any OpenAI-compatible server; recent phrase pairs as context
voices.py     Router: per-language routes to voices (mlx-audio models, worker environments, OpenAI-compatible,
              ElevenLabs, any command, the OS voice); speaker gender for natural voices
aec.py        echo cancellation (WebRTC AEC3 via LiveKit): the microphone never hears our own translation
speakers.py   who is speaking (CAM++ voiceprints): each person keeps their own language and voice
worker.py     the voice worker protocol (JSON lines on stdin/stdout) and its server side
workers/      Qwen3-TTS, Chatterbox, OmniVoice workers for PyTorch (CUDA or CPU)
pipeline.py   lanes (one per direction), voice-activity segmentation, the three-process pipeline, catch-up
stretch.py    pitch-preserving speed-up (WSOLA)
guard.py      stops before the machine thrashes (memory pressure, swap growth, free disk)
hw.py         mlx / cuda / cpu detection
langs.py      language names and which voices cover them
```

Recognition runs in the main engine process; translation and speech run in two more processes, so the next phrase
is heard and translated while the previous one is spoken. Each phrase records its delay: the time from the
speaker finishing it to the listener hearing it.

### Two-way

A conversation is two lanes sharing the loaded models: *them → you* (the call's audio → your speakers) and
*you → them* (your microphone → the virtual microphone the meeting app uses). Each lane identifies its own
speaker's language and clones its own speaker's voice.

### Real rooms and real meetings

Measured on a noisy test meeting (eight turns by six speakers in Chinese, English, Japanese and Hindi, one
speaker 8 dB quieter, an overlap, other people talking across the room, room noise):

| Problem | What handles it |
|---|---|
| Our translation coming back through the microphone | WebRTC echo cancellation fed exactly what the speakers play (24-40 dB removed; a person talking over the translation is still transcribed word for word); without the package, the engine stops listening while it speaks. A text check drops anything matching what was just said. |
| People talking in the background, a TV | A loudness gate follows the person being translated; voices more than 14 dB quieter are background (the bar lowers slowly if they keep talking). If a much louder voice appears early, what came before was background and its language is discarded. |
| Several speakers | CAM++ voiceprints (Apache-2.0): a 3 s phrase that matches no one, or two shorter ones, is a new person with their own language and cloned voice. |
| Switching languages | Every phrase of 2 s or more is identified again; a confident change is confirmed by the two-detector vote. Hindi/Urdu and Mandarin/Cantonese, which Whisper confuses, are settled by Qwen3-ASR. |
| Short replies ("yes", "sí") | They keep the speaker's language and are never dropped; only lone words much quieter than the speaker are. |
| Recognisers inventing text in noise | Phrases that are a known hallucination ("Thank you for watching", subtitle credits, a lone "Music") are dropped. |
| A voice model that never stops | Each sentence is cut at a generous length for its text (one had generated 460 s for one sentence). |
| Long sessions | The translator's prompt stays the same size (the last two phrase pairs), per-phrase records are released once spoken, and a crashed stage stops the session with a message instead of hanging. |

### Keeping up

When translated speech queues up behind the speaker, the sink plays it up to 30% faster at the same pitch and the
translator is asked to condense, until the queue drains.
