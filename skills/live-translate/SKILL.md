---
name: live-translate
description: Real-time speech-to-speech translation on this computer. Use when the user wants to understand or talk with someone in another language — in person, on a call (Zoom, Meet, Teams, Discord), on a stream, or from a recording — or asks to translate what someone is saying live.
---

# Live Translate

Live Translate listens to speech, translates it, and speaks it in the listener's language a couple of seconds
behind, in the speaker's own voice where it can (23 languages) and a natural voice otherwise (600+). It understands
about 100 languages and detects the speaker's language by itself. Everything runs on this computer unless the user
picked a cloud model in Settings.

## Tools

| Tool | Use it to |
|---|---|
| `translate_start` | start translating. `target_language` = what the user wants to HEAR (code or name). |
| `translate_status` | see what is happening: state, detected language, latest captions, delay. |
| `translate_stop` | stop. Stops within about a second. |
| `translate_languages` | answer "which languages?" — counts and which ones keep the speaker's own voice. |

## Voice commands → tool calls

Users often say these to Hermes out loud ("Hey Hermes, …"). Map them directly, without asking follow-up questions
when the meaning is clear:

| The user says | Call |
|---|---|
| "translate this", "what are they saying?", "translate into Spanish" | `translate_start(target_language=<their language>)` (mode `listen`) |
| "video call", "translate this call", "call my grandma in Korean" | `translate_start(mode="call", target_language=<user's language>, source_language=<the other person's>)` |
| "stream", "caption my stream in Spanish" | `translate_start(mode="stream", target_language=<viewers' language>)` |
| "end call", "end stream", "stop translating", "that's enough" | `translate_stop()` |
| "how's it going?", "what did she say?" | `translate_status()` |
| "which languages?" | `translate_languages()` |

The user's language is the one they talk to Hermes in, unless they say otherwise.

## Modes

- **listen** (default): the other person speaks, the user hears it in their language, in the speaker's own voice.
  Leave `source_language` out — it is detected.
- **call**: both ways. The meeting's audio is translated for the user, and the user's speech is translated into the
  call in their own voice through a virtual microphone. Needs `source_language` (the other person's language).
  The virtual audio devices are found automatically; if there aren't two, tell the user the exact message the tool
  returns (it names what to install). The meeting app's own settings are never changed — the user picks the
  virtual mic in the meeting app once.
- **stream**: captions only — OBS closed captions viewers can switch on, plus the caption page (an OBS Browser
  Source). No voice is played, so the stream keeps the streamer's real voice.

## Good habits

- Ask which language the user wants to hear if it is not obvious; never guess the other person's language when
  `source_language` is optional — detection is better.
- After starting, check `translate_status` once: state `starting` means models are loading (20–60 s);
  `live` means it is listening.
- If the state is `error`, read `message` and tell the user plainly (for example "no microphone found: plug one
  in"). Do not retry in a loop.
- Translation uses whatever models the user picked in the app's Settings → Models (built-in, a local server such
  as Ollama, or a subscription). Do not change them unless asked.
- The window and tray icon are the "Live Translate" app; the user can also do everything there.
