"""Tool schemas: what the agent reads to decide when to call each tool."""

TRANSLATE_START = {
    "name": "translate_start",
    "description": (
        "Start live speech-to-speech translation on this computer. Whatever is spoken into the input (the "
        "microphone, or a meeting/stream routed through a virtual audio device) is heard in the target language, "
        "in the speaker's own voice, a few seconds behind. The speaker's language is detected automatically. Use "
        "when the user wants to talk with someone in another language, follow a foreign-language meeting or "
        "stream, or have their own voice translated for others."),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["listen", "call", "stream"],
                     "description": "listen (default): what is heard plays translated, in the speaker's voice "
                                    "('Hey Hermes, translate this'). call: a two-way video/voice call -- the "
                                    "meeting's audio is translated for the user and the user's speech is "
                                    "translated into the call through a virtual microphone ('Hey Hermes, video "
                                    "call'; needs source_language = the other person's language). stream: "
                                    "captions only, to OBS closed captions and the caption page, no voice playback "
                                    "('Hey Hermes, stream')."},
            "target_language": {"type": "string",
                                "description": "Language to translate INTO (code like 'es' or a name like 'Spanish')"},
            "source_language": {"type": "string",
                                "description": "Language being spoken, if known. Omit to detect it automatically."},
            "input_device": {"type": "string",
                             "description": "Audio input: omit for the default microphone; 'BlackHole 2ch' to "
                                            "translate what a meeting app is playing."},
            "output_device": {"type": "string",
                              "description": "Where the translation plays: omit for the speakers; 'BlackHole 2ch' "
                                             "to send it into Zoom/Meet/OBS as a microphone."},
            "voice_clone": {"type": "boolean", "description": "Speak in the speaker's own voice (default true)."},
            "two_way": {"type": "boolean",
                        "description": "Also translate the user's own speech into the other person's language "
                                       "(needs source_language). Use for conversations and meetings."},
            "to_them": {"type": "string",
                        "description": "Two-way: where the user's translated voice goes, e.g. 'BlackHole 2ch' as "
                                       "the meeting app's microphone."},
        },
        "required": ["target_language"],
    },
}

TRANSLATE_STOP = {
    "name": "translate_stop",
    "description": ("Stop live translation, whichever mode it runs in ('Hey Hermes, end call', 'end stream', "
                    "'stop translating'). Phrases already heard are still translated and spoken first."),
    "parameters": {"type": "object", "properties": {}},
}

TRANSLATE_STATUS = {
    "name": "translate_status",
    "description": ("Show whether live translation is running, the detected language, the latest captions "
                    "(original + translation) and the measured delay."),
    "parameters": {"type": "object", "properties": {
        "captions": {"type": "integer", "description": "How many recent captions to return (default 10)."}}},
}

TRANSLATE_LANGUAGES = {
    "name": "translate_languages",
    "description": ("List the languages live translation understands and speaks, and which ones are spoken in "
                    "the speaker's own cloned voice."),
    "parameters": {"type": "object", "properties": {}},
}
