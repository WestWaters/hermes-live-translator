"""Language table: names for the translator prompt, and which voice tier speaks each language."""
from __future__ import annotations

# Whisper's 99 speech-recognition languages (ISO 639-1 / Whisper codes) with English names.
NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani", "ba": "Bashkir",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fo": "Faroese",
    "fr": "French", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jw": "Javanese", "ka": "Georgian",
    "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean", "la": "Latin", "lb": "Luxembourgish",
    "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mi": "Maori",
    "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian",
    "oc": "Occitan", "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian",
    "sn": "Shona", "so": "Somali", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen",
    "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese",
}

# Tier 1: Chatterbox multilingual -- speaks in the speaker's own (cloned) voice. MIT licence.
CLONE_LANGS = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
               "pl", "pt", "ru", "sv", "sw", "tr", "zh"}

# Fast recognition (Qwen3-ASR) and fast own-voice speech (Qwen3-TTS) cover these.
FAST_ASR = {"zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi", "ja", "tr", "hi",
            "ms", "nl", "sv", "da", "fi", "pl", "cs", "tl", "fa", "el", "ro", "hu", "mk"}
FAST_CLONE = {"zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"}

# Languages written without spaces between words: agreement and wait-k work on characters there.
NO_SPACE = {"zh", "ja", "th", "lo", "km", "my", "bo", "yue"}

# Verb-final languages: the translator waits for more words before committing a clause.
SOV = {"ja", "ko", "tr", "hi", "ur", "bn", "fa", "mn", "ta", "te", "kn", "ml", "mr", "ne", "my", "si", "am"}


def name(code: str) -> str:
    return NAMES.get(code, code)


def resolve(lang: str) -> str:
    """Accept a code ('es') or an English name ('Spanish') and return the code."""
    s = (lang or "").strip()
    if s.lower() in NAMES:
        return s.lower()
    for c, n in NAMES.items():
        if n.lower() == s.lower():
            return c
    raise ValueError(f"unknown language: {lang!r}")
