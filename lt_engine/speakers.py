"""Who is speaking: so each person in a conversation keeps their own voice and their own language.

A speaker-embedding model (3D-Speaker CAM++ zh/en "advanced" via sherpa-onnx; Apache-2.0, 28 MB, every platform,
~26 ms a phrase) turns a phrase into a voiceprint. Each person is a running average of their phrases, which gets
steadier the longer they talk. A phrase joins the closest person; someone new is added when a 3 s phrase matches
nobody, or two shorter ones in a row do (the second more like the first than like anyone known). A wrong "new
person" costs little (a fresh voice sample of the same voice); a missed one gives them someone else's voice.
Measured on real speakers: two halves of one person's speech score 0.72 or more, different people 0.18 at most on
clean audio; background chatter pulls both toward ~0.2, so the bar for "new" sits low, and phrases under MIN_S
never create a speaker.
Without the model it reports one speaker -- the behaviour before speaker tracking.
"""
from __future__ import annotations

import numpy as np

REPO = "csukuangfj/speaker-embedding-models"
MODEL = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
NEW_BELOW = 0.30          # best match below this (on a long enough phrase) -> a new person
MIN_S = 1.8               # shorter phrases are assigned, never create a person
SURE_S = 3.0              # a phrase this long that matches nobody is someone new at once (shorter: two in a row)
MAX_SPEAKERS = 6


class SpeakerTracker:
    def __init__(self, threads=2):
        self.ex = None
        self.people = []          # running mean voiceprints (unit length)
        self.counts = []
        try:
            import sherpa_onnx
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(REPO, MODEL)
            self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=path, num_threads=threads))
        except Exception:                                          # noqa: BLE001 -- optional: one speaker
            self.ex = None

    @property
    def available(self):
        return self.ex is not None

    def _embed(self, audio16k):
        s = self.ex.create_stream()
        s.accept_waveform(16000, np.asarray(audio16k, np.float32))
        s.input_finished()
        e = np.asarray(self.ex.compute(s), np.float32)
        n = float(np.linalg.norm(e))
        return e / n if n else e

    def identify(self, audio16k):
        """Speaker number (0, 1, ...) for one phrase."""
        if self.ex is None or len(audio16k) < 16000 * 0.8:
            return 0 if not self.people else self._last
        e = self._embed(audio16k)
        if not self.people:
            self.people, self.counts, self._last = [e], [1], 0
            return 0
        sims = [float(p @ e) for p in self.people]
        best = int(np.argmax(sims))
        if sims[best] < NEW_BELOW and len(audio16k) >= 16000 * MIN_S and len(self.people) < MAX_SPEAKERS:
            # one short odd phrase is not a new person (a cough, a laugh, a noisy second): it takes a long phrase,
            # or two short ones in a row that match nobody and are more like each other than like anyone known
            sure = len(audio16k) >= 16000 * SURE_S
            pair = self._pending is not None and float(self._pending @ e) > sims[best]
            if not (sure or pair):
                self._pending = e
                return self._last
            m = e + self._pending if pair else e
            self.people.append(m / max(float(np.linalg.norm(m)), 1e-6))
            self.counts.append(2 if pair else 1)
            self._pending = None
            best = len(self.people) - 1
        else:
            self._pending = None
            k = self.counts[best]
            m = self.people[best] * k + e
            self.people[best] = m / max(float(np.linalg.norm(m)), 1e-6)
            self.counts[best] = k + 1
        self._last = best
        return best

    _last = 0
    _pending = None           # a phrase that matched nobody, waiting for a second one to confirm a new person
