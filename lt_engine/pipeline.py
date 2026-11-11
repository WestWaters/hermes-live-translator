"""The live pipeline: audio in -> voice activity -> streaming ASR -> phrase segmenter -> translator -> voice -> out.

Recognition runs in this process; translation and speech each run in their own process so the three overlap
(the next phrase is recognised and translated while the previous one is still being spoken). Every phrase
records its lag: the time from the speaker finishing the phrase to the listener hearing its translation."""
from __future__ import annotations

import json
import multiprocessing as mp
import queue
import re
import sys
import threading
import time

import numpy as np

from . import langs
from .audio import FRAME, FileSource, MicSource, Sink, check_input  # noqa: F401  (re-exported)
from .guard import overloaded

SR = 16000
PAUSE_S = 0.45                   # silence that ends a phrase
STEP_S = 1.2                     # long run-on speech: re-recognise this often (pauses commit at once)
LONG_S = 2.4                     # ...once the phrase is this long
MAX_UTT_S = 20.0                 # force a cut in endless speech
# Lines speech recognisers produce from silence, music and noise (the subtitle credits their training data is full
# of; arXiv 2501.11378 "bag of hallucinations"). Dropped only when they are the whole phrase.
HALLUCINATIONS = (
    "thank you for watching", "thanks for watching", "please subscribe", "subscribe to my channel",
    "like and subscribe", "see you in the next video", "subtitles by", "amara.org", "subtítulos por",
    "subtitulado por", "gracias por ver", "sous-titres réalisés par", "sous-titrage", "untertitel im auftrag",
    "untertitel von", "sottotitoli creati", "legendas pela comunidade", "字幕由", "请不吝点赞", "订阅", "ご視聴ありがとうございました",
    "시청해주셔서 감사합니다", "продолжение следует", "редактор субтитров",
    "music", "[music]", "(music)", "música", "musique", "musik", "موسیقی", "موسيقى", "音乐", "音楽", "संगीत")


def max_speech_s(text, lang):
    """The longest a sentence can take to say (slow speech, generous): autoregressive voice models sometimes miss
    their stop token and keep generating for minutes; anything past this is cut. ~5 syllables/s for alphabetic
    text (≈ 0.07 s/char), ~4 chars/s for Chinese/Japanese/Korean (one syllable each)."""
    from .langs import NO_SPACE
    per = 0.3 if lang in NO_SPACE else 0.09
    return 3.0 + 1.5 * per * len(text)


def stray(ln, text):
    """A lone word from a phrase much quieter than the person we follow: a fragment of background talk."""
    from .langs import NO_SPACE
    t = text.strip(" .!?。！？…,、")
    short = len(t) <= 2 if ln.rec.language in NO_SPACE else len(t.split()) <= 1
    return short and ln.level is not None and getattr(ln, "phrase_db", 0.0) < ln.level - STRAY_DB


def hallucinated(text):
    t = text.strip().lower().strip(" .!?。！？…'\"")
    if not t:
        return True
    if t in HALLUCINATIONS or (len(t) < 80 and any(len(h) > 12 and h in t for h in HALLUCINATIONS)):
        return True                                    # a credit line, or a lone "Music" -- not a sentence about it
    words = t.split()                                  # the same word over and over ("you you you you")
    return len(words) >= 4 and len(set(words)) == 1


ABS_MIN_DB = -48.0                # quieter than this is never someone talking to us
BACKGROUND_DB = 14.0             # voices this much quieter than the main speaker are background
STRAY_DB = 8.0                   # a one-word phrase this much quieter than the speaker is background
LEVEL_DECAY_DB_S = 0.6           # ...unless they keep talking: the bar comes down this fast while they do
ECHO_TAIL_S = 0.35               # the room keeps ringing a little after our voice stops
REF_S = 5.0                      # speaker voice sample length (fixed)
BEHIND_S = 4.0                   # translated speech queued this far behind -> fast voice until caught up
CLAUSE = re.compile(r"[,;:.!?，。！？、；：]$")










def emit(ev, **kw):
    kw["event"] = ev
    kw.setdefault("t", round(time.time(), 3))
    sys.stdout.write(json.dumps(kw, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- worker processes
def _mt_worker(model, inq, outq, evq, behind):
    from .mt import make
    tr = make(model)
    evq.put(("ready", {"stage": "translate", "model": model}))
    while True:
        job = inq.get()
        if job is None:
            outq.put(None)
            return
        t0 = time.time()
        out = tr.translate(job["text"], job["src"], job["tgt"], concise=behind.value > 3.0)
        job.update(translation=out, mt_s=round(time.time() - t0, 3))
        evq.put(("caption", {k: job.get(k) for k in ("id", "lane", "speaker", "src", "tgt", "text", "translation",
                                                     "mt_s")}))
        outq.put(job)


def _tts_worker(route, clone, targets, inq, audq, evq, behind):
    """One voice process for every lane; each lane keeps its own speaker sample (two-way: you and them)."""
    from .voices import Router
    routers = {lane: Router(route) for lane in targets}
    refs, current = {}, {}                          # (lane, speaker) -> voice sample; the voice each lane has now
    for lane, target in targets.items():
        try:
            routers[lane].voice(target).warm(target)
        except Exception as e:                          # a slow first phrase beats no voice at all
            evq.put(("error", {"stage": "speak", "warm": True, "error": str(e)[:200]}))
    evq.put(("ready", {"stage": "speak", "voice": ", ".join(routers[l].voice(t).name for l, t in targets.items())}))
    while True:
        job = inq.get()
        if job is None:
            audq.put(None)
            return
        lane, spk = job.get("lane", "a"), job.get("speaker", 0)
        router = routers[lane]
        if job.get("ref") is not None:
            refs[(lane, spk)] = (job["ref"], job.get("ref_text", ""))
            if clone and current.get(lane) is None:          # first voice in this lane: use it right away
                router.set_reference(*refs[(lane, spk)])
                current[lane] = spk
            evq.put(("voice", {"cloned": bool(clone), "seconds": round(len(job["ref"]) / SR, 1), "lane": lane,
                               "speaker": spk}))
            continue
        if clone and (lane, spk) in refs and current.get(lane) != spk:   # this phrase is someone else's
            router.set_reference(*refs[(lane, spk)])
            current[lane] = spk
        t0 = time.time()
        first = True
        voice = router.voice(job["tgt"])
        cap, said = max_speech_s(job["translation"], job["tgt"]), 0.0
        try:
            for chunk in voice.speak(job["translation"], job["tgt"]):
                if said + len(chunk) / SR > cap:                 # the voice model missed its stop: cut it here
                    chunk = chunk[:max(0, int((cap - said) * SR))]
                    evq.put(("runaway", {"id": job["id"], "voice": voice.name, "cap_s": round(cap, 1)}))
                if len(chunk):
                    audq.put({"id": job["id"], "lane": job.get("lane", "a"), "audio": chunk, "first": first,
                              "tts_first_s": round(time.time() - t0, 3)})
                    first = False
                said += len(chunk) / SR
                if said >= cap:
                    break
        except Exception as e:                                   # a voice backend failed: keep going
            evq.put(("error", {"stage": "speak", "voice": voice.name, "error": str(e)[:200]}))
        audq.put({"id": job["id"], "lane": job.get("lane", "a"), "done": True, "tier": voice.name})










# ---------------------------------------------------------------- the loop
class Lane:
    """One direction of translation: a source of speech, a listener's language and where it is played.
    One-way has a single lane; two-way has two (them -> you, and you -> them), sharing the loaded models."""

    def __init__(self, name, source, target, source_lang, sink, rec):
        self.name, self.source, self.target, self.sink, self.rec = name, source, target, sink, rec
        self.source_lang = source_lang
        self.utt, self.pre, self.ref = [], [], []
        self.ref_sent, self.in_speech = False, False
        self.silence, self.since_step, self.last_speech_end, self.t = 0.0, 0.0, 0.0, 0.0
        self.seg, self.done = "", False
        self.speaker = 0                    # who is talking now (speakers.py), their languages and voice samples
        self.lang_by_speaker = {}
        self.refs, self.refs_sent = {}, set()
        self.tracker = None
        self.level = None                   # the main speaker's loudness (dBFS), learned as they talk
        self.heard_s = 0.0                  # seconds of speech followed so far
        self.aec = None                     # echo canceller, when this is a microphone near our speakers


class Pipeline:
    def __init__(self, target, source_lang=None, mt_model=None, tts_model=None, clone=True, fast_asr=True,
                 asr_route="auto", volume=1.0):
        from .voices import PRESETS
        self.volume = volume
        self.target = langs.resolve(target)
        self.source_lang = langs.resolve(source_lang) if source_lang else None
        self.mt_model, self.tts_model = mt_model or None, tts_model or "commercial"   # None: chosen by hardware
        assert self.tts_model in PRESETS or "=" in self.tts_model or ":" in self.tts_model
        self.clone, self.fast_asr, self.asr_route = clone, fast_asr, asr_route or "auto"

    def run(self, source, sink_play=True, out_device=None, record=None, original=None, stop=None,
            back_source=None, back_out_device=None):
        """`source` is what you listen to, translated into `target` and played on `out_device`. With
        `back_source` (two-way), your own speech is translated into the other side's language (`source_lang`,
        or theirs once detected) and played on `back_out_device` -- usually a virtual mic into the meeting."""
        import torch
        from silero_vad import load_silero_vad
        from .asr import Recognizer
        from .voices import OUT_SR

        two_way = back_source is not None
        if two_way and not self.source_lang:
            raise SystemExit("two-way needs the other side's language (--from)")
        overloaded()                                       # remember the swap level we started from
        for src_ in (source, back_source):
            if isinstance(src_, MicSource):
                why = check_input(src_.device)
                if why:
                    emit("stopped", reason=why)
                    return []
        targets = {"a": self.target, **({"b": self.source_lang} if two_way else {})}

        ctx = mp.get_context("spawn")
        mt_in, tts_in, audq, evq = ctx.Queue(), ctx.Queue(), ctx.Queue(), ctx.Queue()
        behind = ctx.Value("d", 0.0)
        procs = [ctx.Process(target=_mt_worker, args=(self.mt_model, mt_in, tts_in, evq, behind), daemon=True),
                 ctx.Process(target=_tts_worker, args=(self.tts_model, self.clone, targets, tts_in, audq, evq,
                                                       behind), daemon=True)]
        for p in procs:
            p.start()
        vad = {"a": load_silero_vad(onnx=True), **({"b": load_silero_vad(onnx=True)} if two_way else {})}
        rec_a = Recognizer(self.source_lang, fast=self.fast_asr, route=self.asr_route)
        rec_a.warm()
        lanes = {"a": Lane("a", source, self.target, self.source_lang,
                           Sink(OUT_SR, device=out_device, play=sink_play, record=record), rec_a)}
        if two_way:
            rec_b = Recognizer(None, fast=self.fast_asr, route=self.asr_route, shared=rec_a.made)
            lanes["b"] = Lane("b", back_source, self.source_lang, None,
                              Sink(OUT_SR, device=back_out_device, play=sink_play), rec_b)
        from .speakers import SpeakerTracker
        for ln in lanes.values():
            ln.tracker = SpeakerTracker()
        emit("speakers", tracking=bool(lanes["a"].tracker.available))
        from . import aec
        loud = next((o.sink for o in lanes.values() if o.sink.audible and o.sink.stream is not None), None)
        if loud is not None and aec.available():           # speakers a microphone can hear: cancel their echo
            for ln in lanes.values():
                if isinstance(ln.source, MicSource):
                    try:
                        lat = loud.stream.latency + (ln.source.stream.latency if getattr(ln.source, "stream", None)
                                                     else 0.02)
                    except Exception:
                        lat = 0.06
                    ln.aec = aec.EchoCanceller(loud.sr, delay_ms=lat * 1000)
                    loud.on_play = ln.aec.played
                    break
        emit("echo_cancelling", on=any(ln.aec is not None for ln in lanes.values()))
        def dead():
            """The stage whose process has exited (crashed), if any: never wait on it forever."""
            for name, p in zip(("translate", "speak"), procs):
                if not p.is_alive():
                    return name
            return None

        ready = 0
        while ready < 2:
            try:
                ev, kw = evq.get(timeout=1.0)
            except queue.Empty:
                if dead():
                    emit("stopped", reason=f"the {dead()} stage stopped unexpectedly (see the log)")
                    for p in procs:
                        p.terminate()
                    return
                continue
            emit(ev, **kw)
            ready += ev == "ready"

        def control():
            """Live commands from the app on stdin, one JSON per line: {"volume": 0.8}."""
            for line in sys.stdin:
                try:
                    cmd = json.loads(line)
                except ValueError:
                    continue
                if "volume" in cmd:
                    v = max(0.0, min(1.5, float(cmd["volume"])))
                    for ln in lanes.values():
                        ln.sink.gain = v
                    emit("volume", value=round(v, 2))
        if not sys.stdin.isatty():
            threading.Thread(target=control, daemon=True).start()
        for ln in lanes.values():
            ln.sink.gain = self.volume

        t0 = time.time()
        for ln in lanes.values():
            ln.source.start(t0)
        emit("listening", target=self.target, source=self.source_lang or "auto", two_way=two_way)

        st = {"next": 0, "done_audio": False}
        spoken = []                                  # (time, normalized translation) of recent output

        def norm(t):
            return re.sub(r"[\W_]+", "", t.lower())

        def is_echo(text):
            from difflib import SequenceMatcher
            n, now = norm(text), time.time()
            return bool(n) and any(now - ts < 15 and SequenceMatcher(None, n, s).ratio() > 0.75 for ts, s in spoken)
        segs, lags = {}, []

        def send(ln, text, src_end):
            text = text.strip()
            if not text:
                return
            if hallucinated(text) or stray(ln, text):   # what recognisers "hear" in silence, noise, chatter
                emit("dropped", lane=ln.name, text=text)
                return
            if is_echo(text):                       # it heard its own translation (speakers + microphone)
                emit("echo", lane=ln.name, text=text)
                return
            src = ln.rec.language or "en"
            i = st["next"]
            st["next"] += 1
            segs[i] = {"src_end": src_end, "lane": ln.name}
            job = {"id": i, "lane": ln.name, "speaker": ln.speaker, "text": text, "src": src, "tgt": ln.target}
            emit("heard", id=i, lane=ln.name, lang=src, text=text, at=round(src_end, 2))
            if src == ln.target:
                job["translation"] = text
                emit("caption", id=i, lane=ln.name, src=src, tgt=ln.target, text=text, translation=text, mt_s=0)
                tts_in.put(job)
            else:
                mt_in.put(job)

        def on_text(ln, new, src_end, final=False):
            """Committed text arrives; whole clauses go to the translator, a pause flushes the rest."""
            ln.seg += new
            seg = ln.seg
            ns = (ln.rec.language or "") in langs.NO_SPACE
            need = (8 if ns else 3) * (2 if (ln.rec.language or "") in langs.SOV else 1)
            cut = -1
            for m in re.finditer(r"[,;:.!?，。！？、；：।]", seg):
                head_ = seg[:m.end()]
                if (len(head_.strip()) if ns else len(head_.split())) >= need:
                    cut = m.end()
            if final:
                cut = len(seg)
            if cut > 0:
                send(ln, seg[:cut], src_end)
                ln.seg = seg[cut:]

        def drain():
            while True:
                try:
                    ev, kw = evq.get_nowait()
                    if ev == "caption" and kw.get("translation"):
                        spoken.append((time.time(), norm(kw["translation"])))
                        del spoken[:-12]
                    emit(ev, **kw)
                except queue.Empty:
                    break
            while True:
                try:
                    m = audq.get_nowait()
                except queue.Empty:
                    break
                if m is None:
                    st["done_audio"] = True
                    break
                if m.get("done"):
                    emit("spoken", id=m["id"], lane=m.get("lane"), tier=m["tier"])
                    segs.pop(m["id"], None)                 # nothing grows with the length of a session
                    continue
                now = time.time() - t0
                start = lanes[m.get("lane", "a")].sink.push(m["audio"], now)
                if m["first"]:
                    lag = start - segs[m["id"]]["src_end"]
                    lags.append(lag)
                    del lags[:-500]
                    emit("latency", id=m["id"], lane=m.get("lane"), lag_s=round(lag, 2),
                         tts_first_s=m["tts_first_s"])
            now = time.time() - t0
            behind.value = max(0.0, max(ln.sink.play_end - now for ln in lanes.values()))
            pace = "catching_up" if behind.value > 1.5 else "on_time"
            if pace != st.get("pace"):
                st["pace"] = pace
                emit("pace", state=pace, behind_s=round(behind.value, 1))

        def recognise(ln, final):
            audio = np.concatenate(ln.utt)
            if final and ln.tracker is not None and ln.tracker.available:
                spk = ln.tracker.identify(audio)
                if spk != ln.speaker:
                    ln.speaker = spk
                    emit("speaker", lane=ln.name, speaker=spk)
                    if ln.rec.fixed is None:            # each person keeps their own language (checked again:
                        ln.rec.language = ln.lang_by_speaker.get(spk)   # voices in noise can be mistaken)
                        ln.rec._pick()
                        ln.rec.checked = False
            ta = time.time()
            ln.phrase_db = 20 * np.log10(float(np.sqrt(np.mean(audio * audio))) + 1e-9)
            new = ln.rec.final(audio) if final else ln.rec.partial(audio)
            if ln.rec.language:
                ln.lang_by_speaker[ln.speaker] = ln.rec.language
            if final:
                voice_sample(ln, audio)
            emit("asr", lane=ln.name, seconds=round(time.time() - ta, 3), audio_s=round(len(audio) / SR, 1),
                 final=final, lang=ln.rec.language, check=getattr(ln.rec, "last_check", None))
            ln.rec.last_check = None
            return new

        def echo_risk(ln):
            """A live microphone while translated speech is playing out loud (plus the room's tail): its own voice
            would be heard and translated again."""
            if not isinstance(ln.source, MicSource) or ln.aec is not None:   # cancelled: keep listening (barge-in)
                return False
            now = time.time() - t0
            return any(o.sink.audible and now < o.sink.play_end + ECHO_TAIL_S for o in lanes.values())

        def voice_sample(ln, audio):
            """Collect each person's first seconds of speech; once there are 2 s, send their voice sample.
            Always exactly REF_S seconds (trimmed or padded with silence): the voice model then sees one reference
            shape, warmed at startup, instead of compiling for a new one on the first phrase."""
            spk = ln.speaker
            if (spk in ln.refs_sent) or not ln.rec.language:
                return
            ln.refs.setdefault(spk, []).append(audio)
            got = sum(len(a) for a in ln.refs[spk])
            if got < 2 * SR:
                return
            ra = np.concatenate(ln.refs[spk])[:int(REF_S * SR)]
            ra = np.pad(ra, (0, int(REF_S * SR) - len(ra)))
            eng = ln.rec.engine or ln.rec.whisper
            tts_in.put({"lane": ln.name, "speaker": spk, "ref": ra,
                        "ref_text": eng.transcribe(ra, ln.rec.language)[0]})
            ln.refs_sent.add(spk)

        def feed(ln, frame):
            ln.t += len(frame) / SR
            if echo_risk(ln):                       # deaf while we are talking out loud
                frame = np.zeros_like(frame)
            speech = float(vad[ln.name](torch.from_numpy(frame), SR).item()) > 0.5
            if speech:                              # near-field gate: voices far quieter than the speaker are
                rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)   # background (TV, other people, chatter)
                db = 20 * np.log10(rms)
                if ln.level is not None and db > ln.level + BACKGROUND_DB and ln.heard_s < 20:
                    # a much louder voice than anything so far: what we were following was the background
                    # (a TV, people across the room) -- start over with this voice, and identify its language
                    ln.level = db
                    ln.rec.held = []
                    ln.rec.utterance(pause_s=99.0)
                    ln.utt, ln.since_step = [], 0.0
                    emit("background", lane=ln.name)
                if ln.level is None:
                    speech = db > ABS_MIN_DB
                else:
                    speech = db > max(ABS_MIN_DB, ln.level - BACKGROUND_DB)
                    if not speech:                  # a voice keeps talking below the bar: lower the bar slowly,
                        ln.level -= LEVEL_DECAY_DB_S * len(frame) / SR   # so a quieter person is heard soon
                if speech:                          # follow the main speaker's level (fast up, slow down)
                    ln.heard_s += len(frame) / SR
                    ln.level = db if ln.level is None else (max(ln.level, db) * 0.3 + ln.level * 0.7
                                                             if db > ln.level else ln.level * 0.995 + db * 0.005)
            if speech:
                if not ln.in_speech:
                    emit("activity", state="hearing", lane=ln.name)
                    ln.rec.utterance(pause_s=ln.t - ln.last_speech_end)
                    ln.utt, ln.pre = ln.pre + [], []
                    ln.in_speech, ln.since_step = True, 0.0
                ln.utt.append(frame)
                ln.silence = 0.0
                ln.since_step += len(frame) / SR
                if len(ln.utt) * FRAME / SR >= MAX_UTT_S:
                    on_text(ln, recognise(ln, True), ln.t, final=True)
                    ln.utt, ln.in_speech = [], False
                elif ln.since_step >= STEP_S and len(ln.utt) * FRAME / SR >= LONG_S:
                    on_text(ln, recognise(ln, False), ln.t)
                    ln.since_step = 0.0
            else:
                ln.pre = (ln.pre + [frame])[-6:]
                if ln.in_speech:
                    ln.utt.append(frame)
                    ln.silence += len(frame) / SR
                    if ln.silence >= PAUSE_S:
                        ln.last_speech_end = ln.t - ln.silence
                        on_text(ln, recognise(ln, True), ln.last_speech_end, final=True)
                        ln.in_speech, ln.utt = False, []

        last_guard = 0.0
        while not all(ln.done for ln in lanes.values()):
            drain()
            if stop is not None and stop.is_set():
                break
            if time.time() - last_guard > 3.0:
                last_guard = time.time()
                why = overloaded()
                if why:
                    emit("stopped", reason=f"protecting the machine: {why}")
                    break
            got = False
            for ln in lanes.values():
                if ln.done:
                    continue
                try:
                    frame = ln.source.q.get_nowait()
                except queue.Empty:
                    continue
                got = True
                if frame is None:
                    ln.done = True
                    continue
                if ln.aec is not None:
                    for f in ln.aec.process(frame):
                        feed(ln, f)
                else:
                    feed(ln, frame)
            if not got:
                time.sleep(0.01)

        if stop is not None and stop.is_set():
            # Stop means stop: drop what is still queued, silence the speakers, end the workers now.
            for p in procs:
                p.terminate()
            for ln in lanes.values():
                if ln.sink.stream is not None:
                    with ln.sink.lock:
                        ln.sink.buf = ln.sink.buf[:0]
                    ln.sink.stream.stop()
                    ln.sink.stream = None
                if getattr(ln.source, "stream", None) is not None:
                    ln.source.stream.stop()
            for p in procs:
                p.join(timeout=3)
        else:
            for ln in lanes.values():
                if ln.in_speech and ln.utt:
                    on_text(ln, recognise(ln, True), ln.t, final=True)
                on_text(ln, "", ln.t, final=True)
            mt_in.put(None)
            while not st["done_audio"]:
                drain()
                if not procs[1].is_alive() and audq.empty():  # the voice stage crashed: it will never finish
                    drain()                                  # (the translator exiting here is normal: it is done)
                    if not st["done_audio"]:
                        emit("error", stage="speak", error="stage stopped unexpectedly; the rest was not spoken")
                    break
                time.sleep(0.02)
            drain()
            for p in procs:
                p.join(timeout=5)
        if lags:
            emit("summary", phrases=len(lags), lag_median_s=round(float(np.median(lags)), 2),
                 lag_max_s=round(float(np.max(lags)), 2), lag_min_s=round(float(np.min(lags)), 2))
        a = lanes["a"]
        if record:
            a.sink.save(record, original=original, length_s=a.t)
            emit("recorded", path=record)
        for ln in lanes.values():
            if ln.sink.stream is not None:
                time.sleep(max(0.0, ln.sink.play_end - (time.time() - t0)) + 0.2)
                ln.sink.stream.stop()
        return lags
