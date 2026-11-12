"""Translation backends. Each phrase of committed speech becomes the listener's language.

  mlx:<hf repo>                     a local MLX model (default: Qwen3.5-4B, 201 languages, Apache-2.0)
  gguf:<hf repo>#<file>             llama.cpp on NVIDIA or CPU (Windows / Linux), e.g. Qwen3.5-4B Q4_K_M
  openai:<base url>#<model>         any OpenAI-compatible server: Ollama, LM Studio, llama-server, vLLM, or a
                                    cloud API (key from LIVE_TRANSLATE_API_KEY)

The last couple of phrase pairs ride along as context so pronouns, names and gender stay consistent, but only the
new phrase is translated -- what was already spoken is never revised."""
from __future__ import annotations

import json
import os
import urllib.request

from . import langs

DEFAULT = "mlx:mlx-community/Qwen3.5-4B-MLX-4bit"
DEFAULT_GGUF = "gguf:unsloth/Qwen3.5-4B-GGUF#Qwen3.5-4B-Q4_K_M.gguf"
DEFAULT_TORCH = "torch:Qwen/Qwen3.5-4B"
DEFAULT_MODEL = DEFAULT


def _messages(history, text, src, tgt, context_pairs, concise=False):
    sys = (f"You are a live simultaneous interpreter. Translate the speaker's {langs.name(src)} into natural "
           f"spoken {langs.name(tgt)}. The input is one phrase of ongoing speech and may be an unfinished "
           f"sentence: translate exactly that phrase, keep names, numbers and tone, add nothing, and output "
           f"only the {langs.name(tgt)} translation.")
    if concise:
        sys += (" You are running behind the speaker: say it as briefly as a skilled interpreter would, keeping "
                "every fact but dropping filler.")
    msgs = [{"role": "system", "content": sys}]
    for s_, t_ in history[-context_pairs:]:
        msgs += [{"role": "user", "content": s_}, {"role": "assistant", "content": t_}]
    msgs.append({"role": "user", "content": text})
    return msgs


def _clean(out):
    return out.strip().split("\n")[0].strip().strip('"').strip()


class _Base:
    context_pairs = 2

    def __init__(self):
        self.history = {}                 # (src, tgt) -> [(source, translation)]

    def translate(self, text, src, tgt, concise=False):
        if src == tgt:
            return text
        h = self.history.setdefault((src, tgt), [])
        out = _clean(self._complete(_messages(h, text, src, tgt, self.context_pairs, concise),
                                    max(24, int(len(text) * 1.5))))
        h.append((text, out))
        del h[:-8]
        return out


class MLXTranslator(_Base):
    def __init__(self, repo):
        super().__init__()
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        from .pins import local
        self.model, self.tok = load(local(repo))
        self.sampler = make_sampler(temp=0.0)

    def _complete(self, msgs, max_tokens):
        from mlx_lm import generate
        try:
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens, sampler=self.sampler)


class OpenAITranslator(_Base):
    def __init__(self, base_url, model):
        super().__init__()
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = os.environ.get("LIVE_TRANSLATE_API_KEY", "")

    def _complete(self, msgs, max_tokens):
        body = json.dumps({"model": self.model, "messages": msgs, "temperature": 0, "max_tokens": max_tokens,
                           "stream": False,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()   # reasoning models: answer
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json",
                                                      **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=30) as r:
            msg = json.load(r)["choices"][0]["message"]
        return msg.get("content") or ""


class GGUFTranslator(_Base):
    def __init__(self, repo, filename):
        super().__init__()
        from llama_cpp import Llama
        from .hw import detect
        import os
        threads = int(os.environ.get("LIVE_TRANSLATE_THREADS", "0") or 0) or None
        from .pins import local
        self.llm = Llama(model_path=local(repo, filename), n_ctx=4096, verbose=False, n_threads=threads,
                         n_gpu_layers=-1 if detect() == "cuda" else 0)

    def _complete(self, msgs, max_tokens):
        p = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        p += "<|im_start|>assistant\n<think>\n\n</think>\n\n"          # thinking off (Qwen3.5 template)
        return self.llm(p, max_tokens=max_tokens, temperature=0.0, stop=["<|im_end|>"])["choices"][0]["text"]


class TorchTranslator(_Base):
    """The same Qwen3.5-4B through Hugging Face transformers on an NVIDIA GPU: for machines llama.cpp publishes no
    prebuilt wheel for (Linux and Windows on ARM64: DGX Spark, RTX Spark). bf16, ~8 GB of GPU memory."""

    def __init__(self, repo):
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        from .pins import local
        path = local(repo)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()

    def _complete(self, msgs, max_tokens):
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                           return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def default_spec():
    """The built-in translator for this machine: MLX on Apple Silicon, llama.cpp where it has a prebuilt wheel,
    otherwise transformers on an NVIDIA GPU."""
    from .hw import detect
    hw = detect()
    if hw == "mlx":
        return DEFAULT
    try:
        import llama_cpp  # noqa: F401
        return DEFAULT_GGUF
    except ImportError:
        return DEFAULT_TORCH if hw == "cuda" else DEFAULT_GGUF


def make(spec=None):
    """Build a translator from a backend spec (see the module docstring)."""
    spec = spec or default_spec()
    kind, _, rest = spec.partition(":")
    if kind == "openai":
        base, _, model = rest.partition("#")
        return OpenAITranslator(base, model or "default")
    if kind == "mlx":
        return MLXTranslator(rest)
    if kind == "gguf":
        repo, _, fn = rest.partition("#")
        return GGUFTranslator(repo, fn)
    if kind == "torch":
        return TorchTranslator(rest)
    return MLXTranslator(spec)                  # a bare repo id means MLX


Translator = MLXTranslator                      # backwards-compatible name
