import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from live_translate import detect


def test_running_local_servers_are_found(monkeypatch):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": "qwen3:8b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(detect, "LOCAL_SERVERS", [("Ollama", f"http://127.0.0.1:{srv.server_port}/v1")])
    found = detect.detect_models()
    srv.shutdown()
    ids = [m["id"] for m in found["translator"]]
    assert any(i.endswith("#qwen3:8b") and "Ollama" in m["label"] for i, m in zip(ids, found["translator"]))


def _hermes_holds(monkeypatch, **keys):
    """Keys as Hermes's secret scope would return them (the only place the plugin reads keys from)."""
    monkeypatch.setattr(detect, "_secret", lambda name: keys.get(name, ""))


def test_subscriptions_appear_only_with_a_key(monkeypatch):
    monkeypatch.setattr(detect, "LOCAL_SERVERS", [])
    _hermes_holds(monkeypatch, OPENAI_API_KEY="sk-test")
    found = detect.detect_models()
    groups = {m["label"].split(" · ")[0] for m in found["translator"] if m["group"] == "Your subscriptions"}
    assert "OpenAI" in groups and "Groq" not in groups and "Groq" in found["missing"]


def test_the_engine_gets_only_the_key_it_needs(monkeypatch):
    _hermes_holds(monkeypatch, OPENAI_API_KEY="sk-openai", GROQ_API_KEY="gsk-groq")
    env = detect.keys_for("openai:https://api.openai.com/v1#gpt-5-mini", "best", "auto")
    assert env == {"LIVE_TRANSLATE_API_KEY": "sk-openai"}


def test_keys_in_the_raw_environment_are_never_read(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-another-profile")   # catalog rule 11: only Hermes's secret scope
    assert detect._secret("OPENAI_API_KEY") == ""


def test_hermes_model_is_offered_only_when_the_plugin_can_call_it(monkeypatch):
    monkeypatch.setattr(detect, "LOCAL_SERVERS", [])
    _hermes_holds(monkeypatch)
    ids = lambda found: [m["id"] for m in found["translator"]]
    assert "hermes" not in ids(detect.detect_models())
    assert "hermes" in ids(detect.detect_models(hermes_llm=True))
