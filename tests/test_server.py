import json
import re
import urllib.error
import urllib.request

import pytest

from live_translate import server
from live_translate.state import State


class FakeSession:
    def __init__(self):
        self.captions, self.lags, self.state = [], [], State("idle")

    def status(self, n=4):
        return {"state": self.state.snapshot(), "captions": []}


@pytest.fixture(scope="module")
def base():
    calls = []
    llm = lambda b: {"choices": [{"message": {"role": "assistant", "content": "hola"}}], "seen": b}
    url, srv = server.serve(FakeSession(), port=7911, actions={"stop": lambda b: calls.append(b) or {"ok": True},
                                                               "llm": llm})
    yield url.rstrip("/"), calls
    srv.shutdown()


def _post(url, headers):
    req = urllib.request.Request(url, b"{}", {"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(req).status
    except urllib.error.HTTPError as e:
        return e.code


def test_the_window_gets_a_token_and_the_languages(base):
    page = urllib.request.urlopen(base[0] + "/app").read().decode()
    assert re.search(r'const TOKEN = "[A-Za-z0-9_-]{20,}"', page) and "__LANGS__" not in page


def test_changes_need_the_token_and_the_same_origin(base):
    url = base[0] + "/api/stop"
    assert _post(url, {}) == 403
    assert _post(url, {"X-LT-Token": server.TOKEN, "Origin": "https://evil.example"}) == 403
    assert _post(url, {"X-LT-Token": server.TOKEN, "Origin": base[0]}) == 200


def test_state_is_readable(base):
    assert json.load(urllib.request.urlopen(base[0] + "/api/state"))["state"]["session"] == "idle"


def test_hermes_model_endpoint_serves_only_local_processes_with_the_token(base):
    url = base[0] + "/v1/chat/completions"
    assert _post(url, {}) == 403                                             # no token
    assert _post(url, {"Authorization": "Bearer wrong"}) == 403
    assert _post(url, {"Authorization": f"Bearer {server.TOKEN}", "Origin": base[0]}) == 403   # a web page: never
    req = urllib.request.Request(url, json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {server.TOKEN}"})
    r = json.load(urllib.request.urlopen(req))
    assert r["choices"][0]["message"]["content"] == "hola" and r["seen"]["messages"][0]["content"] == "hello"
