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


def test_the_window_page_carries_the_languages_but_never_the_token(base):
    page = urllib.request.urlopen(base[0] + "/app").read().decode()
    assert server.TOKEN not in page and "__LANGS__" not in page and '"English"' in page


def test_changes_need_the_token_and_the_same_origin(base):
    url = base[0] + "/api/stop"
    assert _post(url, {}) == 403
    assert _post(url, {"X-LT-Token": server.TOKEN, "Origin": "https://evil.example"}) == 403
    assert _post(url, {"X-LT-Token": server.TOKEN, "Origin": "null"}) == 403      # sandboxed/file pages: no
    assert _post(url, {"X-LT-Token": server.TOKEN, "Origin": base[0]}) == 200


def _get(url, headers=None):
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})).status
    except urllib.error.HTTPError as e:
        return e.code


def test_state_and_events_need_the_token(base):
    assert _get(base[0] + "/api/state") == 403
    assert _get(base[0] + "/events") == 403
    r = urllib.request.urlopen(urllib.request.Request(base[0] + "/api/state", headers={"X-LT-Token": server.TOKEN}))
    assert json.load(r)["state"]["session"] == "idle"


def test_dns_rebinding_is_refused(base):
    """A web page whose domain now resolves to 127.0.0.1 still sends its own name as Host."""
    port = base[0].rsplit(":", 1)[1]
    for host in (f"evil.example:{port}", "127.0.0.1", f"127.0.0.1:{int(port) + 1}", f"localhost.evil.example:{port}"):
        assert _get(base[0] + "/api/state", {"Host": host, "X-LT-Token": server.TOKEN}) == 403
        assert _get(base[0] + "/app", {"Host": host}) == 403
        assert _post(base[0] + "/api/stop", {"Host": host, "X-LT-Token": server.TOKEN}) == 403
    assert _get(base[0] + "/api/state", {"Host": f"localhost:{port}", "X-LT-Token": server.TOKEN}) == 200


def test_the_token_lives_in_an_owner_only_file(base):
    import os
    from live_translate.paths import data_dir
    f = data_dir() / "server.token"
    assert f.read_text().strip() == server.TOKEN and len(server.TOKEN) >= 32
    if os.name != "nt":
        assert (f.stat().st_mode & 0o777) == 0o600


def test_fonts_are_served_locally_and_nothing_else(base):
    import glob, os
    name = os.path.basename(sorted(glob.glob(str(server.FONTS / "*.woff2")))[0])
    assert _get(base[0] + "/fonts/" + name) == 200
    for bad in ("../server.py", "..%2Fserver.py", "OFL-geistmono.txt", "nope.woff2"):
        assert _get(base[0] + "/fonts/" + bad) == 404


def test_hermes_model_endpoint_serves_only_local_processes_with_the_token(base):
    url = base[0] + "/v1/chat/completions"
    assert _post(url, {}) == 403                                             # no token
    assert _post(url, {"Authorization": "Bearer wrong"}) == 403
    assert _post(url, {"Authorization": f"Bearer {server.TOKEN}", "Origin": base[0]}) == 403   # a web page: never
    req = urllib.request.Request(url, json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {server.TOKEN}"})
    r = json.load(urllib.request.urlopen(req))
    assert r["choices"][0]["message"]["content"] == "hola" and r["seen"]["messages"][0]["content"] == "hello"
