"""Local server for the translator, 127.0.0.1 only.

  /              live captions page (OBS Browser Source)
  /app           the control window (the tray app and Hermes Desktop open this)
  /api/...       state, start, stop, setup, devices, languages -- what the window drives

Changing requests need the per-run token the server writes into /app, and a same-origin (or no) Origin header, so
a web page in the user's browser cannot start or stop translation.

Live captions page: http://127.0.0.1:<port>/ shows the original and the translation as they happen.

Add it to OBS as a Browser Source for bilingual subtitles on a stream, or open it next to a meeting. Local only
(binds 127.0.0.1), standard library only."""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Live Translate captions</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--fg:#fff;--dim:rgba(255,255,255,.66);--acc:#ffc46b}
html,body{margin:0;height:100%;background:transparent;color:var(--fg);font:600 30px/1.25 "Atkinson Hyperlegible Next","Noto Sans",system-ui,sans-serif;overflow:hidden}
#wrap{position:fixed;left:24px;right:24px;bottom:24px;display:flex;flex-direction:column;gap:10px}
.c{background:rgba(0,0,0,.72);border-radius:14px;padding:12px 18px;animation:in .25s ease-out}
.o{color:var(--dim);font-size:.6em;font-weight:500;margin-top:3px}
.l{color:var(--acc);font:600 .42em ui-monospace,Menlo,monospace;letter-spacing:.1em;text-transform:uppercase;margin-right:10px;vertical-align:.2em}
#lag{position:fixed;top:10px;right:14px;font:500 13px ui-monospace,Menlo,monospace;color:var(--dim)}
/* ?bar=1: the floating caption bar window in the desktop app */
.bar body{font-size:26px}
.bar #shell{position:fixed;inset:0;border-radius:14px;background:rgba(14,12,20,.86);border:1px solid rgba(255,255,255,.14);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);box-shadow:inset 0 1px 0 rgba(255,255,255,.18)}
.bar #wrap{left:18px;right:18px;bottom:14px;gap:6px}
.bar .c{background:none;padding:0}
.bar .c:not(:last-child){opacity:.45;font-size:.72em}
.bar #lag{top:9px;right:40px}
#close{display:none;position:fixed;top:8px;right:10px;width:22px;height:22px;border-radius:50%;border:0;cursor:pointer;
  background:rgba(255,255,255,.12);color:#fff;font:600 13px system-ui;line-height:22px;padding:0}
.bar #close{display:block}#close:hover{background:#ff6a3d}
#idle{position:fixed;left:18px;bottom:16px;font:500 15px system-ui;color:var(--dim);display:none}.bar #idle{display:block}
@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1}}
@media (prefers-reduced-motion: reduce){.c{animation:none}}
</style></head><body><div id="shell" data-tauri-drag-region></div><div id="lag"></div><button id="close" title="Close">×</button>
<div id="idle">Captions appear here while Live Translate is translating.</div><div id="wrap" data-tauri-drag-region></div><script>
const bar=new URLSearchParams(location.search).has("bar");if(bar)document.documentElement.classList.add("bar");
const wrap=document.getElementById('wrap'),lag=document.getElementById('lag'),idle=document.getElementById('idle');
const N=+(new URLSearchParams(location.search).get('lines')||(bar?2:3));
document.getElementById('close').onclick=()=>{const T=window.__TAURI__;if(T&&T.window)T.window.getCurrentWindow().close();else window.close();};
const es=new EventSource('/events');
es.onmessage=e=>{const d=JSON.parse(e.data);
 if(d.lag!=null)lag.textContent='delay '+d.lag.toFixed(1)+' s';
 if(!d.caption)return;idle.style.display='none';const c=d.caption,el=document.createElement('div');el.className='c';
 el.innerHTML='<div class="t"><span class="l"></span></div><div class="o"></div>';
 el.querySelector('.l').textContent=c.tgt;el.querySelector('.t').append(c.translation);el.querySelector('.o').textContent=c.text;
 wrap.append(el);while(wrap.children.length>N)wrap.firstChild.remove();};
</script></body></html>"""


TOKEN = secrets.token_urlsafe(24)
ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "index.html"


def _app_page():
    import sys
    here = str(ROOT)
    if here not in sys.path:
        sys.path.insert(0, here)
    from lt_engine import langs
    rows = [[c, n, 1 if (c in langs.FAST_CLONE or c in langs.CLONE_LANGS) else 0]
            for c, n in sorted(langs.NAMES.items(), key=lambda kv: kv[1])]
    return (APP.read_text().replace("__LANGS__", json.dumps(rows, ensure_ascii=False))
            .replace('"__TOKEN__"', json.dumps(TOKEN)))


def serve(session, port=7865, actions=None):
    """Start the server for `session` (a tools.Session) in a background thread; returns (base URL, server).
    `actions` maps start/stop/setup/devices/languages to callables (tools.py provides them)."""
    actions = actions or {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            if os.environ.get("LT_HTTP_LOG"):
                import sys
                sys.stderr.write("http " + (fmt % a) + " origin=" + str(self.headers.get("Origin")) + "\n")

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else (json.dumps(body, ensure_ascii=False).encode()
                                                         if ctype == "application/json" else body.encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if self.path == "/v1/chat/completions":         # the engine translating with Hermes's own model
                if origin or self.headers.get("Authorization") != f"Bearer {TOKEN}" or "llm" not in actions:
                    return self._send(403, {"error": "forbidden"})   # local processes only, never a web page
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    return self._send(200, actions["llm"](json.loads(self.rfile.read(n) or b"{}")))
                except Exception as e:
                    return self._send(502, {"error": {"message": str(e)[:300]}})
            host = self.headers.get("Host", "")
            if self.headers.get("X-LT-Token") != TOKEN or (origin and origin not in (f"http://{host}", "null",
                                                                                     "tauri://localhost",
                                                                                     "http://tauri.localhost")):
                return self._send(403, {"error": "forbidden"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._send(400, {"error": "bad json"})
            name = self.path.split("?")[0].removeprefix("/api/")
            if name not in ("start", "stop", "setup", "volume", "models", "obs") or name not in actions:
                return self._send(404, {"error": "no such action"})
            try:
                return self._send(200, actions[name](body))
            except Exception as e:
                return self._send(500, {"error": str(e)[:300]})

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/app":
                return self._send(200, _app_page(), "text/html")
            if path == "/api/state":
                return self._send(200, session.status(4))
            if path in ("/api/devices", "/api/languages", "/api/hermes-skin", "/api/models") and path[5:] in actions:
                try:
                    return self._send(200, actions[path[5:]]({}))
                except Exception as e:
                    return self._send(500, {"error": str(e)[:300]})
            if self.path.startswith("/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                caps = list(session.captions)
                last = caps[-1]["id"] if caps else -1            # only captions from now on
                try:
                    while True:
                        caps = list(session.captions)
                        if caps and caps[-1]["id"] < last:       # a new session restarted the numbering
                            last = -1
                        new = [c for c in caps if c["id"] > last]
                        for c in new:
                            self.wfile.write(f"data: {json.dumps({'caption': c}, ensure_ascii=False)}\n\n"
                                             .encode())
                            last = c["id"]
                        if new and session.lags:
                            self.wfile.write(f"data: {json.dumps({'lag': session.lags[-1]})}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(0.15)
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send(404, {"error": "not found"})

    for p in range(port, port + 20):                      # next free port if 7865 is taken
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), H)
            port = p
            break
        except OSError:
            continue
    else:
        return None, None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/", srv
