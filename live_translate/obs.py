"""Closed captions on a stream: each translated line goes into OBS (obs-websocket 5, built into OBS 28+) with
SendStreamCaption. OBS embeds it in the outgoing stream as CEA-608 closed captions, so YouTube and Twitch show
viewers their own CC button: the stream keeps the streamer's real voice; viewers choose to read the translation.

CEA-608 carries Latin-alphabet text only; for other scripts use the caption page as an OBS Browser Source.
Standard library only (a minimal WebSocket client: OBS only needs text frames)."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import threading
import uuid


class OBSError(RuntimeError):
    pass


def _auth(password, salt, challenge):
    secret = base64.b64encode(hashlib.sha256((password + salt).encode()).digest()).decode()
    return base64.b64encode(hashlib.sha256((secret + challenge).encode()).digest()).decode()


class OBS:
    def __init__(self, host="127.0.0.1", port=4455, password=""):
        self.host, self.port, self.password = host, int(port), password or ""
        self.sock, self.lock = None, threading.Lock()
        self.pending = b""            # bytes read past the HTTP handshake: the start of OBS's first frame

    # -- minimal WebSocket (RFC 6455) ------------------------------------------------------------------------
    def _send(self, obj):
        data = json.dumps(obj).encode()
        head = bytearray([0x81])                                    # FIN + text
        n = len(data)
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)                                        # clients must mask
        self.sock.sendall(bytes(head) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _recv_exact(self, n):
        buf, self.pending = self.pending[:n], self.pending[n:]
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise OBSError("OBS closed the connection")
            buf += chunk
        return buf

    def _recv(self):
        while True:
            b0, b1 = self._recv_exact(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._recv_exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if b1 & 0x80 else None
            data = self._recv_exact(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            op = b0 & 0x0F
            if op == 0x8:
                raise OBSError("OBS closed the connection")
            if op == 0x9:                                           # ping -> pong
                continue
            if op == 0x1:
                return json.loads(data)

    # -- obs-websocket 5 --------------------------------------------------------------------------------------
    def connect(self):
        s = socket.create_connection((self.host, self.port), timeout=3)
        key = base64.b64encode(os.urandom(16)).decode()
        s.sendall((f"GET / HTTP/1.1\r\nHost: {self.host}:{self.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                   f"Sec-WebSocket-Protocol: obswebsocket.json\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = s.recv(1024)
            if not chunk:
                raise OBSError("no answer from OBS")
            resp += chunk
        head, _, rest = resp.partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise OBSError("OBS refused the connection: " + head.split(b"\r\n")[0].decode(errors="replace"))
        self.sock, self.pending = s, rest          # OBS may send its hello in the same packet as the handshake
        hello = self._recv()
        identify = {"rpcVersion": 1}
        a = hello.get("d", {}).get("authentication")
        if a:
            if not self.password:
                raise OBSError("OBS needs its WebSocket password (OBS: Tools > WebSocket Server Settings)")
            identify["authentication"] = _auth(self.password, a["salt"], a["challenge"])
        self._send({"op": 1, "d": identify})
        ack = self._recv()
        if ack.get("op") != 2:
            raise OBSError("OBS did not accept the password")
        return self

    def request(self, kind, data=None):
        try:
            return self._request(kind, data)
        except (OSError, ValueError) as e:                          # timeouts, refused, dropped connections
            self.close()
            raise OBSError(f"OBS did not answer ({e.__class__.__name__}): is OBS running with its WebSocket "
                           "server on?") from e

    def _request(self, kind, data=None):
        with self.lock:
            if self.sock is None:
                self.connect()
            rid = uuid.uuid4().hex
            self._send({"op": 6, "d": {"requestType": kind, "requestId": rid, "requestData": data or {}}})
            while True:
                m = self._recv()
                if m.get("op") == 7 and m["d"].get("requestId") == rid:
                    st = m["d"].get("requestStatus", {})
                    if not st.get("result"):
                        raise OBSError(st.get("comment") or f"OBS rejected {kind}")
                    return m["d"].get("responseData") or {}

    def caption(self, text):
        """Send one line as a closed caption (OBS only embeds it while it is streaming)."""
        return self.request("SendStreamCaption", {"captionText": text[:255]})

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock, self.pending = None, b""


class CaptionFeed:
    """Sends each new translation to OBS in the background; reconnects after errors, never blocks translation."""

    def __init__(self, host, port, password):
        self.obs = OBS(host, port, password)
        self.last_error = ""

    def push(self, text):
        def run():
            try:
                self.obs.caption(text)
                self.last_error = ""
            except Exception as e:                                  # noqa: BLE001 -- report, reconnect next time
                self.last_error = str(e)
                self.obs.close()
        threading.Thread(target=run, daemon=True).start()
