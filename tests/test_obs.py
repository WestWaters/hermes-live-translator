"""The OBS caption feed against a simulated obs-websocket 5 server (handshake, password, SendStreamCaption)."""
import base64
import hashlib
import json
import socket
import struct
import threading

import pytest

from live_translate.obs import OBS, OBSError, _auth

SALT, CHALLENGE = "c2FsdA==", "Y2hhbGxlbmdl"


def _frame(obj):
    data = json.dumps(obj).encode()
    return bytes([0x81, len(data)]) + data if len(data) < 126 else bytes([0x81, 126]) + struct.pack(">H", len(data)) + data


def _exact(c, n):
    """n bytes from the socket: recv may return fewer under load."""
    b = b""
    while len(b) < n:
        chunk = c.recv(n - len(b))
        if not chunk:
            raise ConnectionError("client closed")
        b += chunk
    return b


def _read(c):
    b0, b1 = _exact(c, 2)
    n = b1 & 0x7F
    if n == 126:
        n = struct.unpack(">H", _exact(c, 2))[0]
    mask = _exact(c, 4)
    data = _exact(c, n)
    return json.loads(bytes(b ^ mask[i % 4] for i, b in enumerate(data)))


def fake_obs(password, got):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def run():
        c, _ = srv.accept()
        req = c.recv(4096).decode()
        key = [l.split(": ")[1] for l in req.split("\r\n") if l.lower().startswith("sec-websocket-key")][0]
        acc = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        c.sendall(f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Accept: {acc}\r\n\r\n".encode())
        c.sendall(_frame({"op": 0, "d": {"rpcVersion": 1, "authentication": {"salt": SALT, "challenge": CHALLENGE}}}))
        ident = _read(c)
        if ident["d"].get("authentication") != _auth(password, SALT, CHALLENGE):
            c.close()
            return
        c.sendall(_frame({"op": 2, "d": {"negotiatedRpcVersion": 1}}))
        r = _read(c)
        got.append(r["d"])
        c.sendall(_frame({"op": 7, "d": {"requestType": r["d"]["requestType"], "requestId": r["d"]["requestId"],
                                         "requestStatus": {"result": True, "code": 100}}}))
        c.close()

    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1]


def test_captions_reach_obs_with_the_password():
    got = []
    port = fake_obs("hunter2", got)
    OBS("127.0.0.1", port, "hunter2").caption("Hola a todos")
    assert got == [{"requestType": "SendStreamCaption", "requestId": got[0]["requestId"],
                    "requestData": {"captionText": "Hola a todos"}}]


def test_a_wrong_password_is_a_clear_error():
    port = fake_obs("hunter2", [])
    with pytest.raises(OBSError):
        OBS("127.0.0.1", port, "wrong").caption("x")


def test_a_hello_in_the_same_packet_as_the_handshake_is_not_lost():
    """OBS can send its hello right behind the 101 response; the client must keep those bytes."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    got = []

    def run():
        c, _ = srv.accept()
        req = c.recv(4096).decode()
        key = [l.split(": ")[1] for l in req.split("\r\n") if l.lower().startswith("sec-websocket-key")][0]
        acc = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        c.sendall(f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Accept: {acc}\r\n\r\n".encode() + _frame({"op": 0, "d": {"rpcVersion": 1}}))
        _read(c)
        c.sendall(_frame({"op": 2, "d": {"negotiatedRpcVersion": 1}}))
        r = _read(c)
        got.append(r["d"]["requestType"])
        c.sendall(_frame({"op": 7, "d": {"requestType": r["d"]["requestType"], "requestId": r["d"]["requestId"],
                                         "requestStatus": {"result": True, "code": 100}}}))
        c.close()
    threading.Thread(target=run, daemon=True).start()
    OBS("127.0.0.1", srv.getsockname()[1]).caption("hola")
    assert got == ["SendStreamCaption"]
