#!/usr/bin/env python3
import hashlib
import http.server
import json
import re
import subprocess
import sys
import threading
import time

NONCE = "ptz-test-nonce"

class Handler(http.server.BaseHTTPRequestHandler):
    digest_seen = False
    calls = {"capabilities": 0, "profiles": 0, "status": 0, "move": 0, "stop": 0}

    def log_message(self, *_):
        pass

    @staticmethod
    def valid_digest(header, method):
        values = {key: quoted or plain for key, quoted, plain in
                  ((match.group(1), match.group(2), match.group(3))
                   for match in re.finditer(r'(\w+)=(?:"([^"]*)"|([^,\s]+))', header))}
        required = ("username", "realm", "nonce", "uri", "response", "nc", "cnonce", "qop")
        if any(not values.get(key) for key in required):
            return False
        if values["username"] != "operator" or values["realm"] != "ptz-test" or values["nonce"] != NONCE:
            return False
        md5 = lambda text: hashlib.md5(text.encode()).hexdigest()
        ha1 = md5("operator:ptz-test:secret")
        ha2 = md5(f'{method}:{values["uri"]}')
        expected = md5(f'{ha1}:{NONCE}:{values["nc"]}:{values["cnonce"]}:{values["qop"]}:{ha2}')
        return values["response"] == expected

    def do_POST(self):
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Digest "):
            self.send_response(401)
            self.send_header("WWW-Authenticate", f'Digest realm="ptz-test", nonce="{NONCE}", qop="auth", algorithm=MD5')
            self.end_headers()
            return
        if not self.valid_digest(auth, "POST"):
            self.send_error(403)
            return
        Handler.digest_seen = True
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        base = f"http://127.0.0.1:{self.server.server_port}"
        if "GetCapabilities" in body:
            Handler.calls["capabilities"] += 1
            payload = f"<tds:GetCapabilitiesResponse><tds:Capabilities><tt:Media><tt:XAddr>{base}/media</tt:XAddr></tt:Media><tt:PTZ><tt:XAddr>{base}/ptz</tt:XAddr></tt:PTZ></tds:Capabilities></tds:GetCapabilitiesResponse>"
        elif "GetProfiles" in body:
            Handler.calls["profiles"] += 1
            payload = '<trt:GetProfilesResponse><trt:Profiles token="profile-1"/></trt:GetProfilesResponse>'
        elif "GetStatus" in body:
            Handler.calls["status"] += 1
            if Handler.calls["status"] == 2:
                self.send_error(503)
                return
            if Handler.calls["status"] == 3:
                time.sleep(0.7)
            payload = '<tptz:GetStatusResponse><tptz:PTZStatus><tt:Position><tt:PanTilt x="0.1" y="-0.2"/><tt:Zoom x="0.25"/></tt:Position><tt:MoveStatus><tt:PanTilt>IDLE</tt:PanTilt><tt:Zoom>IDLE</tt:Zoom></tt:MoveStatus></tptz:PTZStatus></tptz:GetStatusResponse>'
        elif "ContinuousMove" in body:
            Handler.calls["move"] += 1
            if Handler.calls["move"] == 1:
                self.send_error(503)
                return
            payload = "<tptz:ContinuousMoveResponse/>"
        elif "<tptz:Stop>" in body:
            Handler.calls["stop"] += 1
            payload = "<tptz:StopResponse/>"
        else:
            self.send_error(500)
            return
        xml = ('<?xml version="1.0"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
               'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
               'xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema"><s:Body>' + payload + '</s:Body></s:Envelope>').encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/soap+xml")
        self.send_header("Content-Length", str(len(xml)))
        self.end_headers()
        try:
            self.wfile.write(xml)
        except (BrokenPipeError, ConnectionResetError):
            pass

def main():
    executable = sys.argv[1]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        proc = subprocess.run([executable, f"http://127.0.0.1:{server.server_port}/device"], timeout=10)
    finally:
        server.shutdown()
    minimum = {"capabilities": 2, "profiles": 2, "status": 4, "move": 2, "stop": 1}
    required = all(Handler.calls[key] >= count for key, count in minimum.items())
    if proc.returncode or not Handler.digest_seen or not required:
        print(json.dumps({"returncode": proc.returncode, "digest": Handler.digest_seen, "calls": Handler.calls}), file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
