"""
Schwab OAuth re-authorization script.

Usage:
    python3 schwab_auth.py                   # default: paste redirect URL after login
    python3 schwab_auth.py --manual        # same (explicit)
    python3 schwab_auth.py --auto          # local HTTPS callback (port 443; use sudo)
    sudo python3 schwab_auth.py --auto     # same with privileges for port 443
    python3 schwab_auth.py --url "https://127.0.0.1/?code=..."  # pass URL directly

What it does:
    1. Opens your browser to the Schwab login page.
    2. Starts a tiny local HTTPS server to capture the OAuth callback.
       (Falls back to manual paste if the port is unavailable.)
    3. Exchanges the auth code for tokens.
    4. Saves tokens locally to .schwab/schwab_tokens.json and pushes them to GitHub Gist.
       The Render deployment pulls tokens from Gist on startup.
"""
from __future__ import annotations

import base64
import json
import os
import ssl
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests

import gist_sync

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECRETS_PATH = os.path.join(os.path.dirname(__file__), ".schwab", "secrets.toml")
TOKEN_PATH = os.path.join(os.path.dirname(__file__), ".schwab", "schwab_tokens.json")
CERT_DIR = os.path.join(os.path.dirname(__file__), ".schwab", "certs")
CERT_FILE = os.path.join(CERT_DIR, "server-cert.pem")
KEY_FILE = os.path.join(CERT_DIR, "server-key.pem")


def _read_secrets():
    with open(SECRETS_PATH, "r") as f:
        lines = f.readlines()
    get = lambda key: next(
        l.split("=", 1)[1].strip().strip('"') for l in lines if key in l
    )
    return get("APP_KEY"), get("APP_SECRET"), get("CALLBACK_URL")


def _ensure_certs():
    """Generate a self-signed cert if one doesn't exist yet."""
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return
    os.makedirs(CERT_DIR, exist_ok=True)
    os.system(
        f'openssl req -x509 -newkey rsa:2048 -keyout "{KEY_FILE}" '
        f'-out "{CERT_FILE}" -days 3650 -nodes -subj "/CN=127.0.0.1" 2>/dev/null'
    )
    print("🔐 Generated self-signed SSL certificate.")


# ---------------------------------------------------------------------------
# Local HTTPS callback server
# ---------------------------------------------------------------------------
_captured_url: str | None = None


class _CallbackHandler(BaseHTTPRequestHandler):
    """Handles the single OAuth redirect from Schwab."""

    def do_GET(self):
        global _captured_url
        _captured_url = self.path
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h2>Tokens captured!</h2>"
            b"<p>You can close this tab and return to the terminal.</p>"
            b"</body></html>"
        )

    def log_message(self, *args):
        pass  # silence request logging


def _try_auto_capture(host: str, port: int, timeout: int = 120) -> str | None:
    """
    Start an HTTPS server, wait for the callback, return the full URL path.
    Returns None if the server can't bind (e.g. port 443 without root).
    """
    global _captured_url
    _captured_url = None
    _ensure_certs()

    try:
        server = HTTPServer((host, port), _CallbackHandler)
    except PermissionError:
        return None
    except OSError as e:
        if e.errno in (13, 48):  # EACCES or EADDRINUSE
            return None
        raise

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT_FILE, KEY_FILE)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)

    def _serve():
        server.handle_request()  # handle exactly one request, then stop

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    deadline = time.time() + timeout
    while _captured_url is None and time.time() < deadline:
        time.sleep(0.3)

    server.server_close()

    if _captured_url:
        if port == 443:
            return f"https://{host}{_captured_url}"
        return f"https://{host}:{port}{_captured_url}"
    return None


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------
def _exchange_code_for_tokens(code: str, app_key: str, app_secret: str, callback_url: str):
    if not code.endswith("@"):
        code += "@"

    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{app_key}:{app_secret}'.encode()).decode()}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": callback_url,
    }

    resp = requests.post(
        "https://api.schwabapi.com/v1/oauth/token",
        headers=headers,
        data=payload,
    )
    return resp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Default is manual paste; opt into local callback with --auto (often needs sudo on 443).
    manual_mode = "--auto" not in sys.argv

    # --url "https://..." lets you pass the redirect URL directly as an arg
    direct_url = None
    if "--url" in sys.argv:
        idx = sys.argv.index("--url")
        if idx + 1 < len(sys.argv):
            direct_url = sys.argv[idx + 1]

    app_key, app_secret, callback_url = _read_secrets()

    parsed_cb = urllib.parse.urlparse(callback_url)
    cb_host = parsed_cb.hostname or "127.0.0.1"
    cb_port = parsed_cb.port or 443

    auth_url = (
        f"https://api.schwabapi.com/v1/oauth/authorize"
        f"?client_id={app_key}&redirect_uri={callback_url}"
    )

    # --- Step 1: Open browser (skip if URL already provided) ---
    if direct_url:
        print(f"\n=== STEP 1: USING PROVIDED REDIRECT URL ===")
        redirected_url = direct_url
    else:
        print("\n=== STEP 1: OPENING BROWSER FOR SCHWAB LOGIN ===")
        print(f"If the browser doesn't open, visit:\n{auth_url}\n")
        webbrowser.open(auth_url)

        # --- Step 2: Capture callback ---
        redirected_url = None

        if not manual_mode:
            print(f"⏳ Waiting for OAuth callback on https://{cb_host}:{cb_port} ...")
            print("   (Log in and approve access in your browser)\n")
            redirected_url = _try_auto_capture(cb_host, cb_port)

            if redirected_url is None:
                print(
                    "⚠️  Could not start local HTTPS server "
                    f"(port {cb_port} requires sudo)."
                )
                print(
                    "   Tip: run 'sudo python3 schwab_auth.py --auto' to bind port 443,\n"
                    "   or omit --auto and paste the redirect URL (default).\n"
                )
                manual_mode = True

        if manual_mode:
            print("After login, your browser will show 'This site can't be reached'.")
            print("Copy the ENTIRE URL from the address bar and paste it below.\n")
            redirected_url = input("Paste redirect URL here > ")

    if not redirected_url:
        print("❌ No callback received. Exiting.")
        sys.exit(1)

    # --- Step 3: Extract code & exchange ---
    parsed = urllib.parse.urlparse(redirected_url)
    code = urllib.parse.parse_qs(parsed.query).get("code", [None])[0]
    if not code:
        print("❌ Could not extract authorization code from URL.")
        sys.exit(1)

    print("=== STEP 2: EXCHANGING CODE FOR TOKENS ===")
    resp = _exchange_code_for_tokens(code, app_key, app_secret, callback_url)

    if resp.status_code != 200:
        print(f"❌ Token exchange failed ({resp.status_code}): {resp.text}")
        sys.exit(1)

    tokens = resp.json()

    # --- Step 4: Save locally ---
    os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        json.dump(tokens, f)
    print(f"✅ Tokens saved to {TOKEN_PATH}")

    # --- Step 5: Push to gist ---
    gist_sync.push_tokens_to_gist(tokens)

    print("\n🎉 Done! Tokens saved locally and pushed to Gist. The dashboard will pick them up automatically.")


if __name__ == "__main__":
    main()
