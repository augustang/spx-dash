"""SPX Dashboard — Flask entry point."""
from __future__ import annotations

import json
import os
import secrets
from datetime import timedelta

from dotenv import load_dotenv
load_dotenv()

# ── Token bootstrap (Render / any host without persistent disk) ─────────────
_token_path = '.streamlit/schwab_tokens.json'
if not os.path.exists(_token_path):
    os.makedirs('.streamlit', exist_ok=True)
    import gist_sync
    _gist_id  = os.environ.get("GIST_ID")
    _gist_tok = os.environ.get("GITHUB_GIST_TOKEN")
    _schwab_key = os.environ.get("SCHWAB_APP_KEY")
    print(f"🔑 Env check — GIST_ID: {'✅' if _gist_id else '❌ MISSING'} | "
          f"GITHUB_GIST_TOKEN: {'✅' if _gist_tok else '❌ MISSING'} | "
          f"SCHWAB_APP_KEY: {'✅' if _schwab_key else '❌ MISSING'}")
    tokens = gist_sync.fetch_tokens_from_gist()
    if tokens:
        with open(_token_path, 'w') as f:
            json.dump(tokens, f)
        print("✅ Schwab tokens loaded from Gist.")
    else:
        print("⚠️  Could not fetch Schwab tokens — charts will be empty until tokens are available.")
else:
    print("✅ Schwab tokens file found on disk.")

from flask import Flask, redirect, render_template, request, session, url_for

from shared.cache import cache
from routes.trading import trading_bp
from routes.study import study_bp

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.permanent_session_lifetime = timedelta(days=90)

cache.init_app(app)

app.register_blueprint(trading_bp)
app.register_blueprint(study_bp)

_DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")


@app.before_request
def require_auth():
    if request.path.startswith('/static') or request.path == '/login':
        return
    if not session.get('auth'):
        return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == _DASHBOARD_PASSWORD and _DASHBOARD_PASSWORD:
            session.permanent = True
            session['auth'] = True
            return redirect(url_for('trading.trading'))
        error = 'Incorrect password.'
    return render_template('login.html', error=error)


if __name__ == "__main__":
    app.run(debug=True)
