import json
import requests
import base64
import os

import gist_sync

TOKEN_PATH = '.schwab/schwab_tokens.json'


def _schwab_keys():
    """Return (APP_KEY, APP_SECRET) from env vars or .schwab/secrets.toml fallback."""
    key = os.environ.get("SCHWAB_APP_KEY")
    secret = os.environ.get("SCHWAB_APP_SECRET")
    if key and secret:
        return key, secret

    # Local dev fallback: parse .schwab/secrets.toml
    secrets_path = os.path.join(os.path.dirname(__file__), ".schwab", "secrets.toml")
    if os.path.exists(secrets_path):
        with open(secrets_path) as f:
            for line in f:
                if "APP_KEY" in line and "=" in line:
                    key = line.split("=", 1)[1].strip().strip('"')
                if "APP_SECRET" in line and "=" in line:
                    secret = line.split("=", 1)[1].strip().strip('"')
    return key, secret


def _load_tokens() -> dict | None:
    """Return parsed token dict or None if the file is missing/corrupt."""
    if not os.path.exists(TOKEN_PATH):
        # Try to re-bootstrap from Gist before giving up
        try:
            import gist_sync as _gs
            tokens = _gs.fetch_tokens_from_gist()
            if tokens:
                os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
                with open(TOKEN_PATH, 'w') as f:
                    json.dump(tokens, f)
                return tokens
        except Exception:
            pass
        return None
    try:
        with open(TOKEN_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def refresh_access_token():
    """Silently trades the refresh token for a brand new access token."""
    print("🔄 Access token expired. Refreshing quietly in the background...")

    tokens = _load_tokens()
    if not tokens:
        print("❌ No token file found — cannot refresh.")
        return None

    APP_KEY, APP_SECRET = _schwab_keys()

    headers = {
        'Authorization': f'Basic {base64.b64encode(f"{APP_KEY}:{APP_SECRET}".encode()).decode()}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': tokens['refresh_token']
    }

    response = requests.post('https://api.schwabapi.com/v1/oauth/token', headers=headers, data=payload, timeout=(5, 30))

    if response.status_code == 200:
        new_tokens = response.json()
        if 'refresh_token' not in new_tokens:
            new_tokens['refresh_token'] = tokens['refresh_token']

        with open(TOKEN_PATH, 'w') as f:
            json.dump(new_tokens, f)

        gist_sync.push_tokens_to_gist(new_tokens)
        return new_tokens['access_token']
    else:
        print("❌ CRITICAL: Failed to refresh token. You may need to run schwab_auth.py again.")
        return None


def fetch_market_hours():
    """Returns {'isOpen': bool, 'start': datetime, 'end': datetime} or None."""
    from datetime import datetime
    import pytz

    tokens = _load_tokens()
    if not tokens:
        return None

    access_token = tokens['access_token']
    eastern = pytz.timezone('America/New_York')
    today = datetime.now(eastern).strftime('%Y-%m-%d')

    url = "https://api.schwabapi.com/marketdata/v1/markets"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    params = {"markets": "equity", "date": today}

    response = requests.get(url, headers=headers, params=params, timeout=(5, 30))

    if response.status_code == 401:
        new_token = refresh_access_token()
        if new_token is None:
            return None
        headers["Authorization"] = f"Bearer {new_token}"
        response = requests.get(url, headers=headers, params=params, timeout=(5, 30))

    if response.status_code != 200:
        return None

    try:
        eq = response.json()['equity']['EQ']
        is_open = eq.get('isOpen', False)
        session = eq.get('sessionHours', {}).get('regularMarket', [{}])[0]
        start = datetime.fromisoformat(session['start']).astimezone(eastern)
        end = datetime.fromisoformat(session['end']).astimezone(eastern)
        return {'isOpen': is_open, 'start': start, 'end': end}
    except (KeyError, IndexError, TypeError):
        return {'isOpen': False, 'start': None, 'end': None}


def fetch_live_quote(symbol="$SPX"):
    """Fetches a live quote, automatically refreshing the token if needed."""
    tokens = _load_tokens()
    if not tokens:
        return None

    access_token = tokens['access_token']
    url = "https://api.schwabapi.com/marketdata/v1/quotes"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    response = requests.get(url, headers=headers, params={"symbols": symbol}, timeout=(5, 30))

    if response.status_code == 401:
        new_token = refresh_access_token()
        if new_token is None:
            return None
        headers["Authorization"] = f"Bearer {new_token}"
        response = requests.get(url, headers=headers, params={"symbols": symbol}, timeout=(5, 30))

    if response.status_code == 200:
        data = response.json()
        quote = data[symbol]['quote']
        return {
            "lastPrice": quote['lastPrice'],
            "openPrice": quote['openPrice'],
            "closePrice": quote['closePrice'],
            "netChange": quote['netChange']
        }
    return None


def fetch_price_history(symbol="$SPX", period_type="day", period=1, freq_type="minute", freq=5, start_date=None, end_date=None):
    """Fetches intraday or historical candles from Schwab using explicit timestamps."""
    tokens = _load_tokens()
    if not tokens:
        return None

    access_token = tokens['access_token']
    url = "https://api.schwabapi.com/marketdata/v1/pricehistory"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    params = {
        "symbol": symbol,
        "periodType": period_type,
        "frequencyType": freq_type,
        "frequency": freq
    }

    if start_date and end_date:
        params["startDate"] = int(start_date)
        params["endDate"] = int(end_date)
    else:
        params["period"] = period

    response = requests.get(url, headers=headers, params=params, timeout=(5, 30))
    if response.status_code == 401:
        new_token = refresh_access_token()
        if new_token is None:
            return None
        headers["Authorization"] = f"Bearer {new_token}"
        response = requests.get(url, headers=headers, params=params, timeout=(5, 30))

    if response.status_code == 200:
        return response.json()
    return None


def fetch_options_chain(symbol="$SPX"):
    """Fetches the near-term Out-Of-The-Money puts."""
    tokens = _load_tokens()
    if not tokens:
        return None

    access_token = tokens['access_token']
    url = "https://api.schwabapi.com/marketdata/v1/chains"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    params = {
        "symbol": symbol,
        "contractType": "PUT",
        "includeQuotes": "TRUE",
        "range": "OTM",
        "strikeCount": 90,
        "daysToExpiration": 1
    }

    response = requests.get(url, headers=headers, params=params, timeout=(5, 30))
    if response.status_code == 401:
        new_token = refresh_access_token()
        if new_token is None:
            return None
        headers["Authorization"] = f"Bearer {new_token}"
        response = requests.get(url, headers=headers, params=params, timeout=(5, 30))

    if response.status_code == 200:
        return response.json()
    return None
