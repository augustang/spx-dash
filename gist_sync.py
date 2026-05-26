"""
Read/write schwab_tokens.json to a private GitHub Gist so that
the Render deployment always has the latest tokens without manual paste.
"""
from __future__ import annotations

import json
import requests
import os

GIST_FILENAME = "schwab_tokens.json"


def _load_gist_config():
    """Return (gist_id, github_token) from secrets.toml or env vars."""
    gist_id = os.environ.get("GIST_ID")
    github_token = os.environ.get("GITHUB_GIST_TOKEN")

    if gist_id and github_token:
        return gist_id, github_token

    # Fall back to reading secrets.toml directly (for CLI scripts)
    secrets_path = os.path.join(
        os.path.dirname(__file__), ".schwab", "secrets.toml"
    )
    if os.path.exists(secrets_path):
        with open(secrets_path, "r") as f:
            for line in f:
                if "GIST_ID" in line and "=" in line:
                    gist_id = line.split("=", 1)[1].strip().strip('"')
                if "GITHUB_GIST_TOKEN" in line and "=" in line:
                    github_token = line.split("=", 1)[1].strip().strip('"')

    return gist_id, github_token


def push_tokens_to_gist(tokens: dict) -> bool:
    """Upload token dict to the private gist. Returns True on success."""
    gist_id, github_token = _load_gist_config()
    if not gist_id or not github_token:
        print("⚠️  Gist sync skipped — GIST_ID or GITHUB_GIST_TOKEN not configured.")
        return False

    resp = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
        },
        json={"files": {GIST_FILENAME: {"content": json.dumps(tokens)}}},
        timeout=(5, 30),
    )

    if resp.status_code == 200:
        print("✅ Tokens pushed to GitHub Gist.")
        return True
    else:
        print(f"❌ Gist push failed ({resp.status_code}): {resp.text}")
        return False


def fetch_tokens_from_gist() -> dict | None:
    """Download the latest token dict from the private gist."""
    gist_id, github_token = _load_gist_config()
    if not gist_id or not github_token:
        return None

    resp = requests.get(
        f"https://api.github.com/gists/{gist_id}",
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=(5, 30),
    )

    if resp.status_code == 200:
        content = resp.json()["files"].get(GIST_FILENAME, {}).get("content")
        if content:
            return json.loads(content)
    return None


def create_gist(tokens: dict, github_token: str) -> str | None:
    """Create a new private gist and return its id."""
    resp = requests.post(
        "https://api.github.com/gists",
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "description": "Schwab API tokens (auto-managed)",
            "public": False,
            "files": {GIST_FILENAME: {"content": json.dumps(tokens)}},
        },
        timeout=(5, 30),
    )
    if resp.status_code == 201:
        return resp.json()["id"]
    print(f"❌ Failed to create gist ({resp.status_code}): {resp.text}")
    return None
