"""Shared page-header context (date/time + market open/close status)."""
from __future__ import annotations

import datetime

import pytz

import schwab_client
from shared.cache import cache

_ET = pytz.timezone("America/New_York")


@cache.memoize(timeout=300)
def _get_market_hours():
    return schwab_client.fetch_market_hours()


def build_header_ctx() -> dict:
    now = datetime.datetime.now(_ET)
    date_str = f"{now.strftime('%A %B')} {now.day}, {now.year}"
    parts = date_str.split(' ')
    try:
        mi = _get_market_hours()
    except Exception:
        mi = None
    if mi and mi.get('start') and mi.get('end'):
        mkt_start, mkt_end = mi['start'], mi['end']
        if now < mkt_start:
            diff = mkt_start - now
            h, m = int(diff.total_seconds() // 3600), int((diff.total_seconds() % 3600) // 60)
            status = f"{h}h {m}m until open"
        elif now <= mkt_end:
            diff = mkt_end - now
            h, m = int(diff.total_seconds() // 3600), int((diff.total_seconds() % 3600) // 60)
            status = f"{h}h {m}m until close"
        else:
            status = "(Market Closed)"
    elif mi:
        status = "(Market Closed)"
    else:
        status = ""
    return {
        "date_bold": parts[0],
        "date_rest": " ".join(parts[1:]),
        "time": now.strftime("%H:%M"),
        "status": status,
    }
