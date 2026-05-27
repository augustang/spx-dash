"""Study page — Flask Blueprint."""
from __future__ import annotations

import datetime


def _most_recent_trading_day() -> datetime.date:
    """Return the most recent weekday whose session has fully closed.

    If today is a weekday but the market hasn't closed yet (before 4 PM ET),
    we step back to the previous trading day so the explorer doesn't open on
    a date with no data.
    """
    import pytz
    now_et = datetime.datetime.now(pytz.timezone("America/New_York"))
    d = now_et.date()
    # Walk back past weekends
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    # If today is a weekday but before market close, use the previous trading day
    if d == now_et.date() and now_et.time() < datetime.time(16, 0):
        d -= datetime.timedelta(days=1)
        while d.weekday() >= 5:
            d -= datetime.timedelta(days=1)
    return d
import html
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Blueprint, render_template, request, session

import schwab_client
from shared.cache import cache
from shared.chart import create_spx_chart, create_long_chart, empty_figure, _HOVERLABEL, GREEN_400, GREEN_600, _fmt_date
from shared.events import FOMC_DATES, get_financial_events

study_bp = Blueprint('study', __name__)

# ── Constants (copied from pages/study.py) ───────────────────────────────────

_DATA_DIR      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_FRD_5MIN_PATH = os.path.join(_DATA_DIR, "SPX_5min.csv")
_FRD_1DAY_PATH = os.path.join(_DATA_DIR, "SPX_1day.csv")
_NOTES_PATH    = os.path.join(_DATA_DIR, "intraday_notes.json")
_FRD_MIN_DATE  = datetime.date(2008, 1, 2)
_FRD_DAILY_START   = datetime.date(2000, 11, 27)
_EVENT_IMPACT_YEARS = 15
MINUTE_HISTORY_DAYS = 240

try:
    with open(_NOTES_PATH) as _f:
        _INTRADAY_NOTES: dict[str, str] = json.load(_f)
except (FileNotFoundError, json.JSONDecodeError):
    _INTRADAY_NOTES = {}

_CC_SNAP_TIMES = [(h, m) for h in range(9, 16) for m in (0, 30) if (h, m) >= (9, 30) and (h, m) <= (15, 30)]
_CC_TIME_OPTS  = [f"{h}:{m:02d}" for h, m in _CC_SNAP_TIMES]
_CC_COND_TYPES = ["% change at time", "Days from event", "Day of week", "Month", "Overnight gap"]
_CC_EVENT_OPTS = ["OPEX", "VIX Exp", "FOMC"]
_CC_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
_CC_MON_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_CC_NORM_OPTS  = ["% from prior close", "% from open"]

_CMP_COLORS      = ["#B71AFF", "#4B7BFF", "#FF6B35", "#11B8A0",
                    "#FF3D54", "#F5A623", GREEN_400, "#888888"]
_CMP_ENTRY_TYPES = ["FOMC", "OPEX", "VIX Exp", "Black Swan", "Specific date"]
_CMP_OFFSET_OPTS = ["-3 days", "-2 days", "-1 day", "Day of", "+1 day", "+2 days"]
_CMP_OFFSET_VALS = {"-3 days": -3, "-2 days": -2, "-1 day": -1, "Day of": 0, "+1 day": 1, "+2 days": 2}
_CMP_RANGE_OPTS  = ["3M", "6M", "1Y", "2Y", "All"]
_CMP_RANGE_DAYS  = {"3M": 91, "6M": 182, "1Y": 365, "2Y": 730, "All": None}
_CMP_GAP_OPTS    = ["All", "Gap up ↑", "Gap down ↓"]

_NOTABLE_EVENTS: list[tuple[datetime.date, str]] = [
    (datetime.date(2019, 8,  5),  "Trade war escalation"),
    (datetime.date(2020, 2, 24),  "COVID fears begin"),
    (datetime.date(2020, 2, 27),  "COVID selloff"),
    (datetime.date(2020, 3,  9),  "Black Monday II"),
    (datetime.date(2020, 3, 12),  "COVID crash"),
    (datetime.date(2020, 3, 16),  "Worst day since '87"),
    (datetime.date(2020, 3, 24),  "Biggest rally since '33"),
    (datetime.date(2020, 3, 26),  "Stimulus rally"),
    (datetime.date(2020, 4,  6),  "Stimulus rally II"),
    (datetime.date(2022, 5,  5),  "Fed hike selloff"),
    (datetime.date(2022, 6, 13),  "Bear mkt confirm"),
    (datetime.date(2022, 9, 13),  "Hot CPI shock"),
    (datetime.date(2024, 12, 18), "FOMC hawkish pivot"),
    (datetime.date(2025, 4,  3),  "Liberation Day"),
    (datetime.date(2025, 4,  4),  "Tariff panic"),
    (datetime.date(2025, 4,  9),  "Tariff pause"),
]

# ── Default session stores ───────────────────────────────────────────────────

_DEFAULT_CMP_STORE = {
    "ids": [0], "next_id": 1,
    "range": "All", "gap": "All",
    "entries": [{"id": 0, "type": "FOMC", "offset": "Day of",
                 "date": (datetime.date.today() - datetime.timedelta(days=1)).isoformat(),
                 "enabled": True}],
}
_DEFAULT_CC_STORE = {"ids": [], "next_id": 0, "entries": [], "range": "All", "gap": "All", "norm": "% from prior close"}


def _get_cmp_store() -> dict:
    if 'cmp' not in session:
        session['cmp'] = dict(_DEFAULT_CMP_STORE)
    return session['cmp']


def _get_cc_store() -> dict:
    if 'cc' not in session:
        session['cc'] = {
            "ids": [], "next_id": 0, "entries": [],
            "range": "All", "gap": "All", "norm": "% from prior close",
        }
    store = session['cc']
    # Sanitize: drop any entry whose type got corrupted (e.g. set to a time
    # string due to the hx-include="select" multi-value bug).
    valid_types = set(_CC_COND_TYPES)
    bad_ids = {e["id"] for e in store.get("entries", []) if e.get("type") not in valid_types}
    if bad_ids:
        store["entries"] = [e for e in store["entries"] if e["id"] not in bad_ids]
        store["ids"]     = [i for i in store.get("ids", []) if i not in bad_ids]
        session.modified = True
    # Ensure auto-mode keys exist for sessions created before this feature.
    store.setdefault("auto_mode",       True)
    store.setdefault("tolerance",       0.20)
    store.setdefault("match_threshold", 0.75)
    return store


def _save_cmp_store(s: dict) -> None:
    session['cmp'] = s
    session.modified = True


def _save_cc_store(s: dict) -> None:
    session['cc'] = s
    session.modified = True


# ── Cached data loaders ──────────────────────────────────────────────────────

@cache.memoize(timeout=0)
def _load_frd_5min() -> pd.DataFrame:
    if not os.path.exists(_FRD_5MIN_PATH):
        return pd.DataFrame()
    df = pd.read_csv(_FRD_5MIN_PATH, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
    df.index = df.index.tz_localize(None)
    return df


@cache.memoize(timeout=0)
def _load_frd_daily() -> pd.DataFrame:
    if not os.path.exists(_FRD_1DAY_PATH):
        return pd.DataFrame()
    df = pd.read_csv(_FRD_1DAY_PATH, parse_dates=["date"])
    df.set_index("date", inplace=True)
    df.index = df.index.normalize()
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
    return df[["Open", "High", "Low", "Close"]].sort_index()


def _append_to_archive(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df_out = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    df_out.index.name = "timestamp"
    if os.path.exists(_FRD_5MIN_PATH):
        existing = pd.read_csv(_FRD_5MIN_PATH, parse_dates=["timestamp"])
        existing.set_index("timestamp", inplace=True)
        existing.index = existing.index.tz_localize(None)
        combined = pd.concat([existing, df_out])
        combined = combined[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)
        combined.to_csv(_FRD_5MIN_PATH)
    else:
        df_out.to_csv(_FRD_5MIN_PATH)
    cache.delete_memoized(_load_frd_5min)
    cache.delete_memoized(_build_daily_snapshots)


@cache.memoize(timeout=86400)
def get_spx_daily(years) -> pd.DataFrame:
    now_ms = int(datetime.datetime.now().timestamp() * 1000)
    if years is None:
        start_ms = int(datetime.datetime(_FRD_DAILY_START.year, _FRD_DAILY_START.month, _FRD_DAILY_START.day).timestamp() * 1000)
    else:
        start_ms = now_ms - 86400 * 1000 * 365 * years
    raw = schwab_client.fetch_price_history(
        symbol="$SPX", period_type="year", freq_type="daily", freq=1,
        start_date=start_ms, end_date=now_ms,
    )
    if raw and 'candles' in raw:
        df = pd.DataFrame(raw['candles'])
        if not df.empty:
            df['datetime'] = (
                pd.to_datetime(df['datetime'], unit='ms')
                .dt.tz_localize('UTC').dt.tz_convert('America/New_York')
                .dt.tz_localize(None).dt.normalize()
            )
            df.set_index('datetime', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
            return df
    frd = _load_frd_daily()
    if not frd.empty:
        if years is None:
            return frd[frd.index >= pd.Timestamp(_FRD_DAILY_START)]
        return frd[frd.index >= pd.Timestamp.now() - pd.DateOffset(years=years)]
    return pd.DataFrame()


@cache.memoize(timeout=86400)
def _get_event_daily_df() -> pd.DataFrame:
    now_ms   = int(datetime.datetime.now().timestamp() * 1000)
    start_ms = now_ms - 86400 * 1000 * 365 * _EVENT_IMPACT_YEARS
    raw = schwab_client.fetch_price_history(
        symbol="$SPX", period_type="year", freq_type="daily", freq=1,
        start_date=start_ms, end_date=now_ms,
    )
    if raw and 'candles' in raw:
        df = pd.DataFrame(raw['candles'])
        if not df.empty:
            df['datetime'] = (
                pd.to_datetime(df['datetime'], unit='ms')
                .dt.tz_localize('UTC').dt.tz_convert('America/New_York')
                .dt.tz_localize(None).dt.normalize()
            )
            df.set_index('datetime', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
            return df
    frd = _load_frd_daily()
    if not frd.empty:
        return frd[frd.index >= pd.Timestamp.now() - pd.DateOffset(years=_EVENT_IMPACT_YEARS)]
    return pd.DataFrame()


@cache.memoize(timeout=0)
def get_spx_5min_for_date(d: datetime.date) -> pd.DataFrame:
    frd = _load_frd_5min()
    if not frd.empty:
        day_df = frd[frd.index.date == d]
        if not day_df.empty:
            return day_df[["Open", "High", "Low", "Close"]]
    start_dt = datetime.datetime.combine(d, datetime.time(0, 0))
    end_dt   = datetime.datetime.combine(d, datetime.time(23, 59, 59))
    raw = schwab_client.fetch_price_history(
        symbol="$SPX", period_type="day", freq_type="minute", freq=5,
        start_date=int(start_dt.timestamp() * 1000),
        end_date=int(end_dt.timestamp() * 1000),
    )
    if raw and 'candles' in raw:
        df = pd.DataFrame(raw['candles'])
        if not df.empty:
            df['datetime'] = (
                pd.to_datetime(df['datetime'], unit='ms')
                .dt.tz_localize('UTC').dt.tz_convert('America/New_York')
                .dt.tz_localize(None)
            )
            df = df[df['datetime'].dt.date == d]
            if not df.empty:
                df.set_index('datetime', inplace=True)
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
                df = df.between_time('09:30', '16:00')
                _append_to_archive(df)
                return df
    return pd.DataFrame()


@cache.memoize(timeout=120)
def _get_today_5min_live() -> pd.DataFrame:
    """Fetch today's 5-min bars with a 2-minute cache so auto-mode stays current."""
    today = datetime.date.today()
    start_dt = datetime.datetime.combine(today, datetime.time(0, 0))
    end_dt   = datetime.datetime.combine(today, datetime.time(23, 59, 59))
    raw = schwab_client.fetch_price_history(
        symbol="$SPX", period_type="day", freq_type="minute", freq=5,
        start_date=int(start_dt.timestamp() * 1000),
        end_date=int(end_dt.timestamp() * 1000),
    )
    if raw and 'candles' in raw:
        df = pd.DataFrame(raw['candles'])
        if not df.empty:
            df['datetime'] = (
                pd.to_datetime(df['datetime'], unit='ms')
                .dt.tz_localize('UTC').dt.tz_convert('America/New_York')
                .dt.tz_localize(None)
            )
            df = df[df['datetime'].dt.date == today]
            if not df.empty:
                df.set_index('datetime', inplace=True)
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
                return df.between_time('09:30', '16:00')
    # Fallback to the CSV archive for today's rows if Schwab is unavailable.
    frd = _load_frd_5min()
    if not frd.empty:
        day_df = frd[frd.index.date == today]
        if not day_df.empty:
            return day_df[["Open", "High", "Low", "Close"]]
    return pd.DataFrame()


@cache.memoize(timeout=0)
def _build_daily_snapshots() -> pd.DataFrame:
    frd5 = _load_frd_5min()
    frd1 = _load_frd_daily()
    if frd5.empty:
        return pd.DataFrame()
    df = frd5.copy()
    df["_date"] = df.index.date
    df["_time"] = df.index.time
    grp_open  = df.groupby("_date")["Open"].first()
    grp_close = df.groupby("_date")["Close"].last()
    grp_low   = df.groupby("_date")["Low"].min()
    snap = pd.DataFrame(index=grp_open.index)
    snap.index.name = "date"
    snap["eod_pct"]     = (grp_close / grp_open - 1) * 100
    snap["eod_chg_pct"] = (grp_close / grp_close.shift(1) - 1) * 100
    snap["eod_low_pct"] = (grp_low   / grp_open - 1) * 100
    snap["prev_close"]  = grp_close.shift(1)
    if not frd1.empty:
        ds = frd1.sort_index()
        gap_s = (ds["Open"] / ds["Close"].shift(1) - 1) * 100
        gap_s.index = gap_s.index.date
        snap["gap_pct"] = gap_s.reindex(snap.index)
    else:
        snap["gap_pct"] = np.nan
    for h, m in _CC_SNAP_TIMES:
        k = f"{h:02d}{m:02d}"
        t = datetime.time(h, m)
        sb = df[df["_time"] <= t]
        if sb.empty:
            snap[f"pct_at_{k}"] = np.nan
            snap[f"range_at_{k}"] = np.nan
            continue
        sc = sb.groupby("_date")["Close"].last()
        sh = sb.groupby("_date")["High"].max()
        sl = sb.groupby("_date")["Low"].min()
        snap[f"pct_at_{k}"]             = ((sc / grp_open - 1) * 100).reindex(snap.index)
        snap[f"pct_from_close_at_{k}"] = ((sc / snap["prev_close"]) - 1) * 100
        snap[f"range_at_{k}"]           = ((sh - sl) / grp_open * 100).reindex(snap.index)
    all_ds = sorted(snap.index.tolist())
    if all_ds:
        ev_s = min(all_ds) - datetime.timedelta(days=90)
        ev_e = max(all_ds) + datetime.timedelta(days=90)
        all_ev  = get_financial_events(ev_s, ev_e)
        opex_ds = sorted(d for d, lbl in all_ev if "OPEX"    in lbl)
        vix_ds  = sorted(d for d, lbl in all_ev if lbl == "VIX Exp")
        fomc_ds = sorted(d for d in FOMC_DATES if ev_s <= d <= ev_e)

        def _near(d, evts):
            return float(min(((d - e).days for e in evts), key=abs)) if evts else np.nan

        snap["days_from_opex"]    = [_near(d, opex_ds) for d in snap.index]
        snap["days_from_vix_exp"] = [_near(d, vix_ds)  for d in snap.index]
        snap["days_from_fomc"]    = [_near(d, fomc_ds) for d in snap.index]
    else:
        snap["days_from_opex"] = snap["days_from_vix_exp"] = snap["days_from_fomc"] = np.nan
    snap["day_of_week"] = [d.weekday() for d in snap.index]
    snap["month"]       = [d.month     for d in snap.index]
    return snap


# ── Helper: build header context (shared with trading) ───────────────────────

@cache.memoize(timeout=300)
def _get_market_hours():
    return schwab_client.fetch_market_hours()


def _build_header_ctx():
    import pytz
    eastern = pytz.timezone('America/New_York')
    now = datetime.datetime.now(eastern)
    date_str = f"{now.strftime('%A %B')} {now.day}, {now.year}"
    parts = date_str.split(' ')
    time_str = now.strftime("%H:%M")
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
        "time": time_str,
        "status": status,
    }


# ── Chart helper ─────────────────────────────────────────────────────────────

def _chart_html(div_id: str, fig: go.Figure, evt_payload=None) -> str:
    fig_json = html.escape(fig.to_json(), quote=True)
    h = fig.layout.height or 400
    evt_attr = (
        f' data-events="{html.escape(json.dumps(evt_payload), quote=True)}"'
        if evt_payload else ''
    )
    return (
        f'<div class="chart-pill-wrap" style="position:relative;width:100%;height:{h}px;overflow:hidden">'
        f'<div id="{div_id}" data-plotly="{fig_json}"{evt_attr} '
        f'style="width:100%;height:{h}px"></div>'
        f'</div>'
    )


# ── Main page ────────────────────────────────────────────────────────────────

@study_bp.route('/study')
def study():
    today = datetime.date.today()
    frd_loaded = os.path.exists(_FRD_5MIN_PATH)
    min_date = _FRD_MIN_DATE if frd_loaded else today - datetime.timedelta(days=MINUTE_HISTORY_DAYS)
    return render_template(
        'study.html',
        active_page='study',
        header=_build_header_ctx(),
        today=today.isoformat(),
        yesterday=_most_recent_trading_day().isoformat(),
        min_date=min_date.isoformat(),
    )


# ── Section 1: Long-term chart ───────────────────────────────────────────────

@cache.memoize(timeout=3600)
def _build_long_chart_html(selected_range: str, show_events: bool, show_line: bool) -> str:
    """Build and serialize the long-term chart HTML.

    Cached for 1 hour — the 20 possible (range × events × line) combinations
    are all warm after first load, making subsequent visits near-instant.
    The underlying get_spx_daily() data cache (86400s) is a separate layer;
    this cache avoids the expensive Plotly figure-build + to_json() each time.
    """
    range_params = {"1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10, "Max": None}
    years   = range_params.get(selected_range, 1)
    df_long = get_spx_daily(years)
    if df_long.empty:
        return _chart_html('study-long-chart', empty_figure(), evt_payload=None)
    last_close = float(df_long['Close'].iloc[-1])
    first_open = float(df_long['Open'].iloc[0])
    is_down    = (last_close - first_open) < 0
    line_color = "#FF3D54" if is_down else GREEN_400
    halo_color = 'rgba(255,61,84,0.3)' if is_down else 'rgba(17,241,133,0.3)'
    events = None
    if show_events:
        lookahead = df_long.index.max() + pd.DateOffset(months=1)
        events = get_financial_events(df_long.index.min(), lookahead)
    ohlc_df = None if show_line else df_long
    fig, evt_payload = create_long_chart(
        df_long['Close'], df_long.index,
        line_color, halo_color,
        events=events, chart_height=500, ohlc_df=ohlc_df,
    )
    return _chart_html('study-long-chart', fig, evt_payload=evt_payload)


@study_bp.route('/api/study/long-chart')
def api_long_chart():
    selected_range = request.args.get('range', '1Y')
    show_events    = bool(request.args.get('show_events'))
    show_line      = bool(request.args.get('show_line'))
    return _build_long_chart_html(selected_range, show_events, show_line)


# ── Section 2: Intraday explorer ─────────────────────────────────────────────

@study_bp.route('/api/study/intraday')
def api_intraday():
    date_str = request.args.get('date')
    if not date_str:
        return ''
    selected_date = datetime.date.fromisoformat(date_str)
    df_day = get_spx_5min_for_date(selected_date)
    today_iso    = datetime.date.today().isoformat()
    min_date_iso = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
    if df_day.empty:
        return render_template('partials/intraday.html',
            error=f"No data for {_fmt_date(selected_date)}. "
                  "Try a US trading day within the last ~9 months.",
            fig_html='', stats=None,
            date_str=date_str, today=today_iso, min_date=min_date_iso)

    day_open  = float(df_day['Open'].iloc[0])
    day_high  = float(df_day['High'].max())
    day_low   = float(df_day['Low'].min())
    day_close = float(df_day['Close'].iloc[-1])

    gap_pts = gap_pct = None
    prior_close = None
    try:
        df_long = get_spx_daily(1)
        if not df_long.empty:
            sel_ts = pd.Timestamp(selected_date)
            prior_days = df_long.index[df_long.index < sel_ts]
            if len(prior_days) > 0:
                prior_close = float(df_long.loc[prior_days[-1], 'Close'])
                gap_pts = day_open - prior_close
                gap_pct = (gap_pts / prior_close) * 100
    except Exception:
        pass

    log_rets = np.log(df_day['Close'] / df_day['Close'].shift(1)).dropna()
    vol_str = "—"
    if len(log_rets) > 1:
        vol_pct = log_rets.std() * np.sqrt(78) * 100
        vol_pts = vol_pct * day_open / 100
        vol_str = f"{vol_pts:.1f} pts ({vol_pct:.2f}%)"

    _ref_price    = prior_close if prior_close is not None else day_open
    day_ch_pts    = day_close - _ref_price
    day_ch_pct    = (day_ch_pts / _ref_price) * 100

    is_down    = day_ch_pts < 0
    line_color = "#FF3D54" if is_down else GREEN_400
    halo_color = 'rgba(255,61,84,0.3)' if is_down else 'rgba(17,241,133,0.3)'
    fig = create_spx_chart(
        _fmt_date(selected_date),
        df_day['Close'], df_day.index, line_color, halo_color,
        chart_height=420, hover_xfmt="%H:%M",
    )

    today_iso    = datetime.date.today().isoformat()
    min_date_iso = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()

    return render_template('partials/intraday.html',
        error=None,
        fig_html=_chart_html('study-intraday-chart', fig),
        date_str=date_str,
        today=today_iso,
        min_date=min_date_iso,
        stats={
            "open": day_open, "close": day_close,
            "high": day_high, "low": day_low,
            "day_ch_pts": day_ch_pts, "day_ch_pct": day_ch_pct,
            "gap_pts": gap_pts, "gap_pct": gap_pct,
            "open_to_low_pts": day_low - day_open,
            "open_to_low_pct": (day_low - day_open) / day_open * 100,
            "high_to_low_pts": day_low - day_high,
            "high_to_low_pct": (day_low - day_high) / day_high * 100,
            "vol": vol_str,
        })


# ── Section 3: Event comparison ──────────────────────────────────────────────

@study_bp.route('/api/study/cmp', methods=['POST'])
def api_cmp_update():
    """Handle add/clear/delete/update events for the comparison store."""
    store = _get_cmp_store()
    action = request.form.get('action')

    if action == 'add':
        cid = store["next_id"]
        store["next_id"] += 1
        store["ids"].append(cid)
        store["entries"].append({
            "id": cid, "type": "FOMC", "offset": "Day of",
            "date": (datetime.date.today() - datetime.timedelta(days=1)).isoformat(),
            "enabled": True,
        })
    elif action == 'clear':
        store["ids"] = []
        store["entries"] = []
    elif action == 'delete':
        cid = int(request.form.get('id', -1))
        store["entries"] = [e for e in store["entries"] if e["id"] != cid]
        store["ids"]     = [i for i in store["ids"]     if i != cid]
    elif action == 'update':
        cid    = int(request.form.get('id', -1))
        field  = request.form.get('field')
        value  = request.form.get('value')
        for entry in store["entries"]:
            if entry["id"] == cid and field:
                if field == "enabled":
                    entry["enabled"] = value == "true"
                else:
                    entry[field] = value
    elif action == 'filter':
        store["range"] = request.form.get('range', store["range"])
        store["gap"]   = request.form.get('gap',   store["gap"])

    _save_cmp_store(store)
    n_badge_html, charts_html = _build_cmp_charts_html(store)
    return render_template('partials/cmp_section.html',
        store=store,
        cmp_colors=_CMP_COLORS,
        entry_types=_CMP_ENTRY_TYPES,
        offset_opts=_CMP_OFFSET_OPTS,
        range_opts=_CMP_RANGE_OPTS,
        gap_opts=_CMP_GAP_OPTS,
        today=datetime.date.today().isoformat(),
        min_date=(_FRD_MIN_DATE if os.path.exists(_FRD_5MIN_PATH)
                  else (datetime.date.today() - datetime.timedelta(days=MINUTE_HISTORY_DAYS))).isoformat(),
        n_badge_html=n_badge_html,
        charts_html=charts_html,
    )


@study_bp.route('/api/study/cmp')
def api_cmp_get():
    store = _get_cmp_store()
    n_badge_html, charts_html = _build_cmp_charts_html(store)
    return render_template('partials/cmp_section.html',
        store=store,
        cmp_colors=_CMP_COLORS,
        entry_types=_CMP_ENTRY_TYPES,
        offset_opts=_CMP_OFFSET_OPTS,
        range_opts=_CMP_RANGE_OPTS,
        gap_opts=_CMP_GAP_OPTS,
        today=datetime.date.today().isoformat(),
        min_date=(_FRD_MIN_DATE if os.path.exists(_FRD_5MIN_PATH)
                  else (datetime.date.today() - datetime.timedelta(days=MINUTE_HISTORY_DAYS))).isoformat(),
        n_badge_html=n_badge_html,
        charts_html=charts_html,
    )


def _cmp_resolve_entry(entry, cmp_daily, td_idx, gap_map):
    today_ = datetime.date.today()
    ctype  = entry.get("type", "FOMC")
    if ctype == "Specific date":
        d_str = entry.get("date")
        if d_str:
            d = datetime.date.fromisoformat(d_str)
            return [d], _fmt_date(d)
        return [], "Specific date"
    rng      = entry.get("range", "All")
    rng_days = _CMP_RANGE_DAYS.get(rng)
    cutoff   = (today_ - datetime.timedelta(days=rng_days)) if rng_days else datetime.date(2000, 1, 1)
    all_ev   = get_financial_events(cutoff, today_)
    if ctype == "FOMC":
        raw = sorted(d for d in FOMC_DATES if cutoff <= d <= today_)
    elif ctype == "OPEX":
        raw = sorted(d for d, lbl in all_ev if "OPEX" in lbl and d <= today_)
    elif ctype == "Black Swan":
        raw = sorted(d for d, _ in _NOTABLE_EVENTS if cutoff <= d <= today_)
    else:
        raw = sorted(d for d, lbl in all_ev if lbl == "VIX Exp" and d <= today_)
    offset_val = _CMP_OFFSET_VALS.get(entry.get("offset", "Day of"), 0)
    resolved = []
    if len(td_idx) > 0:
        for ev_d in raw:
            pos    = int(td_idx.searchsorted(pd.Timestamp(ev_d)))
            target = pos + offset_val
            if 0 <= target < len(td_idx):
                resolved.append(td_idx[target].date())
    gap_f = entry.get("gap", "All")
    if gap_f == "Gap up ↑":
        resolved = [d for d in resolved if gap_map.get(d, 0) > 0]
    elif gap_f == "Gap down ↓":
        resolved = [d for d in resolved if gap_map.get(d, 0) < 0]
    seen, out = set(), []
    for d in resolved:
        if d not in seen:
            seen.add(d)
            out.append(d)
    off = entry.get("offset", "Day of")
    off_label = f" ({off})" if off != "Day of" else ""
    return out, f"{ctype}{off_label}"


def _build_cmp_charts_html(store) -> str:
    cmp_daily = _load_frd_daily()
    cmp_frd5  = _load_frd_5min()
    gap_map: dict[datetime.date, float] = {}
    prev_close_s = pd.Series(dtype=float)
    eod_chg_s    = pd.Series(dtype=float)
    td_idx = pd.DatetimeIndex([])
    if not cmp_daily.empty:
        ds = cmp_daily.sort_index()
        td_idx = ds.index
        gap_s        = (ds["Open"] / ds["Close"].shift(1) - 1) * 100
        prev_close_s = ds["Close"].shift(1)
        eod_chg_s    = (ds["Close"] / prev_close_s - 1) * 100
        for ts, gv in gap_s.items():
            if pd.notna(gv):
                gap_map[ts.date()] = float(gv)

    _ref = datetime.date(2000, 1, 3)
    all_entries = []
    legend_items = []
    n_badge_html = ""
    for i, entry in enumerate(store.get("entries", [])):
        if not entry.get("enabled", True):
            continue
        entry_with_filters = {**entry, "range": store.get("range", "All"), "gap": store.get("gap", "All")}
        dates, lbl = _cmp_resolve_entry(entry_with_filters, cmp_daily, td_idx, gap_map)
        if not dates:
            continue
        color = _CMP_COLORS[i % len(_CMP_COLORS)]
        all_entries.append((color, lbl, dates))
        legend_items.append((color, lbl, len(dates)))

    # Histogram
    hist_html = ""
    if all_entries and not cmp_daily.empty:
        eod_all = []
        for _, _, entry_dates in all_entries:
            for ed in entry_dates:
                ed_ts = pd.Timestamp(ed)
                if ed_ts in eod_chg_s.index and not pd.isna(eod_chg_s.loc[ed_ts]):
                    eod_all.append(float(eod_chg_s.loc[ed_ts]))
        if eod_all:
            eod_s  = pd.Series(eod_all)
            h_mean = eod_s.mean()
            h_med  = eod_s.median()
            h_ppos = (eod_s >= 0).mean() * 100
            h_std  = eod_s.std()
            h_n    = len(eod_s)
            if h_n < 30:
                hbg = "#FFA743"
            elif h_n < 75:
                hbg = "#ecff8b"
            else:
                hbg = "#13ff98"
            hist_fig = _build_histogram_figure(eod_s, x_label="EOD % from prior close")
            n_badge_html = (
                f'<div style="display:inline-block;padding:4px 12px;border-radius:7px;'
                f'background:{hbg};font-size:12px;font-weight:400;'
                f'color:#1A1A1A;">N = {h_n}</div>'
            )
            hist_html = (
                _stat_pills_html(h_mean, h_med, h_ppos, h_std)
                + _chart_html('cmp-histogram-chart', hist_fig)
            )

    # Legend
    leg_html = ""
    if legend_items:
        items = "".join(
            f'<span style="display:flex;align-items:center;gap:5px;">'
            f'<span style="width:12px;height:3px;background:{lc};border-radius:2px;display:inline-block;"></span>'
            f'<span style="font-size:12px;color:#444;">{ll}</span>'
            f'<span style="font-size:11px;color:#aaa;">({ln})</span>'
            f'</span>'
            for lc, ll, ln in legend_items
        )
        leg_html = f'<div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:24px;margin-bottom:8px;">{items}</div>'

    # Overlay chart
    single_entry = len(all_entries) == 1
    if single_entry:
        leg_html = ""  # hide legend when using EOD color coding
    cmp_fig = go.Figure()
    for cmp_color, cmp_lbl, entry_dates in all_entries:
        for cd in entry_dates:
            ots = pd.Timestamp(cd)
            if not cmp_frd5.empty:
                ote = ots + pd.Timedelta(hours=23, minutes=59)
                cdf = cmp_frd5.loc[ots:ote]
            else:
                cdf = get_spx_5min_for_date(cd)
            if cdf.empty:
                continue
            times = [datetime.datetime.combine(_ref, ts.time()) for ts in cdf.index]
            pc = float(prev_close_s.loc[ots]) if ots in prev_close_s.index and not pd.isna(prev_close_s.loc[ots]) else float(cdf["Open"].iloc[0])
            pct = ((cdf["Close"] / pc - 1) * 100).round(2)
            if single_entry:
                eod_val = float(eod_chg_s.loc[ots]) if ots in eod_chg_s.index and not pd.isna(eod_chg_s.loc[ots]) else 0
                line_color = GREEN_400 if eod_val >= 0 else "#FF3D54"
                trace_hoverlabel = dict(font=dict(color="#1E1E1E" if eod_val >= 0 else "white"))
            else:
                line_color = cmp_color
                trace_hoverlabel = {}
            cmp_fig.add_trace(go.Scatter(
                x=times, y=pct, mode="lines",
                legendgroup=cmp_lbl, showlegend=False,
                line=dict(color=line_color, width=0.9), opacity=0.5,
                hoverlabel=trace_hoverlabel if trace_hoverlabel else None,
                hovertemplate=f'{_fmt_date(cd)}: %{{y:+.2f}}%<extra></extra>',
            ))
    cmp_fig.add_hline(y=0, line_dash="dot", line_color="#B2B2B2", line_width=1)
    cmp_fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        dragmode="zoom", height=560,
        margin=dict(l=48, r=0, t=10, b=30),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="closest", hoverdistance=-1,
        hoverlabel=dict(bordercolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", size=12, color="#1E1E1E")),
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0", tickformat="%H:%M", hoverformat="%H:%M",
                   tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                   ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
                   type='date',
                   range=[datetime.datetime.combine(_ref, datetime.time(9, 30)),
                          datetime.datetime.combine(_ref, datetime.time(16, 0))],
                   rangeslider=dict(visible=False)),
        yaxis=dict(automargin=False, showgrid=True, gridcolor="#F0F0F0", side="left",
                   tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                   ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
                   title=dict(text="% from prior close", font=dict(family="Inter, sans-serif", size=10, color="#666")),
                   ticksuffix="%"),
    )
    return n_badge_html, hist_html + leg_html + f'<div style="margin-top:24px">' + _chart_html('cmp-overlay-chart', cmp_fig) + '</div>'


# ── Section 4: Conditional comparison ────────────────────────────────────────

@study_bp.route('/api/study/cc', methods=['POST'])
def api_cc_update():
    store = _get_cc_store()
    action = request.form.get('action')
    if action == 'add':
        cid = store["next_id"]
        store["next_id"] += 1
        store["ids"].append(cid)
        store["entries"].append({
            "id": cid, "type": _CC_COND_TYPES[0], "enabled": True,
            "time": "11:00", "pct_min": 0.2, "pct_max": 0.4,
            "event": "VIX Exp", "days_min": -3, "days_max": 3,
            "gap_min": -1.0, "gap_max": 1.0,
            "dow": list(range(5)), "months": list(range(1, 13)),
        })
    elif action == 'clear':
        store["ids"] = []
        store["entries"] = []
    elif action == 'delete':
        cid = int(request.form.get('id', -1))
        store["entries"] = [e for e in store["entries"] if e["id"] != cid]
        store["ids"]     = [i for i in store["ids"]     if i != cid]
    elif action == 'update':
        cid   = int(request.form.get('id', -1))
        field = request.form.get('field')
        value = request.form.get('value')
        for entry in store["entries"]:
            if entry["id"] == cid and field:
                if field in ("pct_min", "pct_max", "gap_min", "gap_max"):
                    try:
                        entry[field] = float(value)
                    except (ValueError, TypeError):
                        pass
                elif field in ("days_min", "days_max"):
                    try:
                        entry[field] = int(value)
                    except (ValueError, TypeError):
                        pass
                elif field == "enabled":
                    entry["enabled"] = value == "true"
                elif field == "dow_toggle":
                    idx = int(value)
                    dow = list(entry.get("dow", []))
                    if idx in dow:
                        dow.remove(idx)
                    else:
                        dow.append(idx)
                    entry["dow"] = sorted(dow)
                elif field == "month_toggle":
                    mo = int(value)
                    months = list(entry.get("months", []))
                    if mo in months:
                        months.remove(mo)
                    else:
                        months.append(mo)
                    entry["months"] = sorted(months)
                elif field in ("dow", "months"):
                    try:
                        entry[field] = json.loads(value)
                    except Exception:
                        pass
                else:
                    entry[field] = value
    elif action == 'filter':
        store["range"] = request.form.get('range', store.get("range", "All"))
        store["gap"]   = request.form.get('gap',   store.get("gap",   "All"))
        store["norm"]  = request.form.get('norm',  store.get("norm",  "% from prior close"))
    elif action == 'auto':
        store["auto_mode"] = True
    elif action == 'manual':
        store["auto_mode"] = False
    elif action == 'set_threshold':
        try:
            store["match_threshold"] = float(request.form.get("match_threshold", 0.5))
        except (ValueError, TypeError):
            pass
    elif action == 'set_tolerance':
        try:
            store["tolerance"] = max(0.05, round(float(request.form.get("value", 0.10)), 2))
        except (ValueError, TypeError):
            pass
    _save_cc_store(store)
    today_checkpoints = _compute_today_checkpoints() if store.get("auto_mode") else []
    n_badge_html, results_html = _build_cc_results_html(store)
    return render_template('partials/cc_section.html',
        store=store,
        cond_types=_CC_COND_TYPES,
        event_opts=_CC_EVENT_OPTS,
        time_opts=_CC_TIME_OPTS,
        dow_labels=_CC_DOW_LABELS,
        mon_labels=_CC_MON_LABELS,
        range_opts=_CMP_RANGE_OPTS,
        gap_opts=_CMP_GAP_OPTS,
        norm_opts=_CC_NORM_OPTS,
        n_badge_html=n_badge_html,
        results_html=results_html,
        today_checkpoints=today_checkpoints,
    )


@study_bp.route('/api/study/cc')
def api_cc_get():
    store = _get_cc_store()
    today_checkpoints = _compute_today_checkpoints() if store.get("auto_mode") else []
    n_badge_html, results_html = _build_cc_results_html(store)
    return render_template('partials/cc_section.html',
        store=store,
        cond_types=_CC_COND_TYPES,
        event_opts=_CC_EVENT_OPTS,
        time_opts=_CC_TIME_OPTS,
        dow_labels=_CC_DOW_LABELS,
        mon_labels=_CC_MON_LABELS,
        range_opts=_CMP_RANGE_OPTS,
        gap_opts=_CMP_GAP_OPTS,
        norm_opts=_CC_NORM_OPTS,
        n_badge_html=n_badge_html,
        results_html=results_html,
        today_checkpoints=today_checkpoints,
    )


def _compute_today_checkpoints() -> list:
    """Return completed half-hour checkpoints for today as [(time_str, pct_from_prior_close), ...]."""
    import pytz
    now_et = datetime.datetime.now(pytz.timezone("America/New_York"))
    df = _get_today_5min_live()
    if df.empty:
        return []
    # Look up today's prior close from the daily data (same approach as api_intraday).
    prev_close = None
    try:
        today_ts = pd.Timestamp(datetime.date.today())
        dl = get_spx_daily(1)
        if not dl.empty:
            prior_days = dl.index[dl.index < today_ts]
            if len(prior_days) > 0:
                prev_close = float(dl.loc[prior_days[-1], "Close"])
    except Exception:
        pass
    if not prev_close:
        return []
    result = []
    for h, m in _CC_SNAP_TIMES:
        if now_et.time() < datetime.time(h, m):
            break
        sub = df[df.index.time <= datetime.time(h, m)]
        if sub.empty:
            continue
        pct = round((float(sub["Close"].iloc[-1]) / prev_close - 1) * 100, 3)
        result.append((f"{h}:{m:02d}", pct))
    return result


def _score_days_against_checkpoints(snap: pd.DataFrame, checkpoints: list, tolerance: float) -> pd.Series:
    """For each day in snap, count how many checkpoints fall within ±tolerance of today's value."""
    if not checkpoints or snap.empty:
        return pd.Series(0, index=snap.index, dtype=int)
    scores = pd.Series(0, index=snap.index, dtype=int)
    for time_str, target_pct in checkpoints:
        _h, _m = time_str.split(":")
        col = f"pct_from_close_at_{int(_h):02d}{int(_m):02d}"
        if col not in snap.columns:
            continue
        within = snap[col].between(target_pct - tolerance, target_pct + tolerance).fillna(False)
        scores = scores + within.astype(int)
    return scores


def _build_cc_auto_results_html(store, snap) -> tuple:
    checkpoints   = _compute_today_checkpoints()
    tolerance     = float(store.get("tolerance", 0.5))
    threshold     = float(store.get("match_threshold", 0.5))

    if not checkpoints:
        return "", (
            '<p style="font-size:13px;color:#aaa;padding:8px 0">'
            'No checkpoints yet — check back once the market opens.</p>'
        )

    n_checkpoints = len(checkpoints)
    scores = _score_days_against_checkpoints(snap, checkpoints, tolerance)

    rng_days = _CMP_RANGE_DAYS.get(store.get("range", "All"))
    if rng_days:
        cutoff = datetime.date.today() - datetime.timedelta(days=rng_days)
        scores = scores[scores.index >= cutoff]

    gap_filter = store.get("gap", "All")
    if gap_filter != "All" and "gap_pct" in snap.columns:
        gap_s = snap["gap_pct"].reindex(scores.index)
        if gap_filter == "Gap up ↑":
            scores = scores[gap_s > 0]
        elif gap_filter == "Gap down ↓":
            scores = scores[gap_s < 0]

    # Exclude today itself from historical matches
    scores = scores[scores.index != datetime.date.today()]

    min_match     = max(1, int(np.ceil(threshold * n_checkpoints)))
    matched_scores = scores[scores >= min_match].sort_values(ascending=False)
    matched        = snap.loc[matched_scores.index]

    n = len(matched)
    bg = "#ff4646" if n == 0 else ("#FFA743" if n < 30 else ("#ecff8b" if n < 75 else "#13ff98"))
    n_badge_html = (
        f'<div style="display:inline-block;padding:4px 12px;border-radius:7px;'
        f'background:{bg};font-size:12px;font-weight:400;color:#1A1A1A;">N = {n}</div>'
    )
    if n == 0:
        return n_badge_html, ""

    _ref     = datetime.date(2000, 1, 3)
    today_df = _get_today_5min_live()

    # Look up today's prior close once — used for both charts and today's overlay line.
    today_prev_close = None
    try:
        today_ts = pd.Timestamp(datetime.date.today())
        dl = get_spx_daily(1)
        if not dl.empty:
            prior_days = dl.index[dl.index < today_ts]
            if len(prior_days) > 0:
                today_prev_close = float(dl.loc[prior_days[-1], "Close"])
    except Exception:
        pass
    # Fallback: derive from the snapshot's most recent completed day.
    if not today_prev_close:
        try:
            last = snap.iloc[-1]
            pc  = float(last["prev_close"])
            chg = float(last["eod_chg_pct"])
            if not (np.isnan(pc) or np.isnan(chg)):
                today_prev_close = round(pc * (1 + chg / 100), 4)
        except Exception:
            pass


    # ── Stat pills ────────────────────────────────────────────────────────
    eod = matched["eod_chg_pct"].dropna() if "eod_chg_pct" in matched.columns else pd.Series(dtype=float)
    pills_html = ""
    if not eod.empty:
        pills_html = _stat_pills_html(eod.mean(), eod.median(),
                                      (eod >= 0).mean() * 100, eod.std(), margin_top=24)

    # ── Megachart: overlay (col=1) + EOD histogram (col=2) ───────────────
    from plotly.subplots import make_subplots
    megachart_html = ""
    frd5 = _load_frd_5min()
    if not frd5.empty:
        megafig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.80, 0.20],
            shared_yaxes=True,
            horizontal_spacing=0.02,
        )
        all_oy: list[list[float]] = []

        # ── Left panel: historical overlay traces ─────────────────────────
        for od in matched_scores.index:
            score     = int(matched_scores.loc[od])
            match_pct = round(score / n_checkpoints * 100)
            opacity   = 0.12 + (score / n_checkpoints) * 0.55  # 0.12–0.67
            ots = pd.Timestamp(od)
            odf = frd5.loc[ots : ots + pd.Timedelta(hours=23, minutes=59)]
            if odf.empty:
                continue
            ox   = [datetime.datetime.combine(_ref, ts.time()) for ts in odf.index]
            base = float(matched.loc[od, "prev_close"]) if "prev_close" in matched.columns and not pd.isna(matched.loc[od, "prev_close"]) else float(odf["Open"].iloc[0])
            if base == 0:
                continue
            oy  = ((odf["Close"] / base - 1) * 100).round(2).tolist()
            all_oy.append(oy)
            ev  = float(matched.loc[od, "eod_chg_pct"]) if "eod_chg_pct" in matched.columns else 0
            today_equiv = (
                [round(today_prev_close * (1 + p / 100), 2) for p in oy]
                if today_prev_close else None
            )
            megafig.add_trace(go.Scatter(
                x=ox, y=oy, mode="lines",
                line=dict(color=GREEN_400 if ev >= 0 else "#FF3D54", width=0.9),
                opacity=opacity, showlegend=False,
                customdata=today_equiv,
                hoverlabel=dict(font=dict(color="#1E1E1E" if ev >= 0 else "white")),
                hovertemplate=(
                    f'{_fmt_date(od)} ({match_pct}% match):'
                    f' %{{y:+.2f}}%'
                    + (' · %{customdata:,.2f}' if today_prev_close else '')
                    + '<extra></extra>'
                ),
            ), row=1, col=1)

        # ── Today's path ──────────────────────────────────────────────────
        ty: list[float] = []
        if not today_df.empty and today_prev_close:
            if today_prev_close != 0:
                tx = [datetime.datetime.combine(_ref, ts.time()) for ts in today_df.index]
                ty = ((today_df["Close"] / today_prev_close - 1) * 100).round(2).tolist()
                today_prices = today_df["Close"].round(2).tolist()
                all_oy.append(ty)
                megafig.add_trace(go.Scatter(
                    x=tx, y=ty, mode="lines",
                    name="Today", showlegend=False,
                    line=dict(color="#4B7BFF", width=1.5),
                    opacity=1.0,
                    customdata=today_prices,
                    hovertemplate='Today: %{y:+.2f}% · %{customdata:,.2f}<extra></extra>',
                ), row=1, col=1)

        # ── Compute % range ───────────────────────────────────────────────
        flat_y = [v for series in all_oy for v in series]
        if flat_y and today_prev_close:
            buf       = max(0.05, (max(flat_y) - min(flat_y)) * 0.04)
            pct_min   = min(flat_y) - buf
            pct_max   = max(flat_y) + buf
            price_min = today_prev_close * (1 + pct_min / 100)
            price_max = today_prev_close * (1 + pct_max / 100)
            # Dummy trace to force price axis (yaxis2) to render.
            megafig.add_trace(go.Scatter(
                x=[datetime.datetime.combine(_ref, datetime.time(9, 30))],
                y=[price_min],
                yaxis="y2", mode="markers",
                marker=dict(opacity=0, size=1),
                showlegend=False, hoverinfo="skip",
            ))
            shared_y_kw = dict(
                side="right", range=[pct_min, pct_max],
                showgrid=True, gridcolor="#F0F0F0", ticksuffix="%",
                tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
            )
            price_axis_kw = dict(
                side="left", overlaying="y",
                range=[price_min, price_max],
                showgrid=False, tickformat=",.0f",
                tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
            )
            margin_kw = dict(l=0, r=36, t=16, b=30)
        else:
            pct_min = pct_max = None
            shared_y_kw  = dict(
                showgrid=True, gridcolor="#F0F0F0", ticksuffix="%",
                tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
            )
            price_axis_kw = {}
            margin_kw = dict(l=0, r=36, t=16, b=30)

        # ── Right panel: horizontal EOD histogram ─────────────────────────
        if not eod.empty and pct_min is not None:
            bsz      = max(0.1, round((pct_max - pct_min) / 20, 2))
            bins_arr = np.arange(pct_min - bsz, pct_max + bsz * 2, bsz)
            counts, edges = np.histogram(eod.values, bins=bins_arr)
            bin_centers   = (edges[:-1] + edges[1:]) / 2
            bar_colors    = [GREEN_400 if c >= 0 else "#FF3D54" for c in bin_centers]
            megafig.add_trace(go.Bar(
                x=counts, y=bin_centers,
                orientation="h",
                marker_color=bar_colors,
                marker_line_width=0,
                width=bsz * 0.85,
                showlegend=False,
                hovertemplate="%{y:+.2f}%: %{x} days<extra></extra>",
            ), row=1, col=2)

        megafig.add_hline(y=0, line_dash="dot", line_color="#C8C8C8", line_width=1)

        layout_kw: dict = dict(
            font=dict(family="Inter, sans-serif"),
            height=400, margin=margin_kw,
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="closest", hoverdistance=-1,
            showlegend=False, bargap=0.06,
            hoverlabel=dict(bordercolor="rgba(0,0,0,0)",
                            font=dict(family="Inter, sans-serif", size=12, color="#1E1E1E")),
            xaxis=dict(
                showgrid=True, gridcolor="#F0F0F0", tickformat="%H:%M",
                tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
                type='date',
                range=[datetime.datetime.combine(_ref, datetime.time(9, 30)),
                       datetime.datetime.combine(_ref, datetime.time(16, 0))],
                rangeslider=dict(visible=False),
            ),
            xaxis2=dict(
                showgrid=False,
                tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                ticks="outside", ticklen=4, tickcolor="rgba(0,0,0,0)",
            ),
            yaxis=shared_y_kw,
        )
        if price_axis_kw:
            layout_kw["yaxis2"] = price_axis_kw
        megafig.update_layout(**layout_kw)

        today_legend = (
            '<div style="display:flex;align-items:center;gap:6px;margin-top:6px;margin-bottom:8px">'
            '<span style="display:inline-block;width:20px;height:1.5px;background:#4B7BFF;border-radius:1px;"></span>'
            '<span style="font-size:11px;color:#555;">Today</span>'
            '</div>'
        ) if (not today_df.empty and today_prev_close) else ''
        megachart_html = (
            f'<p style="font-size:11px;color:#888;margin-top:24px;margin-bottom:0">'
            f'{n} days matching ≥{int(threshold * 100)}% of {n_checkpoints} checkpoints</p>'
            + today_legend
            + _chart_html('cc-megachart', megafig)
        )

    return n_badge_html, pills_html + megachart_html


def _apply_cc_conditions(snap, store):
    mask = pd.Series(True, index=snap.index)
    ev_map = {"OPEX": "days_from_opex", "VIX Exp": "days_from_vix_exp", "FOMC": "days_from_fomc"}
    for entry in store.get("entries", []):
        if not entry.get("enabled", True):
            continue
        ct = entry.get("type", _CC_COND_TYPES[0])
        if ct == "% change at time":
            _h, _m = entry.get("time", "11:00").split(":")
            col = f"pct_at_{int(_h):02d}{int(_m):02d}"
            lo, hi = float(entry.get("pct_min", -1.0)), float(entry.get("pct_max", -0.1))
            if col in snap.columns:
                mask &= snap[col].between(lo, hi)
        elif ct == "Days from event":
            col = ev_map.get(entry.get("event", "VIX Exp"), "")
            lo, hi = int(entry.get("days_min", -3)), int(entry.get("days_max", 3))
            if col and col in snap.columns:
                mask &= snap[col].between(lo, hi)
        elif ct == "Day of week":
            dows = entry.get("dow", list(range(5)))
            if dows:
                mask &= snap["day_of_week"].isin(dows)
        elif ct == "Month":
            mos = entry.get("months", list(range(1, 13)))
            if mos:
                mask &= snap["month"].isin(mos)
        elif ct == "Overnight gap":
            lo, hi = float(entry.get("gap_min", -1.0)), float(entry.get("gap_max", 1.0))
            if "gap_pct" in snap.columns:
                mask &= snap["gap_pct"].between(lo, hi)
    return snap[mask]


def _build_cc_results_html(store) -> tuple:
    snap = _build_daily_snapshots()
    if snap.empty:
        return "", '<p style="font-size:12px;color:#888">No 5-min historical data available.</p>'

    if store.get("auto_mode"):
        return _build_cc_auto_results_html(store, snap)

    if not store.get("ids"):
        return "", '<p style="font-size:13px;color:#aaa;padding:2px 0 8px">Add a condition above to filter historical days.</p>'

    matched = _apply_cc_conditions(snap, store)
    # Apply time range filter
    rng_days = _CMP_RANGE_DAYS.get(store.get("range", "All"))
    if rng_days:
        cutoff = datetime.date.today() - datetime.timedelta(days=rng_days)
        matched = matched[matched.index >= cutoff]
    # Apply overnight gap filter
    gap_filter = store.get("gap", "All")
    if gap_filter == "Gap up ↑" and "gap_pct" in matched.columns:
        matched = matched[matched["gap_pct"] > 0]
    elif gap_filter == "Gap down ↓" and "gap_pct" in matched.columns:
        matched = matched[matched["gap_pct"] < 0]
    n = len(matched)
    if n == 0:
        bg = "#ff4646"
    elif n < 30:
        bg = "#FFA743"
    elif n < 75:
        bg = "#ecff8b"
    else:
        bg = "#13ff98"

    n_badge_html = (f'<div style="display:inline-block;padding:4px 12px;border-radius:7px;'
                    f'background:{bg};font-size:12px;font-weight:400;'
                    f'color:#1A1A1A;">N = {n}</div>')
    if n == 0:
        return n_badge_html, ""

    norm = "% from prior close"
    eod_col = "eod_chg_pct" if norm == "% from prior close" else "eod_pct"
    eod = matched[eod_col].dropna() if eod_col in matched.columns else matched["eod_pct"].dropna()
    if eod.empty:
        return n_badge_html, ""

    mean = eod.mean()
    med  = eod.median()
    ppos = (eod >= 0).mean() * 100
    std  = eod.std()
    pills = _stat_pills_html(mean, med, ppos, std, margin_top=24)

    hist_x_label = "EOD % from prior close" if norm == "% from prior close" else "EOD % from open"
    hist_fig = _build_histogram_figure(eod, x_label=hist_x_label)
    hist_html = _chart_html('cc-hist-chart', hist_fig)

    # Intraday overlay — all matching days
    overlay_html = ""
    frd5 = _load_frd_5min()
    ov_dates = sorted(matched.index.tolist(), reverse=True)
    _ref = datetime.date(2000, 1, 3)
    use_prev_close = (norm == "% from prior close")
    if ov_dates and not frd5.empty:
        ofig = go.Figure()
        for od in ov_dates:
            ots = pd.Timestamp(od)
            ote = ots + pd.Timedelta(hours=23, minutes=59)
            odf = frd5.loc[ots:ote]
            if odf.empty:
                continue
            ox = [datetime.datetime.combine(_ref, ts.time()) for ts in odf.index]
            if use_prev_close:
                pc = matched.loc[od, "prev_close"] if "prev_close" in matched.columns else None
                base = float(pc) if pc and not pd.isna(pc) else float(odf["Open"].iloc[0])
            else:
                base = float(odf["Open"].iloc[0])
            oy = ((odf["Close"] / base - 1) * 100).round(2).tolist()
            ev = float(matched.loc[od, "eod_pct"])
            ofig.add_trace(go.Scatter(
                x=ox, y=oy, mode="lines",
                line=dict(color=GREEN_400 if ev >= 0 else "#FF3D54", width=0.8),
                opacity=0.4, showlegend=False,
                hoverlabel=dict(font=dict(color="#1E1E1E" if ev >= 0 else "white")),
                hovertemplate=f'{_fmt_date(od)}: %{{y:+.2f}}%<extra></extra>',
            ))
        ofig.add_hline(y=0, line_dash="dot", line_color="#C8C8C8", line_width=1)
        y_title = "% from prior close" if use_prev_close else "% from open"
        ofig.update_layout(
            font=dict(family="Inter, sans-serif"),
            height=400, margin=dict(l=48, r=0, t=16, b=30),
            plot_bgcolor="white", paper_bgcolor="white", hovermode="closest", hoverdistance=-1,
            hoverlabel=dict(bordercolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", size=12, color="#1E1E1E")),
            xaxis=dict(showgrid=True, gridcolor="#F0F0F0", tickformat="%H:%M",
                       tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                       ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
                       type='date',
                       range=[datetime.datetime.combine(_ref, datetime.time(9, 30)),
                              datetime.datetime.combine(_ref, datetime.time(16, 0))],
                       rangeslider=dict(visible=False)),
            yaxis=dict(automargin=False, showgrid=True, gridcolor="#F0F0F0", ticksuffix="%",
                       tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                       ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
                       title=dict(text=y_title, font=dict(family="Inter, sans-serif", size=10, color="#888"))),
        )
        overlay_html = (
            f'<p style="font-size:11px;color:#888;margin-top:24px">'
            f'{len(ov_dates)} matching days</p>'
            + _chart_html('cc-overlay-chart', ofig)
        )
    return n_badge_html, pills + hist_html + overlay_html


# ── Sections 5–8: Static sections ────────────────────────────────────────────

@study_bp.route('/api/study/event-impact')
def api_event_impact():
    event_daily = _get_event_daily_df()
    if event_daily.empty:
        return '<p style="font-size:12px;color:#888">Daily data unavailable.</p>'
    today_ = datetime.date.today()
    impact_start = today_.replace(year=today_.year - _EVENT_IMPACT_YEARS)
    all_events = get_financial_events(impact_start, today_)
    impact_df = _compute_event_impact(event_daily, all_events)
    if impact_df.empty:
        return '<p style="font-size:12px;color:#888">No events found in range.</p>'
    return render_template('partials/event_impact.html',
        impact_df=impact_df,
        n_days=len(event_daily),
        n_events=len(all_events),
        years=_EVENT_IMPACT_YEARS,
    )


@study_bp.route('/api/study/key-dates')
def api_key_dates():
    # Extend through the end of the current year so future-but-scheduled
    # events (e.g. 2026 Thanksgiving/Xmas/NYE, remaining OPEX/VIX/FOMC days)
    # show up in their year column and the section's columns stay aligned.
    today_ = datetime.date.today()
    end_of_year = datetime.date(today_.year, 12, 31)
    kd_events = get_financial_events(datetime.date(2019, 1, 1), end_of_year)
    kd_grouped: dict[str, list[datetime.date]] = {}
    for d, lbl in kd_events:
        key = "OPEX" if "OPEX" in lbl else lbl
        kd_grouped.setdefault(key, []).append(d)
    # Group each event type by year, sorted descending
    kd_by_year: dict[str, dict[int, list[datetime.date]]] = {}
    for key, dates in kd_grouped.items():
        by_year: dict[int, list[datetime.date]] = {}
        for d in sorted(dates):
            by_year.setdefault(d.year, []).append(d)
        kd_by_year[key] = by_year
    return render_template('partials/key_dates.html', grouped=kd_grouped, by_year=kd_by_year)


@study_bp.route('/api/study/notable-events')
def api_notable_events():
    ne_daily = _get_event_daily_df()
    oc_pct: dict[datetime.date, float] = {}
    ol_pct: dict[datetime.date, float] = {}
    if not ne_daily.empty and {"Open", "Close", "Low"}.issubset(ne_daily.columns):
        ne_sorted    = ne_daily.sort_index()
        ne_prev_cls  = ne_sorted["Close"].shift(1)
        for ts, row in ne_sorted.iterrows():
            d = ts.date()
            pc = ne_prev_cls.loc[ts] if not pd.isna(ne_prev_cls.loc[ts]) else None
            ref = pc if pc is not None else (row["Open"] if row["Open"] != 0 else None)
            if ref:
                oc_pct[d] = (row["Close"] - ref) / ref * 100
            if row["Open"] and row["Open"] != 0:
                ol_pct[d] = (row["Low"] - row["Open"]) / row["Open"] * 100
    all_years = list(range(datetime.date.today().year, 2018, -1))
    return render_template('partials/notable_events.html',
        notable_events=_NOTABLE_EVENTS, oc_pct=oc_pct, ol_pct=ol_pct,
        all_years=all_years)


@study_bp.route('/api/study/big-moves')
def api_big_moves():
    bm_df = _get_event_daily_df()
    if bm_df.empty:
        return '<p style="font-size:11px;color:#888">No daily data available.</p>'
    bm = bm_df[["Open", "High", "Low", "Close"]].copy().sort_index()
    bm["chg"]     = (bm["Close"] / bm["Close"].shift(1) - 1) * 100
    bm["chg_low"] = (bm["Low"]   - bm["Open"]) / bm["Open"] * 100
    bm["date"]    = bm.index.date
    bm["year"]    = bm.index.year
    bm_filtered   = bm[(bm["year"] >= 2019) & (bm["chg"].abs() >= 1.5)].copy()
    return render_template('partials/big_moves.html',
        bm_by_year=_group_by_year(bm_filtered),
        intraday_notes=_INTRADAY_NOTES,
    )


# ── Pure-computation helpers ─────────────────────────────────────────────────

def _group_by_year(bm_filtered):
    by_year = {}
    for _, row in bm_filtered.iterrows():
        yr = int(row["year"])
        by_year.setdefault(yr, []).append((row["date"], row["chg"], row["chg_low"]))
    return {yr: sorted(items) for yr, items in sorted(by_year.items(), reverse=True)}


def _build_histogram_figure(eod_series: pd.Series, x_label: str = "EOD % from open") -> go.Figure:
    mean = eod_series.mean()
    med  = eod_series.median()
    rng  = float(eod_series.max() - eod_series.min())
    bsz  = 0.1 if rng < 1.5 else (0.25 if rng < 5.0 else 0.5)
    blo  = np.floor(eod_series.min() / bsz) * bsz - bsz
    bhi  = np.ceil( eod_series.max() / bsz) * bsz + bsz
    bins = np.arange(blo, bhi + bsz, bsz)
    cnts, edges = np.histogram(eod_series.values, bins=bins)
    ctrs  = (edges[:-1] + edges[1:]) / 2
    bclrs = [GREEN_400 if c >= 0 else "#FF3D54" for c in ctrs]
    bpcts = cnts / cnts.sum() * 100 if cnts.sum() > 0 else cnts * 0.0
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ctrs, y=cnts, marker_color=bclrs, marker_line_width=0,
        width=bsz * 0.88, customdata=bpcts,
        hovertemplate="%{y} days (%{customdata:.1f}%)<extra></extra>",
    ))
    fig.add_vline(x=0,    line_color="#C8C8C8", line_width=1, line_dash="dot")
    fig.add_vline(x=mean, line_color="#1A1A1A", line_width=1.5)
    fig.add_vline(x=med,  line_color="#888888", line_width=1, line_dash="dot")
    fig.add_annotation(x=mean, xref="x", y=1.08, yref="paper",
                        text=f"mean {mean:+.2f}%", showarrow=False,
                        xanchor="right", yanchor="bottom", font=dict(size=10, color="#1A1A1A"))
    fig.add_annotation(x=med, xref="x", y=1.08, yref="paper",
                        text=f"median {med:+.2f}%", showarrow=False,
                        xanchor="left", yanchor="bottom", font=dict(size=10, color="#888888"))
    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        height=260, margin=dict(l=0, r=0, t=46, b=40),
        plot_bgcolor="white", paper_bgcolor="white", bargap=0.06,
        hovermode="x", hoverdistance=50,
        hoverlabel=_HOVERLABEL,
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0", ticksuffix="%",
                   tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                   ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
                   title=dict(text=x_label, font=dict(family="Inter, sans-serif", size=11, color="#888"))),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0",
                   tickfont=dict(family="Inter, sans-serif", color="#808080", size=8),
                   ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
                   ),
        showlegend=False,
    )
    return fig


def _stat_pills_html(mean, med, ppos, std, margin_top=24) -> str:
    def _pill(label, val, color="#444"):
        return (f'<span style="font-size:12px;color:#555;margin-right:16px;">'
                f'{label}: <b style="color:{color};">{val}</b></span>')
    mc = GREEN_600 if mean >= 0 else "#FF3D54"
    dc = GREEN_600 if med  >= 0 else "#FF3D54"
    pc = GREEN_600 if ppos >= 50 else "#FF3D54"
    return (
        '<div style="margin-top:{margin_top}px;margin-bottom:12px;">'.format(margin_top=margin_top)
        + _pill("Mean EOD",   f'{"+" if mean >= 0 else ""}{mean:.2f}%', mc)
        + _pill("Median EOD", f'{"+" if med  >= 0 else ""}{med:.2f}%',  dc)
        + _pill("% Positive", f'{ppos:.0f}%',                           pc)
        + _pill("Std Dev",    f'{std:.2f}%')
        + '</div>'
    )


def _compute_event_impact(daily_df, events):
    if daily_df.empty or not events:
        return pd.DataFrame()
    daily_sorted = daily_df.sort_index()
    oc = (daily_sorted['Close'] / daily_sorted['Close'].shift(1) - 1) * 100
    ol = (daily_sorted['Low']   - daily_sorted['Open']) / daily_sorted['Open'] * 100
    daily_df = daily_sorted
    trading_days = daily_df.index

    def _norm(label):
        return "OPEX" if "OPEX" in label else label

    rows = []
    for evt_date, label in events:
        evt_ts = pd.Timestamp(evt_date)
        future = trading_days[trading_days >= evt_ts]
        if len(future) == 0:
            continue
        evt_day = future[0]
        evt_loc = trading_days.get_loc(evt_day)
        rows.append({
            "type":     _norm(label),
            "prior_oc": oc.iloc[evt_loc - 1] if evt_loc - 1 >= 0      else np.nan,
            "evt_oc":   oc.iloc[evt_loc]     if evt_loc < len(oc)      else np.nan,
            "evt_ol":   ol.iloc[evt_loc]     if evt_loc < len(ol)      else np.nan,
            "next_oc":  oc.iloc[evt_loc + 1] if evt_loc + 1 < len(oc) else np.nan,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (
        df.groupby("type").agg(
            Count=("evt_oc", "count"),
            Prior_OC=("prior_oc", "mean"),
            Evt_OC=("evt_oc", "mean"),
            Evt_OL=("evt_ol", "mean"),
            Next_OC=("next_oc", "mean"),
        ).reset_index().rename(columns={
            "type": "Event",
            "Prior_OC": "Prior day C→C",
            "Evt_OC":   "Day of C→C",
            "Evt_OL":   "Day of O→L",
            "Next_OC":  "Next day C→C",
        }).sort_values("Count", ascending=False).reset_index(drop=True)
    )
