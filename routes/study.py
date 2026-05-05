"""Study page — Flask Blueprint."""
from __future__ import annotations

import datetime


def _most_recent_trading_day() -> datetime.date:
    """Return the most recent Mon–Fri (walks back through weekends)."""
    d = datetime.date.today()
    while d.weekday() >= 5:  # 5 = Sat, 6 = Sun
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
from shared.chart import create_spx_chart, create_long_chart
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

_CC_SNAP_TIMES = [(10, 0), (10, 30), (11, 0), (11, 30), (12, 0), (13, 0), (14, 0), (15, 0)]
_CC_TIME_OPTS  = [f"{h}:{m:02d}" for h, m in _CC_SNAP_TIMES]
_CC_COND_TYPES = ["% from open at time", "Days from event", "Day of week", "Month", "Overnight gap"]
_CC_EVENT_OPTS = ["OPEX", "VIX Exp", "FOMC"]
_CC_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
_CC_MON_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_CMP_COLORS      = ["#B71AFF", "#4B7BFF", "#FF6B35", "#11B8A0",
                    "#FF3D54", "#F5A623", "#4CAF50", "#888888"]
_CMP_ENTRY_TYPES = ["FOMC", "OPEX", "VIX Exp", "Specific date"]
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
_DEFAULT_CC_STORE = {"ids": [], "next_id": 0, "entries": []}


def _get_cmp_store() -> dict:
    if 'cmp' not in session:
        session['cmp'] = dict(_DEFAULT_CMP_STORE)
    return session['cmp']


def _get_cc_store() -> dict:
    if 'cc' not in session:
        session['cc'] = dict(_DEFAULT_CC_STORE)
    return session['cc']


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
                _append_to_archive(df)
                return df
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
    snap["eod_low_pct"] = (grp_low   / grp_open - 1) * 100
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
        snap[f"pct_at_{k}"]   = ((sc / grp_open - 1) * 100).reindex(snap.index)
        snap[f"range_at_{k}"] = ((sh - sl) / grp_open * 100).reindex(snap.index)
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
    date_str = now.strftime("%A %B %-d, %Y")
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
    evt_attr = (
        f' data-events="{html.escape(json.dumps(evt_payload), quote=True)}"'
        if evt_payload else ''
    )
    return (
        f'<div class="chart-pill-wrap" style="position:relative;width:100%;height:100%">'
        f'<div id="{div_id}" data-plotly="{fig_json}"{evt_attr} '
        f'style="width:100%;height:100%"></div>'
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

@study_bp.route('/api/study/long-chart')
def api_long_chart():
    selected_range = request.args.get('range', '1Y')
    show_events    = bool(request.args.get('show_events'))
    show_line      = bool(request.args.get('show_line'))
    range_params   = {"1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10, "Max": None}
    years = range_params.get(selected_range, 1)
    df_long = get_spx_daily(years)
    if df_long.empty:
        return _chart_html('study-long-chart', go.Figure(), evt_payload=None)
    last_close = float(df_long['Close'].iloc[-1])
    first_open = float(df_long['Open'].iloc[0])
    is_down    = (last_close - first_open) < 0
    line_color = "#FF3D54" if is_down else "#11F185"
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
            error=f"No data for {selected_date.strftime('%b %-d, %Y')}. "
                  "Try a US trading day within the last ~9 months.",
            fig_html='', stats=None,
            date_str=date_str, today=today_iso, min_date=min_date_iso)

    day_open  = float(df_day['Open'].iloc[0])
    day_high  = float(df_day['High'].max())
    day_low   = float(df_day['Low'].min())
    day_close = float(df_day['Close'].iloc[-1])

    gap_pts = gap_pct = None
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

    day_ch_pts = day_close - day_open
    day_ch_pct = (day_ch_pts / day_open) * 100

    is_down    = day_ch_pts < 0
    line_color = "#FF3D54" if is_down else "#11F185"
    halo_color = 'rgba(255,61,84,0.3)' if is_down else 'rgba(17,241,133,0.3)'
    fig = create_spx_chart(
        selected_date.strftime("%b %-d, %Y"),
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
        charts_html=_build_cmp_charts_html(store),
    )


@study_bp.route('/api/study/cmp')
def api_cmp_get():
    store = _get_cmp_store()
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
        charts_html=_build_cmp_charts_html(store),
    )


def _cmp_resolve_entry(entry, cmp_daily, td_idx, gap_map):
    today_ = datetime.date.today()
    ctype  = entry.get("type", "FOMC")
    if ctype == "Specific date":
        d_str = entry.get("date")
        if d_str:
            d = datetime.date.fromisoformat(d_str)
            return [d], d.strftime("%b %-d, %Y")
        return [], "Specific date"
    rng      = entry.get("range", "All")
    rng_days = _CMP_RANGE_DAYS.get(rng)
    cutoff   = (today_ - datetime.timedelta(days=rng_days)) if rng_days else datetime.date(2000, 1, 1)
    all_ev   = get_financial_events(cutoff, today_)
    if ctype == "FOMC":
        raw = sorted(d for d in FOMC_DATES if cutoff <= d <= today_)
    elif ctype == "OPEX":
        raw = sorted(d for d, lbl in all_ev if "OPEX" in lbl and d <= today_)
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
    td_idx = pd.DatetimeIndex([])
    if not cmp_daily.empty:
        ds = cmp_daily.sort_index()
        td_idx = ds.index
        gap_s = (ds["Open"] / ds["Close"].shift(1) - 1) * 100
        for ts, gv in gap_s.items():
            if pd.notna(gv):
                gap_map[ts.date()] = float(gv)

    _ref = datetime.date(2000, 1, 3)
    all_entries = []
    legend_items = []
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
                if ed_ts in cmp_daily.index:
                    row = cmp_daily.loc[ed_ts]
                    if row["Open"] != 0:
                        eod_all.append(float((row["Close"] - row["Open"]) / row["Open"] * 100))
        if eod_all:
            eod_s  = pd.Series(eod_all)
            h_mean = eod_s.mean()
            h_med  = eod_s.median()
            h_ppos = (eod_s >= 0).mean() * 100
            h_std  = eod_s.std()
            h_n    = len(eod_s)
            if h_n < 30:
                hbg, hfg = "#FF8C0020", "#CC7000"
            elif h_n < 75:
                hbg, hfg = "#F5C51820", "#A08500"
            else:
                hbg, hfg = "#11F18520", "#0AA855"
            hist_fig = _build_histogram_figure(eod_s)
            hist_html = (
                f'<div style="display:inline-block;padding:4px 12px;border-radius:7px;'
                f'background:{hbg};border:1px solid {hfg}44;font-size:12px;font-weight:600;'
                f'color:{hfg};margin-bottom:10px;">N = {h_n}</div>'
                + _stat_pills_html(h_mean, h_med, h_ppos, h_std)
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
        leg_html = f'<div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:8px;">{items}</div>'

    # Overlay chart
    cmp_fig = go.Figure()
    for cmp_color, cmp_lbl, entry_dates in all_entries:
        for cd in entry_dates:
            if not cmp_frd5.empty:
                ots = pd.Timestamp(cd)
                ote = ots + pd.Timedelta(hours=23, minutes=59)
                cdf = cmp_frd5.loc[ots:ote]
            else:
                cdf = get_spx_5min_for_date(cd)
            if cdf.empty:
                continue
            times = [datetime.datetime.combine(_ref, ts.time()) for ts in cdf.index]
            op = float(cdf["Open"].iloc[0])
            pct = ((cdf["Close"] / op - 1) * 100).round(2)
            cmp_fig.add_trace(go.Scatter(
                x=times, y=pct, mode="lines",
                legendgroup=cmp_lbl, showlegend=False,
                line=dict(color=cmp_color, width=0.9), opacity=0.5,
                hovertemplate=f'{cd.strftime("%b %-d, %Y")}: %{{y:+.2f}}%<extra></extra>',
            ))
    cmp_fig.add_hline(y=0, line_dash="dot", line_color="#B2B2B2", line_width=1)
    cmp_fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        dragmode="zoom", uirevision="constant", height=560,
        margin=dict(l=60, r=20, t=10, b=30),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0)", font=dict(color="#1E1E1E")),
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0", tickformat="%H:%M", hoverformat="%H:%M",
                   range=[datetime.datetime.combine(_ref, datetime.time(9, 30)),
                          datetime.datetime.combine(_ref, datetime.time(16, 0))],
                   showspikes=True, spikemode="across", spikesnap="cursor",
                   spikedash="1, 3", spikecolor="#B2B2B2", spikethickness=1,
                   rangeslider=dict(visible=False)),
        yaxis=dict(automargin=False, showgrid=True, gridcolor="#F0F0F0", side="left",
                   title=dict(text="% from open", font=dict(size=10, color="#666")),
                   ticksuffix="%", showspikes=True, spikemode="across", spikesnap="cursor",
                   spikedash="1, 3", spikecolor="#B2B2B2", spikethickness=1),
    )
    return hist_html + leg_html + _chart_html('cmp-overlay-chart', cmp_fig)


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
            "time": "11:00", "pct_min": -1.0, "pct_max": -0.1,
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
    _save_cc_store(store)
    return render_template('partials/cc_section.html',
        store=store,
        cond_types=_CC_COND_TYPES,
        event_opts=_CC_EVENT_OPTS,
        time_opts=_CC_TIME_OPTS,
        dow_labels=_CC_DOW_LABELS,
        mon_labels=_CC_MON_LABELS,
        results_html=_build_cc_results_html(store),
    )


@study_bp.route('/api/study/cc')
def api_cc_get():
    store = _get_cc_store()
    return render_template('partials/cc_section.html',
        store=store,
        cond_types=_CC_COND_TYPES,
        event_opts=_CC_EVENT_OPTS,
        time_opts=_CC_TIME_OPTS,
        dow_labels=_CC_DOW_LABELS,
        mon_labels=_CC_MON_LABELS,
        results_html=_build_cc_results_html(store),
    )


def _apply_cc_conditions(snap, store):
    mask = pd.Series(True, index=snap.index)
    ev_map = {"OPEX": "days_from_opex", "VIX Exp": "days_from_vix_exp", "FOMC": "days_from_fomc"}
    for entry in store.get("entries", []):
        if not entry.get("enabled", True):
            continue
        ct = entry.get("type", _CC_COND_TYPES[0])
        if ct == "% from open at time":
            col = "pct_at_" + entry.get("time", "11:00").replace(":", "")
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


def _build_cc_results_html(store) -> str:
    snap = _build_daily_snapshots()
    if snap.empty:
        return '<p style="font-size:12px;color:#888">No 5-min historical data available.</p>'
    if not store.get("ids"):
        return '<p style="font-size:13px;color:#aaa;padding:2px 0 8px">Add a condition above to filter historical days.</p>'

    matched = _apply_cc_conditions(snap, store)
    n = len(matched)
    if n == 0:
        bg, fg, msg = "#FF3D5420", "#FF3D54", "No matching days"
    elif n < 30:
        bg, fg, msg = "#FF8C0020", "#CC7000", f"N = {n}  ·  thin sample — interpret carefully"
    elif n < 75:
        bg, fg, msg = "#F5C51820", "#A08500", f"N = {n}  ·  moderate sample"
    else:
        bg, fg, msg = "#11F18520", "#0AA855", f"N = {n}  ·  solid sample"

    badge = (f'<div style="display:inline-block;padding:5px 14px;border-radius:8px;'
             f'background:{bg};border:1px solid {fg}44;font-size:13px;font-weight:600;'
             f'color:{fg};margin-bottom:14px;">{msg}</div>')
    if n == 0:
        return badge

    eod = matched["eod_pct"].dropna()
    if eod.empty:
        return badge

    mean = eod.mean()
    med  = eod.median()
    ppos = (eod >= 0).mean() * 100
    std  = eod.std()
    pills = badge + _stat_pills_html(mean, med, ppos, std)

    hist_fig = _build_histogram_figure(eod)
    hist_html = _chart_html('cc-hist-chart', hist_fig)

    # Intraday overlay (up to 25 most recent)
    overlay_html = ""
    frd5 = _load_frd_5min()
    ov_dates = sorted(matched.index.tolist(), reverse=True)[:25]
    _ref = datetime.date(2000, 1, 3)
    if ov_dates and not frd5.empty:
        ofig = go.Figure()
        for od in ov_dates:
            ots = pd.Timestamp(od)
            ote = ots + pd.Timedelta(hours=23, minutes=59)
            odf = frd5.loc[ots:ote]
            if odf.empty:
                continue
            ox = [datetime.datetime.combine(_ref, ts.time()) for ts in odf.index]
            oo = float(odf["Open"].iloc[0])
            oy = ((odf["Close"] / oo - 1) * 100).round(2).tolist()
            ev = float(matched.loc[od, "eod_pct"])
            ofig.add_trace(go.Scatter(
                x=ox, y=oy, mode="lines",
                line=dict(color="#11F185" if ev >= 0 else "#FF3D54", width=0.8),
                opacity=0.4, showlegend=False,
                hovertemplate=f'{od.strftime("%b %-d, %Y")}: %{{y:+.2f}}%<extra></extra>',
            ))
        ofig.add_hline(y=0, line_dash="dot", line_color="#C8C8C8", line_width=1)
        ofig.update_layout(
            font=dict(family="Inter, sans-serif"),
            height=400, margin=dict(l=60, r=20, t=16, b=30),
            plot_bgcolor="white", paper_bgcolor="white", hovermode="closest",
            xaxis=dict(showgrid=True, gridcolor="#F0F0F0", tickformat="%H:%M",
                       range=[datetime.datetime.combine(_ref, datetime.time(9, 30)),
                              datetime.datetime.combine(_ref, datetime.time(16, 0))],
                       rangeslider=dict(visible=False)),
            yaxis=dict(showgrid=True, gridcolor="#F0F0F0", ticksuffix="%",
                       title=dict(text="% from open", font=dict(size=10, color="#888"))),
        )
        overlay_html = (
            f'<p style="font-size:11px;color:#888;margin-top:8px">'
            f'Showing {len(ov_dates)} most recent matching days  ·  '
            f'green = positive EOD, red = negative EOD</p>'
            + _chart_html('cc-overlay-chart', ofig)
        )
    return pills + hist_html + overlay_html


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
    kd_events = get_financial_events(datetime.date(2019, 1, 1), datetime.date.today())
    kd_grouped: dict[str, list[datetime.date]] = {}
    for d, lbl in kd_events:
        key = "OPEX" if "OPEX" in lbl else lbl
        kd_grouped.setdefault(key, []).append(d)
    return render_template('partials/key_dates.html', grouped=kd_grouped)


@study_bp.route('/api/study/notable-events')
def api_notable_events():
    ne_daily = _get_event_daily_df()
    oc_pct: dict[datetime.date, float] = {}
    ol_pct: dict[datetime.date, float] = {}
    if not ne_daily.empty and {"Open", "Close", "Low"}.issubset(ne_daily.columns):
        for ts, row in ne_daily.iterrows():
            if row["Open"] and row["Open"] != 0:
                d = ts.date()
                oc_pct[d] = (row["Close"] - row["Open"]) / row["Open"] * 100
                ol_pct[d] = (row["Low"]   - row["Open"]) / row["Open"] * 100
    return render_template('partials/notable_events.html',
        notable_events=_NOTABLE_EVENTS, oc_pct=oc_pct, ol_pct=ol_pct)


@study_bp.route('/api/study/big-moves')
def api_big_moves():
    bm_df = _get_event_daily_df()
    if bm_df.empty:
        return '<p style="font-size:11px;color:#888">No daily data available.</p>'
    bm = bm_df[["Open", "High", "Low", "Close"]].copy()
    bm["chg"]     = (bm["Close"] - bm["Open"]) / bm["Open"] * 100
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


def _build_histogram_figure(eod_series: pd.Series) -> go.Figure:
    mean = eod_series.mean()
    med  = eod_series.median()
    rng  = float(eod_series.max() - eod_series.min())
    bsz  = 0.1 if rng < 1.5 else (0.25 if rng < 5.0 else 0.5)
    blo  = np.floor(eod_series.min() / bsz) * bsz - bsz
    bhi  = np.ceil( eod_series.max() / bsz) * bsz + bsz
    bins = np.arange(blo, bhi + bsz, bsz)
    cnts, edges = np.histogram(eod_series.values, bins=bins)
    ctrs  = (edges[:-1] + edges[1:]) / 2
    bclrs = ["#11F185" if c >= 0 else "#FF3D54" for c in ctrs]
    bpcts = cnts / cnts.sum() * 100 if cnts.sum() > 0 else cnts * 0.0
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ctrs, y=cnts, marker_color=bclrs, marker_line_width=0,
        width=bsz * 0.88, customdata=bpcts,
        hovertemplate="%{x:+.2f}%  →  %{y} days (%{customdata:.1f}%)<extra></extra>",
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
        height=260, margin=dict(l=50, r=20, t=46, b=40),
        plot_bgcolor="white", paper_bgcolor="white", bargap=0.06,
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0", ticksuffix="%",
                   title=dict(text="EOD % from open", font=dict(size=11, color="#888"))),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0",
                   title=dict(text="# of days", font=dict(size=11, color="#888"))),
        showlegend=False,
    )
    return fig


def _stat_pills_html(mean, med, ppos, std) -> str:
    def _pill(label, val, color="#444"):
        return (f'<span style="display:inline-block;padding:4px 12px;border-radius:6px;'
                f'background:#F1F2F6;font-size:12px;color:#555;margin:0 6px 6px 0;">'
                f'{label}: <b style="color:{color};">{val}</b></span>')
    mc = "#11F185" if mean >= 0 else "#FF3D54"
    dc = "#11F185" if med  >= 0 else "#FF3D54"
    pc = "#11F185" if ppos >= 50 else "#FF3D54"
    return (
        '<div style="margin-bottom:12px;">'
        + _pill("Mean EOD",   f'{"+" if mean >= 0 else ""}{mean:.2f}%', mc)
        + _pill("Median EOD", f'{"+" if med  >= 0 else ""}{med:.2f}%',  dc)
        + _pill("% Positive", f'{ppos:.0f}%',                           pc)
        + _pill("Std Dev",    f'{std:.2f}%')
        + '</div>'
    )


def _compute_event_impact(daily_df, events):
    if daily_df.empty or not events:
        return pd.DataFrame()
    oc = (daily_df['Close'] - daily_df['Open']) / daily_df['Open'] * 100
    ol = (daily_df['Low']   - daily_df['Open']) / daily_df['Open'] * 100
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
            "Prior_OC": "Prior day O→C",
            "Evt_OC":   "Event O→C",
            "Evt_OL":   "Event O→L",
            "Next_OC":  "Next day O→C",
        }).sort_values("Count", ascending=False).reset_index(drop=True)
    )
