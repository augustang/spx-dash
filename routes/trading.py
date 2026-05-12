"""Trading page — Flask Blueprint."""
from __future__ import annotations

import html
import math
import time
import datetime

import pandas as pd
import plotly.graph_objects as go
import pytz
from flask import Blueprint, render_template, request, session, jsonify, make_response

import schwab_client
from shared.cache import cache
from shared.chart import create_spx_chart, empty_figure, GREEN_400, GREEN_600
from shared.events import get_financial_events

_ET = pytz.timezone("America/New_York")

def _market_is_open() -> bool:
    """True only during regular NYSE session (Mon–Fri 9:30–16:00 ET)."""
    now = datetime.datetime.now(_ET)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return datetime.time(9, 30) <= t < datetime.time(16, 0)

trading_bp = Blueprint('trading', __name__)

# ── Default session store ────────────────────────────────────────────────────

_DEFAULT_STORE = {
    "selected_short": None, "selected_long": None, "selected_spread_px": 0.0,
    "saved_entry": 0.0, "saved_close": 0.05,
    "saved_bp": 150000, "saved_spread": 10, "saved_contracts": 150,
    "saved_target": 1500, "last_selected_short": None,
}


def _get_store() -> dict:
    if 'trade' not in session:
        session['trade'] = dict(_DEFAULT_STORE)
    return session['trade']


def _save_store(store: dict) -> None:
    session['trade'] = store
    session.modified = True


# ── Cached data fetchers ─────────────────────────────────────────────────────

@cache.memoize(timeout=60)
def get_spx_metrics():
    try:
        q = schwab_client.fetch_live_quote("$SPX")
        if q and q.get('lastPrice'):
            spx_prior = q['closePrice']  # always previous session close from Schwab

            if _market_is_open():
                # Live session: use real-time last and today's open
                spx_open = q['openPrice'] or spx_prior
                spx_last = q['lastPrice']
            else:
                # Outside regular hours: derive open/close from the regular session
                # intraday bars (already filtered to 09:30–16:00) so we never show
                # extended-hours prices on the metrics card.
                df_intra = get_spx_history_intraday("1d")
                if not df_intra.empty:
                    spx_open = float(df_intra['Open'].iloc[0])
                    spx_last = float(df_intra['Close'].iloc[-1])
                else:
                    spx_open = q['openPrice'] or spx_prior
                    spx_last = q['lastPrice']

            pts = spx_last - spx_prior
            pct = (pts / spx_prior * 100) if spx_prior else 0.0
            arrow = "↑" if pts >= 0 else "↓"
            return (
                spx_last, spx_open, spx_prior,
                f"{arrow} {abs(pts):.2f} pts ({abs(pct):.2f}%)"
            )
    except Exception:
        pass
    return 6850.00, 6860.00, 6800.00, "0 pts (0%)"


@cache.memoize(timeout=60)
def get_spx_history_intraday(period="1d"):
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (86400 * 1000 * 10)
    data = schwab_client.fetch_price_history(
        symbol="$SPX", period_type="day", freq_type="minute", freq=5,
        start_date=start_ms, end_date=now_ms,
    )
    if data and 'candles' in data:
        df = pd.DataFrame(data['candles'])
        if df.empty:
            return df
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
        unique_dates = sorted(df['datetime'].dt.date.unique())
        day_map = {"1d": -1, "3d": -3, "5d": -5}
        target_dates = unique_dates[day_map.get(period, -1):]
        df = df[df['datetime'].dt.date.isin(target_dates)]
        df.set_index('datetime', inplace=True)
        df = df.between_time('09:30', '16:00')
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        return df
    return pd.DataFrame()


@cache.memoize(timeout=3600)
def get_spx_history_historical():
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (86400 * 1000 * 365)
    data = schwab_client.fetch_price_history(
        symbol="$SPX", period_type="year", freq_type="daily", freq=1,
        start_date=start_ms, end_date=now_ms,
    )
    if data and 'candles' in data:
        df = pd.DataFrame(data['candles'])
        if df.empty:
            return df
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df['datetime'] = (
            df['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            .dt.tz_localize(None).dt.normalize()
        )
        df.set_index('datetime', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        return df
    return pd.DataFrame()


@cache.memoize(timeout=60)
def get_spx_puts():
    try:
        chain = schwab_client.fetch_options_chain("$SPX")
        if not chain or 'putExpDateMap' not in chain or not chain['putExpDateMap']:
            return pd.DataFrame()
        exp = sorted(chain['putExpDateMap'].keys())[0]
        puts = chain['putExpDateMap'][exp]
        rows = []
        for strike, data in puts.items():
            opt = data[0]
            rows.append({
                'strike': float(strike),
                'lastPrice': opt['last'] if opt['last'] > 0 else opt['mark'],
                'bid': opt['bid'], 'ask': opt['ask'],
                'delta': opt.get('delta', 0),
            })
        df = pd.DataFrame(rows)
        return df.sort_values('strike', ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ── Header helper ────────────────────────────────────────────────────────────

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


# ── Main page ────────────────────────────────────────────────────────────────

@trading_bp.route('/')
def trading():
    store = _get_store()
    return render_template(
        'trading.html',
        active_page='trading',
        header=_build_header_ctx(),
        store=store,
    )


# ── HTMX partials ────────────────────────────────────────────────────────────

@trading_bp.route('/api/trading/metrics')
def api_metrics():
    spx_last, spx_open, spx_prior, _ = get_spx_metrics()
    try:
        vix_q   = schwab_client.fetch_live_quote("$VIX")
        vix9d_q = schwab_client.fetch_live_quote("$VIX9D")
        vix_last   = vix_q['lastPrice']   if vix_q   else 0.0
        vix9d_last = vix9d_q['lastPrice'] if vix9d_q else 0.0
    except Exception:
        vix_last = vix9d_last = 0.0

    def _pill(label, pts, pct):
        bg   = "#13FF98" if pts >= 0 else "#FF4646"
        text = "#000" if pts >= 0 else "#FFF"
        arr  = "↑" if pts >= 0 else "↓"
        return f'''<div>
          <p style="font-size:11px;color:#888;margin:0 0 2px">{label}</p>
          <div style="background:{bg};color:{text};padding:4px 8px;border-radius:8px;
                      display:inline-block;font-size:12px;">
            {arr} {abs(pts):.2f} pts ({abs(pct):.2f}%)
          </div>
        </div>'''

    gap_pts = spx_open - spx_prior
    gap_pct = (gap_pts / spx_prior * 100) if spx_prior else 0
    pts_ch  = spx_last - spx_open
    pct_ch  = (pts_ch / spx_open * 100) if spx_open else 0
    pr_pts  = spx_last - spx_prior
    pr_pct  = (pr_pts / spx_prior * 100) if spx_prior else 0

    return render_template('partials/metrics.html',
        spx_prior=spx_prior, spx_open=spx_open, spx_last=spx_last,
        vix_last=vix_last, vix9d_last=vix9d_last,
        gap_pts=gap_pts, gap_pct=gap_pct,
        pts_ch=pts_ch, pct_ch=pct_ch,
        pr_pts=pr_pts, pr_pct=pr_pct,
    )


@trading_bp.route('/api/trading/inputs')
def api_inputs_get():
    """Initial render of the inputs panel from session state."""
    store = _get_store()
    return render_template('partials/inputs.html', **store)


@trading_bp.route('/api/trading/save-inputs', methods=['POST'])
def api_save_inputs():
    """Single source of truth for input changes. Persists to session,
    recalculates derived values, and triggers the spreads panel to refresh."""
    store = _get_store()
    bp        = _fnum(request.form.get('bp'),        store['saved_bp'])
    sw        = _fnum(request.form.get('sw'),        store['saved_spread'])
    contracts = _fnum(request.form.get('contracts'), store['saved_contracts'])
    target    = _fnum(request.form.get('target'),    store['saved_target'])
    entry     = _fnum(request.form.get('entry'),     store['saved_entry'])
    close     = _fnum(request.form.get('close'),     store['saved_close'])
    triggered = request.form.get('_triggered', '')

    if triggered in ('bp', 'sw') and bp and sw:
        contracts = int(bp / (sw * 100))
        target    = int(contracts * 0.10 * 100)

    store.update(
        saved_bp=bp, saved_spread=int(sw),
        saved_contracts=int(contracts), saved_target=int(target),
        saved_entry=entry, saved_close=close,
    )
    _save_store(store)

    resp = make_response(render_template('partials/inputs.html', **store))
    resp.headers['HX-Trigger'] = 'inputs-saved'
    return resp


@trading_bp.route('/api/trading/spreads')
def api_spreads():
    store = _get_store()
    bp        = store["saved_bp"]
    sw        = store["saved_spread"]
    contracts = store["saved_contracts"]
    target    = store["saved_target"]

    spx_last, spx_open, spx_prior, _ = get_spx_metrics()
    ref = spx_prior if spx_prior else spx_last
    live_puts_df = get_spx_puts()

    spreads_list, seen_strikes = [], set()
    if not live_puts_df.empty:
        for pct in [x / 10.0 for x in range(5, 201)]:
            target_price = spx_last * (1 - pct / 100)
            ci = (live_puts_df['strike'] - target_price).abs().idxmin()
            short = live_puts_df.loc[ci]
            ss = short['strike']
            if ss in seen_strikes:
                continue
            seen_strikes.add(ss)
            ls = ss - sw
            long_match = live_puts_df[live_puts_df['strike'] == ls]
            if not long_match.empty:
                lp = long_match.iloc[0]
                sp = short['lastPrice'] - lp['lastPrice']
                pts_out = ss - ref
                pct_out = (pts_out / ref) * 100
                spreads_list.append({
                    "Pts": int(pts_out), "(%)": f"{pct_out:+.1f}%",
                    "Strike": int(ss), "Leg": int(ls),
                    "Short PX": round(short['lastPrice'], 2),
                    "Long PX": round(lp['lastPrice'], 2),
                    "Spread": round(sp, 2),
                    "Premiums": round(sp * contracts * 100, 0),
                })

    spreads_list.sort(key=lambda r: r['Pts'], reverse=True)
    selected_short = store.get('selected_short')
    if selected_short and not live_puts_df.empty:
        ss = float(selected_short)
        if not any(r['Strike'] == int(ss) for r in spreads_list):
            sm = live_puts_df[live_puts_df['strike'] == ss]
            ls = ss - sw
            lm = live_puts_df[live_puts_df['strike'] == ls]
            if not sm.empty and not lm.empty:
                short = sm.iloc[0]
                lp = lm.iloc[0]
                sp = short['lastPrice'] - lp['lastPrice']
                pts_out = ss - ref
                pct_out = (pts_out / ref) * 100
                spreads_list.append({
                    "Pts": int(pts_out), "(%)": f"{pct_out:+.1f}%",
                    "Strike": int(ss), "Leg": int(ls),
                    "Short PX": round(short['lastPrice'], 2),
                    "Long PX": round(lp['lastPrice'], 2),
                    "Spread": round(sp, 2),
                    "Premiums": round(sp * contracts * 100, 0),
                })
                spreads_list.sort(key=lambda r: r['Pts'], reverse=True)

    return render_template('partials/spreads.html',
        spreads=spreads_list, target=target,
        selected_short=selected_short)


@trading_bp.route('/api/trading/select-spread', methods=['POST'])
def api_select_spread():
    store = _get_store()
    short = request.form.get('short', type=float)
    long_ = request.form.get('long', type=float)
    spread_px = request.form.get('spread_px', type=float)
    if short is not None:
        if store.get('selected_short') == short:
            store['selected_short'] = None
            store['selected_long'] = None
            store['selected_spread_px'] = 0.0
        else:
            store['selected_short'] = short
            store['selected_long'] = long_
            store['selected_spread_px'] = spread_px or 0.0
            store['saved_entry'] = math.floor((spread_px or 0.0) * 20) / 20
            store['saved_close'] = 0.05
        _save_store(store)
    resp = make_response('', 204)
    resp.headers['HX-Trigger'] = 'spread-selected'
    return resp


@trading_bp.route('/api/trading/trade', methods=['GET', 'POST'])
def api_trade():
    """Renders the full trade card (Entry PX / Current PX / Close / P/L).
    Saves entry/close edits to session on POST."""
    store = _get_store()
    if request.method == 'POST':
        entry = _fnum(request.form.get('entry'), store["saved_entry"])
        close = _fnum(request.form.get('close'), store["saved_close"])
        store["saved_entry"] = entry
        store["saved_close"] = close
        _save_store(store)
    entry     = store["saved_entry"]
    close     = store["saved_close"]
    contracts = store["saved_contracts"]
    pl = (entry - close) * contracts * 100
    pl_str = f"+${pl:,.0f}" if pl >= 0 else f"-${abs(pl):,.0f}"
    return render_template('partials/trade.html', store=store, pl=pl_str)


@trading_bp.route('/api/trading/prob')
def api_prob():
    store = _get_store()
    selected_short = store.get("selected_short")
    selected_long  = store.get("selected_long")
    live_puts_df = get_spx_puts()

    short_prob = long_prob = "—"
    if selected_short and not live_puts_df.empty and 'delta' in live_puts_df.columns:
        m = live_puts_df[live_puts_df['strike'] == float(selected_short)]
        if not m.empty:
            short_prob = f"{(1 - abs(m.iloc[0]['delta'])) * 100:.1f}%"
    if selected_long and not live_puts_df.empty and 'delta' in live_puts_df.columns:
        m = live_puts_df[live_puts_df['strike'] == float(selected_long)]
        if not m.empty:
            long_prob = f"{(1 - abs(m.iloc[0]['delta'])) * 100:.1f}%"

    strike_val = str(int(selected_short)) if selected_short else "—"
    leg_val    = str(int(selected_long))  if selected_long  else "—"
    return render_template('partials/prob.html',
        strike=strike_val, short_prob=short_prob,
        leg=leg_val, long_prob=long_prob)


@trading_bp.route('/api/trading/day-chart')
def api_day_chart():
    period_label = request.args.get('day_period', '1 Day')
    day_map = {"1 Day": "1d", "3 Days": "3d", "5 Days": "5d"}
    df = get_spx_history_intraday(period=day_map.get(period_label, "1d"))
    spx_last, spx_open, _, _ = get_spx_metrics()
    is_down = (spx_last - spx_open) < 0
    color = "#FF3D54" if is_down else GREEN_400
    halo  = 'rgba(255,61,84,0.3)' if is_down else 'rgba(17,241,133,0.3)'

    store = _get_store()
    selected_short = store.get("selected_short")
    selected_long  = store.get("selected_long")

    if df.empty:
        fig = empty_figure()
    else:
        fig = create_spx_chart(
            period_label, df['Close'], df.index, color, halo,
            selected_short=selected_short, selected_long=selected_long,
        )
    return _chart_html('day-chart', fig, selected_short=selected_short, selected_long=selected_long)


@trading_bp.route('/api/trading/month-chart')
def api_month_chart():
    period_label = request.args.get('month_period', '6 Months')
    show_events  = bool(request.args.get('show_events'))
    show_line    = bool(request.args.get('show_line'))

    month_params = {"12 Months": "12mo", "8 Months": "8mo", "6 Months": "6mo",
                    "3 Months": "3mo", "1 Month": "1mo"}
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "8mo": 240, "12mo": 365}

    df = get_spx_history_historical()
    spx_last, spx_open, _, _ = get_spx_metrics()

    if df.empty:
        return _chart_html('month-chart', empty_figure())

    now_ts = pd.Timestamp.now('America/New_York').tz_localize(None).normalize()
    if now_ts not in df.index:
        live_row = pd.DataFrame(
            {'Open': spx_last, 'High': spx_last, 'Low': spx_last, 'Close': spx_last},
            index=[now_ts]
        )
        df = pd.concat([df, live_row])
    else:
        df.loc[now_ts, 'Close'] = spx_last

    is_down = (spx_last - spx_open) < 0
    color = "#FF3D54" if is_down else GREEN_400
    halo  = 'rgba(255,61,84,0.3)' if is_down else 'rgba(17,241,133,0.3)'

    period    = month_params.get(period_label, "6mo")
    view_days = days_map[period]
    view_start = now_ts - pd.Timedelta(days=view_days)

    events = None
    if show_events:
        lookahead = df.index.max() + pd.DateOffset(months=1)
        events = get_financial_events(df.index.min(), lookahead)

    candle_data = None if show_line else df

    store = _get_store()
    selected_short = store.get("selected_short")
    selected_long  = store.get("selected_long")

    fig = create_spx_chart(
        period_label, df['Close'], df.index, color, halo,
        events=events, chart_height=500, view_range=view_start,
        ohlc_df=candle_data,
        selected_short=selected_short, selected_long=selected_long,
    )
    return _chart_html('month-chart', fig, selected_short=selected_short, selected_long=selected_long)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fnum(val, default=0):
    try:
        return float(val) if val is not None else float(default)
    except (ValueError, TypeError):
        return float(default)


def _chart_html(div_id: str, fig: go.Figure, selected_short=None, selected_long=None) -> str:
    # data-short / data-long are read by positionStrikePills() in base.html to
    # render HTML pill overlays (more reliable than Plotly layout.annotations).
    fig_json = html.escape(fig.to_json(), quote=True)
    h = fig.layout.height or 500
    strike_attrs = (
        f' data-short="{int(selected_short)}" data-long="{int(selected_long)}"'
        if selected_short is not None else ''
    )
    # The wrapper holds position:relative so pills can be absolutely positioned
    # over the chart without being children of the Plotly div itself (which
    # would interfere with Plotly's internal hover/event overlay).
    return (
        f'<div class="chart-pill-wrap" style="position:relative;width:100%;height:{h}px;overflow:hidden">'
        f'<div id="{div_id}" data-plotly="{fig_json}"{strike_attrs} '
        f'style="width:100%;height:{h}px"></div>'
        f'</div>'
    )
