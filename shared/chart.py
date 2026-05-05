"""Shared SPX chart engine used by both trading and study pages."""
import datetime
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go


# ── Axis / hover style shared by both chart functions ─────────────────────────
_AXIS_STYLE = dict(
    showgrid=True, gridcolor="#F0F0F0",
    tickfont=dict(color="#808080", size=8),
    ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
    showspikes=True, spikemode="across", spikesnap="cursor",
    spikedash="1,3", spikecolor="#B2B2B2", spikethickness=1,
)


def create_long_chart(
    prices,
    dates,
    line_color,
    halo_color,
    events=None,
    chart_height=500,
    ohlc_df=None,
):
    """Dedicated chart for the study-tab long-term view.

    Returns (fig, evt_payload) where evt_payload is a list of
    {"d": iso-date, "l": label-text} dicts for events within the visible range.
    The route embeds evt_payload as a data-events JSON attribute on the div;
    positionEventLabels() in base.html then injects HTML label divs at the
    correct pixel x-positions after Plotly renders.
    """
    fig = go.Figure()

    # ── Price trace ───────────────────────────────────────────────────────────
    if ohlc_df is not None:
        hover_texts = [
            f"{d.strftime('%b %-d, %Y')}<br>"
            f"O: {o:,.2f}  H: {h:,.2f}  L: {l:,.2f}  C: {c:,.2f}"
            for d, o, h, l, c in zip(
                ohlc_df.index,
                ohlc_df["Open"], ohlc_df["High"],
                ohlc_df["Low"],  ohlc_df["Close"],
            )
        ]
        fig.add_trace(go.Candlestick(
            x=ohlc_df.index,
            open=ohlc_df["Open"], high=ohlc_df["High"],
            low=ohlc_df["Low"],   close=ohlc_df["Close"],
            increasing=dict(line=dict(color="#11F185", width=1), fillcolor="#11F185"),
            decreasing=dict(line=dict(color="#FF3D54", width=1), fillcolor="#FF3D54"),
            showlegend=False,
            text=hover_texts,
            hoverinfo="text",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines",
            line=dict(color=line_color, width=2),
            showlegend=False,
            hovertemplate="%{x|%b %-d, %Y}<br>%{y:,.2f}<extra></extra>",
        ))
        if len(dates) > 0:
            fig.add_trace(go.Scatter(
                x=[dates[-1]], y=[prices.iloc[-1]], mode="markers",
                marker=dict(color=line_color, size=4,
                            line=dict(color=halo_color, width=8)),
                showlegend=False, hoverinfo="skip",
            ))

    # ── Date range ────────────────────────────────────────────────────────────
    if len(dates) == 0:
        fig.update_layout(height=chart_height)
        return fig, []

    min_date = dates.min()
    max_date = dates.max()
    span     = max_date - min_date
    x_end    = max_date + span * 0.04   # small right padding

    # ── Explicit y range (needed so annotations can reference y_top) ──────────
    if ohlc_df is not None:
        y_lo, y_hi = ohlc_df["Low"].min(), ohlc_df["High"].max()
    else:
        y_lo, y_hi = prices.min(), prices.max()
    y_pad   = (y_hi - y_lo) * 0.04
    y_range = [y_lo - y_pad, y_hi + y_pad]

    # ── Events — shapes only; labels are injected as HTML by positionEventLabels() in JS ──
    has_events = bool(events)
    evt_payload = []   # returned so the route can embed as data-events
    if has_events:
        grouped = defaultdict(list)
        for evt_date, evt_label in events:
            if min_date <= pd.Timestamp(evt_date) <= x_end:
                grouped[evt_date].append(evt_label)

        for evt_date, labels in grouped.items():
            ts = pd.Timestamp(evt_date)
            fig.add_shape(
                type="line",
                x0=ts, x1=ts, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(dash="dot", color="#CCCCCC", width=1),
                layer="below",
            )
            evt_payload.append({"d": evt_date.isoformat(), "l": ", ".join(labels)})

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=chart_height,
        margin=dict(l=4, r=4, t=50 if has_events else 12, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        hovermode="x",
        hoverdistance=50,
        spikedistance=50,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color="#1E1E1E", size=11),
        ),
        xaxis=dict(
            **_AXIS_STYLE,
            rangeslider=dict(visible=False),
            range=[min_date, x_end],
        ),
        yaxis=dict(
            **_AXIS_STYLE,
            automargin=True,
            side="left",
            range=y_range,
        ),
    )
    return fig, evt_payload


def create_spx_chart(
    title,
    prices,
    dates,
    line_color,
    halo_color,
    events=None,
    chart_height=420,
    view_range=None,
    ohlc_df=None,
    selected_short=None,
    selected_long=None,
    hover_xfmt=None,
):
    """Render an SPX price chart (line or candlestick) with optional event markers
    and optional short/long strike lines. Identical look across pages."""
    fig = go.Figure()

    if ohlc_df is not None:
        hover_texts = [
            f"{d.strftime('%b %-d, %Y')}<br>Open: {o:,.2f}<br>High: {h:,.2f}<br>Low: {l:,.2f}<br>Close: {c:,.2f}"
            for d, o, h, l, c in zip(ohlc_df.index, ohlc_df['Open'], ohlc_df['High'], ohlc_df['Low'], ohlc_df['Close'])
        ]
        fig.add_trace(go.Candlestick(
            x=ohlc_df.index,
            open=ohlc_df['Open'], high=ohlc_df['High'],
            low=ohlc_df['Low'], close=ohlc_df['Close'],
            increasing=dict(line=dict(color="#11F185", width=1), fillcolor="#11F185"),
            decreasing=dict(line=dict(color="#FF3D54", width=1), fillcolor="#FF3D54"),
            line=dict(width=1),
            showlegend=False,
            text=hover_texts,
            hoverinfo="text",
        ))
    else:
        ht = (
            f"%{{x|{hover_xfmt}}}<br>%{{y:,.2f}}<extra></extra>"
            if hover_xfmt
            else "%{x}<br>%{y:,.2f}<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=dates, y=prices, mode='lines',
            line=dict(color=line_color, width=2),
            showlegend=False,
            hovertemplate=ht,
        ))
        if len(dates) > 0 and len(prices) > 0:
            last_date = dates[-1]
            last_price = prices.iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_date], y=[last_price], mode='markers',
                marker=dict(
                    color=line_color,
                    size=4,
                    line=dict(color=halo_color, width=8)
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

    y_range = None
    if len(dates) > 0:
        min_date = dates.min()
        max_date = dates.max()
        date_range = max_date - min_date
        padded_max_date = max_date + (date_range * 0.05)
        if events:
            last_evt = pd.Timestamp(max(e[0] for e in events))
            if last_evt > padded_max_date:
                padded_max_date = last_evt + (date_range * 0.02)
        view_left = view_range if view_range is not None else min_date
        if ohlc_df is not None:
            vis_df = ohlc_df[(ohlc_df.index >= view_left) & (ohlc_df.index <= padded_max_date)]
            if len(vis_df) > 0:
                y_min, y_max = vis_df['Low'].min(), vis_df['High'].max()
            else:
                y_min, y_max = prices.min(), prices.max()
        else:
            visible = prices[(dates >= view_left) & (dates <= padded_max_date)]
            if len(visible) > 0:
                y_min, y_max = visible.min(), visible.max()
            else:
                y_min, y_max = prices.min(), prices.max()
        if selected_short is not None:
            y_min = min(y_min, float(selected_long or selected_short))
            y_max = max(y_max, float(selected_short))
        y_pad = (y_max - y_min) * 0.05
        y_range = [y_min - y_pad, y_max + y_pad]
    else:
        min_date, padded_max_date = None, None

    strike_margin_r = 4
    if selected_short is not None and selected_long is not None and min_date is not None:
        # Lines stay as Scatter traces (data) — those are diffed reliably by
        # Plotly.react and never flicker. Labels use xref='paper' so they sit
        # outside the data grid, anchored to the right edge of the plot area.
        # The right margin is expanded to give the pills room.
        # Rounded corners come from the CSS rule in style.css targeting
        # rect.bg inside .annotation (SVG rx/ry).
        strike_margin_r = 36
        x_left = view_range if view_range is not None else min_date
        x_line = [x_left, padded_max_date]
        fig.add_trace(go.Scatter(
            x=x_line, y=[selected_short, selected_short],
            mode='lines',
            line=dict(color="#3f73fa", width=1),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=x_line, y=[selected_long, selected_long],
            mode='lines',
            line=dict(color="#ff633d", width=1),
            showlegend=False, hoverinfo='skip',
        ))
        # Pills are rendered as HTML overlays in JS (see base.html positionStrikePills).
        # The right margin whitespace is kept so the lines don't run into the pill area.

    if events:
        # Only render events within the visible view window — events before view_range
        # would be clipped anyway and add noise to the layout.
        x_left = view_range if view_range is not None else min_date
        grouped = defaultdict(list)
        for evt_date, evt_label in events:
            grouped[evt_date].append(evt_label)
        line_xs, line_ys = [], []   # batched vertical lines (None-separated segments)
        lbl_xs, lbl_ys, lbl_texts = [], [], []
        y_lo = y_range[0] if y_range else (prices.min() if len(prices) > 0 else 0)
        y_hi = y_range[1] if y_range else (prices.max() if len(prices) > 0 else 0)
        for evt_date, labels in grouped.items():
            evt_dt = (datetime.datetime.combine(evt_date, datetime.time())
                      if isinstance(evt_date, datetime.date) else evt_date)
            if x_left is None or not (pd.Timestamp(x_left) <= pd.Timestamp(evt_dt) <= pd.Timestamp(padded_max_date)):
                continue
            line_xs += [evt_dt, evt_dt, None]
            line_ys += [y_lo, y_hi, None]
            lbl_xs.append(evt_dt)
            lbl_ys.append(y_hi)
            lbl_texts.append(", ".join(labels))
        # Vertical dashed lines as Scatter traces (layout.shapes are dropped by Plotly.react)
        if line_xs:
            fig.add_trace(go.Scatter(
                x=line_xs, y=line_ys, mode='lines',
                line=dict(dash='dot', color="#B2B2B2", width=1),
                showlegend=False, hoverinfo='skip', cliponaxis=False,
            ))
        # Labels above the chart, rotated — cliponaxis=False lets them sit in the top margin
        if lbl_xs:
            fig.add_trace(go.Scatter(
                x=lbl_xs, y=lbl_ys, mode='text',
                text=lbl_texts,
                textposition='top center',
                textangle=-90,
                textfont=dict(size=7, color='#888888', family='Inter, sans-serif'),
                showlegend=False, hoverinfo='skip', cliponaxis=False,
            ))

    # uirevision: preserve user-driven UI state (zoom, pan) across Plotly.react
    # calls *as long as the value doesn't change*. We tie it to the selected
    # strikes so that picking a new spread RESETS the view (allowing the new
    # strike line to expand the y-range), while routine 60s price refreshes
    # keep the same uirevision and preserve any zoom the user has applied.
    ui_rev = f"strikes-{selected_short}-{selected_long}"
    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        dragmode="zoom",
        uirevision=ui_rev,
        height=chart_height,
        margin=dict(l=4, r=strike_margin_r, t=70 if events else 10, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x" if ohlc_df is not None else "x unified",
        hoverdistance=100 if ohlc_df is not None else -1,
        spikedistance=100 if ohlc_df is not None else -1,
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(0, 0, 0, 0)",
            font=dict(color="#1E1E1E")
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#F0F0F0",
            tickfont=dict(color="#808080", size=8),
            ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
            tickformat=hover_xfmt if hover_xfmt else None,
            range=[view_range if view_range is not None else min_date, padded_max_date] if min_date else None,
            rangeslider=dict(visible=False),
            showspikes=True, spikemode="across",
            spikesnap="data" if ohlc_df is not None else "cursor",
            spikedash="1, 3", spikecolor="#B2B2B2", spikethickness=1
        ),
        yaxis=dict(
            automargin=True,
            showgrid=True, gridcolor="#F0F0F0", side="left",
            tickfont=dict(color="#808080", size=8),
            ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)",
            range=y_range,
            showspikes=True, spikemode="across", spikesnap="cursor", spikedash="1, 3",
            spikecolor="#B2B2B2", spikethickness=1
        )
    )
    return fig
