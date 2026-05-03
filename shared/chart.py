"""Shared SPX chart engine used by both trading and study pages."""
import datetime
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go


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

    if selected_short is not None and selected_long is not None:
        fig.add_hline(
            y=selected_short, line_dash="solid", line_color="#4B7BFF",
            annotation_text=f"Short strike ({selected_short})",
            annotation_position="top left",
            annotation=dict(font_size=8, font_color="white", bgcolor="#4B7BFF", borderpad=3, bordercolor="#4B7BFF")
        )
        fig.add_hline(
            y=selected_long, line_dash="solid", line_color="#FF6347",
            annotation_text=f"Long strike ({selected_long})",
            annotation_position="bottom left",
            annotation=dict(font_size=8, font_color="white", bgcolor="#FF6347", borderpad=3, bordercolor="#FF6347")
        )

    if events:
        grouped = defaultdict(list)
        for evt_date, evt_label in events:
            grouped[evt_date].append(evt_label)
        for evt_date, labels in grouped.items():
            evt_dt = datetime.datetime.combine(evt_date, datetime.time()) if isinstance(evt_date, datetime.date) else evt_date
            if min_date is not None and pd.Timestamp(min_date) <= pd.Timestamp(evt_dt) <= pd.Timestamp(padded_max_date):
                combined = ", ".join(labels)
                fig.add_shape(
                    type="line", x0=evt_dt, x1=evt_dt, y0=0, y1=1,
                    yref="paper", line=dict(dash="1px,3px", color="#B2B2B2", width=1),
                )
                fig.add_annotation(
                    x=evt_dt, y=1.01, yref="paper", text=combined,
                    textangle=-90, font=dict(size=7, color="#888888"),
                    showarrow=False, yanchor="bottom", xanchor="center",
                )

    fig.update_layout(
        dragmode="zoom",
        uirevision="constant",
        height=chart_height,
        margin=dict(l=4, r=4, t=70 if events else 10, b=30),
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
            range=y_range,
            showspikes=True, spikemode="across", spikesnap="cursor", spikedash="1, 3",
            spikecolor="#B2B2B2", spikethickness=1
        )
    )
    return fig
