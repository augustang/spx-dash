"""Convert raw Schwab price-history responses into OHLC DataFrames."""
from __future__ import annotations

import pandas as pd

_RENAME = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}


def candles_to_df(raw: dict | None, normalize: bool = False) -> pd.DataFrame:
    """Return an ET-naive, datetime-indexed OHLC DataFrame from a Schwab
    pricehistory response, or an empty DataFrame if the payload has no candles.

    normalize=True truncates timestamps to midnight (daily bars).
    """
    if not raw or "candles" not in raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw["candles"])
    if df.empty:
        return df
    dt = (
        pd.to_datetime(df["datetime"], unit="ms")
        .dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    df["datetime"] = dt.dt.normalize() if normalize else dt
    df.set_index("datetime", inplace=True)
    df.rename(columns=_RENAME, inplace=True)
    return df
