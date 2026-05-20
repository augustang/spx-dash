"""
Update data/SPX_1day.csv and data/SPX_5min.csv with the latest Schwab data,
then commit and push to GitHub.

Run this weekly so the local CSVs never drift past Schwab's rolling history
window (Schwab only serves the last ~20 trading days of intraday data).

Usage:
    python3 update_csvs.py            # update both, commit, push
    python3 update_csvs.py --no-push  # update + commit, skip git push
    python3 update_csvs.py --dry-run  # report only (no writes, no commit)
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time

import pandas as pd
import pytz

import schwab_client

EASTERN = pytz.timezone("America/New_York")

REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(REPO_DIR, "data")
DAILY_PATH = os.path.join(DATA_DIR, "SPX_1day.csv")
MIN5_PATH  = os.path.join(DATA_DIR, "SPX_5min.csv")

# Re-fetch this many trailing days/rows to absorb any late corrections or
# previously-partial candles. Cheap insurance.
DAILY_OVERLAP_DAYS = 5
MIN5_OVERLAP_DAYS  = 2

# Schwab intraday history works most reliably with windows <= 10 days.
MIN5_CHUNK_DAYS = 10


# ---------------------------------------------------------------------------
# Schwab helpers
# ---------------------------------------------------------------------------
def _candles_to_df(raw) -> pd.DataFrame:
    """Convert a Schwab /pricehistory response into a tz-naive ET DataFrame."""
    if not raw or "candles" not in raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw["candles"])
    if df.empty:
        return df
    df["datetime"] = (
        pd.to_datetime(df["datetime"], unit="ms")
        .dt.tz_localize("UTC")
        .dt.tz_convert(EASTERN)
        .dt.tz_localize(None)
    )
    return df


# ---------------------------------------------------------------------------
# Daily
# ---------------------------------------------------------------------------
def update_daily(dry_run: bool) -> int:
    print("=== SPX_1day.csv ===")
    existing = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    last_date = existing["date"].max().normalize()
    today_et  = dt.datetime.now(EASTERN)
    today     = pd.Timestamp(today_et.date())
    print(f"  last row: {last_date.date()}   today: {today.date()}")

    start    = last_date - pd.Timedelta(days=DAILY_OVERLAP_DAYS)
    end      = today + pd.Timedelta(days=1)
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp() * 1000)

    raw = schwab_client.fetch_price_history(
        symbol="$SPX", period_type="year", freq_type="daily", freq=1,
        start_date=start_ms, end_date=end_ms,
    )
    if raw is None:
        print("  Schwab API call failed (token expired? run schwab_auth.py).")
        return 0

    df = _candles_to_df(raw)
    if df.empty:
        print("  no candles returned")
        return 0

    df["date"] = df["datetime"].dt.normalize()
    df = df[["date", "open", "high", "low", "close"]]

    # Drop today's candle if the market hasn't closed yet — it's incomplete.
    if today_et.time() < dt.time(16, 0):
        df = df[df["date"] < today]

    combined = pd.concat([existing, df], ignore_index=True)
    combined.drop_duplicates(subset=["date"], keep="last", inplace=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    new_count = len(combined) - len(existing)

    if new_count <= 0:
        print("  already up to date")
        return 0

    added = combined.tail(new_count).copy()
    added["date"] = added["date"].dt.strftime("%Y-%m-%d")
    print(f"  + {new_count} new row(s):")
    print(added.to_string(index=False))

    if dry_run:
        return new_count

    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    combined.to_csv(DAILY_PATH, index=False, float_format="%.2f")
    print(f"  wrote {DAILY_PATH}")
    return new_count


# ---------------------------------------------------------------------------
# 5-minute
# ---------------------------------------------------------------------------
def update_5min(dry_run: bool) -> int:
    print("\n=== SPX_5min.csv ===")
    existing = pd.read_csv(MIN5_PATH, parse_dates=["timestamp"])
    last_ts = existing["timestamp"].max()
    now     = dt.datetime.now(EASTERN).replace(tzinfo=None)
    print(f"  last row: {last_ts}   now: {now:%Y-%m-%d %H:%M:%S}")

    fetch_start = (last_ts - pd.Timedelta(days=MIN5_OVERLAP_DAYS)).to_pydatetime()
    fetch_end   = now - dt.timedelta(minutes=5)  # avoid in-progress candle
    if fetch_end <= fetch_start:
        print("  already up to date")
        return 0

    chunks = []
    chunk_start = fetch_start
    while chunk_start < fetch_end:
        chunk_end = min(chunk_start + dt.timedelta(days=MIN5_CHUNK_DAYS), fetch_end)
        s_ms = int(chunk_start.timestamp() * 1000)
        e_ms = int(chunk_end.timestamp() * 1000)
        print(f"  fetching {chunk_start:%Y-%m-%d %H:%M} → {chunk_end:%Y-%m-%d %H:%M}")
        raw = schwab_client.fetch_price_history(
            symbol="$SPX", period_type="day", freq_type="minute", freq=5,
            start_date=s_ms, end_date=e_ms,
        )
        if raw is None:
            print("  Schwab API call failed (token expired? run schwab_auth.py).")
            return 0
        chunk_df = _candles_to_df(raw)
        if not chunk_df.empty:
            chunks.append(chunk_df)
        chunk_start = chunk_end
        time.sleep(0.4)

    if not chunks:
        print("  no candles returned")
        return 0

    new = pd.concat(chunks, ignore_index=True)
    new.rename(columns={"datetime": "timestamp"}, inplace=True)
    new.set_index("timestamp", inplace=True)
    new = new.between_time("09:30", "16:00")
    new.reset_index(inplace=True)
    if "volume" not in new.columns:
        new["volume"] = 0.0
    else:
        new["volume"] = new["volume"].fillna(0.0).astype(float)
    new = new[["timestamp", "open", "high", "low", "close", "volume"]]

    combined = pd.concat([existing, new], ignore_index=True)
    combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    new_count = len(combined) - len(existing)

    if new_count <= 0:
        print("  already up to date")
        return 0

    added = combined.tail(new_count).copy()
    added["timestamp"] = added["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  + {new_count} new row(s):")
    if new_count <= 12:
        print(added.to_string(index=False))
    else:
        print(added.head(5).to_string(index=False))
        print("  ...")
        print(added.tail(5).to_string(index=False))

    if dry_run:
        return new_count

    combined["timestamp"] = pd.to_datetime(combined["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    combined.to_csv(MIN5_PATH, index=False)
    print(f"  wrote {MIN5_PATH}")
    return new_count


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------
def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_DIR, check=True)


def commit_and_push(rows_daily: int, rows_5min: int, push: bool) -> None:
    if rows_daily == 0 and rows_5min == 0:
        print("\nNo new rows — skipping commit.")
        return

    parts = []
    if rows_daily: parts.append(f"{rows_daily} daily")
    if rows_5min:  parts.append(f"{rows_5min} 5min")
    msg = f"Update SPX CSVs ({' + '.join(parts)})"

    print(f"\n=== Committing: {msg} ===")
    _git("add", DAILY_PATH, MIN5_PATH)

    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=REPO_DIR,
    )
    if staged.returncode == 0:
        print("  files unchanged on disk — nothing to commit.")
        return

    _git("commit", "-m", msg)

    if push:
        print("=== Pushing to origin ===")
        _git("push", "origin", "HEAD")
        print("Pushed.")
    else:
        print("Skipped push (--no-push).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    dry_run = "--dry-run" in sys.argv
    push    = "--no-push" not in sys.argv

    if schwab_client._load_tokens() is None:
        print("No Schwab tokens found. Run: python3 schwab_auth.py")
        sys.exit(1)

    rows_daily = update_daily(dry_run)
    rows_5min  = update_5min(dry_run)

    if dry_run:
        print(f"\nDry run complete. Would add {rows_daily} daily + {rows_5min} 5min rows.")
        return

    commit_and_push(rows_daily, rows_5min, push)
    print("\nDone.")


if __name__ == "__main__":
    main()
