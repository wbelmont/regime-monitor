"""Data loading from FREE sources (FRED + Yahoo Finance) with local caching.

We cache to parquet so repeated runs are fast and you can work offline. If a
source is down, we fall back gracefully (FRED -> Yahoo).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from . import config


def _today() -> str:
    return dt.date.today().isoformat()


def _cache_path(name: str) -> "config.Path":
    return config.CACHE_DIR / f"{name}.parquet"


def _is_fresh(path, max_age_hours: int = 18) -> bool:
    """Cache is 'fresh' if updated within max_age_hours (markets close daily)."""
    if not path.exists():
        return False
    age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age.total_seconds() < max_age_hours * 3600


def _fred(series: str, start: str, end: str) -> pd.Series:
    import pandas_datareader.data as web

    df = web.DataReader(series, "fred", start, end)
    return df.iloc[:, 0].rename(series)


def _yahoo_close(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    col = "Close" if "Close" in df.columns else df.columns[0]
    s = df[col]
    if isinstance(s, pd.DataFrame):  # flatten multiindex
        s = s.iloc[:, 0]
    return s.rename(ticker)


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """Return a daily DataFrame with all raw inputs we need.

    Columns: market, vix, y10, y3m, hy_oas
    """
    cache = _cache_path("raw_inputs")
    if not refresh and _is_fresh(cache):
        return pd.read_parquet(cache)

    start, end = config.START_DATE, _today()

    # --- Market (S&P 500) ---
    market = _yahoo_close(config.MARKET_TICKER, start, end).rename("market")

    # --- VIX: FRED first, Yahoo fallback ---
    try:
        vix = _fred(config.FRED_VIX, start, end).rename("vix")
    except Exception:
        vix = _yahoo_close(config.VIX_TICKER, start, end).rename("vix")

    # --- 10y yield ---
    try:
        y10 = _fred(config.FRED_10Y, start, end).rename("y10")
    except Exception:
        y10 = (_yahoo_close(config.TNX_TICKER, start, end) / 10.0).rename("y10")

    # --- 3m yield (for curve slope); optional ---
    try:
        y3m = _fred(config.FRED_YC_SLOPE_3M, start, end).rename("y3m")
    except Exception:
        y3m = pd.Series(dtype=float, name="y3m")

    # --- High-yield credit spread (stress gauge); optional ---
    try:
        hy_oas = _fred(config.FRED_HY_OAS, start, end).rename("hy_oas")
    except Exception:
        hy_oas = pd.Series(dtype=float, name="hy_oas")

    df = pd.concat([market, vix, y10, y3m, hy_oas], axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Forward-fill macro series (they update on different calendars), then drop
    # rows with no market price.
    df[["vix", "y10", "y3m", "hy_oas"]] = df[["vix", "y10", "y3m", "hy_oas"]].ffill()
    df = df.dropna(subset=["market"])

    df.to_parquet(cache)
    return df
