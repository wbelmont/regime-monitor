"""Data loading from FREE sources (FRED + Yahoo Finance) with local caching.

We cache to parquet so repeated runs are fast and you can work offline. If a
source is down, we fall back gracefully (FRED -> Yahoo).
"""

from __future__ import annotations

import datetime as dt
import io
import urllib.request

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
    """Fetch a FRED series via the public fredgraph CSV endpoint.

    We do NOT use `pandas_datareader` here: it imports `distutils`, which was
    removed from the stdlib in Python 3.12+, so it raises ``ModuleNotFoundError``
    on this project's 3.13 venv and made every FRED pull fail *silently* (the
    callers below swallow the exception and fall back), which is why `y3m` and
    `hy_oas` were all-NaN in the cache. The fredgraph CSV endpoint needs no key
    and returns a simple two-column (date, value) table. A short timeout means
    that if FRED is unreachable we fail fast and let the Yahoo fallback run.
    """
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series}&cosd={start}&coed={end}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col].replace(".", pd.NA), errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col]),
        name=series,
    )
    return s.dropna()


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
        # ^TNX is quoted as 10x the yield historically in this cache's lineage
        # (e.g. 4.49% -> 0.449). Keep that /10 scale so the whole y10 series is
        # internally consistent over time (the CJM standardizes features, so the
        # absolute scale is irrelevant — only consistency + sign matter).
        y10 = (_yahoo_close(config.TNX_TICKER, start, end) / 10.0).rename("y10")

    # --- 3m yield (for curve slope); optional ---
    try:
        y3m = _fred(config.FRED_YC_SLOPE_3M, start, end).rename("y3m")
    except Exception:
        # ^IRX = 13-week T-bill discount rate (full history back to 2000). Use
        # the SAME /10 scale as the y10 fallback so curve_slope = y10 - y3m is
        # on a consistent scale with the correct sign (inversion < 0).
        y3m = (_yahoo_close("^IRX", start, end) / 10.0).rename("y3m")

    # --- High-yield credit spread (stress gauge); optional ---
    # FRED-only series (ICE BofA OAS). No free non-FRED equivalent, so if the
    # FRED fetch is unavailable this stays empty and the model simply omits it
    # (features.available() guards on notna()).
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
