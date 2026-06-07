"""Feature engineering.

Design goals:
  * Every feature uses ONLY past data (no look-ahead). All rolling windows and
    EWMAs look backward by construction.
  * Features are interpretable so you can reason about why the model says what
    it says.
  * We fixed the original project's `active_beta` bug (it regressed the market
    on itself and was always ~1). It is removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw inputs into model-ready features.

    Input columns expected: market, vix, y10, y3m (optional), hy_oas (optional).
    """
    d = raw.copy()

    # --- Returns / trend ---
    d["mkt_ret"] = d["market"].pct_change()
    d["cum_return"] = (1 + d["mkt_ret"]).cumprod()
    d["vol_21"] = d["mkt_ret"].rolling(21).std()
    d["vol_63"] = d["mkt_ret"].rolling(63).std()

    d["ma_50"] = d["market"].rolling(50).mean()
    d["ma_200"] = d["market"].rolling(200).mean()
    d["ma_ratio"] = d["ma_50"] / d["ma_200"]  # >1 uptrend, <1 downtrend
    d["above_200ma"] = (d["market"] > d["ma_200"]).astype(float)

    d["mom_63"] = d["market"].pct_change(63)  # ~3 month momentum
    d["mom_126"] = d["market"].pct_change(126)  # ~6 month momentum
    d["mom_252"] = d["market"].pct_change(252)  # ~12 month momentum

    macd, macd_sig = _macd(d["market"])
    d["macd"] = macd
    d["macd_diff"] = macd - macd_sig

    # --- Downside / drawdown (asymmetric risk) ---
    # Shu & Mulvey lean on downside-focused features. These react to *grinding*
    # declines faster than symmetric momentum/vol (which need a big accumulated
    # move to flip), so they target the slow short-ENTRY timing on slow sell-offs.
    #   * drawdown_63: current price vs its trailing-63d high (<=0; deeper = more
    #     bearish). Turns negative on day one of a pullback from a local peak.
    #   * downside_dev_21: rolling std of ONLY the negative daily returns
    #     (semi-deviation) — rises when losses cluster, ignoring upside vol.
    roll_high_63 = d["market"].rolling(63, min_periods=10).max()
    d["drawdown_63"] = d["market"] / roll_high_63 - 1.0
    neg_ret = d["mkt_ret"].where(d["mkt_ret"] < 0, 0.0)
    d["downside_dev_21"] = neg_ret.rolling(21, min_periods=5).std()

    # --- Volatility / fear ---
    d["vix_chg"] = d["vix"].diff()
    d["vix_ma_21"] = d["vix"].rolling(21).mean()
    # vol risk premium: implied (VIX, annualized %) vs realized (annualized %)
    d["vol_premium"] = d["vix"] - d["vol_21"] * np.sqrt(252) * 100

    # --- Rates / credit (macro stress) ---
    d["yield_chg"] = d["y10"].diff()
    if "y3m" in d and d["y3m"].notna().any():
        d["curve_slope"] = d["y10"] - d["y3m"]  # inversion = recession risk
    if "hy_oas" in d and d["hy_oas"].notna().any():
        d["hy_oas_level"] = d["hy_oas"]
        d["hy_oas_chg"] = d["hy_oas"].diff(21)  # widening = stress

    return d


# Features fed to the unsupervised regime labeler (describe the *state* of the
# market). Kept compact and standardized downstream.
#
# NOTE (2026-06-07): a coarse A/B (scripts/eval_features.py) showed that simply
# APPENDING drawdown_63/downside_dev_21/curve_slope to the 2-state CJM did NOT
# improve short-entry timing (delta ~0 on the grinding declines) and slightly
# regressed 2022 (+5d) and 2025-26 (failed to cross 0.60). The extra features
# are highly correlated with the existing vol/mom set, so they dilute rather
# than sharpen the equal-weighted Euclidean clustering. Production therefore
# keeps the BASELINE set; the experimental superset is kept for the next
# iteration (try REPLACE-not-append, or a separate short-entry overlay).
REGIME_FEATURES_BASELINE = [
    "mkt_ret",
    "vol_21",
    "vix",
    "vol_premium",
    "macd",
    "ma_ratio",
    "mom_63",
    "mom_126",
]

# Experimental superset (NOT in production — see note above). Used by the
# signal-quality A/B harness (scripts/eval_features.py) only.
REGIME_FEATURES_EXPERIMENTAL = REGIME_FEATURES_BASELINE + [
    "drawdown_63",
    "downside_dev_21",
    "curve_slope",
]

# Production feature set fed to the labeler.
REGIME_FEATURES = list(REGIME_FEATURES_BASELINE)


# Features fed to the supervised next-regime predictor.
PREDICTOR_FEATURES = [
    "vix",
    "vix_chg",
    "vix_ma_21",
    "mkt_ret",
    "vol_21",
    "vol_63",
    "vol_premium",
    "yield_chg",
    "macd",
    "macd_diff",
    "ma_ratio",
    "above_200ma",
    "mom_63",
    "mom_126",
    "mom_252",
]


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return only the requested columns that actually exist & are non-empty."""
    return [c for c in cols if c in df.columns and df[c].notna().any()]
