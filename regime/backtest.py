"""Honest, simple backtest of a regime-tilt overlay.

This answers the only question that matters: *would following this signal have
helped me make money / avoid losses?*

Strategy under test (continuous equity scaling, tradeable daily with an index
ETF + margin):
    * Very low bear probability -> lean in (leveraged baseline).
    * Rising bear probability    -> scale equity down linearly toward cash.
    * Very high bear probability -> fully defensive (cash proxy).

Tailored to the live use case: accounts are checked daily, adjustments are
frequent, and equity trades are effectively frictionless — so we evaluate at
*daily* resolution, use *continuous* target weights, and charge *no* trading
cost. We DO charge financing on the leverage sleeve: any borrowed exposure
(weight > 1) pays a margin rate, and any idle cash (weight < 1) earns it. We
compare against simple buy-and-hold.

All predictions come from `pipeline.walk_forward`, which is leak-free, and the
overlay trades strictly on the *prior day's* probability (see `run`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Continuous target-weight schedule as a function of P(bear next period).
BEAR_LOW = 0.20  # below this: leveraged baseline
BEAR_HIGH = 0.80  # above this: fully defensive (cash)
LEVERED_WEIGHT = 1.5  # equity exposure when bears look very unlikely
TRADING_DAYS_PER_YEAR = 252


def _daily_rate(annual: float) -> float:
    """Annual rate -> equivalent daily compounding rate."""
    return (1.0 + annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


# The non-equity sleeve `(1 - weight)` is now charged ASYMMETRICALLY, because a
# margin borrow rate and the yield on idle cash are NOT the same number:
#   * weight > 1  -> the borrowed portion (1 - weight) < 0 pays BORROW_RATE.
#   * weight < 1  -> the idle cash    (1 - weight) > 0 earns CASH_YIELD.
# Previously a single 10% rate was applied to both legs, so sitting in cash
# (e.g. during 2008/2020 de-risking) silently compounded at +10%/yr risk-free —
# an unrealistic tailwind that inflated the strategy's apparent edge. CASH_YIELD
# defaults to 0% (the most conservative, honest assumption); raise it via config
# for a T-bill sensitivity. BORROW_RATE keeps the real ~10% margin cost.
BORROW_RATE = getattr(config, "ANNUAL_FINANCING_RATE", 0.10)
CASH_YIELD = getattr(config, "ANNUAL_CASH_YIELD", 0.0)
DAILY_BORROW_RATE = _daily_rate(BORROW_RATE)
DAILY_CASH_YIELD = _daily_rate(CASH_YIELD)

# Backwards-compatible alias (older callers referenced this name).
ANNUAL_FINANCING_RATE = BORROW_RATE
DAILY_FINANCING_RATE = DAILY_BORROW_RATE


def _drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def _metrics(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Annualized performance metrics for a return series at any frequency.

    `periods_per_year` controls annualization (252 for daily, 12 for monthly).
    Defaults to daily, matching this module's daily engine.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return {}
    growth = float(np.prod(1.0 + returns.to_numpy(dtype=float)))
    ann_ret = growth ** (periods_per_year / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + returns).cumprod()
    return {
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": _drawdown(equity),
        "total_return": float(equity.iloc[-1] - 1),
    }


def _target_weight(bear_prob: np.ndarray) -> np.ndarray:
    """Continuous, piecewise equity target as a function of bear probability.

        bear_prob < 0.20          -> 1.5            (leveraged baseline)
        0.20 <= bear_prob <= 0.80 -> linear 1.0 -> 0.0
        bear_prob > 0.80          -> 0.0            (defensive / cash)

    Fully vectorized; the linear leg satisfies w(0.20)=1.0 and w(0.80)=0.0.
    """
    bp = np.asarray(bear_prob, dtype=float)
    linear = 1.0 - (bp - BEAR_LOW) / (BEAR_HIGH - BEAR_LOW)
    return np.select(
        [bp < BEAR_LOW, (bp >= BEAR_LOW) & (bp <= BEAR_HIGH), bp > BEAR_HIGH],
        [LEVERED_WEIGHT, linear, 0.0],
        default=0.0,
    )


def run(
    feat: pd.DataFrame,
    signals: pd.DataFrame,
    cash_yield_daily: float = DAILY_CASH_YIELD,
    borrow_rate_daily: float = DAILY_BORROW_RATE,
) -> dict:
    """DAILY backtest with continuous equity scaling and asymmetric financing.

    feat must contain 'mkt_ret' (daily). signals from walk_forward() must
    contain 'bear_prob'. Returns dict with 'strategy'/'buy_hold' metric dicts,
    an 'equity' frame, and the per-day 'daily' frame.

    Trading is frictionless, but the non-equity sleeve `(1 - weight)` is charged
    ASYMMETRICALLY: idle cash (weight < 1) earns `cash_yield_daily` (default 0%),
    while borrowed exposure (weight > 1) pays `borrow_rate_daily` (~10%/yr). This
    fixes the prior bug where idle cash earned the full margin rate, inflating
    the strategy's edge whenever it de-risked. Returns are net of financing.

    Look-ahead guarantee: today's target weight is a function of today's
    bear_prob, but we trade on the *prior* day's target (`.shift(1)`), so the
    return earned on day t uses only information known at the close of day t-1.
    """
    daily = feat[["mkt_ret"]].copy()
    # Align the (already leak-free) daily signal onto the price index.
    daily["bear_prob"] = signals["bear_prob"].reindex(daily.index).ffill()
    daily = daily.dropna(subset=["mkt_ret", "bear_prob"])

    # Continuous target weight from today's probability...
    daily["target_weight"] = _target_weight(daily["bear_prob"].to_numpy())
    # ...but the position actually held today was set using YESTERDAY's signal.
    # This single shift is the entire zero-look-ahead guarantee at daily freq.
    daily["weight"] = daily["target_weight"].shift(1).fillna(1.0)

    # Non-equity sleeve, charged by sign: cash (>0) earns cash_yield_daily,
    # borrowed (<0) pays borrow_rate_daily. These are NOT the same rate.
    sleeve = 1.0 - daily["weight"]
    sleeve_rate = np.where(sleeve >= 0.0, cash_yield_daily, borrow_rate_daily)
    daily["strat_net"] = daily["weight"] * daily["mkt_ret"] + sleeve * sleeve_rate

    equity = pd.DataFrame(
        {
            "strategy": (1 + daily["strat_net"]).cumprod(),
            "buy_hold": (1 + daily["mkt_ret"]).cumprod(),
        }
    )

    return {
        "strategy": _metrics(daily["strat_net"]),
        "buy_hold": _metrics(daily["mkt_ret"]),
        "equity": equity,
        "daily": daily,
    }
