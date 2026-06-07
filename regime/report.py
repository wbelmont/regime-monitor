"""Chart generation for at-a-glance review (saved into reports/)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no GUI needed; we save PNGs
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from . import config


def regime_chart(
    feat: pd.DataFrame, labels: pd.Series, filename: str = "regime_history.png"
) -> str:
    """Cumulative market return shaded by regime (red = bear)."""
    d = feat.loc[labels.index].copy()
    d["regime"] = labels
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(d.index, d["cum_return"], color="black", linewidth=1, alpha=0.8)
    ax.fill_between(
        d.index,
        d["cum_return"].min(),
        d["cum_return"].max(),
        where=(d["regime"] == 1),
        color="red",
        alpha=0.25,
        label="Bear regime",
    )
    ax.set_title("Market history shaded by detected regime")
    ax.set_ylabel("Cumulative return")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = config.REPORTS_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def equity_chart(equity: pd.DataFrame, filename: str = "backtest_equity.png") -> str:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(equity.index, equity["strategy"], label="Regime strategy", linewidth=1.5)
    ax.plot(
        equity.index, equity["buy_hold"], label="Buy & hold", linewidth=1.0, alpha=0.7
    )
    ax.set_title("Regime-switch strategy vs. buy & hold (out-of-sample, net of costs)")
    ax.set_ylabel("Growth of $1")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = config.REPORTS_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
