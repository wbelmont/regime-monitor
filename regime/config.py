"""Central configuration.

Everything tunable lives here so you (a non-CS user) can change behavior in ONE
place without hunting through code. Edit the values, re-run `regime update`.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths   (everything is stored relative to the project, so it's portable)
# --------------------------------------------------------------------------- #
# config.py lives at <project>/regime/config.py, so the project root is one
# level up from the package directory. (Previously this used parents[2], which
# only matched a src/-layout and resolved to ~/Desktop for this layout — that
# scattered data/ and reports/ outside the project and broke `regime tune`'s
# config rewrite. Fixed to parents[1].)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
SIGNAL_HISTORY_FILE = DATA_DIR / "signal_history.csv"

for _d in (DATA_DIR, CACHE_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data window
# --------------------------------------------------------------------------- #
START_DATE = "2000-01-01"

# Tickers / FRED series we pull (all FREE data).
MARKET_TICKER = "^GSPC"  # S&P 500 index
VIX_TICKER = "^VIX"  # volatility index
TNX_TICKER = "^TNX"  # 10y treasury yield (yfinance fallback)
FRED_VIX = "VIXCLS"  # VIX from FRED (preferred)
FRED_10Y = "DGS10"  # 10y yield from FRED
FRED_HY_OAS = "BAMLH0A0HYM2"  # high-yield credit spread (stress gauge)
FRED_YC_SLOPE_10Y = "DGS10"  # for 10y - 3m slope
FRED_YC_SLOPE_3M = "DGS3MO"

# --------------------------------------------------------------------------- #
# Model settings
# --------------------------------------------------------------------------- #
# Number of regimes. 2 = Bull / Bear (recommended to keep it interpretable).
N_REGIMES = 2

# Jump penalty: higher = stickier regimes (fewer whipsaws). 0 = no penalty.
# Tune this with `regime tune` (sweeps lambda via leak-free time-series CV and
# can write the chosen value here with --write-config). Default inherited from
# the discrete JM; the CJM keeps lambda roughly on the same scale via its 1/4
# factor, so 50 is a reasonable starting point until you confirm a tuned value.
JUMP_PENALTY = 50.0

# Which signal `bear_prob` represents:
#   "cjm_nowcast"  -> the Continuous Jump Model's OWN bear probability (online
#                     inference, the paper's intended use). Default. Empirically
#                     better calibrated and far less whipsaw than the forecast.
#   "gbm_forecast" -> legacy: a gradient-boosted classifier's P(next-day regime
#                     == bear), trained on the CJM's hard labels. Kept for
#                     comparison; it hardens the CJM's continuous output and
#                     added whipsaw/miscalibration in the signal-quality study.
SIGNAL_MODE = "cjm_nowcast"

# Walk-forward settings (in trading days).
TRAIN_MIN_DAYS = 1260  # ~5 years before we make any prediction
REFIT_EVERY_DAYS = 21  # refit the full pipeline monthly (rigor over speed)

# Confidence threshold: only act when bear probability is convincingly high/low.
# Between these two bands we say "no change / hold current allocation".
BEAR_THRESHOLD = 0.60  # above this -> call it Bear
BULL_THRESHOLD = 0.40  # below this -> call it Bull

# --------------------------------------------------------------------------- #
# Re-entry / cover-short OVERLAY (opt-in, default OFF). Separate from the signal.
# --------------------------------------------------------------------------- #
# A leak-free post-processing layer that helps time the *exit of shorts / re-entry
# into longs* after a bottom. It does NOT touch `bear_prob` (the product stays a
# pure CJM nowcast); it produces a separate `bear_prob_overlay` and a plain
# re-entry flag. Validated on 7 corrections (2011–2026): never worsens re-entry
# timing, big wins on deep recoveries (COVID/GFC), silent on shallow ones.
#
# Rule: once the S&P is >= REENTRY_REBOUND above its trailing REENTRY_LOOKBACK-day
# low AND (if required) VIX is below its 21d average (fear receding), cap the bear
# reading at REENTRY_CAP — i.e. "the bounce is confirmed, stand down".
# CAVEAT: faster re-entry can mean re-entering a dead-cat bounce; this is a timing
# aid, not a guarantee. It does NOT address short-ENTRY timing (a separate gap).
# Enabled for this personal instance: it is DISPLAY-ONLY — it surfaces a
# cover-short/re-enter flag + `bear_prob_overlay` but does NOT change `bear_prob`,
# the printed stance, the allocation suggestions, the backtest, or the tuner.
REENTRY_OVERLAY = True  # set False to hide the overlay flag
REENTRY_REBOUND = 0.10  # fraction above trailing low to confirm a bounce
REENTRY_LOOKBACK = 42  # trailing-low window in trading days
REENTRY_CAP = 0.20  # cap bear_prob at this once a rebound is confirmed
REENTRY_REQUIRE_VIX = True  # also require VIX < its 21d average (fear receding)

# --------------------------------------------------------------------------- #
# Your personal allocation playbook (edit to match YOUR risk tolerance)
# --------------------------------------------------------------------------- #
# These are *suggestions* the tool prints. They are not trades.
ALLOCATION_PLAYBOOK = {
    "BULL": {
        "fidelity_401k": "Risk-on: ~90% equity index funds / ~10% bonds.",
        "thinkorswim": "Hold core equity exposure (e.g., SPY/QQQ). Full satellite risk budget.",
    },
    "NEUTRAL": {
        "fidelity_401k": "Stay the course at your baseline (~70% equity / ~30% bonds).",
        "thinkorswim": "No change. Avoid trading on a low-conviction signal.",
    },
    "BEAR": {
        "fidelity_401k": "Risk-off: shift toward ~40% equity / ~60% bonds or stable value.",
        "thinkorswim": "Reduce/hedge satellite risk. Raise cash. Consider defensive sectors.",
    },
}

# --------------------------------------------------------------------------- #
# Backtest economics (the regime SIGNAL is the product; these only affect how
# the illustrative equity-scaling overlay is scored — not the CJM detector).
# --------------------------------------------------------------------------- #
# Margin/borrow rate paid on the BORROWED sleeve when the overlay is levered
# (weight > 1). Real cost structure: equity trades are free, but margin isn't.
ANNUAL_FINANCING_RATE = 0.10
# Yield EARNED on the IDLE-CASH sleeve when defensive (weight < 1). Kept at 0%
# by default (conservative/honest): earlier the cash leg silently earned the
# full 10% margin rate, which inflated the strategy whenever it de-risked. Set
# to a small positive value (e.g. ~0.04) only for a T-bill sensitivity check.
ANNUAL_CASH_YIELD = 0.0

# --------------------------------------------------------------------------- #
# Daily digest delivery (operational use)
# --------------------------------------------------------------------------- #
# Where the morning iMessage digest is sent. Use your phone number in the form
# "+15551234567" or the Apple ID email associated with iMessage. Leave blank to
# disable sending (the digest can still be previewed with `regime digest --dry-run`).
# iMessage can only be sent from a signed-in macOS Messages app, so the digest
# job runs locally (scheduled via deploy/launchd); the hosted dashboard is separate.
IMESSAGE_RECIPIENT = ""

# --------------------------------------------------------------------------- #
# Local overrides (private / per-machine). `regime/local_settings.py` is
# gitignored; if present, any UPPER_CASE names it defines override the public
# defaults above. This keeps secrets (e.g. your phone number) out of the repo
# that publishes the dashboard.
# --------------------------------------------------------------------------- #
try:
    from . import local_settings as _local  # type: ignore

    for _name in dir(_local):
        if _name.isupper():
            globals()[_name] = getattr(_local, _name)
except Exception:
    pass
