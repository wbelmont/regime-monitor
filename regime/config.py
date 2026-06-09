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

# Extra (Yahoo, all FREE) inputs for the short-entry FRAGILITY score only. These
# are NOT fed to the CJM (the regime signal stays a pure CJM nowcast); they feed
# a separate, display-only early-warning overlay. Cached in their own parquet so
# the main raw-inputs lineage / backtest is untouched. Each is used as a rolling
# z-score of its RECENT CHANGE (not its level), so slow structural drift — e.g.
# the AI/electricity re-rating of utilities, or secular credit compression — is
# continuously re-baselined out and only "something is shifting now" registers.
FRAGILITY_TICKERS = {
    "vix3m": "^VIX3M",  # 3-month VIX (term-structure vs spot VIX); from 2006
    "vvix": "^VVIX",  # vol-of-vol (convexity/tail demand); from 2007
    "skew": "^SKEW",  # CBOE SKEW (cost of tail puts); long history
    "spy": "SPY",  # cap-weighted S&P (breadth/credit ratio denominators)
    "rsp": "RSP",  # equal-weighted S&P (breadth: RSP/SPY); from 2003
    "hyg": "HYG",  # high-yield credit ETF (credit stress); from 2007
    "lqd": "LQD",  # investment-grade credit ETF (HYG/LQD divergence)
    "xlp": "XLP",  # staples — CLEAN defensive-rotation tell (no AI tailwind)
    "xlu": "XLU",  # utilities — defensive, but AI-distorted → velocity + low wt
    "xly": "XLY",  # consumer discretionary — CYCLICAL anchor (XLP/XLY = clean,
    #               beta-neutral risk-on/off rotation; from 1998)
    "move": "^MOVE",  # ICE BofA MOVE — bond-market implied vol; LEADS equity
    #                  vol on rates/credit-led stress (e.g. +18d lead on '07 GFC
    #                  in the horse-race). Rising = stress. From 2002.
}

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
# Short-ENTRY OVERLAY (display-only) — a graded FRAGILITY SCORE.
# --------------------------------------------------------------------------- #
# Buying protection (puts / VIX calls / raising cash) has the OPPOSITE loss
# function from re-entry: being early is cheap (a little theta), being late is
# expensive (implied vol has already exploded). So this overlay is NOT a mirror
# of the re-entry confirmation gate — it is a LEADING "fragility" detector meant
# to nudge while the market still looks calm and protection is still cheap. It
# is allowed to fire with the S&P near all-time highs and VIX low.
#
# It outputs a graded 0-100% FRAGILITY SCORE (like the CJM dial) with three
# bands — WATCH / LEAN / ACT — blended from drift-robust, leak-free components:
#   * VIX term structure (VIX3M/VIX flattening → near-term fear bid),
#   * VIX velocity (spot VIX rising off a low base),
#   * VVIX (vol-of-vol; convexity/tail demand),
#   * SKEW (cost of tail puts rising),
#   * credit divergence (HYG/LQD weakening),
#   * breadth narrowing (RSP/SPY weakening),
#   * defensive rotation (XLP/SPY; XLU down-weighted + velocity-only because the
#     AI/electricity re-rating has structurally lifted utilities).
# EVERY component is a rolling z-score of its RECENT CHANGE (not its level), so
# slow structural drift is continuously re-baselined out — only "shifting now"
# registers. The score uses whatever components have data on a given date.
#
# DISPLAY-ONLY: never alters `bear_prob`, the stance, the allocation, the
# backtest, or the tuner. CAVEAT: a leading signal WILL have false positives
# (that's the point — they're cheap); treat ACT as "scale into protection",
# not "all in".
SHORT_ENTRY_OVERLAY = True  # master switch for the short-entry overlay
FRAGILITY_Z_WINDOW = 252  # rolling window (trading days) for the z-scores
FRAGILITY_K = 1.4  # logistic steepness mapping a component z-score -> 0..1
FRAGILITY_Z0 = 0.75  # z-score at which a component sub-score crosses 0.5
# Component weights (renormalized over whichever components have data). Tier-1
# vol-structure/hedging tells lead; Tier-2 divergence tells confirm. Defensive
# rotation is measured BETA-NEUTRALLY as XLP/XLY (staples vs discretionary), and
# XLU is gated by staples confirmation (it's AI-distorted, so it only counts
# when staples corroborate) — see pipeline.fragility_score.
FRAGILITY_WEIGHTS = {
    "term_structure": 0.22,  # VIX3M/VIX flattening
    "vix_velocity": 0.18,  # spot VIX rising
    "vvix": 0.12,  # vol-of-vol rising
    "skew": 0.10,  # tail-put cost rising
    "bond_vol": 0.10,  # MOVE rising — bond-market vol (leads on rates/credit stress)
    "credit": 0.16,  # HYG/LQD weakening
    "breadth": 0.12,  # RSP/SPY weakening
    "defensive_staples": 0.07,  # XLP/XLY rotation (clean, beta-neutral tell)
    "defensive_xlu": 0.03,  # utilities bid, GATED by staples (AI-distorted)
}
# Grade thresholds on the 0..1 composite. WATCH = early heads-up; LEAN = start
# scaling into protection; ACT = fragility is broad/elevated.
FRAGILITY_WATCH = 0.35
FRAGILITY_LEAN = 0.55
FRAGILITY_ACT = 0.70
# `short_entry_flag` (logged to history / lit on the dashboard) fires at >= LEAN.

# Secondary "decline-confirmed" tell (the original drawdown trigger). This is a
# LATER-stage confirmation (a decline is already underway), kept distinct from
# the leading fragility score above. Surfaced as its own boolean, not the flag.
SHORT_ENTRY_DRAWDOWN = 0.07  # fraction below trailing high = decline confirmed
SHORT_ENTRY_LOOKBACK = 63  # trailing-high window in trading days (== drawdown_63)
SHORT_ENTRY_REQUIRE_VIX = True  # also require VIX > its 21d average (fear rising)

# --------------------------------------------------------------------------- #
# Your personal allocation playbook (edit to match YOUR risk tolerance)
# --------------------------------------------------------------------------- #
# These are *suggestions* the tool prints. They are not trades. This portfolio
# holds NO bonds by design — de-risking means moving toward CASH / lower beta /
# hedges, never into fixed income. Aggressiveness is expressed as a target
# EQUITY BETA (portfolio sensitivity to the market) and, for the trading
# account, a net DELTA bias plus whether options/leverage are appropriate.
#
# The 3-way stance keys the qualitative playbook; `recommend.py` additionally
# derives a continuous target beta/delta from the bear-probability dial so the
# guidance scales smoothly within a regime (e.g. "deep BULL near 0%" is more
# aggressive than "BULL near the 40% line").
ALLOCATION_PLAYBOOK = {
    "BULL": {
        "fidelity_401k": (
            "Risk-on: 100% equity (broad index + growth tilt). No bonds. In a "
            "deep/confirmed bull (dial near 0%), full aggression is appropriate — "
            "leveraged/levered-index or call exposure acceptable. As the dial "
            "lifts toward the 40% line, stay 100% invested but DROP options/"
            "leverage and trim toward plain beta."
        ),
        "thinkorswim": (
            "Carry a long, high-beta core (SPY/QQQ + leaders). Full satellite "
            "risk budget; net delta clearly LONG. Options/leverage on the long "
            "side acceptable when the dial is near 0%."
        ),
    },
    "NEUTRAL": {
        "fidelity_401k": (
            "Hold 100% equity but DE-BETA: rotate from high-growth toward "
            "lower-beta / quality / equal-weight; no new options or leverage. "
            "Keep a little dry powder (cash from contributions). No bonds."
        ),
        "thinkorswim": (
            "Low-conviction zone — don't initiate fresh directional risk. Trim "
            "net delta toward neutral, let theta/hedges define the book. Avoid "
            "adding leverage; consider a cheap collar on the long core."
        ),
    },
    "BEAR": {
        "fidelity_401k": (
            "Risk-off WITHOUT bonds: cut equity beta — raise CASH / stable value "
            "and shift the remaining equity toward defensives (staples, min-vol). "
            "No bonds, no leverage."
        ),
        "thinkorswim": (
            "Reduce/hedge the long core; take net delta toward flat or short. "
            "Raise cash, buy protection (puts / VIX calls), or run defined-risk "
            "shorts. This is the account that expresses the bearish view."
        ),
    },
}

# Continuous aggressiveness targets derived from the bear-probability dial. The
# tool interpolates a target equity BETA between these anchors (1.0 = market;
# >1 = leveraged/high-beta tilt; <1 = de-risked). Net DELTA bias (trading book)
# follows the same shape. No bonds anywhere — the low-aggression end is CASH.
TARGET_BETA_MAX = 1.30  # at bear_prob = 0% (deep bull): leaned-in, leverage OK
TARGET_BETA_MIN = 0.15  # at bear_prob = 100% (deep bear): mostly cash + defensives
# Below this dial reading, options/leverage are considered appropriate (deep,
# confirmed bull only); above it, stay long but plain-beta (no options/leverage).
LEVERAGE_OK_BELOW = 0.15

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
