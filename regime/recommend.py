"""Turn the model's probability into a plain-English, account-specific suggestion.

This is the layer that makes the tool *usable*. It applies your confidence
thresholds (config) so you don't react to low-conviction noise, and prints what
to consider doing in each account.

IMPORTANT: these are suggestions for YOU to review and act on manually. The tool
does not place trades. This is not financial advice.
"""

from __future__ import annotations

from . import config


def classify(next_bear_prob: float) -> str:
    """Map probability -> BULL / NEUTRAL / BEAR using your thresholds."""
    if next_bear_prob >= config.BEAR_THRESHOLD:
        return "BEAR"
    if next_bear_prob <= config.BULL_THRESHOLD:
        return "BULL"
    return "NEUTRAL"


def exposure_targets(next_bear_prob: float) -> dict:
    """Continuous aggressiveness targets derived from the dial.

    Interpolates a target equity BETA between ``config.TARGET_BETA_MAX`` (dial
    0%) and ``config.TARGET_BETA_MIN`` (dial 100%), a suggested net DELTA bias
    for the trading book (same shape, signed), and whether options/leverage are
    appropriate (only in a deep/confirmed bull). No bonds — the low end is cash.
    """
    p = max(0.0, min(1.0, float(next_bear_prob)))
    beta = config.TARGET_BETA_MAX + (config.TARGET_BETA_MIN - config.TARGET_BETA_MAX) * p
    # Net delta bias: long when risk-on, flat/short as the dial climbs. Centered
    # so it crosses zero around the bear threshold and goes net-short beyond it.
    delta = round((config.BEAR_THRESHOLD - p) / config.BEAR_THRESHOLD, 2)
    delta = max(-1.0, min(1.0, delta))
    return {
        "target_beta": round(beta, 2),
        "net_delta": delta,
        "leverage_ok": p <= config.LEVERAGE_OK_BELOW,
    }


def build_recommendation(signal: dict) -> dict:
    stance = classify(signal["next_bear_prob"])
    playbook = config.ALLOCATION_PLAYBOOK[stance]
    rec = {
        "as_of": signal["as_of"],
        "stance": stance,
        "next_bear_prob": signal["next_bear_prob"],
        "current_regime": "Bear" if signal["current_regime"] == 1 else "Bull",
        # Numeric 0/1 form of the hard regime label, kept distinct from the
        # continuous probability and the 3-way stance, so the dashboard can plot
        # the binary Bull/Bear call as its OWN layer.
        "regime_binary": int(signal["current_regime"]),
        "fidelity_401k": playbook["fidelity_401k"],
        "thinkorswim": playbook["thinkorswim"],
        "exposure": exposure_targets(signal["next_bear_prob"]),
        "top_drivers": list(signal.get("feature_importances", {}).items())[:5],
    }
    # CJM per-feature attribution (why today leans bear/bull). Present in both
    # signal modes; the display layer decides how many to show.
    if signal.get("drivers"):
        rec["drivers"] = signal["drivers"]
    # Opt-in re-entry / cover-short overlay (separate from the stance above).
    if "reentry_flag" in signal:
        rec["reentry_flag"] = signal["reentry_flag"]
        rec["bear_prob_overlay"] = signal.get("bear_prob_overlay")
        if "reentry_diag" in signal:
            rec["reentry_diag"] = signal["reentry_diag"]
    # Short-ENTRY overlay (a future, separate layer — mirror of the re-entry
    # overlay). Passed through when the signal provides it so the dashboard and
    # history can track it as its own signal; absent/0 until that overlay lands.
    if "short_entry_flag" in signal:
        rec["short_entry_flag"] = signal["short_entry_flag"]
    # Graded short-entry FRAGILITY score (the leading early-warning) + its
    # component attribution, plus the later-stage decline-confirmed tell.
    if "fragility_score" in signal:
        rec["fragility_score"] = signal["fragility_score"]
        rec["fragility_grade"] = signal.get("fragility_grade", "none")
        rec["fragility_drivers"] = signal.get("fragility_drivers", [])
        rec["fragility_pctile"] = signal.get("fragility_pctile")
        rec["fragility_pctiles"] = signal.get("fragility_pctiles", {})
    if "decline_confirmed" in signal:
        rec["decline_confirmed"] = signal["decline_confirmed"]
    return rec
