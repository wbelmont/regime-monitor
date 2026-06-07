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
    # Short-ENTRY overlay (a future, separate layer — mirror of the re-entry
    # overlay). Passed through when the signal provides it so the dashboard and
    # history can track it as its own signal; absent/0 until that overlay lands.
    if "short_entry_flag" in signal:
        rec["short_entry_flag"] = signal["short_entry_flag"]
    return rec
