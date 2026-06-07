"""Coarse, CHEAP signal-quality A/B: baseline REGIME_FEATURES vs the new set
(adds drawdown_63, downside_dev_21, curve_slope).

This is an ITERATION tool, not a conclusion: short OOS window, fewer CJM inits,
a larger refit step — fast enough to read in a couple of minutes. It judges the
SIGNAL (not P&L), focused on the open gap: short-ENTRY timeliness on grinding
declines, plus whipsaw. Confirm any winner later at full rigor.

Run:  PYTHONPATH=. .venv/bin/python scripts/eval_features.py
"""

from __future__ import annotations

import pandas as pd

from regime import data, features, pipeline

# Known local market PEAKS that preceded a correction/bear (S&P). Entry lag =
# trading days from the peak until bear_prob first crosses ENTER_THRESHOLD.
PEAKS = {
    "2007 top (GFC)": "2007-10-09",
    "2011 (debt ceiling)": "2011-04-29",
    "2015-16": "2015-11-03",
    "2018 Q4": "2018-09-20",
    "2020 (COVID)": "2020-02-19",
    "2022 bear": "2022-01-03",
    "2025-26": "2025-12-15",
}
ENTER_THRESHOLD = 0.60  # config.BEAR_THRESHOLD

# Fast/coarse settings (iteration only — NOT full rigor).
N_INIT = 3
REFIT_EVERY = 63  # ~quarterly instead of 21
MAX_OOS_DAYS = None  # full OOS span so we hit the older peaks; coarse via above


def entry_lag(bear_prob: pd.Series, peak: str) -> int | float:
    peak_ts = pd.Timestamp(peak)
    after = bear_prob[bear_prob.index >= peak_ts]
    crossed = after[after >= ENTER_THRESHOLD]
    if crossed.empty:
        return float("nan")
    return int((crossed.index[0] - peak_ts).days)


def transitions_per_year(bear_prob: pd.Series) -> float:
    hard = (bear_prob >= 0.5).astype(int)
    n_trans = int((hard.diff().abs() > 0).sum())
    years = (bear_prob.index[-1] - bear_prob.index[0]).days / 365.25
    return n_trans / max(years, 1e-9)


def run_for(feat: pd.DataFrame, reg_cols: list[str], label: str) -> pd.Series:
    orig = features.REGIME_FEATURES
    features.REGIME_FEATURES = reg_cols  # monkeypatch the labeler's feature set
    try:
        wf = pipeline.walk_forward(
            feat,
            n_init=N_INIT,
            refit_every=REFIT_EVERY,
            max_oos_days=MAX_OOS_DAYS,
            signal_mode="cjm_nowcast",
        )
    finally:
        features.REGIME_FEATURES = orig
    bp = wf["bear_prob"]
    print(f"\n=== {label} ({len(reg_cols)} feats, OOS {bp.index.min().date()}..{bp.index.max().date()}) ===")
    print(f"  transitions/yr: {transitions_per_year(bp):.2f}")
    print("  entry lag (days from peak to P(bear)>=0.60):")
    for name, peak in PEAKS.items():
        lag = entry_lag(bp, peak)
        print(f"    {name:<22} {lag}")
    return bp


def main() -> None:
    raw = data.load_raw(refresh=False)
    feat = features.build_features(raw)

    base_cols = features.available(feat, features.REGIME_FEATURES_BASELINE)
    new_cols = features.available(feat, features.REGIME_FEATURES_EXPERIMENTAL)
    print("baseline feats:", base_cols)
    print("new feats:     ", new_cols)

    bp_base = run_for(feat, base_cols, "BASELINE")
    bp_new = run_for(feat, new_cols, "NEW (downside+curve)")

    print("\n=== summary: entry-lag delta (NEW - BASELINE; negative = faster) ===")
    for name, peak in PEAKS.items():
        lb, ln = entry_lag(bp_base, peak), entry_lag(bp_new, peak)
        delta = (ln - lb) if (lb == lb and ln == ln) else float("nan")
        print(f"  {name:<22} base={lb}  new={ln}  delta={delta}")


if __name__ == "__main__":
    main()
