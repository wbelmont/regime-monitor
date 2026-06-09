"""Horse-race for the SHORT-ENTRY fragility overlay's DEFENSIVE-ROTATION tell.

Question being answered: "What is the best-performing way to measure defensive
rotation for the leading short signal — and has it degraded as market structure
changed since ~2020?" The utilities (XLU) tell is suspected to be contaminated
by the AI/electricity power-demand re-rating (utilities now rise for a RISK-ON
reason), so we test cleaner, beta-neutral and confirmation-gated alternatives.

This is CHEAP: the fragility score is pure pandas (NO CJM walk-forward). It
mirrors `pipeline.fragility_score` exactly for the shared components (same
`_roll_z`, `_logistic`, window, k, z0, weights, thresholds) and only swaps the
DEFENSIVE components per candidate, so the comparison is apples-to-apples. It
leaves `pipeline.py` / `config.py` UNCHANGED — we implement only the winner.

Candidates (defensive block only; all other components identical):
  A  current:        XLP/SPY  + XLU/SPY                 (weights 0.07 / 0.03)
  B  beta-neutral:   XLP/XLY  + XLU/SPY                 (staples vs cyclicals)
  C  B + XLU gated:  XLU sub-score *= staples sub-score (utilities only counts
                     when staples ALSO rotate defensive -> kills AI-only pops)
  D  C + risk-on veto: whole defensive block dampened when NO other stress
                     (credit/breadth/vix) corroborates.

Scoring (protection loss fn: early is cheap, late is expensive):
  * lead = trading days the composite first reaches WATCH BEFORE each known peak
    (positive = early = good; NaN = never armed in the 120d before the peak).
  * hit rate = fraction of peaks armed (WATCH+) before the peak.
  * calm FP rate = fraction of days at WATCH+ during designated CALM windows
    (cheap false alarms, but still a cost to compare).
  All reported OVERALL and split by ERA (2007-2019 vs 2020-2026) to test the
  "structure changed" hypothesis. Thresholds are held FIXED (we compare designs,
  not re-tuned thresholds).

Run:  PYTHONPATH=. .venv/bin/python scripts/eval_short_signal.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime import config, data, features
from regime.pipeline import _logistic, _roll_z

CHG = 5  # measure the recent CHANGE over a trading week, then z-score it
ERA_SPLIT = pd.Timestamp("2020-01-01")
LEAD_WINDOW = 120  # trading days before a peak to look for the first WATCH arm

# Known local market PEAKS preceding a correction/bear (same set as
# eval_features.py, minus dot-com: the vol/credit components didn't exist pre-2007).
PEAKS = {
    "2007 top (GFC)": "2007-10-09",
    "2011 (debt ceiling)": "2011-04-29",
    "2015-16": "2015-11-03",
    "2018 Q4": "2018-09-20",
    "2020 (COVID)": "2020-02-19",
    "2022 bear": "2022-01-03",
    "2025-26": "2025-12-15",
}
# Designated CALM windows (no major drawdown began here) for false-positive rate.
CALM = {
    "2013 calm": ("2013-02-01", "2013-12-15"),
    "2014 calm": ("2014-03-01", "2014-09-15"),
    "2017 calm": ("2017-02-01", "2017-12-31"),
    "2019 calm": ("2019-03-01", "2019-12-15"),
    "2021 calm": ("2021-04-01", "2021-12-31"),
    "2023-24 calm": ("2023-06-01", "2024-06-30"),
}


def _stress(raw: pd.Series, window: int, rising_is_stress: bool) -> pd.Series:
    delta = raw - raw.shift(CHG)
    z = _roll_z(delta, window)
    return z if rising_is_stress else -z


def build_composite(extra: pd.DataFrame, feat: pd.DataFrame, candidate: str) -> pd.Series:
    """Return the 0..1 fragility composite for a given defensive-design candidate."""
    window = config.FRAGILITY_Z_WINDOW
    k, z0 = config.FRAGILITY_K, config.FRAGILITY_Z0
    w = dict(config.FRAGILITY_WEIGHTS)  # base weights for shared components

    e = extra.sort_index().ffill()
    vix = feat["vix"].reindex(e.index).ffill()

    def sub(z: pd.Series) -> pd.Series:
        return _logistic(z, k, z0)

    # --- shared components (identical across candidates) ---
    comp_sub: dict[str, pd.Series] = {}
    if "vix3m" in e:
        comp_sub["term_structure"] = sub(
            _stress(e["vix3m"] / vix.replace(0.0, np.nan), window, False)
        )
    comp_sub["vix_velocity"] = sub(_stress(vix, window, True))
    if "vvix" in e:
        comp_sub["vvix"] = sub(_stress(e["vvix"], window, True))
    if "skew" in e:
        comp_sub["skew"] = sub(_stress(e["skew"], window, True))
    if {"hyg", "lqd"}.issubset(e.columns):
        comp_sub["credit"] = sub(_stress(e["hyg"] / e["lqd"], window, False))
    if {"rsp", "spy"}.issubset(e.columns):
        comp_sub["breadth"] = sub(_stress(e["rsp"] / e["spy"], window, False))

    # --- defensive block (varies by candidate) ---
    weights = {kk: vv for kk, vv in w.items() if kk in comp_sub}
    DEF_STAPLES_W = 0.07
    DEF_XLU_W = 0.03

    if candidate == "A":  # current
        comp_sub["def_staples"] = sub(_stress(e["xlp"] / e["spy"], window, True))
        comp_sub["def_xlu"] = sub(_stress(e["xlu"] / e["spy"], window, True))
    else:  # B/C/D share the beta-neutral staples tell
        staples = sub(_stress(e["xlp"] / e["xly"], window, True))  # XLP/XLY
        xlu = sub(_stress(e["xlu"] / e["spy"], window, True))
        comp_sub["def_staples"] = staples
        if candidate == "B":
            comp_sub["def_xlu"] = xlu
        else:  # C and D: gate XLU by staples confirmation
            comp_sub["def_xlu"] = xlu * staples  # only counts when staples confirm
    weights["def_staples"] = DEF_STAPLES_W
    weights["def_xlu"] = DEF_XLU_W

    sub_df = pd.DataFrame(comp_sub)

    # --- weighted average over available components ---
    num = pd.Series(0.0, index=e.index)
    den = pd.Series(0.0, index=e.index)
    for name, s in sub_df.items():
        wt = float(weights.get(str(name), 0.0))
        valid = s.notna()
        num = num.add((s.fillna(0.0) * wt).where(valid, 0.0), fill_value=0.0)
        den = den.add(pd.Series(np.where(valid, wt, 0.0), index=e.index), fill_value=0.0)
    composite = (num / den.replace(0.0, np.nan)).clip(0.0, 1.0)

    if candidate == "D":  # risk-on veto: dampen if no corroborating stress
        corrob = pd.concat(
            [comp_sub.get("credit"), comp_sub.get("breadth"), comp_sub.get("vix_velocity")],
            axis=1,
        ).max(axis=1)
        # Soft veto: floor at 0.4 so corroboration scales 0.4..1.0 (never fully zeroed).
        veto = (0.4 + 0.6 * corrob.clip(0, 1)).reindex(composite.index).fillna(1.0)
        composite = (composite * veto).clip(0.0, 1.0)

    return composite


def lead_days(comp: pd.Series, peak: str, thresh: float) -> float:
    peak_ts = pd.Timestamp(peak)
    win = comp[(comp.index >= peak_ts - pd.Timedelta(days=int(LEAD_WINDOW * 1.5)))
               & (comp.index <= peak_ts)]
    armed = win[win >= thresh]
    if armed.empty:
        return float("nan")
    return float((peak_ts - armed.index[0]).days)


def calm_fp_rate(comp: pd.Series, start: str, end: str, thresh: float) -> float:
    win = comp[(comp.index >= pd.Timestamp(start)) & (comp.index <= pd.Timestamp(end))]
    win = win.dropna()
    if win.empty:
        return float("nan")
    return float((win >= thresh).mean())


def main() -> None:
    feat = features.build_features(data.load_raw(refresh=False))
    extra = data.load_extra(refresh=False)
    if "xly" not in extra.columns:
        raise SystemExit("XLY missing from extra cache — run load_extra(refresh=True).")

    comps = {c: build_composite(extra, feat, c) for c in ("A", "B", "C", "D")}
    last = min(c.dropna().index.max() for c in comps.values())
    TH = config.FRAGILITY_LEAN  # score at the bar that actually FIRES the flag
    print(f"data through {last.date()} | scoring at LEAN={TH} "
          f"(the short_entry_flag bar) | WATCH={config.FRAGILITY_WATCH} "
          f"ACT={config.FRAGILITY_ACT}\n")

    # --- lead time to each peak ---
    print("=== LEAD (days the composite first hit LEAN before the peak; higher=earlier) ===")
    hdr = f"{'peak':<22}" + "".join(f"{c:>8}" for c in comps)
    print(hdr)
    era_hits = {c: {"pre": [0, 0], "post": [0, 0]} for c in comps}  # [hits, total]
    for name, peak in PEAKS.items():
        row = f"{name:<22}"
        era = "post" if pd.Timestamp(peak) >= ERA_SPLIT else "pre"
        for c in comps:
            ld = lead_days(comps[c], peak, TH)
            row += f"{('' if np.isnan(ld) else int(ld)):>8}"
            era_hits[c][era][1] += 1
            if not np.isnan(ld):
                era_hits[c][era][0] += 1
        print(row)

    print("\n=== HIT RATE by era (reached LEAN before the peak) ===")
    print(f"{'era':<14}" + "".join(f"{c:>8}" for c in comps))
    for era, lbl in (("pre", "2007-2019"), ("post", "2020-2026")):
        row = f"{lbl:<14}"
        for c in comps:
            h, t = era_hits[c][era]
            row += f"{(f'{h}/{t}'):>8}"
        print(row)

    print("\n=== CALM-PERIOD FALSE-POSITIVE RATE (fraction of days at LEAN+; lower=better) ===")
    print(f"{'calm window':<16}" + "".join(f"{c:>8}" for c in comps))
    for name, (s, e) in CALM.items():
        row = f"{name:<16}"
        for c in comps:
            fp = calm_fp_rate(comps[c], s, e, TH)
            row += f"{('' if np.isnan(fp) else f'{fp:.0%}'):>8}"
        print(row)

    # overall calm FP by era
    print("\n=== CALM FP RATE by era (avg over calm windows) ===")
    print(f"{'era':<14}" + "".join(f"{c:>8}" for c in comps))
    for era, lbl, wins in (
        ("pre", "2007-2019", [v for k, v in CALM.items() if pd.Timestamp(v[0]) < ERA_SPLIT]),
        ("post", "2020-2026", [v for k, v in CALM.items() if pd.Timestamp(v[0]) >= ERA_SPLIT]),
    ):
        row = f"{lbl:<14}"
        for c in comps:
            rates = [calm_fp_rate(comps[c], s, e, TH) for s, e in wins]
            rates = [r for r in rates if not np.isnan(r)]
            row += f"{(f'{np.mean(rates):.0%}' if rates else '-'):>8}"
        print(row)


if __name__ == "__main__":
    main()
