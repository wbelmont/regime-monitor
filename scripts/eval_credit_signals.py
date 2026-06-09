"""Horse-race: do JPY carry-unwind and/or the MOVE index earn a spot in the
short-entry FRAGILITY composite?

Same cheap, leak-free, pure-pandas methodology as scripts/eval_short_signal.py
(no CJM walk-forward). Baseline = the CURRENT LIVE composite (candidate "C":
beta-neutral XLP/XLY staples + XLU gated by staples). We test adding:

  * MOVE  (^MOVE, ICE BofA bond-market implied vol): rising = stress. Bond vol
    often LEADS equity vol, so it could improve LEAD time.
  * JPYs  (yen-strength velocity): USD/JPY (JPY=X) FALLING fast = yen
    appreciating = carry-trade unwind = forced global de-risking.
  * JPYv  (yen realized-vol spike): rolling vol of USD/JPY returns rising — the
    Aug-2024 carry-blowup signature (violent, direction-agnostic).

Reported: lead-to-LEAN before each known peak, hit rate, calm-period
false-positive rate, ALL split by era (2007-2019 vs 2020-2026), PLUS a
STANDALONE section (each new tell alone) so we can tell raw signal quality from
mere composite dilution. Thresholds held FIXED (compare designs, not thresholds).
This experiment is ISOLATED: it does NOT modify config.py / pipeline.py. If a
tell wins, we wire it in afterward.

Run:  PYTHONPATH=. .venv/bin/python scripts/eval_credit_signals.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime import config, data, features
from regime.pipeline import _logistic, _roll_z

CHG = 5
ERA_SPLIT = pd.Timestamp("2020-01-01")
LEAD_WINDOW = 120
TH = config.FRAGILITY_LEAN  # score at the firing bar

PEAKS = {
    "2007 top (GFC)": "2007-10-09",
    "2011 (debt ceiling)": "2011-04-29",
    "2015-16": "2015-11-03",
    "2018 Q4": "2018-09-20",
    "2020 (COVID)": "2020-02-19",
    "2022 bear": "2022-01-03",
    "2025-26": "2025-12-15",
}
CALM = {
    "2013": ("2013-02-01", "2013-12-15"),
    "2014": ("2014-03-01", "2014-09-15"),
    "2017": ("2017-02-01", "2017-12-31"),
    "2019": ("2019-03-01", "2019-12-15"),
    "2021": ("2021-04-01", "2021-12-31"),
    "2023-24": ("2023-06-01", "2024-06-30"),
}


def _stress(raw: pd.Series, window: int, rising_is_stress: bool) -> pd.Series:
    z = _roll_z(raw - raw.shift(CHG), window)
    return z if rising_is_stress else -z


def _sub(z: pd.Series) -> pd.Series:
    return _logistic(z, config.FRAGILITY_K, config.FRAGILITY_Z0)


def base_subscores(extra: pd.DataFrame, feat: pd.DataFrame) -> dict[str, pd.Series]:
    """Candidate-C sub-scores (the current LIVE design)."""
    window = config.FRAGILITY_Z_WINDOW
    e = extra.sort_index().ffill()
    vix = feat["vix"].reindex(e.index).ffill()
    s: dict[str, pd.Series] = {}
    if "vix3m" in e:
        s["term_structure"] = _sub(
            _stress(e["vix3m"] / vix.replace(0, np.nan), window, False)
        )
    s["vix_velocity"] = _sub(_stress(vix, window, True))
    if "vvix" in e:
        s["vvix"] = _sub(_stress(e["vvix"], window, True))
    if "skew" in e:
        s["skew"] = _sub(_stress(e["skew"], window, True))
    if {"hyg", "lqd"}.issubset(e.columns):
        s["credit"] = _sub(_stress(e["hyg"] / e["lqd"], window, False))
    if {"rsp", "spy"}.issubset(e.columns):
        s["breadth"] = _sub(_stress(e["rsp"] / e["spy"], window, False))
    # candidate C defensive block
    staples = _sub(_stress(e["xlp"] / e["xly"], window, True))
    xlu = _sub(_stress(e["xlu"] / e["spy"], window, True))
    s["def_staples"] = staples
    s["def_xlu"] = xlu * staples  # gated by staples confirmation
    return s


def base_weights() -> dict[str, float]:
    w = {
        k: v
        for k, v in config.FRAGILITY_WEIGHTS.items()
        if k not in ("defensive_xlp", "defensive_xlu")
    }
    w["def_staples"] = 0.07
    w["def_xlu"] = 0.03
    return w


def composite(subs: dict[str, pd.Series], weights: dict[str, float], idx) -> pd.Series:
    num = pd.Series(0.0, index=idx)
    den = pd.Series(0.0, index=idx)
    for name, s in subs.items():
        s = s.reindex(idx)
        wt = float(weights.get(name, 0.0))
        valid = s.notna()
        num = num.add((s.fillna(0.0) * wt).where(valid, 0.0), fill_value=0.0)
        den = den.add(pd.Series(np.where(valid, wt, 0.0), index=idx), fill_value=0.0)
    return (num / den.replace(0.0, np.nan)).clip(0.0, 1.0)


def lead_days(comp: pd.Series, peak: str) -> float:
    p = pd.Timestamp(peak)
    win = comp[
        (comp.index >= p - pd.Timedelta(days=int(LEAD_WINDOW * 1.5)))
        & (comp.index <= p)
    ]
    armed = win[win >= TH]
    return float("nan") if armed.empty else float((p - armed.index[0]).days)


def calm_fp(comp: pd.Series, s: str, e: str) -> float:
    win = comp[
        (comp.index >= pd.Timestamp(s)) & (comp.index <= pd.Timestamp(e))
    ].dropna()
    return float("nan") if win.empty else float((win >= TH).mean())


def report(name: str, comps: dict[str, pd.Series]) -> None:
    print(f"\n############ {name} ############")
    print("=== LEAD to LEAN (days before peak; higher=earlier) ===")
    print(f"{'peak':<22}" + "".join(f"{c:>10}" for c in comps))
    hits = {c: {"pre": [0, 0], "post": [0, 0]} for c in comps}
    for nm, pk in PEAKS.items():
        era = "post" if pd.Timestamp(pk) >= ERA_SPLIT else "pre"
        row = f"{nm:<22}"
        for c in comps:
            ld = lead_days(comps[c], pk)
            row += f"{('' if np.isnan(ld) else int(ld)):>10}"
            hits[c][era][1] += 1
            hits[c][era][0] += 0 if np.isnan(ld) else 1
        print(row)
    print("--- hit rate by era ---")
    for era, lbl in (("pre", "2007-2019"), ("post", "2020-2026")):
        print(
            f"{lbl:<22}"
            + "".join(f"{f'{hits[c][era][0]}/{hits[c][era][1]}':>10}" for c in comps)
        )
    print("--- calm FP rate by era (lower=better) ---")
    for era, lbl, wins in (
        (
            "pre",
            "2007-2019",
            [v for v in CALM.values() if pd.Timestamp(v[0]) < ERA_SPLIT],
        ),
        (
            "post",
            "2020-2026",
            [v for v in CALM.values() if pd.Timestamp(v[0]) >= ERA_SPLIT],
        ),
    ):
        row = f"{lbl:<22}"
        for c in comps:
            r = [calm_fp(comps[c], s, e) for s, e in wins]
            r = [x for x in r if not np.isnan(x)]
            row += f"{(f'{np.mean(r):.0%}' if r else '-'):>10}"
        print(row)


def main() -> None:
    feat = features.build_features(data.load_raw(refresh=False))
    extra = data.load_extra(refresh=False)
    idx = extra.index
    window = config.FRAGILITY_Z_WINDOW

    # New tells (fetched here; NOT persisted to config).
    usdjpy = (
        data._yahoo_close("JPY=X", config.START_DATE, data._today())
        .reindex(idx)
        .ffill()
    )
    move = (
        data._yahoo_close("^MOVE", config.START_DATE, data._today())
        .reindex(idx)
        .ffill()
    )
    jpy_ret = usdjpy.pct_change()
    sub_move = _sub(_stress(move, window, True))
    sub_jpys = _sub(
        _stress(usdjpy, window, False)
    )  # USD/JPY falling = yen strength = stress
    sub_jpyv = _sub(_stress(jpy_ret.rolling(10).std(), window, True))  # yen vol spike

    base = base_subscores(extra, feat)
    bw = base_weights()
    C = composite(base, bw, idx)

    def aug(extra_subs: dict[str, pd.Series], extra_w: dict[str, float]) -> pd.Series:
        return composite({**base, **extra_subs}, {**bw, **extra_w}, idx)

    # ---- composite augmentation horse-race ----
    candidates = {
        "C (live)": C,
        "C+MOVE": aug({"move": sub_move}, {"move": 0.10}),
        "C+JPYs": aug({"jpys": sub_jpys}, {"jpys": 0.06}),
        "C+JPYv": aug({"jpyv": sub_jpyv}, {"jpyv": 0.06}),
        "C+MOVE+JPYs": aug(
            {"move": sub_move, "jpys": sub_jpys}, {"move": 0.10, "jpys": 0.06}
        ),
    }
    report("COMPOSITE AUGMENTATION (does adding the tell help C?)", candidates)

    # ---- standalone raw quality (each tell alone, scored at LEAN) ----
    standalone = {
        "MOVE": sub_move,
        "JPYstrength": sub_jpys,
        "JPYvol": sub_jpyv,
        "credit(HYG/LQD)": base["credit"],
        "VIXvel": base["vix_velocity"],
    }
    report("STANDALONE (single-tell lead — raw quality, not composite)", standalone)


if __name__ == "__main__":
    main()
