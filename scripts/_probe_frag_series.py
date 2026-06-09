"""One-off probe: confirm the fragility series is dense (multi-day) for the
dashboard chart. Cached data only; writes a short summary to stdout and exits.

Run: PYTHONPATH=. .venv/bin/python scripts/_probe_frag_series.py
"""

from __future__ import annotations

from regime import data, features, pipeline


def main() -> None:
    feat = features.build_features(data.load_raw(refresh=False))
    extra = data.load_extra(refresh=False)
    fr = pipeline.fragility_score(extra, feat, index=feat.index)
    cols = [c for c in fr.columns if c not in ("fragility", "grade")]
    tail = fr.tail(10)
    print(f"rows={len(fr)} last={fr.index[-1].date()} n_components={len(cols)}")
    print("components:", ",".join(cols))
    comp = (tail["fragility"] * 100).round(0)
    print("last10_composite_pct:", list(comp.astype("float").tolist()))


if __name__ == "__main__":
    main()
