"""Throwaway probe: time each step of the fragility wiring and print results.

Run: PYTHONPATH=. .venv/bin/python scripts/_probe_fragility.py > reports/_probe.log 2>&1
"""

import time
import warnings

warnings.filterwarnings("ignore")

from regime import data, features, pipeline  # noqa: E402


def stamp(label, t0):
    print(f"{label}: {time.time() - t0:.1f}s", flush=True)
    return time.time()


t = time.time()
print("start", flush=True)

feat = features.build_features(data.load_raw(refresh=False))
t = stamp("feat", t)

extra = data.load_extra(refresh=False)
print("extra shape:", extra.shape, flush=True)
t = stamp("extra", t)

sig = pipeline.latest_signal(feat, extra=extra)
t = stamp("latest_signal", t)

for k in (
    "fragility_score",
    "fragility_grade",
    "short_entry_flag",
    "decline_confirmed",
):
    print(k, "=", sig.get(k), flush=True)
print("drivers:", sig.get("fragility_drivers"), flush=True)
print("done", flush=True)
