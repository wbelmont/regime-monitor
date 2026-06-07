"""Full-rigor walk-forward runner with live progress, cached to parquet.

Runs the leak-free walk-forward at PRODUCTION settings (the way the signal is
actually used: refit_every=21, n_init=10, max_iter=50) and also captures the
CJM's own bear-probability nowcast in the same pass (`return_nowcast=True`), so
the signal-quality harness can compare the GBM forecast vs the pure CJM nowcast
without a second ~30-45 min run.

Designed to be run in the background and watched via its log:

    PYTHONPATH=. .venv/bin/python scripts/run_walkforward.py 2>&1 | tee reports/walkforward.log

Progress prints every few windows with elapsed time, ETA, and the current test
date so it is obvious the run is alive (a prior full run looked "hung" at ~45m).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regime import config, data, features, pipeline  # noqa: E402

# Production / "as actually used" settings.
N_INIT = 10
MAX_ITER = 50
REFIT_EVERY = None  # None -> config.REFIT_EVERY_DAYS (21), the production cadence
PRINT_EVERY = 5  # windows between progress prints


def main() -> None:
    t0 = time.time()
    print(
        f"[{time.strftime('%H:%M:%S')}] loading cached data (no refresh)...", flush=True
    )
    raw = data.load_raw(refresh=False)
    feat = features.build_features(raw)
    print(
        f"[{time.strftime('%H:%M:%S')}] feat: {len(feat):,} rows, "
        f"{feat.index.min().date()} -> {feat.index.max().date()}",
        flush=True,
    )
    print(
        f"settings: jump_penalty={config.JUMP_PENALTY:g}, "
        f"refit_every={config.REFIT_EVERY_DAYS}, n_init={N_INIT}, "
        f"max_iter={MAX_ITER}, train_min={config.TRAIN_MIN_DAYS}",
        flush=True,
    )

    # Map each progress step to the test-block date so the log shows WHERE we are.
    reg_cols = features.available(feat, features.REGIME_FEATURES)
    pred_cols = features.available(feat, features.PREDICTOR_FEATURES)
    df = feat.dropna(subset=reg_cols + pred_cols + ["mkt_ret"])
    start = config.TRAIN_MIN_DAYS
    step = config.REFIT_EVERY_DAYS
    step_dates = [df.index[i].date() for i in range(start, len(df), step)]

    def on_step(done: int, total: int) -> None:
        if done == 1 or done % PRINT_EVERY == 0 or done == total:
            elapsed = time.time() - t0
            rate = elapsed / done
            eta = rate * (total - done)
            where = step_dates[done - 1] if done - 1 < len(step_dates) else "?"
            print(
                f"[{time.strftime('%H:%M:%S')}] window {done:>3}/{total} "
                f"({100 * done / total:4.0f}%) | at {where} | "
                f"{elapsed / 60:5.1f}m elapsed | ~{eta / 60:4.1f}m left | "
                f"{rate:4.1f}s/window",
                flush=True,
            )

    print(f"[{time.strftime('%H:%M:%S')}] starting walk-forward...", flush=True)
    signals = pipeline.walk_forward(
        feat,
        progress=on_step,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        refit_every=REFIT_EVERY,
        return_nowcast=True,
    )

    out_path = config.CACHE_DIR / f"signals_jp{config.JUMP_PENALTY:g}_full.parquet"
    signals.to_parquet(out_path)
    print(
        f"[{time.strftime('%H:%M:%S')}] DONE in {(time.time() - t0) / 60:.1f}m. "
        f"{len(signals):,} rows, cols={list(signals.columns)} -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
