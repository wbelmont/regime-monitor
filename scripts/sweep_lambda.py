"""Sweep the jump penalty (lambda) and cache each CJM-nowcast signal.

Re-tuning lambda IS the re-entry-lag lever: lambda sets how sticky the regime
path is. Higher lambda -> stickier (less whipsaw, but slower to flip both into
AND out of bear, i.e. slower re-entry). Lower lambda -> more responsive (faster
re-entry, more whipsaw). This script runs one leak-free walk-forward per lambda
(production signal = CJM nowcast) and caches the result so the notebook harness
can score re-entry lag / whipsaw / calibration across the grid without re-running.

Usage (background + live log):

    PYTHONPATH=. .venv/bin/python scripts/sweep_lambda.py \
        --grid 5,12.5,25,50,100 --refit-every 63 --n-init 5 --max-iter 30 \
        2>&1 | tee reports/lambda_sweep.log

Defaults are a FAST/coarse preset for cheap directional iteration; confirm the
chosen lambda at full rigor (refit_every=21, n_init=10) before adopting it.
Each cached file: data/cache/signals_jp{lambda}_sweep.parquet.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regime import config, data, features, pipeline  # noqa: E402

PRINT_EVERY = 10  # windows between progress prints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--grid", default="5,12.5,25,50,100", help="comma-separated jump penalties"
    )
    p.add_argument("--refit-every", type=int, default=63)
    p.add_argument("--n-init", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument(
        "--suffix",
        default="sweep",
        help="cache filename suffix: signals_jp{lam}_{suffix}.parquet",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grid = [float(x) for x in args.grid.split(",") if x.strip()]
    t_all = time.time()

    print(
        f"[{time.strftime('%H:%M:%S')}] loading cached data (no refresh)...", flush=True
    )
    raw = data.load_raw(refresh=False)
    feat = features.build_features(raw)
    print(
        f"[{time.strftime('%H:%M:%S')}] feat: {len(feat):,} rows, "
        f"{feat.index.min().date()} -> {feat.index.max().date()}\n"
        f"grid={grid}  refit_every={args.refit_every}  n_init={args.n_init}  "
        f"max_iter={args.max_iter}  signal_mode={config.SIGNAL_MODE}",
        flush=True,
    )

    for li, lam in enumerate(grid, start=1):
        t0 = time.time()

        def on_step(done: int, total: int, _lam=lam, _li=li, _t0=t0) -> None:
            if done == 1 or done % PRINT_EVERY == 0 or done == total:
                elapsed = time.time() - _t0
                eta = (elapsed / done) * (total - done)
                print(
                    f"[{time.strftime('%H:%M:%S')}] lambda {_lam:>6g} "
                    f"({_li}/{len(grid)}) | window {done:>3}/{total} "
                    f"({100 * done / total:3.0f}%) | {elapsed / 60:4.1f}m | "
                    f"~{eta / 60:4.1f}m left",
                    flush=True,
                )

        print(
            f"[{time.strftime('%H:%M:%S')}] === lambda={lam:g} starting ===", flush=True
        )
        sig = pipeline.walk_forward(
            feat,
            progress=on_step,
            jump_penalty=lam,
            n_init=args.n_init,
            max_iter=args.max_iter,
            refit_every=args.refit_every,
            signal_mode="cjm_nowcast",
        )
        out = config.CACHE_DIR / f"signals_jp{lam:g}_{args.suffix}.parquet"
        sig.to_parquet(out)
        print(
            f"[{time.strftime('%H:%M:%S')}] lambda={lam:g} DONE in "
            f"{(time.time() - t0) / 60:.1f}m -> {out.name} ({len(sig):,} rows)",
            flush=True,
        )

    print(
        f"[{time.strftime('%H:%M:%S')}] ALL DONE in {(time.time() - t_all) / 60:.1f}m "
        f"({len(grid)} lambdas).",
        flush=True,
    )


if __name__ == "__main__":
    main()
