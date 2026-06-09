# regime-monitor

A personal **market-regime monitor + research harness**. It detects Bull/Bear
regimes with a **Continuous Jump Model (CJM)**, turns that into a daily
bear-probability signal, scales equity exposure accordingly, and backtests the
result leak-free. It is *decision support* for a personal 401k + trading
account — **not** an automated trader and **not** financial advice.

> **For an AI assistant picking this up:** read this file, then `NEXT_STEPS.md`
> (the live handoff with current numbers and the next task). Read source in this
> order: `regime/jump_model.py`, `regime/pipeline.py`, `regime/backtest.py`,
> `regime/tune.py`, `regime/config.py`, `regime/cli.py`. Do **not** read the
> whole repo or re-read the Shu & Mulvey PDF — the relevant methodology is
> summarized below.

---

## How to run

```bash
# Live signal (today's regime + suggested stance)
PYTHONPATH=. .venv/bin/python -m regime.cli update --no-refresh

# Leak-free out-of-sample backtest (~13 min full; uses cached data)
PYTHONPATH=. .venv/bin/python -m regime.cli backtest --no-refresh

# Tune the jump penalty (lambda) via nested time-series CV
PYTHONPATH=. .venv/bin/python -m regime.cli tune --no-refresh

# Regenerate the regime-history chart
PYTHONPATH=. .venv/bin/python -m regime.cli chart --no-refresh
```

- Env: `uv` venv at `.venv`, Python 3.13. Package is `./regime` (run with
  `PYTHONPATH=.`). There is a stale `src/regime/` duplicate — **ignore it**; the
  live package is `./regime`.
- `--no-refresh` uses the on-disk cache in `data/cache/raw_inputs.parquet`
  instead of re-downloading (free yfinance/FRED data).
- Double-clicking `regime-monitor.command` runs `update`.

---

## Architecture (data → signal → strategy → evaluation)

| File | Role |
| --- | --- |
| `regime/data.py` | Pull/cache raw inputs (S&P 500, VIX, 10y, 3m, HY OAS). `load_extra()` caches separate free Yahoo inputs (VIX term structure, VVIX, SKEW, credit/breadth/defensive ETFs) for the short-entry fragility score only. |
| `regime/features.py` | Backward-looking features. `REGIME_FEATURES` feed the labeler; `PREDICTOR_FEATURES` feed the next-regime classifier. |
| `regime/jump_model.py` | `JumpModel` (discrete) and **`ContinuousJumpModel`** (the one in use). **Do not change the model math.** |
| `regime/pipeline.py` | `walk_forward` (leak-free OOS signal), `latest_signal` (live), `label_full_sample` (charts only), and the display-only overlays: `reentry_overlay` + the leading `fragility_score` (short-entry). |
| `regime/backtest.py` | Daily, continuous-weight, financing-aware backtest. |
| `regime/tune.py` | Nested-CV jump-penalty tuner (`regime tune`). |
| `regime/cli.py` | CLI: `update`, `backtest`, `tune`, `chart`. |
| `regime/config.py` | All tunables (paths, `JUMP_PENALTY`, windows, thresholds, playbook). |
| `notebooks/performance_analysis.ipynb` | Interpretability: equity curves, per-period signal→exposure, return-gap decomposition. |

**Two leak-free conventions** (enforced in `pipeline.walk_forward`):

1. The regime labeler is fit **inside each training window only** (online
   inference — the paper's requirement; forward-looking labels inflate results).
2. The **production signal is the CJM's own bear probability nowcast**
   (`config.SIGNAL_MODE='cjm_nowcast'`), produced by the train-window-fitted
   model — leak-free. A legacy one-step-ahead GBM forecast (`gbm_forecast`) is
   kept for comparison but was shown to add whipsaw and miscalibration.
