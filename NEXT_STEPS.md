# Next steps & handoff notes

**Last updated: 2026-06-07.** Read `README.md` first for full project + paper
context, then this file for current state and the next task.

---

## Resume prompt (paste into a FRESH chat)

```text
I have a Python research project at ~/Desktop/regime-monitor (uv venv at .venv,
package in ./regime, run with `PYTHONPATH=. .venv/bin/python -m regime.cli ...`).
It detects market regimes with a Continuous Jump Model (Shu & Mulvey) and
backtests a continuous equity-scaling overlay.

Please read README.md and NEXT_STEPS.md first (they summarize the paper, our
deliberate deviations, current results, and the next task) — do NOT re-read the
Shu & Mulvey PDF. Then read only the relevant source files for the task:
regime/jump_model.py, pipeline.py, backtest.py, tune.py, config.py, cli.py.

Cost rules: always use --no-refresh (cached data); iterate on a coarse grid /
short CV window before any full run; ask before kicking off a ~13-min backtest;
summarize terminal output. Don't change the CJM model math.

Today's task: <FILL IN — see "Prioritized next steps" in NEXT_STEPS.md>.
```

---

## Use case (what the signals are FOR)

Decision support for a personal 401k + thinkorswim account — **not** auto-trading,
**not** financial advice. The owner uses the signals to:

- **Set allocation aggressiveness** day to day (risk-on vs risk-off).
- **Go short / buy puts** (thinkorswim) and **raise cash** (401k) when a bear
  regime fires.
- **Time the exit of shorts / how long to hold them**, and **when to re-enter
  longs** — i.e. avoid covering or re-entering too early or too late.

Mapping today: `bear_prob` (CJM nowcast) = the risk dial; the **re-entry overlay**
= cover-short / re-enter timing (now enabled, display-only). **Short-ENTRY timing
is still an OPEN GAP** (the detector is slow to call tops on grinding declines).

## Preferences & locked decisions (honor these)

**Priorities / philosophy**

- **The regime SIGNAL is the product**, not the backtest. Optimize regime
  detection first; the allocation overlay is a validation lens and may change.
  Judge ideas by **signal quality** (calibration, whipsaw, vol/forward-risk link,
  turning-point timeliness), not just P&L.
- **Do NOT change the CJM model math** (`jump_model.py`).
- **Rigor / representativeness matters:** results meant as conclusions must use
  production settings (`refit_every=21`, full-rigor walk-forward). Fast presets
  (fewer inits, larger refit step, capped window) are for *iteration only*, then
  confirm at full rigor. Guard against look-ahead and selection-on-evaluation.

**Cost / working style (token & compute sensitive)**

- Always `--no-refresh` (use cached data). Iterate on a coarse grid / short
  window, then ONE full-rigor confirmation.
- **Ask before kicking off any ~13-min (or longer) run.** Long runs go in the
  background with per-window progress to `reports/*.log` (a prior run "looked
  hung" at 45 min — it was real, just slow).
- Summarize terminal output; read only the few relevant files; one fresh chat
  per task (paste the resume prompt above).

**Decisions locked in this work (don't relitigate without reason)**

- `JUMP_PENALTY = 50` (λ sweep showed re-entry/quality ~flat in λ; kept 50).
- `SIGNAL_MODE = "cjm_nowcast"` (beats the legacy GBM forecast on every axis).
- Financing: `ANNUAL_CASH_YIELD = 0.0`, `ANNUAL_FINANCING_RATE = 0.10` (idle
  cash must NOT earn the borrow rate — that was a bug).
- `REENTRY_OVERLAY = True` (display-only; separate from the pure signal).

---

## What's DONE (so the next chat doesn't redo it)

### ✦ Dashboard redesigned into 3 independently-tracked layers (2026-06-07)

The dashboard + history now separate the three things the owner reasons about,
each tracked on its own so you can watch signals fire one by one:

- **Layer 1 — P(bear), the continuous risk dial** (top). Big number + a
  red/amber/green gauge with a needle, shown with adaptive precision
  (`_fmt_prob`: <1% renders as e.g. `0.3%` instead of a bare `0%`, so the dial
  is informative in calm bulls). This is the aggressiveness dial.
- **Layer 2 — binary Bull/Bear regime** (its own card + step sparkline). The
  hard CJM argmax label (`regime_binary` 0/1), distinct from the dial and from
  the 3-way `stance`.
- **Layer 3 — Signals overlays** (its own card + event-timeline sparkline):
  **Short-entry** (shown "not yet built" until that overlay lands) and **Long
  re-entry** (live; armed/fired + last-fired date). Each tracked independently.
- **History schema expanded** (`cli._log_history` now read-modify-writes the
  whole file, back-filling older rows): added `regime_binary`, `reentry_flag`,
  `short_entry_flag`, `bear_prob_overlay`. `recommend.build_recommendation` adds
  `regime_binary` and passes through a future `short_entry_flag`. Sparklines
  dedup to one row/day. `notify.py` unaffected (reads the rec/state, not the CSV).
- No CJM math changed; the dashboard is display-only.
- **Note:** sparklines look flat right now because `signal_history.csv` only has
  one calendar day (2026-06-05) of near-0% reads; they fill in as the daily job
  accumulates dates. The hosted GitHub Pages action commits the CSV so the
  hosted history grows over time.

### ✦ FRED data-layer fix + downside/curve features + coarse entry-timing A/B (2026-06-07)

Investigated whether to chase NEW signals vs extend the sample back to 1998.
Conclusion: extending to 1998 is LOW value for short-entry timing (dot-com is
already in-sample from 2000; expanding window means old data barely shifts
today's centroids; λ is flat; the entry lag is structural). New
signals/features are higher value — so pursued those.

- **Root-cause data bug fixed.** `hy_oas` and `y3m` were all-NaN because
  `pandas_datareader` is **broken on Python 3.13** (`No module named
  'distutils'`), so EVERY FRED pull silently failed and fell back. Rewrote
  `data._fred` to hit the keyless `fredgraph.csv` endpoint directly (short
  timeout → fast fallback). Also fixed the Yahoo yield fallbacks to a
  consistent `/10` scale (`^TNX/10` for y10, **new `^IRX/10` for y3m**) so
  `curve_slope = y10 - y3m` has the right sign/scale.
- **`y3m` / `curve_slope` now populated** (backfilled from `^IRX/10` into the
  cache, 6646/6646 rows; curve inverts ~14% of history — 2000/2006-07/2019/
  2022-23, correct sign). **`hy_oas` STILL empty**: FRED's `fredgraph.csv`
  download times out from this environment and there is no free non-FRED OAS
  equivalent. It will populate automatically once FRED is reachable.
- **New features added** in `features.py` (always computed): `drawdown_63`
  (price vs trailing-63d high), `downside_dev_21` (semi-deviation of negative
  returns), plus `curve_slope` now usable. Intended to react to GRINDING
  declines faster than symmetric vol/momentum.
- **Coarse A/B (`scripts/eval_features.py`, n_init=3, refit=63d, full OOS):**
  appending the 3 features to the 2-state CJM **did NOT help** short-entry
  timing (Δlag ~0 on 2007/2011/2015/2018), **regressed 2022 (+5d)** and
  **failed to cross 0.60 at all on 2025-26**; whipsaw unchanged (1.37/yr). The
  new features are highly correlated with the existing vol/mom set, so they
  **dilute** the equal-weighted Euclidean clustering rather than sharpen it.
  This confirms the entry lag is **structural**, not an info deficit.
- **Decision: production keeps the BASELINE 8 features** (`REGIME_FEATURES`
  reverted; `REGIME_FEATURES_BASELINE` + `REGIME_FEATURES_EXPERIMENTAL` kept
  for the harness). No regression shipped; no CJM math changed.
- **Next to try (didn't help → so don't re-append):** (a) REPLACE-not-append —
  swap a redundant momentum feature for `drawdown_63` and re-run the A/B; or
  (b) a **separate short-ENTRY overlay** (mirror of the re-entry overlay) that
  fires on `drawdown_63` crossing a threshold, kept OUT of the pure CJM signal.

### ✦ OPERATIONALIZED — daily iMessage digest + hosted dashboard + driver panel (2026-06-07)

The signal is now a live daily tool, not just research. No CJM math changed.

- **CJM driver attribution (the "why").** `pipeline.cjm_feature_drivers()`
  decomposes the live model's bear-vs-bull lean per feature as
  `bear_pull_i = (z_i-mu_bull_i)^2 - (z_i-mu_bear_i)^2` (sums to the exact
  distance gap the CJM compares). Leak-free; attached to `latest_signal()` as
  `drivers`. Surfaced in `regime update` (a "Why" table) and the dashboard.
  Validated: COVID/GFC days show VIX/vol/MACD pushing BEAR; today pushes BULL.
- **`regime digest`** (`notify.py`) — formats a short text + sends via **iMessage**
  (AppleScript/osascript, macOS-only), **change-gated** (only pings on stance
  flip, threshold cross, new re-entry confirm, or ≥15-pt 1-day move; remembers
  yesterday in `data/notify_state.json`). Flags: `--to`, `--force`, `--dry-run`.
- **`regime dashboard`** (`dashboard.py`) — renders a self-contained, phone-
  friendly `reports/site/index.html` (dial, re-entry banner, 6-mo P(bear)
  sparkline, driver table, recent calls).
- **Delivery split (decided + deployed):** iMessage MUST run on the Mac
  (`deploy/com.regimemonitor.digest.plist` launchd @ 9:00 AM local + `pmset`
  wake @ 8:55 + Full Disk Access granted); the 24/7 dashboard is hosted free on
  **GitHub Pages** via `.github/workflows/dashboard.yml` (daily, commits
  `signal_history.csv` so the hosted sparkline accumulates). Live at
  `https://wbelmont.github.io/regime-monitor/`.
- **Secrets:** phone number lives in gitignored `regime/local_settings.py`
  (config.py loads it as an override); committed `IMESSAGE_RECIPIENT=""`.
- Notebook §15 = new-model backtest + the bear-regime/re-entry tape visual.

### ✦ Re-entry/cover-short OVERLAY added — now ENABLED (display-only) (2026-06-06)

Investigated re-entry lag (signal stands down ~45–65 d after a bottom):
- **λ is NOT the lever.** Swept λ∈{5,12.5,25,50,100} at 45-day refit
  (`scripts/sweep_lambda.py`, cached `signals_jp{λ}_sweep.parquet`): re-entry
  ~flat across a 20× λ range. Re-entry lag is **structural** (backward-looking
  features stay bearish ~2–3 mo after a trough). **Kept λ=50.**
- **Smoothing can't help** (an EMA only lags/matches the raw signal). The fix
  is a **price-rebound override** using info the nowcast lags on.
- **Winner: VIX-receding rebound gate.** Cap `bear_prob` at 0.20 once S&P is
  ≥ `REENTRY_REBOUND` above its trailing `REENTRY_LOOKBACK`-day low AND VIX <
  its 21d avg. Param sweep → robust defaults **rebound=0.10, lookback=42**
  (lookback insensitive; 8–12% is the safe band, 10% the peak).
- **Validated out-of-sample on 7 corrections** (2011, 2015–16, 2018, 2026 +
  the 3 tuned): never worsens re-entry timing; big wins on deep recoveries
  (COVID 65→7 d, GFC 46→3 d), silent on shallow ones. Caveat: faster re-entry
  can mean re-entering a dead-cat bounce (DD risk); it does **not** touch
  short-ENTRY timing (slow on grinding declines: 2011 97 d, 2015 92 d).
- **Implemented as a SEPARATE layer** (signal stays a pure CJM nowcast):
  `config.REENTRY_OVERLAY=True` now (was opt-in/off; **display-only** — surfaces
  a flag + `bear_prob_overlay`, does NOT change `bear_prob`, stance, allocation,
  backtest, or tuner) + `REENTRY_REBOUND/LOOKBACK/CAP/REQUIRE_VIX`;
  `pipeline.reentry_overlay()` returns `bear_prob` / `bear_prob_overlay` /
  `reentry_flag`; `latest_signal` adds overlay fields when enabled;
  `recommend`/`cli update` print a cover-short/re-enter flag. Notebook §10–14
  has all the analysis.

### ✦ Signal architecture switched to the CJM nowcast (2026-06-06)

Highest-value change so far. A P&L-free signal-quality study (full-rigor
walk-forward, `signals_jp50_full.parquet`) compared the legacy GBM one-step
forecast vs the CJM's OWN bear probability (`predict_proba`, online inference):

- CJM nowcast is **better calibrated** (Brier 0.020 vs 0.033), **3× less
  whipsaw** (1.5 vs 4.5 transitions/yr; ~166- vs 55-day regimes), stronger
  forward-**vol** link (0.57 vs 0.50), and calls crises just as fast.
- **Neither signal predicts forward returns** (~0 corr) — this is a
  risk/volatility detector; use it to size risk, not call direction.
- `config.SIGNAL_MODE` selects `cjm_nowcast` (default) vs `gbm_forecast`
  (legacy). `pipeline.walk_forward(..., signal_mode=, return_nowcast=)` and
  `latest_signal` honor it. `walk_forward(return_nowcast=True)` adds a
  `cjm_bear_nowcast` diagnostic column (leak-free).
- Corrected backtest of the switch (no allocation change): **12.6% / Sharpe
  0.69 / −27% DD** vs legacy 7.0% / 0.37 / −43% and B&H 9.0% / 0.46 / −57%.
- Notebook §8–9: the allocation-agnostic harness (`forward_stats`,
  `decile_table`, `timeliness`, `calibration`, `stability`) + the comparison.
  `scripts/run_walkforward.py` is the background full-rigor runner (≈ 13 min).

### ✦ Financing bug fixed (2026-06-06)

`backtest.py` applied a single 10%/yr rate to the whole `(1-weight)` sleeve, so
idle cash silently earned 10%/yr — inflating the strategy exactly when it
de-risked. Now asymmetric: borrowed leg pays `ANNUAL_FINANCING_RATE=10%`, idle
cash earns `ANNUAL_CASH_YIELD=0%` (config). This flipped the headline read (the
old "beats B&H" was the cash tailwind). Look-ahead audited clean (`.shift(1)` +
contemporaneous `bear_prob`, one-step-ahead label target — no leak).

### ✦ Earlier scaffolding (still valid)

1. **`regime tune`** — nested-CV jump-penalty tuner (`tune.py`, in `cli.py`).
   Leak-free, nested (select vs held-out eval span, `eval_frac=0.30`),
   `--select-by sharpe|jumps`. *Note:* it scores on the OVERLAY; for
   signal-first tuning use the notebook harness instead.
2. **`backtest.py`** — daily, vectorized, continuous piecewise weight from
   `bear_prob` (`<0.20→1.5`; `0.20–0.80→lin 1→0`; `>0.80→0`), trades on prior
   day's weight (`.shift(1)`), asymmetric financing.
3. **Config path** fixed (`PROJECT_ROOT = parents[1]`).
4. **`performance_analysis.ipynb`** — the live research harness (§8–14).

---

## Current results (FULL-RIGOR OOS, JUMP_PENALTY=50, daily, 0% cash / 10% borrow)

Production signal = **CJM nowcast** (`SIGNAL_MODE='cjm_nowcast'`).

| Metric | CJM nowcast (prod) | GBM forecast (legacy) | Buy & hold |
| --- | --- | --- | --- |
| Annual return | 12.6% | 7.0% | 9.0% |
| Sharpe | 0.69 | 0.37 | 0.46 |
| Max drawdown | −27% | −43% | −57% |

Signal-quality (allocation-agnostic): nowcast Brier 0.020, 1.5 transitions/yr,
~166-day regimes. Detects **volatility** regimes well; does **not** predict
forward returns. Re-entry overlay (opt-in) cuts re-entry ~50→~13 d avg at
roughly neutral Sharpe/DD.

---

## Prioritized next steps

> **North star:** optimize the CJM bear-probability nowcast as a daily
> risk-tolerance dial. Judge ideas by signal quality, not just backtest P&L.

1. **Short-ENTRY timing (the open gap).** The signal is slow to call the top on
   grinding declines (2011 97 d, 2015 92 d, 2026 56 d after the peak). NOTE
   (2026-06-07): appending downside/drawdown/curve features to the CJM did NOT
   help (see "What's DONE"). Best remaining bets: a **separate short-entry
   overlay** (mirror of the re-entry overlay, fires on `drawdown_63`), or
   REPLACE-not-append feature selection. Highest-value remaining timing work.
2. **Feature review for the CJM** (`REGIME_FEATURES`). `drawdown_63`,
   `downside_dev_21`, `curve_slope` now EXIST (in
   `REGIME_FEATURES_EXPERIMENTAL`); a coarse A/B showed *appending* them dilutes
   the 2-state clustering. Try REPLACE-not-append next, or feature weighting.
3. **(Mostly done) Re-entry overlay is live** — `REENTRY_OVERLAY=True` and now
   surfaced in the daily digest + dashboard (display-only; doesn't alter the
   traded `bear_prob`). Remaining option: also write `bear_prob_overlay` into the
   history CSV / regime chart if you want it persisted there too.
4. **(Optional) revisit allocation** (`LEVERED_WEIGHT`, weight map) — only after
   the signal/timing is locked.
5. **(Optional) 3-state CJM** (bull/neutral/bear) for a finer dial.
6. **HY OAS still empty** (`hy_oas` all-NaN). ROOT CAUSE FOUND & partly fixed
   (2026-06-07): `pandas_datareader` is broken on Py3.13, so FRED pulls failed
   silently; `data._fred` now uses `fredgraph.csv` directly. `y3m`/`curve_slope`
   are backfilled (Yahoo `^IRX`), but `hy_oas` is FRED-only and `fredgraph.csv`
   times out from this environment — it will populate once FRED is reachable.
