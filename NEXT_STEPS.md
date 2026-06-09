# Next steps & handoff notes

**Last updated: 2026-06-09 (session: full dashboard REDESIGN shipped — true dark
mode, SVG risk dial, rigorous methodology explainer, no-bonds beta/delta
playbook, re-entry "why" checklist, fragility ignition heatmap, percentiles
everywhere. NEXT: evaluate whether a GARCH model adds anything the CJM doesn't —
see PICKING UP NEXT).** Read `README.md` first for full project + paper context,
then this file for current state and the next task.

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

Today's task: a THINK-FIRST research-scoping question — does a GARCH model add
anything the CJM doesn't already do? See "PICKING UP NEXT" at the top of
"Prioritized next steps" in NEXT_STEPS.md for the full brief. Scope it before
building: a GARCH(1,1)/EGARCH gives a parametric conditional-vol forecast, but
the CJM already consumes realized vol, VIX, and the vol risk premium and is a
strong volatility-regime detector — so measure whether a rolling GARCH sigma adds
INCREMENTAL info (correlate it with vol_21, realized forward vol, and the CJM
bear nowcast in the notebook) before wiring it in as a CJM feature, a fragility
component, or a risk-sizing transform. Only adopt it if it demonstrably improves
a signal-quality axis (earlier fragility lead / calibration / less whipsaw) the
current features don't capture; otherwise document "CJM dominates" and stop. The
`arch` package may need installing into .venv. Don't change the CJM math.

NOTE: the 2026-06-09 dashboard redesign is built but NOT yet committed/pushed —
push it when the owner signs off.
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
= cover-short / re-enter timing (enabled, display-only); the **short-entry
fragility score** = get-short / buy-protection timing (enabled, display-only) — a
graded 0–100% LEADING early-warning (WATCH/LEAN/ACT) that can fire while price is
near highs and VIX is low, because buying protection has the OPPOSITE loss
function from re-entry (early is cheap, late is expensive). All overlays are
independent, display-only layers; the traded `bear_prob` stays a pure CJM nowcast.

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

**Operational guardrails (avoid hangs) — learned the hard way**

- **Every network call MUST carry an explicit timeout.** A blocking read with no
  timeout will hang the whole session if the remote stalls (this bit us polling
  the live Pages site). Use `curl --max-time 15 --retry 2 …`, and in Python
  `urllib.request.urlopen(url, timeout=15)` — never a bare `urlopen`/`curl`.
- **Poll with a bounded loop, not an open-ended one** (cap attempts AND a total
  deadline; print progress; exit non-zero if unreachable). No infinite `while`.
- **Long jobs run in the BACKGROUND**, never as a foreground blocking call:
  write per-step progress to `reports/*.log` and poll the log/terminal output.
  (Distinct from a real slow job like the ~13-min backtest — that's not a hang,
  it's just slow; still background it.)
- **Distinguish the two "hang" classes** when one happens: (1) a stalled network
  read with no timeout (avoidable — add a timeout/retry), vs (2) legitimately
  long compute (background + log). Don't treat #2 as broken.

**Decisions locked in this work (don't relitigate without reason)**

- `JUMP_PENALTY = 50` (λ sweep showed re-entry/quality ~flat in λ; kept 50).
- `SIGNAL_MODE = "cjm_nowcast"` (beats the legacy GBM forecast on every axis).
- Financing: `ANNUAL_CASH_YIELD = 0.0`, `ANNUAL_FINANCING_RATE = 0.10` (idle
  cash must NOT earn the borrow rate — that was a bug).
- `REENTRY_OVERLAY = True` (display-only; separate from the pure signal).
- `SHORT_ENTRY_OVERLAY = True` (display-only; separate from the pure signal).
  The short-entry layer is now a graded **FRAGILITY SCORE** (0–100%, WATCH/LEAN/
  ACT), NOT a reversed re-entry gate. It's a LEADING early-warning (no drawdown
  gate) built from drift-robust z-scores of vol-structure + hedging-demand +
  divergence components (see `FRAGILITY_*` in `config.py`). The old price-
  drawdown trigger is retained only as a secondary `decline_confirmed` boolean.
  `short_entry_flag` (logged/dashboard) fires at grade ≥ LEAN.
- **Fragility components (2026-06-09):** defensive tell is **cand. C** — staples
  rotation = beta-neutral **`XLP/XLY`**, and **XLU gated by staples confirmation**
  (AI-power contamination fix). **`bond_vol` = MOVE** (`^MOVE`, rising = stress,
  w=0.10) is IN. **JPY carry-unwind is OUT** as a blended component (it dilutes /
  delays the composite; see DONE). Don't re-tune the WATCH/LEAN/ACT thresholds
  on the same ~7 episodes.
- **`REFIT_EVERY_DAYS = 21`** (refit=15 tested 2026-06-09 → identical results,
  ~35% more compute; refit cadence doesn't affect the live signal anyway).

---

## What's DONE (so the next chat doesn't redo it)

### ✦ DASHBOARD REDESIGN shipped + interpretability pass (2026-06-09 PM)

A true clean dark-mode rework of `regime/dashboard.py` (display-only; no CJM math
changed) plus a no-bonds allocation playbook and percentile interpretability.

- **True dark theme.** New cohesive palette (`dashboard.C`): near-black canvas,
  one elevated card surface, a single risk ramp (mint `#21d07a` → amber → coral
  `#ff5d63`) reused by every element; **Inter** font (non-blocking load with a
  system fallback so the page always renders); generous whitespace. Replaced the
  generic blue/"Claude" look.
- **The CJM risk dial is the hero.** Inline **SVG semicircular arc gauge**
  (`_arc_gauge` + `_risk_color`/`_arc_point`) — crisp, self-contained (no PNG),
  with threshold ticks + a colored needle; the big adaptive-precision % number is
  anchored on the gauge pivot (`.dial-center`, `top:72%` + transform) so it sits
  centered in the bowl at any width.
- **Module order** (matches the brief): dial → **How this works (methodology)** →
  Why drivers → timing overlay → fragility → suggested stance → **Recent calls
  (bottom)**. Dropped the redundant binary Bull/Bear sparkline + the near-empty
  overlay event timeline.
- **Rigorous methodology explainer.** 5 steps that emphasize the CJM's edge:
  distribution-free jump penalty vs an HMM's parametric emissions/EM/Viterbi-vs-
  smoothing mismatch; continuous calibrated probability (Brier ≈0.02) vs the
  discrete SJM's coarse 0/1; leak-free walk-forward; volatility (not direction)
  detector. Inline `.mono` formula styling.
- **Re-entry overlay now shows WHY.** `pipeline._reentry_diagnostics()` (leak-free)
  exposes rebound-%-off-the-low vs threshold + VIX-vs-21d-MA; passed via
  `recommend` → dashboard renders a ✓/○ **condition checklist** (e.g. "S&P +8.9%
  off its 42-day low (needs ≥+10%) ○ / VIX still elevated ○ → waiting"). No longer
  an opaque "armed".
- **Fragility chart = component ignition heatmap** (`_frag_ignition`): replaced the
  tangle of lines with rows = stress tells (ordered structural→late), x = time,
  cell shade = 0..1 sub-score (dark→amber→coral), composite track on top — so you
  can trace which tell lit up first, then next.
- **No-bonds allocation playbook.** `config.ALLOCATION_PLAYBOOK` rewritten: 100%
  equity always, NEVER bonds — de-risking = cash / lower beta / hedges. Added
  `recommend.exposure_targets()` deriving a continuous **target equity beta**
  (`TARGET_BETA_MAX=1.30` at dial 0% → `TARGET_BETA_MIN=0.15` at 100%), a net
  **delta** bias (long→flat→short as the dial climbs), and an **options/leverage**
  flag (OK only when dial ≤ `LEVERAGE_OK_BELOW=0.15`, i.e. deep/confirmed bull;
  otherwise "100% invested but plain beta, no options/leverage"). Dashboard shows
  a beta meter (with the market-1.0× tick) + delta + leverage chip. **Owner can
  tune the beta anchors / leverage cutoff to taste.**
- **Percentiles for interpretability (2026-06-09).** Empirical percentiles (where
  today sits in its OWN trailing distribution — no normality assumption) added
  alongside the z-scores/sub-scores: `cjm_feature_drivers(..., history=)` adds a
  `pctile` per feature (shown under the σ in the Why table's "vs normal"); the
  fragility block adds `fragility_pctile` (composite) + `fragility_pctiles`
  (per-component), shown as a "Nth pct vs history" chip + a "vs history" column.
  Live read: fragility 63% = **97th pct** of its history; VIX term-structure tell
  **99th pct** — reinforces the LEAN. `_fmt_pctile` does the ordinal formatting.
- **Render/verify:** `dashboard --no-refresh --no-log`; previewed via a local
  `python3 -m http.server` (Simple Browser white-screened on `file://`; Safari +
  `http://localhost:PORT` works). **Not yet committed/pushed** at session end —
  push when ready (rebase+retry guardrails in the CI note below).

### ✦ SHORT-ENTRY redesigned as a graded FRAGILITY SCORE (2026-06-08)

Replaced the first-cut short-entry overlay (a literal reverse of the re-entry
confirmation gate) after realizing the two problems have OPPOSITE loss
functions: for **buying protection** (puts / VIX calls / raising cash), being
early is cheap (a little theta) and being late is expensive (implied vol already
exploded). So the short-entry layer is now a LEADING "fragility" detector meant
to nudge while the market still looks calm and protection is cheap — it does NOT
require a price drawdown and can read elevated with the S&P near highs + VIX low.
Still display-only; the CJM signal/math is untouched.

- **Output:** a graded **0–100% fragility score** (like the CJM dial) with
  **WATCH / LEAN / ACT** bands (`config.FRAGILITY_WATCH/LEAN/ACT` = 0.35/0.55/
  0.70). `short_entry_flag` fires at grade ≥ LEAN.
- **Components (all drift-robust):** VIX term structure (`VIX3M/VIX`), VIX
  velocity, VVIX, SKEW, credit (`HYG/LQD`), breadth (`RSP/SPY`), defensive
  rotation (XLP primary; **XLU down-weighted + velocity-only** because the
  AI/electricity re-rating structurally lifted utilities — the owner's call).
  **Every component is a trailing z-score of its RECENT CHANGE, not its level**,
  so slow structural drift is continuously re-baselined out (the general fix for
  the XLU concern). Each z → 0..1 via a logistic, weight-averaged over whichever
  components have data (`config.FRAGILITY_WEIGHTS/K/Z0/Z_WINDOW`).
- **New data:** `data.load_extra()` pulls/caches 9 free Yahoo series
  (`^VIX3M ^VVIX ^SKEW SPY RSP HYG LQD XLP XLU`) into a SEPARATE
  `data/cache/extra_inputs.parquet` so the CJM raw-inputs lineage / backtest is
  untouched. (No CBOE put/call — owner's call; too fiddly for now.) History
  depth: SKEW→1990, XLP/XLU→1998, RSP→2003, VIX3M→2006, VVIX/HYG→2007 (so all
  six eval grinds are covered; pre-2006 just uses fewer components).
- **Code:** `pipeline.fragility_score()` (pure, leak-free) + `_roll_z`/
  `_logistic` helpers; `latest_signal(feat, extra=)` sets `fragility_score`,
  `fragility_grade`, `fragility_drivers`, `short_entry_flag` (≥LEAN), and a
  secondary `decline_confirmed` (the old drawdown trigger, kept as the
  later-stage tell). `cli update` loads `extra` (respects `--no-refresh`) and
  prints a graded banner + top drivers; `_log_history` adds a `fragility_score`
  column; `recommend` passes the fields through. Dashboard renders fine.
- **Coarse validation (cached `signals_jp50_full.parquet` + cached inputs):**
  - **Base rate healthy** — LEAN ~5–7% of days in calm bull years (2013/17/19/
    21/24), ACT ~1–2%; full-sample median score 0.28. Not crying wolf.
  - **Leads on grinding tops** (the open gap): sustained LEAN fired with **VIX
    still 14–19** in 2018 / 2020-COVID / 2025-26 (protection cheap). Honestly
    LATE only on violent gap-downs (2015 China deval ~VIX 40, 2022 ~VIX 27) —
    which no leading signal can front-run. Mean VIX at first sustained LEAN ≈ 23.
  - **Stopped tuning at v1** to avoid selection-on-evaluation over 6 episodes.
- **Live read (2026-06-08): fragility 64% → LEAN**, driven by VIX term structure
  (96%), VIX velocity (90%), VVIX (77%) — firing while the CJM is still BULL/0%,
  i.e. exactly the intended LEADING behavior.
- **Caveat:** a leading signal WILL have false positives (that's the point —
  they're cheap). Treat ACT as "scale into protection", not "all in".

### ✦ SHORT-ENTRY overlay built — now ENABLED (display-only) (2026-06-07)

> SUPERSEDED 2026-06-08 by the fragility score above. The price-drawdown trigger
> below is retained only as the secondary `decline_confirmed` boolean.

Mirror of the re-entry overlay, in the opposite direction: an EARLIER
"consider getting short / buying puts / raising cash" flag for the slow
short-ENTRY timing (the open gap). The pure CJM signal is untouched — this is
a separate, display-only layer (does NOT change `bear_prob`, stance,
allocation, backtest, or tuner). No CJM math changed.

- **Rule (leak-free, backward-looking):** fire when the S&P is ≥
  `SHORT_ENTRY_DRAWDOWN` BELOW its trailing `SHORT_ENTRY_LOOKBACK`-day high AND
  (if `SHORT_ENTRY_REQUIRE_VIX`) VIX > its 21d average (fear rising). With
  `lookback == 63` the trailing-high drawdown equals the existing `drawdown_63`
  feature. It's the cover-short overlay run in reverse (drawdown-from-high vs
  rebound-from-low; VIX rising vs receding).
- **Code:** `config.SHORT_ENTRY_OVERLAY=True` plus `SHORT_ENTRY_DRAWDOWN/
  LOOKBACK/REQUIRE_VIX`; `pipeline.short_entry_overlay()` returns `bear_prob`
  (passthrough) and `short_entry_flag`; `latest_signal` sets `short_entry_flag`
  when enabled; `recommend`/`cli update` print a short-entry banner. Downstream
  wiring (`_log_history`, dashboard Signals card + timeline) was already in
  place, so the dashboard now shows short-entry as BUILT/tracked instead of
  "not yet built".
- **Coarse validation (cached `signals_jp50_full.parquet` OOS bear_prob +
  cached features; entry lag from the PEAK, lower = earlier):** the overlay
  fires earlier than (or equal to) the raw 0.60 crossing in EVERY episode and
  never later. At the chosen **drawdown=0.07**: mean lag-from-peak 16.0 d vs raw
  22.8 d (**~6.8 d earlier**); big wins 2022 +22 d, 2025-26 +9 d, 2011/2018 +4 d;
  2015-16 tie; 2020 COVID +2 d. **drawdown=0.05** is strictly better (13.2 vs
  22.8 d, ~9.7 d earlier, never worse) but fires more often (~2.25 vs ~1.76
  episodes/yr). **drawdown=0.10 REGRESSES** (24.3 d, later than raw on 2018) →
  don't go that wide.
- **Threshold decision: kept 0.07** (conservative balance — earlier than raw in
  every case, won't cry wolf on every shallow dip given the VIX-rising gate).
  The owner can drop to 0.05 for a more sensitive flag (it dominates on the eval
  episodes); left at 0.07 to avoid over-fitting the threshold to 6 episodes.
- **Caveat:** a drawdown trigger can fire on a shallow dip that recovers (a
  false top); it's a timing aid, not a guarantee. Display-only, so no P&L risk.

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

### ✦ Fragility dashboard line + post-close CI run + pushed the overlay (2026-06-08 PM)

Surfacing item (B) is partly done; the fragility overlay is now actually LIVE on
the hosted dashboard (it had only ever existed in the local working tree before
this session — uncommitted/unpushed, so the hosted page still said "short-entry
not yet built"). No CJM math changed.

- **Pushed the fragility work** (`config/data/pipeline/recommend/cli` + docs +
  `scripts/_probe_fragility.py`) to `origin/main`. The hosted Short-entry overlay
  row flipped from "not yet built" → real armed/fired state.
- **Graded fragility CARD on the dashboard** (`dashboard._fragility_card`): its
  own 0–100% card with a WATCH/LEAN/ACT **banded gauge** (bands from
  `config.FRAGILITY_*`), a grade chip, and the **top component drivers** (VIX
  term structure, VIX velocity, VVIX, SKEW, credit, breadth, defensive). Graceful
  "inputs unavailable" state. Display-only; never touches the risk dial. Verified
  live on `https://wbelmont.github.io/regime-monitor/`.
- **Post-close hosted refresh (CI fix for the stale-date confusion):** added a
  SECOND daily cron to `.github/workflows/dashboard.yml` at **21:30 UTC (~5:30 PM
  ET, after the close)** alongside the morning 13:30 UTC run. Root cause of "the
  dashboard still says 06-05 after the close": the only run was at ~9:30 AM ET
  (BEFORE the close), so it always showed the prior trading day; it also never
  re-ran post-close. Two runs/day now keeps the hosted "As of" current-day by
  evening. (The time-sensitive iMessage digest still runs on the Mac at 9 AM ET.)
- **Process lesson captured** as "Operational guardrails (avoid hangs)" above:
  a bare `urllib.urlopen`/`curl` with no timeout hung the session while polling
  the live page — all network calls now require explicit timeouts + bounded
  loops; long jobs go to the background.

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
>
> **▶ PICKING UP NEXT (saved 2026-06-09): EVALUATE whether a GARCH model adds
> anything the CJM doesn't.** This is a THINK-FIRST / research-scoping task, not a
> build — the owner's hypothesis is that "the CJM might already be better at
> everything GARCH can do," and we should confirm or refute that before writing
> code. Don't change the CJM math. Frame it as signal-quality, not P&L.
>
> Questions to answer (cheaply, with cached data + the notebook harness):
>
> 1. **What would GARCH actually add?** GARCH(1,1)/EGARCH/GJR give a *parametric
>    conditional-volatility* forecast (vol clustering + mean reversion, with an
>    explicit 1-day/h-day-ahead σ and leverage asymmetry). The CJM already
>    consumes realized vol (`vol_21`), implied vol (`vix`) and the vol risk
>    premium, and its bear nowcast is a strong *volatility-regime* detector
>    (forward-vol corr ≈0.57). So the honest question is whether a GARCH σ
>    forecast adds INCREMENTAL info beyond those features.
> 2. **Three candidate roles, in priority order:**
>    (a) **A new CJM feature** — feed a GARCH conditional-σ (or its innovation /
>        σ-vs-realized gap) into `REGIME_FEATURES` via the REPLACE-not-append
>        discipline (item C); test on the eval episodes for timeliness/whipsaw.
>        Likely redundant with `vol_21`/`vix` — measure the correlation first.
>    (b) **A fragility component** — a GARCH-implied short-horizon vol JUMP
>        (forecast σ rising fast, or realized > GARCH-expected) as another
>        drift-robust leading tell. Plausibly the best fit, since fragility is
>        explicitly a leading-vol story.
>    (c) **A risk-sizing transform** on the dial (scale exposure by forecast σ).
>        Lowest priority — the beta playbook already does discretionary sizing.
> 3. **Cheap first cut:** in the notebook, fit a rolling GARCH(1,1) on S&P returns
>    (the `arch` package), get the 1-step σ forecast, and correlate it with
>    (i) `vol_21`, (ii) realized fwd 21d vol, (iii) the CJM bear nowcast. If GARCH
>    σ is ~collinear with existing features AND doesn't lead the nowcast, the CJM
>    likely dominates and GARCH is not worth wiring in — record that and move on.
> 4. **Decision rule:** only adopt GARCH somewhere if it demonstrably improves a
>    signal-quality axis (earlier fragility lead, better calibration, less
>    whipsaw) that the current features don't already capture. Otherwise document
>    "CJM dominates" and close the thread.
>
> Cost rules as always: `--no-refresh`, coarse first, ask before any ~13-min run.
> `arch` may need installing into `.venv` (cheap, pure-Python-ish).
>
> **Also still open from the redesign session:** the redesign is built but
> **NOT yet committed/pushed** — push it to `origin/main` when the owner signs
> off (hosted page will then show the new design). Owner may also want to tune the
> exposure anchors (`TARGET_BETA_MAX/MIN`, `LEVERAGE_OK_BELOW`) to taste.
>
> **Deferred (still valuable):**
> **(A) Harder validation of the fragility score** — per-component ablation,
> false-positive clustering, behavior where component coverage is partial. (The
> 2026-06-09 horse-races did a coarse version of this; a fuller pass is still
> worthwhile.) **Do NOT re-tune thresholds on the same episodes.**
> **(B) Fragility grade in the iMessage digest** (`notify.py`), change-gated on
> grade transitions (WATCH→LEAN→ACT).
> **(C) REPLACE-not-append feature selection** for the CJM (the remaining lever
> on the structural short-ENTRY lag INSIDE the pure signal; don't change CJM math).
> **(D) JPY as a SEPARATE alert** (not a blended fragility component — see DONE).
