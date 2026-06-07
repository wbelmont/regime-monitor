"""Tune the Continuous Jump Model jump penalty (lambda) via NESTED time-series CV.

Why this exists
---------------
Shu & Mulvey select the jump penalty by *time-series cross-validation*, rather
than reusing a value from the discrete jump model. The CJM's penalty enters the
objective on a different scale than the discrete JM (it acts on simplex
probability vectors, not hard 0/1 labels — see `ContinuousJumpModel`), so the
inherited `JUMP_PENALTY = 50` is not guaranteed to be a good operating point.
This module sweeps a small, log-spaced grid and selects lambda under one of two
criteria (`--select-by`):

  * ``sharpe`` — maximize the strategy's out-of-sample Sharpe (P&L-aligned).
  * ``jumps``  — match a realistic number of regime transitions per year, with
    whipsaws (too-many jumps) penalized harder than over-smoothing. This
    *decouples* the regime detector's smoothness from the P&L, which is the
    less overfit-prone, more strategy-agnostic choice.

Nested, leak-free by construction
----------------------------------
Two independent guarantees stack here:

1. **Within a walk-forward**, `pipeline.walk_forward` fits the regime labeler
   and the next-regime classifier on an *expanding training window* that always
   ends strictly before the block it predicts. So every signal value is
   produced from data available at that time (the paper's online-inference
   requirement).

2. **Across the lambda search**, we split the out-of-sample timeline into two
   disjoint, time-ordered spans:
     * a *selection* span (the earlier portion) on which lambda is chosen, and
     * an *evaluation* span (the later `eval_frac` portion) that is NEVER read
       during selection and is used only to report the headline metrics of the
       already-chosen lambda.
   Because the lambda decision is a function of the selection span alone, no
   evaluation-period information can leak into model selection. The reported
   `oos_sharpe` / `max_drawdown` therefore measure a penalty that was picked
   without ever seeing that window — a genuine nested out-of-sample estimate.

Speed
-----
The full backtest is ~12 min. For tuning we expose coarser knobs (fewer
k-means++ restarts, a larger refit step, and a capped recent CV window) so a
sweep of 5–8 lambdas iterates in a couple of minutes. Run a final confirmation
with default settings on the chosen lambda before trusting it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest, config, pipeline


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
# Coarse, log-spaced grid bracketing the inherited value (50). The CJM keeps
# lambda roughly on the discrete-model scale via its 1/4 factor, so spanning
# ~10x–0.25x of 50 covers "very jumpy" to "very sticky".
DEFAULT_GRID: list[float] = [12.5, 25.0, 50.0, 100.0, 200.0]

# Fast CV preset (override per-call). Full confirmation should use the
# pipeline/backtest defaults instead.
FAST_CV = dict(n_init=5, max_iter=30, refit_every=63, max_oos_days=252 * 8)

# Fraction of the out-of-sample timeline reserved (at the END, time-ordered) as
# the held-out *evaluation* span. Lambda selection never sees this window.
DEFAULT_EVAL_FRAC = 0.30

# `jumps` criterion: a realistic target for regime switches per year, plus an
# asymmetric penalty so whipsaws (over-target) hurt more than over-smoothing.
DEFAULT_TARGET_JUMPS_PER_YEAR = 2.0
JUMP_OVER_WEIGHT = 3.0  # penalty weight for jumps ABOVE target (whipsaws)
JUMP_UNDER_WEIGHT = 1.0  # penalty weight for jumps BELOW target (too sticky)

TRADING_DAYS_PER_YEAR = 252.0


@dataclass
class LambdaResult:
    jump_penalty: float
    select_metric: str  # 'sharpe' or 'jumps' — the criterion used
    select_score: float  # criterion score on the SELECTION span (maximized)
    cv_sharpe: float  # mean per-fold Sharpe on the SELECTION span
    cv_sharpe_std: float  # dispersion across selection folds (consistency)
    sel_jumps_per_year: float  # regime transitions/yr on the SELECTION span
    oos_sharpe: float  # Sharpe on the held-out EVALUATION span
    max_drawdown: float  # worst peak-to-trough on the EVALUATION span
    annual_return: float  # annualized return on the EVALUATION span
    eval_jumps_per_year: float
    n_folds: int
    fold_sharpes: list[float] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "jump_penalty": self.jump_penalty,
            "select_metric": self.select_metric,
            "select_score": self.select_score,
            "cv_sharpe": self.cv_sharpe,
            "cv_sharpe_std": self.cv_sharpe_std,
            "sel_jumps_per_year": self.sel_jumps_per_year,
            "oos_sharpe": self.oos_sharpe,
            "max_drawdown": self.max_drawdown,
            "annual_return": self.annual_return,
            "eval_jumps_per_year": self.eval_jumps_per_year,
            "n_folds": self.n_folds,
        }


def _sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe of a DAILY return series (backtest is now daily)."""
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    vol = returns.std()
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return float((returns.mean() / vol) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _fold_sharpes(returns: pd.Series, n_folds: int) -> list[float]:
    """Split the OOS DAILY returns into `n_folds` contiguous (time-ordered)
    chunks and compute Sharpe within each. Contiguous chunks preserve the
    time-series structure — we never shuffle, which would leak across time."""
    returns = returns.dropna()
    if len(returns) == 0:
        return []
    n_folds = max(1, min(n_folds, len(returns)))
    chunks = np.array_split(np.arange(len(returns)), n_folds)
    out = []
    for idx in chunks:
        if len(idx) >= 2:
            out.append(_sharpe(returns.iloc[idx]))
    return [s for s in out if np.isfinite(s)]


def _jumps_per_year(regime) -> float:
    """Annualized count of regime transitions in a daily 0/1 label path.

    This measures the *detector's* raw smoothness (what lambda controls),
    independent of P&L — so it can decouple regime selection from the strategy.
    """
    regime = pd.Series(regime).dropna()
    if len(regime) < 2:
        return float("nan")
    transitions = int((regime.to_numpy()[1:] != regime.to_numpy()[:-1]).sum())
    years = len(regime) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    return transitions / years


def _jumps_score(
    jumps_per_year: float,
    target: float,
    *,
    over_weight: float = JUMP_OVER_WEIGHT,
    under_weight: float = JUMP_UNDER_WEIGHT,
) -> float:
    """Asymmetric closeness-to-target score (higher = better, max 0 at target).

    Deviations ABOVE the target (whipsaws) are penalized harder than deviations
    below it (over-smoothing), so the criterion prefers stickier regimes when in
    doubt — matching the jump penalty's whole purpose.
    """
    if not np.isfinite(jumps_per_year):
        return float("-inf")
    dev = jumps_per_year - target
    w = over_weight if dev > 0 else under_weight
    return -w * (dev**2)


def _span_metrics(strat_net: pd.Series) -> dict:
    """Backtest metrics (Sharpe, drawdown, annual return) on a sub-span of the
    DAILY strategy returns. Drawdown is computed *within* the span."""
    return backtest._metrics(pd.Series(strat_net).dropna())


def evaluate_lambda(
    feat: pd.DataFrame,
    jump_penalty: float,
    *,
    select_by: str = "sharpe",
    target_jumps: float = DEFAULT_TARGET_JUMPS_PER_YEAR,
    n_folds: int = 3,
    n_init: int = 5,
    max_iter: int = 30,
    refit_every: int | None = 63,
    max_oos_days: int | None = 252 * 8,
    eval_frac: float = DEFAULT_EVAL_FRAC,
    progress=None,
) -> LambdaResult:
    """Run one leak-free walk-forward for `jump_penalty`, then score it under
    NESTED splitting: choose on the earlier *selection* span, report on the
    later, held-out *evaluation* span.

    The walk-forward itself is already leak-free (expanding training windows
    that end before each predicted block). Here we additionally partition that
    OOS path in time so the selection criterion never reads the evaluation span.
    """
    signals = pipeline.walk_forward(
        feat,
        progress=progress,
        jump_penalty=jump_penalty,
        n_init=n_init,
        max_iter=max_iter,
        refit_every=refit_every,
        max_oos_days=max_oos_days,
    )
    res = backtest.run(feat, signals)
    rets = res["daily"]["strat_net"].dropna()

    # ---- nested split: earlier = selection, later = held-out evaluation ---- #
    n_total = len(rets)
    n_eval = int(np.ceil(n_total * eval_frac)) if n_total else 0
    n_eval = max(1, min(n_eval, n_total - 1)) if n_total > 1 else 0
    n_select = n_total - n_eval

    sel_rets = rets.iloc[:n_select]
    eval_rets = rets.iloc[n_select:]

    # Date boundary between the two spans, used to split the daily signal path
    # for the jump count (and to guarantee the spans are disjoint in time).
    split_ts = eval_rets.index[0] if len(eval_rets) else None
    if split_ts is not None:
        sel_reg = signals.loc[signals.index < split_ts, "predicted_regime"]
        eval_reg = signals.loc[signals.index >= split_ts, "predicted_regime"]
    else:
        sel_reg = signals["predicted_regime"]
        eval_reg = signals["predicted_regime"].iloc[0:0]

    # ---- selection-span statistics (the ONLY thing selection may use) ------ #
    sel_folds = _fold_sharpes(sel_rets, n_folds)
    cv_sharpe = float(np.mean(sel_folds)) if sel_folds else float("nan")
    cv_sharpe_std = float(np.std(sel_folds)) if len(sel_folds) > 1 else 0.0
    sel_jpy = _jumps_per_year(sel_reg)

    if select_by == "jumps":
        select_metric = "jumps"
        select_score = _jumps_score(sel_jpy, target_jumps)
    else:
        select_metric = "sharpe"
        select_score = cv_sharpe if np.isfinite(cv_sharpe) else float("-inf")

    # ---- evaluation-span metrics (reported only; never used for selection) - #
    em = _span_metrics(eval_rets)

    return LambdaResult(
        jump_penalty=float(jump_penalty),
        select_metric=select_metric,
        select_score=float(select_score),
        cv_sharpe=cv_sharpe,
        cv_sharpe_std=cv_sharpe_std,
        sel_jumps_per_year=sel_jpy,
        oos_sharpe=em.get("sharpe", float("nan")),
        max_drawdown=em.get("max_drawdown", float("nan")),
        annual_return=em.get("annual_return", float("nan")),
        eval_jumps_per_year=_jumps_per_year(eval_reg),
        n_folds=len(sel_folds),
        fold_sharpes=sel_folds,
    )


def tune_jump_penalty(
    feat: pd.DataFrame,
    grid: list[float] | None = None,
    *,
    select_by: str = "sharpe",
    target_jumps: float = DEFAULT_TARGET_JUMPS_PER_YEAR,
    n_folds: int = 3,
    n_init: int = 5,
    max_iter: int = 30,
    refit_every: int | None = 63,
    max_oos_days: int | None = 252 * 8,
    eval_frac: float = DEFAULT_EVAL_FRAC,
    progress=None,
) -> tuple[list[LambdaResult], LambdaResult]:
    """Sweep `grid` under NESTED CV and return (all_results_sorted, best).

    Selection rule (maximized, computed on the SELECTION span only):
      * ``select_by="sharpe"`` -> mean per-fold selection-span Sharpe.
      * ``select_by="jumps"``  -> asymmetric closeness to `target_jumps`/yr,
        which penalizes whipsaws harder than over-smoothing.

    The winner's headline metrics (`oos_sharpe`, `max_drawdown`, `annual_return`)
    come from the held-out EVALUATION span, which played no role in selection —
    so they are a genuine nested out-of-sample estimate.

    Results are returned sorted best-first. `progress` is a callable(done, total)
    over (lambda, window) pairs so the CLI can show a single combined bar.
    """
    grid = list(DEFAULT_GRID if grid is None else grid)
    results: list[LambdaResult] = []

    # Build a combined progress counter across all lambdas x windows.
    state = {"done": 0}
    total_windows = _estimate_total_windows(feat, refit_every, max_oos_days) * len(grid)

    def make_inner():
        def inner(_done_in_lambda, _total_in_lambda):
            state["done"] += 1
            if progress is not None:
                progress(state["done"], max(total_windows, state["done"]))

        return inner

    for lam in grid:
        results.append(
            evaluate_lambda(
                feat,
                lam,
                select_by=select_by,
                target_jumps=target_jumps,
                n_folds=n_folds,
                n_init=n_init,
                max_iter=max_iter,
                refit_every=refit_every,
                max_oos_days=max_oos_days,
                eval_frac=eval_frac,
                progress=make_inner(),
            )
        )

    def score(r: LambdaResult) -> tuple[float, float]:
        # Primary: the chosen selection criterion (selection span only).
        # Tie-break: lower whipsaw rate (stickier) on the selection span, so a
        # near-tie resolves toward the more-regularized, more-robust lambda.
        primary = r.select_score if np.isfinite(r.select_score) else -np.inf
        jpy = r.sel_jumps_per_year if np.isfinite(r.sel_jumps_per_year) else np.inf
        return (primary, -jpy)

    results.sort(key=score, reverse=True)
    return results, results[0]


def _estimate_total_windows(
    feat: pd.DataFrame, refit_every: int | None, max_oos_days: int | None
) -> int:
    """Estimate the number of walk-forward windows for one lambda (progress)."""
    from . import features

    reg_cols = features.available(feat, features.REGIME_FEATURES)
    pred_cols = features.available(feat, features.PREDICTOR_FEATURES)
    n = len(feat.dropna(subset=reg_cols + pred_cols + ["mkt_ret"]))
    start = config.TRAIN_MIN_DAYS
    if max_oos_days is not None and n - start > max_oos_days:
        start = n - int(max_oos_days)
    step = config.REFIT_EVERY_DAYS if refit_every is None else int(refit_every)
    return max(1, len(range(start, n, step)))


def results_to_frame(results: list[LambdaResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in results])


# --------------------------------------------------------------------------- #
# Persistence + config wiring
# --------------------------------------------------------------------------- #
def save_results(results: list[LambdaResult], best: LambdaResult) -> tuple[str, str]:
    """Save the sweep to reports/ as both CSV and a small JSON summary."""
    csv_path = config.REPORTS_DIR / "lambda_tuning.csv"
    json_path = config.REPORTS_DIR / "lambda_tuning.json"
    results_to_frame(results).to_csv(csv_path, index=False)
    summary = {
        "best_jump_penalty": best.jump_penalty,
        "select_metric": best.select_metric,
        "best_select_score": best.select_score,
        "best_cv_sharpe": best.cv_sharpe,
        "best_oos_sharpe_eval": best.oos_sharpe,
        "best_max_drawdown_eval": best.max_drawdown,
        "best_sel_jumps_per_year": best.sel_jumps_per_year,
        "grid": [r.jump_penalty for r in results],
        "results": [r.as_row() for r in results],
    }
    json_path.write_text(json.dumps(summary, indent=2))
    return str(csv_path), str(json_path)


def write_jump_penalty_to_config(value: float) -> str:
    """Update the `JUMP_PENALTY = ...` line in regime/config.py in place.

    We only touch that single assignment via a line-anchored regex, so the rest
    of the file (and its comments) is left untouched.
    """
    cfg_path = config.PROJECT_ROOT / "regime" / "config.py"
    text = cfg_path.read_text()
    pattern = re.compile(r"^(JUMP_PENALTY\s*=\s*).*$", re.MULTILINE)
    if not pattern.search(text):
        raise RuntimeError(f"Could not find JUMP_PENALTY assignment in {cfg_path}")
    new_text = pattern.sub(
        lambda m: f"{m.group(1)}{float(value)}  # tuned via `regime tune`",
        text,
        count=1,
    )
    cfg_path.write_text(new_text)
    return str(cfg_path)
