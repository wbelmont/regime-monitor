"""The end-to-end pipeline: data -> features -> regime labels -> next-regime model.

Two leak-free conventions are enforced here:

1. **Regime labels** are produced by a jump model that is fit *inside each
   training window* during the walk-forward (never on future data). This fixes
   the original project's biggest issue, where labels were fit on the full
   sample and then "predicted" out-of-sample (look-ahead bias that inflates the
   backtest).

2. The supervised model predicts the regime ONE STEP AHEAD at the cadence we
   actually act on (monthly), and is trained only on data available at the time.

The convention: regime 1 = "Bear" (lower average return), regime 0 = "Bull".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features
from .jump_model import ContinuousJumpModel


def _new_jump_model(**overrides):
    """Construct the regime labeler. We use the Continuous Jump Model (Shu &
    Mulvey, Sec. 2.4): it yields calibrated regime *probabilities* and smooth,
    consistent transitions, which the paper shows are more reliable than the
    discrete JM / HMM — and which directly feed this tool's bear-probability."""
    overrides.setdefault("n_states", config.N_REGIMES)
    # Treat an explicit None jump_penalty as "use the configured default", so
    # callers (e.g. the tuner) can pass an optional override uniformly.
    if overrides.get("jump_penalty") is None:
        overrides["jump_penalty"] = config.JUMP_PENALTY
    return ContinuousJumpModel(**overrides)


def _orient_bear_as_1(labels: np.ndarray, returns) -> np.ndarray:
    """Make the lower-mean-return regime = 1 (Bear) for consistent semantics."""
    returns = np.asarray(returns, dtype=float)
    r0 = returns[labels == 0].mean() if (labels == 0).any() else 0.0
    r1 = returns[labels == 1].mean() if (labels == 1).any() else 0.0
    # we want the WORSE regime to be labeled 1
    return labels if r1 <= r0 else 1 - labels


def cjm_feature_drivers(jm, reg_cols, x_row, bear_state) -> list[dict]:
    """Per-feature attribution of *why* the latest day leans bear vs bull.

    The CJM assigns a day's regime by its squared distance (in standardized
    feature space) to each centroid: a day is "more bear" when it sits closer to
    the bear centroid than the bull one. We decompose that distance gap per
    feature, so each feature gets a signed **bear pull**:

        bear_pull_i = (z_i - mu_bull_i)^2 - (z_i - mu_bear_i)^2

    where ``z`` is today's standardized feature vector and ``mu_*`` are the
    fitted bull/bear centroids (also standardized). ``bear_pull_i > 0`` means
    feature ``i`` pushes the day toward **bear** (it's closer to the bear
    centroid on that axis); ``< 0`` pushes toward bull. Summed over features it
    equals ``||z-mu_bull||^2 - ||z-mu_bear||^2`` — the exact quantity the model
    compares — so the per-feature parts are a faithful additive breakdown.

    Leak-free: uses ONLY the live-fitted scaler/centroids and today's features
    (no future data, no separate model). Does not alter the CJM in any way.

    Returns a list of dicts (one per feature), sorted by |bear_pull| desc:
        feature, value (raw), z (standardized today), bull_centroid_z,
        bear_centroid_z, bear_pull (signed, std units), share (|pull| fraction).
    """
    x_row = np.asarray(x_row, dtype=float).reshape(1, -1)
    z = jm.scaler.transform(x_row)[0]
    mu = np.asarray(jm.centroids_, dtype=float)  # (K, F), standardized space
    bear_mu = mu[bear_state]
    bull_mu = mu[1 - bear_state]
    pull = (z - bull_mu) ** 2 - (z - bear_mu) ** 2  # >0 -> pulls bear
    denom = float(np.abs(pull).sum()) or 1.0
    rows = [
        {
            "feature": reg_cols[i],
            "value": float(x_row[0, i]),
            "z": float(z[i]),
            "bull_centroid_z": float(bull_mu[i]),
            "bear_centroid_z": float(bear_mu[i]),
            "bear_pull": float(pull[i]),
            "share": float(abs(pull[i]) / denom),
        }
        for i in range(len(reg_cols))
    ]
    rows.sort(key=lambda r: abs(r["bear_pull"]), reverse=True)
    return rows


def reentry_overlay(
    bear_prob: pd.Series,
    feat: pd.DataFrame,
    *,
    rebound: float | None = None,
    lookback: int | None = None,
    cap: float | None = None,
    require_vix: bool | None = None,
) -> pd.DataFrame:
    """Leak-free re-entry / cover-short overlay on a bear-probability series.

    Returns a DataFrame aligned to `bear_prob.index` with:
      * ``bear_prob``        — the unchanged input (the product stays pure);
      * ``bear_prob_overlay``— the input, capped at ``cap`` whenever a rebound is
        CONFIRMED; and
      * ``reentry_flag``     — bool, True on days the override fires ("the bounce
        is confirmed → consider covering shorts / re-entering").

    Confirmation (all backward-looking → no look-ahead):
      * price (S&P) >= ``rebound`` above its trailing ``lookback``-day low, AND
      * (``require_vix``) VIX < its 21-day average (fear receding).

    Parameters default to ``config.REENTRY_*``. This only affects the OVERLAY
    column; it never modifies ``bear_prob`` itself. It addresses exit-of-short /
    re-entry timing only — NOT short-entry timing.
    """
    rebound = config.REENTRY_REBOUND if rebound is None else rebound
    lookback = config.REENTRY_LOOKBACK if lookback is None else lookback
    cap = config.REENTRY_CAP if cap is None else cap
    require_vix = config.REENTRY_REQUIRE_VIX if require_vix is None else require_vix

    idx = bear_prob.index
    price = feat["market"].reindex(idx).ffill()
    trail_low = price.rolling(lookback, min_periods=1).min()
    confirmed = (price / trail_low - 1.0) >= rebound
    if require_vix and "vix" in feat.columns:
        vix = feat["vix"].reindex(idx).ffill()
        vix_ma = vix.rolling(21, min_periods=1).mean()
        confirmed = confirmed & (vix < vix_ma)

    overlay = bear_prob.copy()
    overlay[confirmed] = np.minimum(overlay[confirmed], cap)
    return pd.DataFrame(
        {
            "bear_prob": bear_prob,
            "bear_prob_overlay": overlay,
            "reentry_flag": confirmed.reindex(idx).fillna(False),
        }
    )


def label_full_sample(
    feat: pd.DataFrame, jump_penalty: float | None = None
) -> pd.Series:
    """Label the entire dataset (used for *charts/insight*, not for backtesting).

    Note: because this uses all data it is in-sample; do not use it to judge
    performance. Use walk_forward() for honest out-of-sample results.
    """
    cols = features.available(feat, features.REGIME_FEATURES)
    sub = feat[cols].dropna()
    jm = _new_jump_model(
        jump_penalty=config.JUMP_PENALTY if jump_penalty is None else jump_penalty,
    )
    jm.fit(sub.values)
    labels = jm.predict(sub.values)
    labels = _orient_bear_as_1(labels, feat.loc[sub.index, "mkt_ret"].values)
    return pd.Series(labels, index=sub.index, name="regime")


def walk_forward(
    feat: pd.DataFrame,
    progress=None,
    *,
    jump_penalty: float | None = None,
    n_init: int = 10,
    max_iter: int = 50,
    refit_every: int | None = None,
    train_min: int | None = None,
    max_oos_days: int | None = None,
    return_nowcast: bool = False,
    signal_mode: str | None = None,
) -> pd.DataFrame:
    """Honest out-of-sample regime predictions.

    Returns a DataFrame indexed by date with columns:
        predicted_regime (0/1), bear_prob (0..1)

    If `return_nowcast=True`, also returns a `cjm_bear_nowcast` column: the
    Continuous Jump Model's OWN bear probability for each test day, produced by
    the train-window-fitted CJM via `predict_proba` (the paper's online
    inference). This is leak-free for the same reason as `bear_prob` — the CJM
    and its scaler were fit only on the training window that ends before the
    test block. It lets us compare the "pure CJM nowcast" against the
    GBM-forecast `bear_prob` from a SINGLE walk-forward pass.

    `progress` (optional): a callable(done, total) invoked after each step so a
    caller can render a progress bar. This keeps the long backtest from ever
    *looking* stalled.

    The keyword-only arguments let the tuner sweep the jump penalty and trade
    accuracy for speed during cross-validation WITHOUT changing the default
    behavior used by `regime backtest`:

    * `jump_penalty` overrides `config.JUMP_PENALTY` (this is what `regime tune`
      sweeps). `None` keeps the configured value.
    * `n_init`, `max_iter` control the CJM fit effort (lower = faster, coarser).
    * `refit_every` / `train_min` override the refit cadence / minimum training
      window (both in trading days). `None` uses the config defaults.
    * `max_oos_days` caps how many trading days of out-of-sample signal we
      generate, counted back from the end of the sample. This lets the tuner
      evaluate only a recent CV window so iterating stays fast; `None` uses the
      whole available out-of-sample span.
    """
    reg_cols = features.available(feat, features.REGIME_FEATURES)
    pred_cols = features.available(feat, features.PREDICTOR_FEATURES)
    df = feat.dropna(subset=reg_cols + pred_cols + ["mkt_ret"]).copy()

    mode = signal_mode or config.SIGNAL_MODE
    if mode not in ("cjm_nowcast", "gbm_forecast"):
        raise ValueError(f"unknown signal_mode: {mode!r}")

    dates, preds, probs = [], [], []
    nowcasts: list[float] = []
    n = len(df)
    start = config.TRAIN_MIN_DAYS if train_min is None else int(train_min)
    step = config.REFIT_EVERY_DAYS if refit_every is None else int(refit_every)
    # Optionally restrict to a recent out-of-sample window (counted in days of
    # OOS signal, i.e. from `start` to the end of the sample) for fast tuning.
    if max_oos_days is not None and n - start > max_oos_days:
        start = n - int(max_oos_days)
    steps = list(range(start, n, step))
    total = len(steps)

    for done, i in enumerate(steps, start=1):
        train = df.iloc[:i]
        test = df.iloc[i : i + step]
        if len(test) == 0:
            break

        # 1) Label regimes using ONLY the training window (leak-free, per the
        #    paper's online-inference requirement). Full rigor: 10 k-means++
        #    restarts as in Shu & Mulvey — accuracy over speed.
        jm = _new_jump_model(
            jump_penalty=jump_penalty, n_init=n_init, max_iter=max_iter
        )
        jm.fit(train[reg_cols].values)
        raw_labels = jm.predict(train[reg_cols].values)
        train_labels = _orient_bear_as_1(raw_labels, train["mkt_ret"].values)

        # The CJM's centroid index for "bear" = whichever raw state has the
        # lower mean training return (this is exactly what _orient_bear_as_1
        # used to decide the flip), so we can read the matching proba column.
        rets = np.asarray(train["mkt_ret"].values, dtype=float)
        r0 = rets[raw_labels == 0].mean() if (raw_labels == 0).any() else 0.0
        r1 = rets[raw_labels == 1].mean() if (raw_labels == 1).any() else 0.0
        bear_state = 1 if r1 <= r0 else 0

        # CJM's OWN bear probability over the test block (online inference with
        # the train-fitted centroids + scaler). Leak-free; cheap (block-sized).
        nowcast = jm.predict_proba(test[reg_cols].values)[:, bear_state]

        if mode == "gbm_forecast":
            # Legacy: gradient-boosted one-step-ahead forecast of the next-day
            # regime, trained on the CJM's hard labels.
            y = pd.Series(train_labels, index=train.index).shift(-1).dropna()
            X = train.loc[y.index, pred_cols]
            model = _new_classifier()
            model.fit(X, y)
            chosen = model.predict_proba(test[pred_cols])[:, 1]
        else:  # cjm_nowcast (default): the CJM's own probability IS the signal.
            chosen = nowcast

        chosen = np.asarray(chosen, dtype=float)
        dates.extend(test.index)
        probs.extend(chosen)
        preds.extend((chosen >= 0.5).astype(int))

        if return_nowcast:
            # Expose the CJM nowcast as a diagnostic column (equals bear_prob in
            # cjm_nowcast mode; the alternative signal in gbm_forecast mode).
            nowcasts.extend(nowcast)

        if progress is not None:
            progress(done, total)

    data = {"predicted_regime": preds, "bear_prob": probs}
    if return_nowcast:
        data["cjm_bear_nowcast"] = nowcasts
    out = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    out.index.name = "date"
    return out


def latest_signal(feat: pd.DataFrame) -> dict:
    """Train on ALL available history and report the *current* regime + the
    bear probability. This is what the daily/monthly monitor reports.

    Respects `config.SIGNAL_MODE`:
      * ``cjm_nowcast`` (default): `next_bear_prob` is the CJM's OWN bear
        probability for the latest day (online inference) — the same signal the
        backtest/harness use. No second model; feature importances are omitted.
      * ``gbm_forecast`` (legacy): `next_bear_prob` is the gradient-boosted
        one-step-ahead forecast, with permutation feature importances.
    """
    from sklearn.inspection import permutation_importance

    reg_cols = features.available(feat, features.REGIME_FEATURES)
    pred_cols = features.available(feat, features.PREDICTOR_FEATURES)
    df = feat.dropna(subset=reg_cols + pred_cols + ["mkt_ret"]).copy()

    jm = _new_jump_model()
    jm.fit(df[reg_cols].values)
    raw_labels = jm.predict(df[reg_cols].values)
    labels_arr = _orient_bear_as_1(raw_labels, df["mkt_ret"].values)
    labels = pd.Series(labels_arr, index=df.index)
    current_regime = int(labels.iloc[-1])

    rets = np.asarray(df["mkt_ret"].values, dtype=float)
    r0 = rets[raw_labels == 0].mean() if (raw_labels == 0).any() else 0.0
    r1 = rets[raw_labels == 1].mean() if (raw_labels == 1).any() else 0.0
    bear_state = 1 if r1 <= r0 else 0

    mode = config.SIGNAL_MODE
    importances: dict = {}

    if mode == "gbm_forecast":
        y = labels.shift(-1).dropna()
        X = df.loc[y.index, pred_cols]
        model = _new_classifier()
        model.fit(X, y)
        latest_row = df[pred_cols].iloc[[-1]]
        bear_prob = float(model.predict_proba(latest_row)[:, 1][0])
        # Which features mattered most (permutation importance for any model).
        try:
            pi = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            means = np.asarray(pi["importances_mean"])
            importances = dict(
                sorted(
                    zip(pred_cols, means.tolist()),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
            )
        except Exception:
            importances = {}
    else:  # cjm_nowcast: the CJM's own probability for the latest day.
        bear_prob = float(jm.predict_proba(df[reg_cols].values)[-1, bear_state])

    out = {
        "as_of": df.index[-1],
        "current_regime": current_regime,  # 0 bull, 1 bear (today's state)
        "next_bear_prob": bear_prob,  # bear probability (nowcast or forecast)
        "feature_importances": importances,
    }

    # Per-feature attribution of the live CJM regime label: *why* today leans
    # bear vs bull. Leak-free (uses only the fitted scaler/centroids + today's
    # features) and available in BOTH signal modes, since the CJM is always fit.
    try:
        out["drivers"] = cjm_feature_drivers(
            jm, reg_cols, df[reg_cols].iloc[-1].to_numpy(), bear_state
        )
    except Exception:
        out["drivers"] = []

    # Opt-in re-entry / cover-short overlay (default OFF). Separate from the
    # signal: reports a confirmed-rebound flag + the capped overlay reading for
    # the latest day, without altering `next_bear_prob`.
    if config.REENTRY_OVERLAY:
        bp_series = pd.Series([bear_prob], index=[df.index[-1]])
        ov = reentry_overlay(bp_series, feat)
        out["reentry_flag"] = bool(ov["reentry_flag"].iloc[-1])
        out["bear_prob_overlay"] = float(ov["bear_prob_overlay"].iloc[-1])

    return out


def _new_classifier():
    """Gradient-boosted trees for the next-regime prediction.

    We use scikit-learn's HistGradientBoostingClassifier rather than XGBoost: it
    is just as strong on this small tabular problem, ships with scikit-learn, and
    has NO native/OpenMP dependency — so the tool runs reliably on your Mac with
    nothing extra to install.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
