"""Static, phone-friendly dashboard for the regime monitor.

Renders a single self-contained `index.html` (inline CSS, embedded sparklines as
base64 PNGs) into `reports/site/`. Fully static, so it publishes anywhere free —
GitHub Pages, Netlify drop, or opened from iCloud Drive on your phone.

The page is organized as three LAYERED, independently-tracked signals so you can
see them firing one by one as regimes change:

  1. **P(bear) — the risk dial** (top). The continuous CJM bear probability, the
     thing to size aggressiveness on. Shown with fine precision so it's useful
     even when it sits near 0% in calm bulls.
  2. **Regime — Bull / Bear** (binary). The hard regime label, with its own
     step sparkline.
  3. **Signals** — the short-entry and long re-entry overlays, each tracked
     separately with a fired/armed state, last-fired date, and an event timeline.

Plus the per-feature "why" table (live CJM driver attribution) and recent calls.

Decision support only. Not financial advice.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io

import pandas as pd

from . import config

SITE_DIR = config.REPORTS_DIR / "site"

_STANCE_COLOR = {"BULL": "#16a34a", "NEUTRAL": "#ca8a04", "BEAR": "#dc2626"}
_REGIME_COLOR = {"Bull": "#16a34a", "Bear": "#dc2626"}
_FEATURE_LABELS = {
    "mkt_ret": "Daily return",
    "vol_21": "Realized vol (21d)",
    "vix": "VIX (implied vol)",
    "vol_premium": "Vol risk premium",
    "macd": "MACD (trend)",
    "ma_ratio": "50/200-day MA ratio",
    "mom_63": "Momentum (3mo)",
    "mom_126": "Momentum (6mo)",
    "mom_252": "Momentum (12mo)",
    "drawdown_63": "Drawdown (63d high)",
    "downside_dev_21": "Downside deviation",
    "curve_slope": "Yield-curve slope",
    "hy_oas_level": "HY credit spread",
}


def _flabel(key: str) -> str:
    return _FEATURE_LABELS.get(key, key)


def _fmt_prob(p: float) -> str:
    """Format a probability so the risk dial stays informative at LOW values.

    Rounding everything to whole percents shows ``0%`` ~90% of the time, which
    hides the dial's gradations. So: < 1% -> one decimal (e.g. ``0.3%``); the
    rest -> whole percent.
    """
    if p < 0.01:
        return f"{p * 100:.1f}%"
    return f"{p:.0%}"


def _dedup_daily(history: "pd.DataFrame | None") -> "pd.DataFrame | None":
    """Keep one row per date (the last run that day) and sort by date, so the
    sparklines/history aren't distorted by multiple intraday runs."""
    if history is None or history.empty or "date" not in history.columns:
        return history
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"]).sort_values("date")
    h = h.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return h


# --------------------------------------------------------------------------- #
# Sparklines (one per layer)
# --------------------------------------------------------------------------- #
def _new_ax(title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 1.6))
    ax.margins(x=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, fontsize=9, loc="left")
    return plt, fig, ax


def _encode(plt, fig) -> str:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _spark_prob(h: "pd.DataFrame | None", days: int = 180) -> str | None:
    """Continuous P(bear) line with the bull/bear threshold bands."""
    if h is None or h.empty or "next_bear_prob" not in h.columns:
        return None
    d = h.tail(days)
    plt, fig, ax = _new_ax("P(bear) — last 6 months (continuous risk dial)")
    ax.plot(d["date"], d["next_bear_prob"], color="#2563eb", lw=1.6)
    ax.axhline(config.BEAR_THRESHOLD, color="#dc2626", ls=":", lw=0.8)
    ax.axhline(config.BULL_THRESHOLD, color="#16a34a", ls=":", lw=0.8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1.0])
    return _encode(plt, fig)


def _spark_regime(h: "pd.DataFrame | None", days: int = 180) -> str | None:
    """Binary Bull(0)/Bear(1) step line."""
    if h is None or h.empty or "regime_binary" not in h.columns:
        return None
    d = h.tail(days)
    reg = pd.to_numeric(d["regime_binary"], errors="coerce")
    if reg.notna().sum() == 0:
        return None
    plt, fig, ax = _new_ax("Regime — Bull (0) / Bear (1)")
    ax.step(d["date"], reg, where="post", color="#7c3aed", lw=1.8)
    ax.fill_between(d["date"], 0, reg, step="post", color="#7c3aed", alpha=0.15)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Bull", "Bear"])
    return _encode(plt, fig)


def _spark_signals(h: "pd.DataFrame | None", days: int = 180) -> str | None:
    """Event timeline: markers when the short-entry / re-entry overlays fire."""
    if h is None or h.empty:
        return None
    has_re = "reentry_flag" in h.columns
    has_se = "short_entry_flag" in h.columns
    if not (has_re or has_se):
        return None
    d = h.tail(days)
    plt, fig, ax = _new_ax("Signals — short-entry & long re-entry events")
    plotted = False
    if has_se:
        se = pd.to_numeric(d["short_entry_flag"], errors="coerce").fillna(0)
        fired = d["date"][se > 0]
        ax.scatter(
            fired, [1] * len(fired), marker="v", color="#dc2626", s=42
        )
        plotted = plotted or len(fired) > 0
    if has_re:
        re = pd.to_numeric(d["reentry_flag"], errors="coerce").fillna(0)
        fired = d["date"][re > 0]
        ax.scatter(
            fired, [0] * len(fired), marker="^", color="#16a34a", s=42
        )
        plotted = plotted or len(fired) > 0
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Re-entry", "Short-entry"])
    if not plotted:
        ax.text(
            0.5, 0.5, "no signals fired in this window", ha="center",
            va="center", transform=ax.transAxes, fontsize=9, color="#9ca3af",
        )
    return _encode(plt, fig)


def _img(b64: str | None, empty_msg: str, *, style: str = "") -> str:
    if b64:
        st = f" style='{style}'" if style else ""
        return f"<img class='spark'{st} src='data:image/png;base64,{b64}' alt='sparkline'/>"
    return f"<div class='sub'>{html.escape(empty_msg)}</div>"


# --------------------------------------------------------------------------- #
# Signal-state helpers (current state + last-fired date)
# --------------------------------------------------------------------------- #
def _last_fired(h: "pd.DataFrame | None", col: str) -> str | None:
    if h is None or h.empty or col not in h.columns:
        return None
    flags = pd.to_numeric(h[col], errors="coerce").fillna(0)
    fired_dates = h["date"][flags > 0]
    if fired_dates.empty:
        return None
    return str(pd.to_datetime(fired_dates.iloc[-1]).date())


def _signal_row(
    name: str, active: bool, available: bool, last: str | None, hint: str
) -> str:
    if not available:
        state, color = "not yet built", "#64748b"
    elif active:
        state, color = "FIRED today", "#16a34a"
    else:
        state, color = "armed (idle)", "#9ca3af"
    last_txt = f"last fired {last}" if last else "never fired"
    return (
        f"<tr><td><b>{html.escape(name)}</b>"
        f"<div class='sub'>{html.escape(hint)}</div></td>"
        f"<td style='color:{color};font-weight:700;white-space:nowrap'>{state}</td>"
        f"<td class='num sub'>{html.escape(last_txt)}</td></tr>"
    )


def _num0(value) -> float:
    n = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(n) else float(n)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _drivers_rows(drivers: list[dict]) -> str:
    rows = []
    for d in drivers[:6]:
        toward = "BEAR" if d["bear_pull"] > 0 else "BULL"
        color = "#dc2626" if d["bear_pull"] > 0 else "#16a34a"
        zlabel = f"{d['z']:+.1f}\u03c3"
        rows.append(
            f"<tr><td>{html.escape(_flabel(d['feature']))}</td>"
            f"<td class='num'>{d['value']:.2f}</td>"
            f"<td class='num'>{zlabel}</td>"
            f"<td style='color:{color};font-weight:600'>{toward}</td>"
            f"<td class='num'>{d['share']:.0%}</td></tr>"
        )
    return "\n".join(rows)


def _history_rows(h: "pd.DataFrame | None", n: int = 12) -> str:
    if h is None or h.empty:
        return "<tr><td colspan='5'>No history yet.</td></tr>"
    tail = h.tail(n).iloc[::-1]
    rows = []
    for _, r in tail.iterrows():
        stance = str(r.get("stance", ""))
        scolor = _STANCE_COLOR.get(stance, "#64748b")
        regime = str(r.get("current_regime", ""))
        rcolor = _REGIME_COLOR.get(regime, "#64748b")
        bp = pd.to_numeric(pd.Series([r.get("next_bear_prob")]), errors="coerce").iloc[0]
        bp_txt = _fmt_prob(float(bp)) if pd.notna(bp) else "-"
        sigs = []
        if _num0(r.get("short_entry_flag", 0)) > 0:
            sigs.append("<span style='color:#dc2626'>short</span>")
        if _num0(r.get("reentry_flag", 0)) > 0:
            sigs.append("<span style='color:#16a34a'>re-entry</span>")
        sig_txt = " &middot; ".join(sigs) if sigs else "<span class='sub'>&mdash;</span>"
        date_txt = (
            str(pd.to_datetime(r["date"]).date()) if pd.notna(r.get("date")) else ""
        )
        rows.append(
            f"<tr><td>{html.escape(date_txt)}</td>"
            f"<td class='num'>{bp_txt}</td>"
            f"<td style='color:{rcolor};font-weight:600'>{html.escape(regime)}</td>"
            f"<td style='color:{scolor};font-weight:600'>{html.escape(stance)}</td>"
            f"<td>{sig_txt}</td></tr>"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def render(
    rec: dict, history: pd.DataFrame | None = None, filename: str = "index.html"
) -> str:
    """Write the dashboard HTML and return its path."""
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    h = _dedup_daily(history)

    stance = rec["stance"]
    scolor = _STANCE_COLOR.get(stance, "#64748b")
    bp = float(rec["next_bear_prob"])
    bp_pct = max(0.0, min(1.0, bp)) * 100.0
    regime = rec["current_regime"]
    rcolor = _REGIME_COLOR.get(regime, "#64748b")
    as_of = str(rec["as_of"])[:10]
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Layer sparklines ---
    spark_prob = _img(
        _spark_prob(h), "Run the monitor a few days to build this.",
        style="margin-top:12px",
    )
    spark_regime = _img(_spark_regime(h), "Regime history will appear here.")
    spark_signals = _img(_spark_signals(h), "Signal events will appear here.")

    # --- Signal states (each layer tracked independently) ---
    re_available = "reentry_flag" in rec or (
        h is not None and "reentry_flag" in h.columns
    )
    re_active = bool(rec.get("reentry_flag"))
    re_last = _last_fired(h, "reentry_flag")
    se_available = "short_entry_flag" in rec  # the overlay isn't built yet
    se_active = bool(rec.get("short_entry_flag"))
    se_last = _last_fired(h, "short_entry_flag")

    signal_rows = _signal_row(
        "Short-entry (call the top)", se_active, se_available, se_last,
        "Get short / buy puts when a top is confirmed.",
    ) + _signal_row(
        "Long re-entry (cover / re-enter)", re_active, re_available, re_last,
        "Cover shorts / re-enter longs once a rebound is confirmed.",
    )

    reentry_banner = ""
    if re_active:
        reentry_banner = (
            "<div class='card'><div class='banner ok'>\u2705 Long re-entry confirmed "
            f"(overlay {rec.get('bear_prob_overlay', 0):.0%}) — consider covering "
            "shorts / re-entering longs.</div></div>"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta http-equiv="refresh" content="3600"/>
<title>Regime Monitor — P(bear) {_fmt_prob(bp)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
          margin: 0; padding: 16px; background: #0b1020; color: #e5e7eb; }}
  .wrap {{ max-width: 720px; margin: 0 auto; }}
  .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 16px;
           padding: 18px; margin-bottom: 14px; }}
  .dial {{ text-align: center; }}
  .prob {{ font-size: 60px; font-weight: 800; margin: 2px 0; color: #2563eb; }}
  .label {{ font-size: 13px; text-transform: uppercase; letter-spacing: .08em;
            color: #9ca3af; }}
  .chip {{ display: inline-block; padding: 3px 12px; border-radius: 999px;
           font-weight: 700; font-size: 13px; margin-top: 6px; }}
  .gauge {{ position: relative; height: 12px; border-radius: 999px; margin: 14px 0 6px;
            background: linear-gradient(90deg,#16a34a 0%,#16a34a 40%,#ca8a04 40%,
            #ca8a04 60%,#dc2626 60%,#dc2626 100%); }}
  .needle {{ position: absolute; top: -4px; width: 3px; height: 20px;
             background: #e5e7eb; border-radius: 2px; box-shadow: 0 0 0 2px #0b1020; }}
  .gauge-scale {{ display: flex; justify-content: space-between; font-size: 11px;
                  color: #6b7280; }}
  .regime-badge {{ font-size: 30px; font-weight: 800; }}
  .sub {{ color: #9ca3af; font-size: 13px; }}
  .banner {{ padding: 10px 12px; border-radius: 10px; font-weight: 600; }}
  .banner.ok {{ background: #064e3b; color: #d1fae5; }}
  h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: .06em;
        color: #9ca3af; margin: 0 0 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  td, th {{ padding: 7px 6px; border-bottom: 1px solid #1f2937; text-align: left;
            vertical-align: top; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .spark {{ width: 100%; border-radius: 10px; background: #fff; margin-top: 4px; }}
  .actions td:first-child {{ color: #9ca3af; width: 96px; }}
  .foot {{ color: #6b7280; font-size: 12px; text-align: center; margin-top: 18px; }}
</style>
</head>
<body>
<div class="wrap">

  <div class="card dial">
    <div class="label">Probability of bear regime &middot; the risk dial</div>
    <div class="prob">{_fmt_prob(bp)}</div>
    <div class="gauge"><div class="needle" style="left:calc({bp_pct:.1f}% - 1.5px)"></div></div>
    <div class="gauge-scale"><span>0% risk-on</span><span>50%</span><span>100% risk-off</span></div>
    <div class="chip" style="background:{scolor}22;color:{scolor}">Stance: {stance}</div>
    <div class="sub" style="margin-top:8px">Size aggressiveness on this number. As of {as_of}.</div>
    {spark_prob}
  </div>

  <div class="card">
    <h2>Regime — binary call</h2>
    <div class="regime-badge" style="color:{rcolor}">{html.escape(regime)}</div>
    <div class="sub">The hard Bull/Bear label from the live CJM (argmax), separate from the dial above.</div>
    {spark_regime}
  </div>

  {reentry_banner}

  <div class="card">
    <h2>Signals — layered overlays</h2>
    <table>
      <tr><th>Signal</th><th>State</th><th class="num">History</th></tr>
      {signal_rows}
    </table>
    {spark_signals}
    <div class="sub" style="margin-top:8px">Each overlay is tracked independently of the dial and the regime label. Short-entry timing is still in development.</div>
  </div>

  <div class="card">
    <h2>Why — what's driving today's read (live CJM)</h2>
    <table>
      <tr><th>Feature</th><th class="num">Current</th><th class="num">vs normal</th><th>Pushing</th><th class="num">Weight</th></tr>
      {_drivers_rows(rec.get("drivers") or [])}
    </table>
    <div class="sub" style="margin-top:8px">Each feature's lean toward the bear vs bull centroid. Leak-free attribution of the live model.</div>
  </div>

  <div class="card">
    <h2>Suggested stance</h2>
    <table class="actions">
      <tr><td>401k</td><td>{html.escape(rec["fidelity_401k"])}</td></tr>
      <tr><td>thinkorswim</td><td>{html.escape(rec["thinkorswim"])}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Recent calls — all layers</h2>
    <table>
      <tr><th>Date</th><th class="num">P(bear)</th><th>Regime</th><th>Stance</th><th>Signals</th></tr>
      {_history_rows(h)}
    </table>
  </div>

  <div class="foot">Generated {generated} &middot; decision support only — not financial advice.</div>
</div>
</body>
</html>
"""
    path = SITE_DIR / filename
    path.write_text(doc, encoding="utf-8")
    return str(path)


def load_history() -> pd.DataFrame | None:
    """Read the signal-history CSV the CLI maintains, if present."""
    f = config.SIGNAL_HISTORY_FILE
    if not f.exists():
        return None
    try:
        return pd.read_csv(f)
    except Exception:
        return None
