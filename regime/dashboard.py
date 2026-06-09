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

# Sparklines now default to a ~2-week window (10 trading days) so day-to-day
# moves are legible; pass a larger `days` for more context where useful.
DEFAULT_SPARK_DAYS = 10

# Dark chart palette (matches the dashboard cards so charts blend in rather than
# sitting on a jarring white block).
_CHART = {
    "bg": "none",  # transparent → shows the card background through the PNG
    "fg": "#cbd5e1",  # axis text / ticks
    "grid": "#1f2937",  # subtle gridlines
    "ink": "#e5e7eb",  # last-value annotation text
    "prob": "#60a5fa",  # P(bear) line
    "regime": "#a78bfa",  # regime step
    "frag": "#f59e0b",  # fragility composite
    "bull": "#34d399",
    "bear": "#f87171",
}
# Distinct hues for the fragility component lines (kept color-blind-friendly-ish).
_FRAG_COMPONENT_COLORS = {
    "term_structure": "#f59e0b",
    "vix_velocity": "#fb7185",
    "vvix": "#f472b6",
    "skew": "#c084fc",
    "credit": "#22d3ee",
    "breadth": "#34d399",
    "defensive_staples": "#a3e635",
    "defensive_xlu": "#94a3b8",
}

_STANCE_COLOR = {"BULL": "#16a34a", "NEUTRAL": "#ca8a04", "BEAR": "#dc2626"}
_REGIME_COLOR = {"Bull": "#16a34a", "Bear": "#dc2626"}
# Short-entry FRAGILITY overlay: grade colors + human-readable driver labels.
_FRAGILITY_GRADE = {
    "none": ("calm", "#64748b"),
    "watch": ("WATCH", "#ca8a04"),
    "lean": ("LEAN", "#ea580c"),
    "act": ("ACT", "#dc2626"),
}
_FRAGILITY_LABELS = {
    "term_structure": "VIX term structure (curve flattening)",
    "vix_velocity": "VIX velocity (spot rising)",
    "vvix": "VVIX (vol-of-vol / tail demand)",
    "skew": "SKEW (cost of tail puts)",
    "credit": "Credit (HYG/LQD weakening)",
    "breadth": "Breadth (RSP/SPY narrowing)",
    "defensive_staples": "Defensive rotation (staples vs cyclicals)",
    "defensive_xlu": "Defensive rotation (utilities, gated)",
}
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
def _new_ax(title: str, *, height: float = 1.7):
    """A small, dark-themed chart that blends into the dashboard cards."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, height))
    fig.patch.set_alpha(0.0)  # transparent figure → card bg shows through
    ax.set_facecolor(_CHART["bg"])
    ax.margins(x=0.02)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_CHART["grid"])
    ax.tick_params(colors=_CHART["fg"], labelsize=8, length=0)
    ax.grid(True, axis="y", color=_CHART["grid"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9.5, loc="left", color=_CHART["ink"], pad=8)
    return plt, fig, ax


def _fmt_date_axis(ax, dates) -> None:
    """Daily ticks with short labels so each day in the ~2-week window is read."""
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d"))
    for lbl in ax.get_xticklabels():
        lbl.set_color(_CHART["fg"])


def _annot_last(ax, x, y, text: str, color: str) -> None:
    """Label the most-recent point so the current value is unmistakable."""
    ax.scatter([x], [y], s=28, color=color, zorder=5, edgecolors="none")
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9,
        fontweight="bold",
        color=color,
        clip_on=False,
    )


def _encode(plt, fig) -> str:
    fig.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _spark_prob(h: "pd.DataFrame | None", days: int = DEFAULT_SPARK_DAYS) -> str | None:
    """Continuous P(bear) line (last ~2 weeks) with the bull/bear bands."""
    if h is None or h.empty or "next_bear_prob" not in h.columns:
        return None
    d = h.tail(days)
    y = pd.to_numeric(d["next_bear_prob"], errors="coerce")
    plt, fig, ax = _new_ax("P(bear) — last 2 weeks (continuous risk dial)")
    # Threshold bands as soft fills so the green/amber/red zones read at a glance.
    ax.axhspan(config.BEAR_THRESHOLD, 1.03, color=_CHART["bear"], alpha=0.07)
    ax.axhspan(config.BULL_THRESHOLD, config.BEAR_THRESHOLD,
               color="#ca8a04", alpha=0.06)
    ax.axhspan(-0.03, config.BULL_THRESHOLD, color=_CHART["bull"], alpha=0.06)
    ax.plot(d["date"], y, color=_CHART["prob"], lw=2.0,
            marker="o", ms=3.5, mfc=_CHART["prob"], mec="none")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"])
    if y.notna().any():
        _annot_last(ax, d["date"].iloc[-1], float(y.iloc[-1]),
                    _fmt_prob(float(y.iloc[-1])), _CHART["prob"])
    _fmt_date_axis(ax, d["date"])
    return _encode(plt, fig)


def _spark_regime(
    h: "pd.DataFrame | None", days: int = DEFAULT_SPARK_DAYS
) -> str | None:
    """Binary Bull(0)/Bear(1) step line (last ~2 weeks)."""
    if h is None or h.empty or "regime_binary" not in h.columns:
        return None
    d = h.tail(days)
    reg = pd.to_numeric(d["regime_binary"], errors="coerce")
    if reg.notna().sum() == 0:
        return None
    plt, fig, ax = _new_ax("Regime — Bull (0) / Bear (1), last 2 weeks")
    ax.step(d["date"], reg, where="post", color=_CHART["regime"], lw=2.0)
    ax.fill_between(d["date"], 0, reg, step="post",
                    color=_CHART["regime"], alpha=0.18)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Bull", "Bear"])
    _fmt_date_axis(ax, d["date"])
    return _encode(plt, fig)


def _spark_fragility(
    frag: "pd.DataFrame | None", days: int = DEFAULT_SPARK_DAYS
) -> str | None:
    """Dense fragility chart (last ~2 weeks): composite + top component drivers.

    Unlike the other sparklines (which read the sparse logged history CSV), this
    is computed from cached price data, so it's rich on day one and lets you
    watch each early-warning tell move day to day. `frag` is the DataFrame from
    `pipeline.fragility_score()` (composite ``fragility`` + per-component
    sub-scores). Display-only.
    """
    if frag is None or len(frag) == 0 or "fragility" not in frag.columns:
        return None
    d = frag.tail(days)
    comp = float(pd.to_numeric(d["fragility"], errors="coerce").iloc[-1])
    plt, fig, ax = _new_ax(
        "Fragility — composite & drivers (last 2 weeks)", height=2.2
    )
    # Threshold bands (WATCH/LEAN/ACT) as soft horizontal zones.
    ax.axhspan(config.FRAGILITY_ACT, 1.0, color=_CHART["bear"], alpha=0.08)
    ax.axhspan(config.FRAGILITY_LEAN, config.FRAGILITY_ACT,
               color="#ea580c", alpha=0.07)
    ax.axhspan(config.FRAGILITY_WATCH, config.FRAGILITY_LEAN,
               color="#ca8a04", alpha=0.06)
    # Component lines: the most-active ones (highest latest sub-score) on top.
    comp_cols = [
        c for c in frag.columns
        if c not in ("fragility", "grade") and pd.to_numeric(d[c], errors="coerce").notna().any()
    ]
    comp_cols.sort(
        key=lambda c: float(pd.to_numeric(d[c], errors="coerce").iloc[-1] or 0),
        reverse=True,
    )
    for c in comp_cols:
        ys = pd.to_numeric(d[c], errors="coerce")
        ax.plot(d.index, ys, lw=1.1, alpha=0.65,
                color=_FRAG_COMPONENT_COLORS.get(c, "#94a3b8"),
                label=_FRAGILITY_LABELS.get(c, c).split(" (")[0])
    # Composite on top, bold, with the current value labeled.
    ax.plot(d.index, pd.to_numeric(d["fragility"], errors="coerce"),
            color=_CHART["frag"], lw=2.6, marker="o", ms=3.5, mec="none",
            label="Composite", zorder=6)
    _annot_last(ax, d.index[-1], comp, f"{comp * 100:.0f}%", _CHART["frag"])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, config.FRAGILITY_WATCH, config.FRAGILITY_LEAN,
                   config.FRAGILITY_ACT, 1.0])
    ax.set_yticklabels(["0", "WATCH", "LEAN", "ACT", "1"])
    _fmt_date_axis(ax, d.index)
    leg = ax.legend(loc="upper left", ncol=2, fontsize=7, frameon=False,
                    labelcolor=_CHART["fg"], handlelength=1.2,
                    columnspacing=1.0, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(_CHART["fg"])
    return _encode(plt, fig)


def _spark_signals(
    h: "pd.DataFrame | None", days: int = DEFAULT_SPARK_DAYS
) -> str | None:
    """Event timeline: markers when the short-entry / re-entry overlays fire."""
    if h is None or h.empty:
        return None
    has_re = "reentry_flag" in h.columns
    has_se = "short_entry_flag" in h.columns
    if not (has_re or has_se):
        return None
    d = h.tail(days)
    plt, fig, ax = _new_ax("Overlay events — last 2 weeks")
    plotted = False
    if has_se:
        se = pd.to_numeric(d["short_entry_flag"], errors="coerce").fillna(0)
        fired = d["date"][se > 0]
        ax.scatter(fired, [1] * len(fired), marker="v",
                   color=_CHART["bear"], s=60, zorder=5)
        plotted = plotted or len(fired) > 0
    if has_re:
        re = pd.to_numeric(d["reentry_flag"], errors="coerce").fillna(0)
        fired = d["date"][re > 0]
        ax.scatter(fired, [0] * len(fired), marker="^",
                   color=_CHART["bull"], s=60, zorder=5)
        plotted = plotted or len(fired) > 0
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Re-entry", "Short-entry"])
    _fmt_date_axis(ax, d["date"])
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "no overlay events in this window",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="#9ca3af",
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


def _fragility_card(rec: dict, frag: "pd.DataFrame | None" = None) -> str:
    """Dedicated card for the LEADING short-entry FRAGILITY score (0-100%).

    Graded WATCH / LEAN / ACT against the ``config.FRAGILITY_*`` thresholds, with
    a banded gauge and the top component drivers. Display-only — never touches
    `bear_prob`. Renders a graceful 'inputs unavailable' state when the extra
    Yahoo inputs aren't present (so the card always appears once the overlay is
    enabled).
    """
    if "fragility_score" not in rec:
        return ""  # overlay not enabled / not emitted by this signal

    score = rec.get("fragility_score")
    grade = str(rec.get("fragility_grade", "none"))
    glabel, gcolor = _FRAGILITY_GRADE.get(grade, ("calm", "#64748b"))
    watch = config.FRAGILITY_WATCH * 100.0
    lean = config.FRAGILITY_LEAN * 100.0
    act = config.FRAGILITY_ACT * 100.0

    # Threshold-banded gauge: calm -> WATCH -> LEAN -> ACT (gold -> orange -> red).
    gauge_bg = (
        "linear-gradient(90deg,"
        f"#1f2937 0%,#1f2937 {watch:.0f}%,"
        f"#ca8a04 {watch:.0f}%,#ca8a04 {lean:.0f}%,"
        f"#ea580c {lean:.0f}%,#ea580c {act:.0f}%,"
        f"#dc2626 {act:.0f}%,#dc2626 100%)"
    )

    if score is None or pd.isna(pd.to_numeric(pd.Series([score]), errors="coerce")[0]):
        body = (
            "<div class='prob' style='color:#64748b;font-size:40px'>—</div>"
            "<div class='sub'>Fragility inputs (VIX term structure, VVIX, SKEW, "
            "credit/breadth/defensive ETFs) are unavailable right now; the score "
            "will populate once they refresh.</div>"
        )
    else:
        s_pct = max(0.0, min(1.0, float(score))) * 100.0
        drivers = rec.get("fragility_drivers") or []
        rows = []
        for name, sub in drivers[:4]:
            sval = _num0(sub)
            scol = "#dc2626" if sval >= 0.5 else "#9ca3af"
            label = _FRAGILITY_LABELS.get(str(name), str(name))
            rows.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td class='num' style='color:{scol};font-weight:600'>"
                f"{sval:.0%}</td></tr>"
            )
        drivers_tbl = (
            "<table style='margin-top:10px'>"
            "<tr><th>Top fragility drivers</th><th class='num'>Stress</th></tr>"
            + "\n".join(rows)
            + "</table>"
            if rows
            else "<div class='sub' style='margin-top:8px'>No component drivers "
            "available.</div>"
        )
        body = (
            f"<div class='prob' style='color:{gcolor}'>{s_pct:.0f}%</div>"
            f"<div class='gauge' style='background:{gauge_bg}'>"
            f"<div class='needle' style='left:calc({s_pct:.1f}% - 1.5px)'></div></div>"
            "<div class='gauge-scale'><span>calm</span>"
            f"<span>WATCH {watch:.0f}</span><span>LEAN {lean:.0f}</span>"
            f"<span>ACT {act:.0f}</span></div>"
            f"<div class='chip' style='background:{gcolor}22;color:{gcolor};"
            f"margin-top:10px'>{glabel}</div>"
            f"{drivers_tbl}"
        )

    return f"""
  <div class="card dial">
    <h2 style="text-align:left">Short-entry fragility &middot; leading early-warning</h2>
    {body}
    {_img(_spark_fragility(frag), "Fragility history will appear here.", style="margin-top:12px")}
    <div class="sub" style="margin-top:10px;text-align:left">A LEADING gauge for
      buying protection while it's still cheap — the opposite loss function from
      re-entry, so it can read elevated with stocks near highs and VIX low. Each
      driver is a drift-robust z-score of a recent <i>change</i>. Display-only:
      it never moves the risk dial. Early false positives are expected (and
      cheap) — treat ACT as &ldquo;scale into protection&rdquo;, not all-in.</div>
  </div>"""


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
        bp = pd.to_numeric(pd.Series([r.get("next_bear_prob")]), errors="coerce").iloc[
            0
        ]
        bp_txt = _fmt_prob(float(bp)) if pd.notna(bp) else "-"
        sigs = []
        if _num0(r.get("short_entry_flag", 0)) > 0:
            sigs.append("<span style='color:#dc2626'>short</span>")
        if _num0(r.get("reentry_flag", 0)) > 0:
            sigs.append("<span style='color:#16a34a'>re-entry</span>")
        sig_txt = (
            " &middot; ".join(sigs) if sigs else "<span class='sub'>&mdash;</span>"
        )
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
    rec: dict,
    history: pd.DataFrame | None = None,
    filename: str = "index.html",
    fragility: pd.DataFrame | None = None,
) -> str:
    """Write the dashboard HTML and return its path.

    `fragility` (optional) is the dense per-day fragility series from
    `pipeline.fragility_score()` — used for the rich fragility chart. When
    omitted, the fragility card still renders (gauge + drivers), just without
    the time-series chart.
    """
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
        _spark_prob(h),
        "Run the monitor a few days to build this.",
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
        "Short-entry (call the top)",
        se_active,
        se_available,
        se_last,
        "Get short / buy puts when a top is confirmed.",
    ) + _signal_row(
        "Long re-entry (cover / re-enter)",
        re_active,
        re_available,
        re_last,
        "Cover shorts / re-enter longs once a rebound is confirmed.",
    )

    reentry_banner = ""
    if re_active:
        reentry_banner = (
            "<div class='card'><div class='banner ok'>\u2705 Long re-entry confirmed "
            f"(overlay {rec.get('bear_prob_overlay', 0):.0%}) — consider covering "
            "shorts / re-entering longs.</div></div>"
        )

    fragility_card = _fragility_card(rec, fragility)

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta http-equiv="refresh" content="3600"/>
<title>Regime Monitor — P(bear) {_fmt_prob(bp)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900&display=swap" rel="stylesheet"/>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Lato', -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
          margin: 0; padding: 18px; background: #0b1020; color: #e5e7eb;
          -webkit-font-smoothing: antialiased; letter-spacing: .1px; }}
  .wrap {{ max-width: 720px; margin: 0 auto; }}
  .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 16px;
           padding: 18px; margin-bottom: 14px;
           box-shadow: 0 1px 2px rgba(0,0,0,.35); }}
  .dial {{ text-align: center; }}
  .prob {{ font-size: 60px; font-weight: 900; margin: 2px 0; color: #2563eb;
           font-variant-numeric: tabular-nums; line-height: 1.05; }}
  .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .12em;
            color: #9ca3b8; }}
  .chip {{ display: inline-block; padding: 3px 12px; border-radius: 999px;
           font-weight: 700; font-size: 13px; margin-top: 6px;
           letter-spacing: .04em; }}
  .gauge {{ position: relative; height: 12px; border-radius: 999px; margin: 14px 0 6px;
            background: linear-gradient(90deg,#16a34a 0%,#16a34a 40%,#ca8a04 40%,
            #ca8a04 60%,#dc2626 60%,#dc2626 100%); }}
  .needle {{ position: absolute; top: -4px; width: 3px; height: 20px;
             background: #e5e7eb; border-radius: 2px; box-shadow: 0 0 0 2px #0b1020; }}
  .gauge-scale {{ display: flex; justify-content: space-between; font-size: 11px;
                  color: #6b7280; }}
  .regime-badge {{ font-size: 30px; font-weight: 900; }}
  .sub {{ color: #9ca3b8; font-size: 13px; line-height: 1.5; }}
  .banner {{ padding: 10px 12px; border-radius: 10px; font-weight: 700; }}
  .banner.ok {{ background: #064e3b; color: #d1fae5; }}
  h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: .1em;
        font-weight: 700; color: #9ca3b8; margin: 0 0 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  td, th {{ padding: 7px 6px; border-bottom: 1px solid #1f2937; text-align: left;
            vertical-align: top; }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
        color: #6b7280; font-weight: 700; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .spark {{ width: 100%; margin-top: 4px; background: transparent; }}
  .actions td:first-child {{ color: #9ca3b8; width: 96px; }}
  .foot {{ color: #6b7280; font-size: 12px; text-align: center; margin-top: 18px; }}
</style>
</head>
<body>
<div class="wrap">

  <div class="card dial">
    <div class="label">Probability of bear regime &middot; the risk dial</div>
    <div class="prob>{_fmt_prob(bp)}</div>
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
    <div class="sub" style="margin-top:8px">Each overlay is tracked independently of the dial and the regime label. The short-entry timing is the graded fragility score below.</div>
  </div>

  {fragility_card}

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
