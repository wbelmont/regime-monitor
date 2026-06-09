"""Static, phone-friendly dashboard for the regime monitor.

Renders a single self-contained `index.html` (inline CSS + SVG, embedded charts
as base64 PNGs) into `reports/site/`. Fully static, so it publishes anywhere
free — GitHub Pages, Netlify drop, or opened from iCloud Drive on your phone.

Design language: a true dark "quant terminal" look (Robinhood / Loop / Opto /
Coatue-ish) — near-black canvas, one elevated card surface, a single risk
gradient (mint → amber → coral), tabular numerals, generous whitespace.

Information architecture, top → bottom:

  1. **The CJM risk dial** — big, centered. A continuous bear-probability arc
     gauge; the single most valuable, finely-tuned number in the system. Size
     aggressiveness on this. Carries a small 2-week trend line.
  2. **How this works** — a plain-English explainer of the CJM + the leak-free
     walk-forward nowcast, so the dial is maximally interpretable.
  3. **Why — today's drivers** — the live, leak-free per-feature attribution of
     the dial (which features pull bear vs bull, and how hard).
  4. **Overlays** — the long re-entry timing tile (cover / re-enter), kept as a
     clean status tile rather than a near-empty event timeline.
  5. **Short-entry fragility** — the leading early-warning score, with a
     component "ignition" heatmap so you can trace which stress tells lit up
     first, then next.
  6. **Suggested stance** — the 401k / thinkorswim playbook.
  7. **Recent calls** — the running history (kept at the bottom).

Decision support only. Not financial advice.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import math

import pandas as pd

from . import config

SITE_DIR = config.REPORTS_DIR / "site"

# Sparklines / charts default to a ~2-week window (10 trading days) so day-to-day
# moves are legible; pass a larger `days` for more context where useful.
DEFAULT_SPARK_DAYS = 10

# --------------------------------------------------------------------------- #
# Palette — one cohesive dark theme, reused by CSS + matplotlib so charts blend
# seamlessly into the cards. The risk ramp (mint → amber → coral) is the spine
# of the whole design.
# --------------------------------------------------------------------------- #
C = {
    "bg": "#06070a",          # page canvas (near-black)
    "surface": "#0e1015",     # card surface
    "surface_2": "#13161d",   # nested / inset surface
    "line": "#1d212b",        # hairline borders / gridlines
    "ink": "#eef1f6",         # primary text
    "ink_dim": "#9aa1b1",     # secondary text
    "ink_faint": "#5b6271",   # tertiary / captions
    "risk_lo": "#21d07a",     # risk-on (low P(bear)) — refined mint
    "risk_mid": "#f3b13c",    # neutral — amber
    "risk_hi": "#ff5d63",     # risk-off (high P(bear)) — coral
    "accent": "#7aa2ff",      # cool indigo accent (P(bear) line, links)
}

# Stance + regime chip colors map onto the same risk ramp.
_STANCE_COLOR = {"BULL": C["risk_lo"], "NEUTRAL": C["risk_mid"], "BEAR": C["risk_hi"]}
_REGIME_COLOR = {"Bull": C["risk_lo"], "Bear": C["risk_hi"]}

# Short-entry FRAGILITY grade chips.
_FRAGILITY_GRADE = {
    "none": ("CALM", C["ink_faint"]),
    "watch": ("WATCH", C["risk_mid"]),
    "lean": ("LEAN", "#ff8a3d"),
    "act": ("ACT", C["risk_hi"]),
}

# Chart palette (matplotlib) derived from the same theme.
_CHART = {
    "bg": "none",            # transparent → card surface shows through
    "fg": C["ink_dim"],
    "grid": C["line"],
    "ink": C["ink"],
    "prob": C["accent"],
    "frag": "#ff8a3d",
}

# Order fragility components from "structural / earliest-leading" to "late",
# used to lay out the ignition heatmap rows consistently.
_FRAG_ORDER = [
    "term_structure",
    "bond_vol",
    "skew",
    "vvix",
    "vix_velocity",
    "credit",
    "breadth",
    "defensive_staples",
    "defensive_xlu",
]

_FRAGILITY_LABELS = {
    "term_structure": "VIX term structure (curve flattening)",
    "vix_velocity": "VIX velocity (spot rising)",
    "vvix": "VVIX (vol-of-vol / tail demand)",
    "skew": "SKEW (cost of tail puts)",
    "bond_vol": "MOVE (bond-market vol)",
    "credit": "Credit (HYG/LQD weakening)",
    "breadth": "Breadth (RSP/SPY narrowing)",
    "defensive_staples": "Defensive rotation (staples vs cyclicals)",
    "defensive_xlu": "Defensive rotation (utilities, gated)",
}
# Compact labels for tight chart rows.
_FRAGILITY_SHORT = {
    "term_structure": "VIX term structure",
    "vix_velocity": "VIX velocity",
    "vvix": "VVIX",
    "skew": "SKEW",
    "bond_vol": "MOVE (bonds)",
    "credit": "Credit (HY/IG)",
    "breadth": "Breadth",
    "defensive_staples": "Staples rotation",
    "defensive_xlu": "Utilities (gated)",
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


def _fmt_pctile(p) -> str | None:
    """Format an empirical percentile (0–1) as an ordinal, e.g. ``92nd pct``."""
    n = pd.to_numeric(pd.Series([p]), errors="coerce").iloc[0]
    if pd.isna(n):
        return None
    v = int(round(max(0.0, min(1.0, float(n))) * 100))
    if 10 <= v % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(v % 10, "th")
    return f"{v}{suf} pct"


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
# The risk dial — an inline SVG semicircular gauge (crisp, self-contained, no
# PNG). The arc sweeps mint → amber → coral across 0 → 100% P(bear); a needle
# marks the live reading. This is THE product, so it's rendered carefully.
# --------------------------------------------------------------------------- #
def _risk_color(p: float) -> str:
    """Blend the risk ramp (mint → amber → coral) for a 0..1 probability."""
    def _hex2rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    stops = [(0.0, C["risk_lo"]), (0.5, C["risk_mid"]), (1.0, C["risk_hi"])]
    p = max(0.0, min(1.0, p))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p <= p1:
            t = 0.0 if p1 == p0 else (p - p0) / (p1 - p0)
            a, b = _hex2rgb(c0), _hex2rgb(c1)
            rgb = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
            return "#%02x%02x%02x" % rgb
    return stops[-1][1]


def _arc_point(cx: float, cy: float, r: float, frac: float) -> tuple[float, float]:
    """Point on a 180° arc (frac 0 = left/0%, frac 1 = right/100%)."""
    ang = math.pi * (1.0 - frac)  # 180° (left) → 0° (right)
    return cx + r * math.cos(ang), cy - r * math.sin(ang)


def _arc_gauge(p: float) -> str:
    """Return an inline SVG semicircular risk gauge for probability ``p``."""
    w, h = 320.0, 188.0
    cx, cy, r = w / 2.0, 168.0, 132.0
    stroke = 18.0
    p = max(0.0, min(1.0, p))
    needle = _risk_color(p)

    # Track arc (full 180°).
    tx0, ty0 = _arc_point(cx, cy, r, 0.0)
    tx1, ty1 = _arc_point(cx, cy, r, 1.0)
    track = f"M {tx0:.1f} {ty0:.1f} A {r} {r} 0 0 1 {tx1:.1f} {ty1:.1f}"

    # Threshold ticks (bull / bear cut-offs) as faint marks on the arc.
    ticks = []
    for thr in (config.BULL_THRESHOLD, config.BEAR_THRESHOLD):
        ix, iy = _arc_point(cx, cy, r - stroke / 2 - 2, thr)
        ox, oy = _arc_point(cx, cy, r + stroke / 2 + 2, thr)
        ticks.append(
            f"<line x1='{ix:.1f}' y1='{iy:.1f}' x2='{ox:.1f}' y2='{oy:.1f}' "
            f"stroke='{C['bg']}' stroke-width='2.4' />"
        )
    ticks_svg = "".join(ticks)

    # Needle.
    nx, ny = _arc_point(cx, cy, r + 6, p)
    bx, by = _arc_point(cx, cy, 14, p)

    return f"""
<svg viewBox="0 0 {w:.0f} {h:.0f}" class="gauge-svg" role="img" aria-label="risk gauge">
  <defs>
    <linearGradient id="riskramp" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"  stop-color="{C['risk_lo']}"/>
      <stop offset="50%" stop-color="{C['risk_mid']}"/>
      <stop offset="100%" stop-color="{C['risk_hi']}"/>
    </linearGradient>
  </defs>
  <path d="{track}" fill="none" stroke="{C['surface_2']}" stroke-width="{stroke + 6}"
        stroke-linecap="round"/>
  <path d="{track}" fill="none" stroke="url(#riskramp)" stroke-width="{stroke}"
        stroke-linecap="round" opacity="0.92"/>
  {ticks_svg}
  <line x1="{bx:.1f}" y1="{by:.1f}" x2="{nx:.1f}" y2="{ny:.1f}"
        stroke="{needle}" stroke-width="4" stroke-linecap="round"/>
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{C['ink']}"/>
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{needle}"/>
</svg>"""


# --------------------------------------------------------------------------- #
# Charts (dark-themed matplotlib, transparent so the card surface shows through)
# --------------------------------------------------------------------------- #
def _new_ax(*, height: float = 1.5, width: float = 7.0):
    """A small, dark-themed chart that blends into the dashboard cards."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_alpha(0.0)  # transparent figure → card surface shows through
    ax.set_facecolor(_CHART["bg"])
    ax.margins(x=0.02)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_CHART["grid"])
    ax.tick_params(colors=_CHART["fg"], labelsize=8, length=0)
    ax.set_axisbelow(True)
    return plt, fig, ax


def _fmt_date_axis(ax) -> None:
    """Daily ticks with short labels so each day in the ~2-week window is read."""
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d"))
    for lbl in ax.get_xticklabels():
        lbl.set_color(_CHART["fg"])


def _annot_last(ax, x, y, text: str, color: str) -> None:
    """Label the most-recent point so the current value is unmistakable."""
    ax.scatter([x], [y], s=26, color=color, zorder=5, edgecolors="none")
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
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _spark_prob(h: "pd.DataFrame | None", days: int = DEFAULT_SPARK_DAYS) -> str | None:
    """A minimal continuous P(bear) trend line for the last ~2 weeks."""
    if h is None or h.empty or "next_bear_prob" not in h.columns:
        return None
    d = h.tail(days)
    y = pd.to_numeric(d["next_bear_prob"], errors="coerce")
    if y.notna().sum() == 0:
        return None
    plt, fig, ax = _new_ax(height=1.25)
    ax.grid(True, axis="y", color=_CHART["grid"], lw=0.6, alpha=0.6)
    # Soft fill under the line for a richer, fintech-y read.
    ax.fill_between(d["date"], 0, y, color=_CHART["prob"], alpha=0.10)
    ax.plot(
        d["date"], y, color=_CHART["prob"], lw=2.0, marker="o", ms=3.0, mec="none"
    )
    hi = max(0.08, float(y.max()) * 1.35)
    ax.set_ylim(0, hi)
    ax.set_yticks([0, hi / 2, hi])
    ax.set_yticklabels([_fmt_prob(0.0), _fmt_prob(hi / 2), _fmt_prob(hi)])
    _annot_last(
        ax, d["date"].iloc[-1], float(y.iloc[-1]), _fmt_prob(float(y.iloc[-1])),
        _CHART["prob"],
    )
    _fmt_date_axis(ax)
    return _encode(plt, fig)


def _frag_ignition(
    frag: "pd.DataFrame | None", days: int = 30
) -> str | None:
    """Component-ignition heatmap: trace WHICH stress tells lit up, and WHEN.

    Replaces the old tangle-of-lines fragility chart. Rows are the stress
    components (ordered structural/early-leading → late), the x-axis is time,
    and each cell is shaded by that component's 0..1 stress sub-score
    (dark → amber → coral). Reading left→right per row shows the progression of
    each tell; reading top→bottom at a date shows which tells are lit. A thin
    composite track sits on top. Display-only; computed from cached price data
    so it's rich on day one.
    """
    if frag is None or len(frag) == 0 or "fragility" not in frag.columns:
        return None
    import numpy as np
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = frag.tail(days)
    comp_cols = [
        c
        for c in _FRAG_ORDER
        if c in d.columns and pd.to_numeric(d[c], errors="coerce").notna().any()
    ]
    # Append any components not in the canonical order (forward-compatible).
    comp_cols += [
        c
        for c in d.columns
        if c not in ("fragility", "grade")
        and c not in comp_cols
        and pd.to_numeric(d[c], errors="coerce").notna().any()
    ]
    if not comp_cols:
        return None

    mat = (
        d[comp_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .T  # rows = components, cols = time
    )
    comp = pd.to_numeric(d["fragility"], errors="coerce").to_numpy(dtype=float)

    nrows = len(comp_cols)
    fig, (axc, axh) = plt.subplots(
        2,
        1,
        figsize=(7.0, 0.42 * nrows + 1.5),
        gridspec_kw={"height_ratios": [1.1, max(2.2, 0.42 * nrows)], "hspace": 0.12},
    )
    fig.patch.set_alpha(0.0)

    # --- Composite track (top strip) over WATCH/LEAN/ACT bands ---
    axc.set_facecolor("none")
    for spine in ("top", "right", "left"):
        axc.spines[spine].set_visible(False)
    axc.spines["bottom"].set_visible(False)
    x = np.arange(len(comp))
    axc.axhspan(config.FRAGILITY_ACT, 1.0, color=C["risk_hi"], alpha=0.10)
    axc.axhspan(config.FRAGILITY_LEAN, config.FRAGILITY_ACT, color="#ff8a3d", alpha=0.10)
    axc.axhspan(config.FRAGILITY_WATCH, config.FRAGILITY_LEAN, color=C["risk_mid"], alpha=0.10)
    axc.fill_between(x, 0, comp, color=_CHART["frag"], alpha=0.14)
    axc.plot(x, comp, color=_CHART["frag"], lw=2.2)
    axc.set_xlim(-0.5, len(comp) - 0.5)
    axc.set_ylim(0, 1.0)
    axc.set_yticks([config.FRAGILITY_WATCH, config.FRAGILITY_LEAN, config.FRAGILITY_ACT])
    axc.set_yticklabels(["W", "L", "A"], fontsize=7.5, color=_CHART["fg"])
    axc.tick_params(length=0, colors=_CHART["fg"])
    axc.set_xticks([])
    if np.isfinite(comp[-1]):
        axc.scatter([x[-1]], [comp[-1]], s=24, color=_CHART["frag"], zorder=5)
        axc.annotate(
            f"{comp[-1] * 100:.0f}%",
            xy=(x[-1], comp[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=_CHART["frag"],
            clip_on=False,
        )
    axc.set_title("Composite", fontsize=8.5, loc="left", color=_CHART["fg"], pad=4)

    # --- Component ignition heatmap ---
    cmap = LinearSegmentedColormap.from_list(
        "stress", [C["surface_2"], "#3a2a18", C["risk_mid"], "#ff8a3d", C["risk_hi"]]
    )
    axh.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=(-0.5, mat.shape[1] - 0.5, nrows - 0.5, -0.5),
    )
    axh.set_yticks(range(nrows))
    axh.set_yticklabels(
        [_FRAGILITY_SHORT.get(c, c) for c in comp_cols], fontsize=8, color=_CHART["fg"]
    )
    # Date ticks along the bottom.
    dates = list(d.index)
    nticks = min(6, len(dates))
    tick_idx = np.linspace(0, len(dates) - 1, nticks).round().astype(int)
    axh.set_xticks(tick_idx)
    axh.set_xticklabels(
        [pd.to_datetime(dates[i]).strftime("%-m/%-d") for i in tick_idx],
        fontsize=8,
        color=_CHART["fg"],
    )
    axh.tick_params(length=0)
    for spine in axh.spines.values():
        spine.set_visible(False)
    # Faint row separators for legibility.
    for yy in range(nrows + 1):
        axh.axhline(yy - 0.5, color=C["bg"], lw=1.4)

    fig.subplots_adjust(left=0.20, right=0.98, top=0.93, bottom=0.10, hspace=0.12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(b64: str | None, empty_msg: str, *, style: str = "") -> str:
    if b64:
        st = f" style='{style}'" if style else ""
        return f"<img class='spark'{st} src='data:image/png;base64,{b64}' alt='chart'/>"
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


def _cond_row(ok: bool, text: str) -> str:
    mark = "✓" if ok else "○"
    col = C["risk_lo"] if ok else C["ink_faint"]
    return (
        f"<div class='cond'><span class='cmark' style='color:{col}'>{mark}</span>"
        f"<span style='color:{C['ink_dim'] if ok else C['ink_faint']}'>{text}</span></div>"
    )


def _overlay_tile(
    title: str,
    sub: str,
    *,
    active: bool,
    last: str | None,
    accent: str,
    diag: dict | None = None,
) -> str:
    """A clean status tile for an overlay (re-entry / cover-short timing).

    Shows the current armed/fired state, the last-fired date, and — when a
    ``diag`` dict is supplied — a per-condition checklist that explains WHY the
    gate is or isn't firing (so it's never an opaque "armed").
    """
    if active:
        state, scol = "FIRED today", accent
        dot = accent
    else:
        state, scol = "armed", C["ink_dim"]
        dot = C["ink_faint"]
    last_txt = f"Last fired {last}" if last else "Not yet fired"

    conds = ""
    if diag:
        reb = float(diag.get("rebound_pct", 0.0)) * 100.0
        thr = float(diag.get("rebound_threshold", 0.0)) * 100.0
        lb = int(diag.get("lookback", 0))
        rows = [
            _cond_row(
                bool(diag.get("cond_price")),
                f"S&amp;P {reb:+.1f}% off its {lb}-day low "
                f"(needs ≥ +{thr:.0f}%)",
            )
        ]
        if diag.get("require_vix"):
            vr = diag.get("vix_receding")
            rows.append(
                _cond_row(
                    bool(vr),
                    "VIX receding (below its 21-day average)"
                    if vr
                    else "VIX still elevated (≥ its 21-day average)",
                )
            )
        verdict = (
            "All conditions met — rebound confirmed."
            if active
            else "Waiting on the unmet condition(s) above."
        )
        conds = (
            "<div class='conds'>"
            + "".join(rows)
            + f"<div class='cond-foot'>{verdict}</div></div>"
        )

    return f"""
    <div class="tile">
      <div class="tile-head">
        <span class="dot" style="background:{dot}"></span>
        <span class="tile-title">{html.escape(title)}</span>
        <span class="tile-state" style="color:{scol}">{state}</span>
      </div>
      <div class="sub">{sub}</div>
      {conds}
      <div class="tile-foot">{html.escape(last_txt)}</div>
    </div>"""


def _num0(value) -> float:
    n = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(n) else float(n)


def _fragility_card(rec: dict, frag: "pd.DataFrame | None" = None) -> str:
    """Dedicated card for the LEADING short-entry FRAGILITY score (0-100%).

    Graded WATCH / LEAN / ACT against the ``config.FRAGILITY_*`` thresholds, with
    a slim banded meter, a grade chip, the component "ignition" heatmap (which
    tells lit up, and when), and the top drivers. Display-only — never touches
    `bear_prob`. Renders a graceful 'inputs unavailable' state when the extra
    Yahoo inputs aren't present.
    """
    if "fragility_score" not in rec:
        return ""  # overlay not enabled / not emitted by this signal

    score = rec.get("fragility_score")
    grade = str(rec.get("fragility_grade", "none"))
    glabel, gcolor = _FRAGILITY_GRADE.get(grade, ("CALM", C["ink_faint"]))
    watch = config.FRAGILITY_WATCH * 100.0
    lean = config.FRAGILITY_LEAN * 100.0
    act = config.FRAGILITY_ACT * 100.0

    # Threshold-banded meter: calm → WATCH → LEAN → ACT.
    meter_bg = (
        "linear-gradient(90deg,"
        f"{C['surface_2']} 0%,{C['surface_2']} {watch:.0f}%,"
        f"{C['risk_mid']} {watch:.0f}%,{C['risk_mid']} {lean:.0f}%,"
        f"#ff8a3d {lean:.0f}%,#ff8a3d {act:.0f}%,"
        f"{C['risk_hi']} {act:.0f}%,{C['risk_hi']} 100%)"
    )

    if score is None or pd.isna(pd.to_numeric(pd.Series([score]), errors="coerce")[0]):
        body = (
            f"<div class='megan' style='color:{C['ink_faint']};font-size:44px'>—</div>"
            "<div class='sub'>Fragility inputs (VIX term structure, VVIX, SKEW, "
            "credit/breadth/defensive ETFs) are unavailable right now; the score "
            "will populate once they refresh.</div>"
        )
    else:
        s_pct = max(0.0, min(1.0, float(score))) * 100.0
        drivers = rec.get("fragility_drivers") or []
        pctiles = rec.get("fragility_pctiles") or {}
        rows = []
        for name, sub in drivers[:4]:
            sval = _num0(sub)
            scol = C["risk_hi"] if sval >= 0.5 else C["ink_dim"]
            label = _FRAGILITY_LABELS.get(str(name), str(name))
            pct = _fmt_pctile(pctiles.get(name)) or "—"
            rows.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td class='num' style='color:{scol};font-weight:600'>"
                f"{sval:.0%}</td>"
                f"<td class='num sub'>{pct}</td></tr>"
            )
        drivers_tbl = (
            "<table style='margin-top:14px'>"
            "<tr><th>Top fragility drivers</th><th class='num'>Stress</th>"
            "<th class='num'>vs history</th></tr>"
            + "\n".join(rows)
            + "</table>"
            if rows
            else "<div class='sub' style='margin-top:8px'>No component drivers "
            "available.</div>"
        )
        comp_pct = _fmt_pctile(rec.get("fragility_pctile"))
        comp_pct_html = (
            f"<span class='chip' style='background:{C['surface_2']};"
            f"color:{C['ink_dim']}'>{comp_pct} vs history</span>"
            if comp_pct
            else ""
        )
        body = (
            f"<div class='frag-head'>"
            f"<div class='megan' style='color:{gcolor}'>{s_pct:.0f}%</div>"
            f"<span class='chip' style='background:{gcolor}1f;color:{gcolor}'>"
            f"{glabel}</span>{comp_pct_html}</div>"
            f"<div class='meter' style='background:{meter_bg}'>"
            f"<div class='meter-needle' style='left:calc({s_pct:.1f}% - 1.5px)'></div></div>"
            "<div class='meter-scale'><span>calm</span>"
            f"<span>watch {watch:.0f}</span><span>lean {lean:.0f}</span>"
            f"<span>act {act:.0f}</span></div>"
            f"{drivers_tbl}"
        )

    heat = _img(
        _frag_ignition(frag),
        "Fragility history will appear here.",
        style="margin-top:16px",
    )

    return f"""
  <section class="card">
    <h2>Short-entry fragility &middot; leading early-warning</h2>
    {body}
    <div class="chart-cap" style="margin-top:6px">Component ignition — which stress
      tells lit up, and when (left → right). Darker = calm, coral = stressed.</div>
    {heat}
    <p class="sub" style="margin-top:12px">A LEADING gauge for buying protection
      while it's still cheap — the opposite loss function from re-entry, so it can
      read elevated with stocks near highs and VIX low. Each driver is a
      drift-robust z-score of a recent <i>change</i>. Display-only: it never moves
      the risk dial. Early false positives are expected (and cheap) — treat ACT as
      &ldquo;scale into protection,&rdquo; not all-in.</p>
  </section>"""


def _stance_card(rec: dict) -> str:
    """Suggested stance: continuous exposure targets + the account playbook.

    No bonds anywhere — de-risking moves toward CASH / lower beta / hedges. The
    target beta and net-delta scale continuously with the dial (via
    ``recommend.exposure_targets``).
    """
    exp = rec.get("exposure") or {}
    beta = exp.get("target_beta")
    delta = exp.get("net_delta")
    lev_ok = bool(exp.get("leverage_ok"))

    metrics = ""
    if beta is not None:
        # Beta meter: 0 → TARGET_BETA_MAX, with 1.0 (market) marked.
        bmax = float(config.TARGET_BETA_MAX)
        bpos = max(0.0, min(1.0, float(beta) / bmax)) * 100.0
        mkt = max(0.0, min(1.0, 1.0 / bmax)) * 100.0
        dcol = (
            C["risk_lo"]
            if (delta or 0) > 0.05
            else C["risk_hi"]
            if (delta or 0) < -0.05
            else C["ink_dim"]
        )
        dword = (
            "net long"
            if (delta or 0) > 0.05
            else "net short"
            if (delta or 0) < -0.05
            else "neutral"
        )
        lev_chip = (
            f"<span class='chip' style='background:{C['risk_lo']}1f;color:{C['risk_lo']}'>"
            "options / leverage OK</span>"
            if lev_ok
            else f"<span class='chip' style='background:{C['ink_faint']}22;"
            f"color:{C['ink_dim']}'>no options / no leverage</span>"
        )
        metrics = f"""
    <div class="targets">
      <div class="target">
        <div class="t-val" style="color:{C['ink']}">{beta:.2f}<span class="t-unit">&beta;</span></div>
        <div class="t-lab">target equity beta</div>
        <div class="meter" style="background:{C['surface_2']}">
          <div class="meter-fill" style="width:{bpos:.0f}%;background:{C['accent']}"></div>
          <div class="meter-tick" style="left:{mkt:.0f}%"></div>
        </div>
        <div class="meter-scale"><span>cash</span><span>market 1.0&times;</span><span>{bmax:.2f}&times;</span></div>
      </div>
      <div class="target">
        <div class="t-val" style="color:{dcol}">{delta:+.2f}</div>
        <div class="t-lab">net delta &middot; {dword}</div>
        <div style="margin-top:8px">{lev_chip}</div>
      </div>
    </div>"""

    return f"""
  <section class="card">
    <h2>Suggested stance &middot; no bonds</h2>
    {metrics}
    <table class="actions" style="margin-top:6px">
      <tr><td>401k</td><td>{html.escape(rec["fidelity_401k"])}</td></tr>
      <tr><td>thinkorswim</td><td>{html.escape(rec["thinkorswim"])}</td></tr>
    </table>
    <p class="sub" style="margin-top:10px">Aggressiveness scales continuously with
      the dial: deep bull (≈0%) → leaned-in, options/leverage OK; toward the 40%
      line → 100% invested but plain beta, no options/leverage; bear → cut beta,
      raise cash &amp; hedge. Targets are derived, not advice.</p>
  </section>"""


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _drivers_rows(drivers: list[dict]) -> str:
    rows = []
    shown = drivers[:6]
    # Scale bars to the largest share among the shown drivers so the leader's
    # bar fills the track and the rest read proportionally — otherwise, with 8+
    # features, every share is ~10–15% and the bars all look uniformly tiny.
    max_share = max((float(d["share"]) for d in shown), default=0.0) or 1.0
    for d in shown:
        toward = "BEAR" if d["bear_pull"] > 0 else "BULL"
        color = C["risk_hi"] if d["bear_pull"] > 0 else C["risk_lo"]
        zlabel = f"{d['z']:+.1f}\u03c3"
        pct = _fmt_pctile(d.get("pctile"))
        vs_normal = (
            f"{zlabel}<div class='sub2'>{pct}</div>" if pct else zlabel
        )
        share = max(0.0, min(1.0, float(d["share"])))
        width = max(6.0, min(100.0, share / max_share * 100.0))
        bar = (
            f"<div class='bar'><div class='bar-fill' "
            f"style='width:{width:.0f}%;background:{color}'></div></div>"
        )
        rows.append(
            f"<tr><td>{html.escape(_flabel(d['feature']))}</td>"
            f"<td class='num'>{d['value']:.2f}</td>"
            f"<td class='num'>{vs_normal}</td>"
            f"<td style='color:{color};font-weight:600'>{toward}</td>"
            f"<td class='num'>{bar}<span class='bar-pct'>{share:.0%}</span></td></tr>"
        )
    return "\n".join(rows)


def _history_rows(h: "pd.DataFrame | None", n: int = 12) -> str:
    if h is None or h.empty:
        return "<tr><td colspan='5'>No history yet.</td></tr>"
    tail = h.tail(n).iloc[::-1]
    rows = []
    for _, r in tail.iterrows():
        stance = str(r.get("stance", ""))
        scolor = _STANCE_COLOR.get(stance, C["ink_faint"])
        regime = str(r.get("current_regime", ""))
        rcolor = _REGIME_COLOR.get(regime, C["ink_faint"])
        bp = pd.to_numeric(pd.Series([r.get("next_bear_prob")]), errors="coerce").iloc[
            0
        ]
        bp_txt = _fmt_prob(float(bp)) if pd.notna(bp) else "-"
        sigs = []
        if _num0(r.get("short_entry_flag", 0)) > 0:
            sigs.append(f"<span style='color:{C['risk_hi']}'>short</span>")
        if _num0(r.get("reentry_flag", 0)) > 0:
            sigs.append(f"<span style='color:{C['risk_lo']}'>re-entry</span>")
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
    `pipeline.fragility_score()` — used for the component-ignition heatmap. When
    omitted, the fragility card still renders (meter + drivers), just without
    the time-series heatmap.
    """
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    h = _dedup_daily(history)

    stance = rec["stance"]
    scolor = _STANCE_COLOR.get(stance, C["ink_faint"])
    bp = float(rec["next_bear_prob"])
    regime = rec["current_regime"]
    rcolor = _REGIME_COLOR.get(regime, C["ink_faint"])
    as_of = str(rec["as_of"])[:10]
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    gauge_svg = _arc_gauge(bp)
    spark_prob = _img(_spark_prob(h), "", style="margin-top:8px")

    # --- Overlay states (re-entry timing tile) ---
    re_active = bool(rec.get("reentry_flag"))
    re_last = _last_fired(h, "reentry_flag")
    reentry_tile = _overlay_tile(
        "Long re-entry — cover / re-enter",
        "Caps the risk dial once a rebound is confirmed (S&amp;P up off its low + "
        "VIX receding). The timing aid for getting back in.",
        active=re_active,
        last=re_last,
        accent=C["risk_lo"],
        diag=rec.get("reentry_diag"),
    )

    reentry_banner = ""
    if re_active:
        reentry_banner = (
            "<div class='banner ok'>Long re-entry confirmed "
            f"(overlay {rec.get('bear_prob_overlay', 0):.0%}) — consider covering "
            "shorts / re-entering longs.</div>"
        )

    fragility_card = _fragility_card(rec, fragility)
    stance_card = _stance_card(rec)

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta http-equiv="refresh" content="3600"/>
<title>Regime Monitor — P(bear) {_fmt_prob(bp)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link rel="stylesheet" media="print" onload="this.media='all'"
      href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap"/>
<noscript><link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap"/></noscript>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Inter', -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
          margin: 0; padding: 20px 16px 36px; color: {C['ink']};
          background:
            radial-gradient(1200px 600px at 50% -10%, #11141d 0%, {C['bg']} 60%),
            {C['bg']};
          -webkit-font-smoothing: antialiased; letter-spacing: .1px; }}
  .wrap {{ max-width: 680px; margin: 0 auto; }}
  .top {{ display: flex; align-items: baseline; justify-content: space-between;
          padding: 2px 4px 16px; }}
  .brand {{ font-size: 14px; font-weight: 700; letter-spacing: .02em; color: {C['ink']}; }}
  .brand .mono {{ color: {C['accent']}; }}
  .step .t .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                    font-size: 12px; color: {C['ink']};
                    background: {C['surface_2']}; padding: 1px 5px;
                    border-radius: 5px; }}
  .asof {{ font-size: 12px; color: {C['ink_faint']}; font-variant-numeric: tabular-nums; }}

  .card {{ background: linear-gradient(180deg, {C['surface']} 0%, #0b0d12 100%);
           border: 1px solid {C['line']}; border-radius: 20px;
           padding: 22px; margin-bottom: 14px;
           box-shadow: 0 1px 0 rgba(255,255,255,.02) inset, 0 10px 30px rgba(0,0,0,.35); }}

  /* ---- The risk dial (hero) ---- */
  .hero {{ text-align: center; padding-bottom: 18px; }}
  .eyebrow {{ font-size: 11px; text-transform: uppercase; letter-spacing: .18em;
              color: {C['ink_faint']}; font-weight: 600; }}
  .gauge-wrap {{ position: relative; width: 320px; max-width: 100%; margin: 8px auto 0;
                 aspect-ratio: 320 / 188; }}
  .gauge-svg {{ width: 100%; display: block; }}
  /* Anchor the number on the gauge's pivot (cx=160, cy=168 in a 320x188 box →
     ~85% of width, ~72% of height) and center it there, so it always sits in
     the bowl of the arc regardless of render width. */
  .dial-center {{ position: absolute; left: 50%; top: 72%;
                  transform: translate(-50%, -50%); text-align: center;
                  width: 100%; pointer-events: none; }}
  .dial-num {{ font-size: 58px; font-weight: 900; line-height: 1;
               font-variant-numeric: tabular-nums; letter-spacing: -.02em; }}
  .dial-cap {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: .14em;
               color: {C['ink_faint']}; margin-top: 4px; }}
  .dial-ends {{ display: flex; justify-content: space-between; width: 320px;
                max-width: 100%; margin: 2px auto 0; font-size: 11px;
                color: {C['ink_faint']}; letter-spacing: .04em; }}
  .chip {{ display: inline-block; padding: 4px 13px; border-radius: 999px;
           font-weight: 700; font-size: 12px; letter-spacing: .08em;
           text-transform: uppercase; }}
  .hero-chips {{ display: flex; gap: 8px; justify-content: center; margin-top: 14px; }}
  .hero-sub {{ color: {C['ink_dim']}; font-size: 13px; line-height: 1.5; margin-top: 12px; }}

  /* ---- generic ---- */
  h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .14em;
        font-weight: 700; color: {C['ink_dim']}; margin: 0 0 14px; }}
  .sub {{ color: {C['ink_dim']}; font-size: 13px; line-height: 1.6; margin: 0; }}
  .sub2 {{ color: {C['ink_faint']}; font-size: 11px; font-weight: 500;
           margin-top: 1px; }}
  .chart-cap {{ color: {C['ink_faint']}; font-size: 11.5px; line-height: 1.5; }}
  .megan {{ font-size: 52px; font-weight: 900; line-height: 1;
            font-variant-numeric: tabular-nums; letter-spacing: -.02em; }}
  .frag-head {{ display: flex; align-items: center; gap: 12px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  td, th {{ padding: 9px 6px; border-bottom: 1px solid {C['line']}; text-align: left;
            vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  th {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: .08em;
        color: {C['ink_faint']}; font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .bar {{ display: inline-block; vertical-align: middle; width: 92px; height: 7px;
          background: {C['surface_2']}; border-radius: 99px; overflow: hidden;
          margin-right: 8px; }}
  .bar-fill {{ height: 100%; border-radius: 99px; }}
  .bar-pct {{ font-size: 12px; color: {C['ink_dim']}; }}

  .spark {{ width: 100%; display: block; }}

  /* ---- explainer ---- */
  .steps {{ display: grid; gap: 12px; margin-top: 2px; }}
  .step {{ display: flex; gap: 12px; align-items: flex-start; }}
  .step .n {{ flex: 0 0 24px; height: 24px; border-radius: 8px;
              background: {C['surface_2']}; color: {C['accent']};
              font-size: 12px; font-weight: 800; display: flex;
              align-items: center; justify-content: center; }}
  .step .t {{ font-size: 13.5px; color: {C['ink_dim']}; line-height: 1.55; }}
  .step .t b {{ color: {C['ink']}; font-weight: 600; }}

  /* ---- overlay tile ---- */
  .tile {{ background: {C['surface_2']}; border: 1px solid {C['line']};
           border-radius: 14px; padding: 14px 16px; }}
  .tile-head {{ display: flex; align-items: center; gap: 9px; margin-bottom: 6px; }}
  .dot {{ width: 9px; height: 9px; border-radius: 99px; flex: 0 0 auto; }}
  .tile-title {{ font-weight: 600; font-size: 14px; }}
  .tile-state {{ margin-left: auto; font-weight: 700; font-size: 12px;
                 letter-spacing: .04em; text-transform: uppercase; }}
  .tile-foot {{ margin-top: 8px; font-size: 12px; color: {C['ink_faint']};
                font-variant-numeric: tabular-nums; }}
  .conds {{ margin-top: 10px; padding: 10px 12px; background: {C['bg']};
            border: 1px solid {C['line']}; border-radius: 10px; }}
  .cond {{ display: flex; gap: 8px; align-items: baseline; font-size: 12.5px;
           line-height: 1.7; }}
  .cmark {{ flex: 0 0 14px; font-weight: 700; }}
  .cond-foot {{ margin-top: 6px; font-size: 11.5px; color: {C['ink_faint']};
                font-style: italic; }}

  /* ---- fragility meter ---- */
  .meter {{ position: relative; height: 10px; border-radius: 99px;
            margin: 16px 0 6px; }}
  .meter-needle {{ position: absolute; top: -4px; width: 3px; height: 18px;
                   background: {C['ink']}; border-radius: 2px;
                   box-shadow: 0 0 0 2px {C['surface']}; }}
  .meter-scale {{ display: flex; justify-content: space-between; font-size: 10.5px;
                  color: {C['ink_faint']}; text-transform: uppercase;
                  letter-spacing: .04em; }}
  .meter-fill {{ position: absolute; left: 0; top: 0; height: 100%;
                 border-radius: 99px; }}
  .meter-tick {{ position: absolute; top: -3px; width: 2px; height: 16px;
                 background: {C['ink_dim']}; opacity: .7; }}

  /* ---- exposure targets ---- */
  .targets {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
              padding: 4px 0 12px; }}
  .target {{ background: {C['surface_2']}; border: 1px solid {C['line']};
             border-radius: 14px; padding: 14px 16px; }}
  .t-val {{ font-size: 34px; font-weight: 800; line-height: 1;
            font-variant-numeric: tabular-nums; letter-spacing: -.02em; }}
  .t-unit {{ font-size: 17px; font-weight: 700; color: {C['ink_faint']};
             margin-left: 3px; }}
  .t-lab {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
            color: {C['ink_faint']}; margin-top: 6px; }}
  .target .meter {{ overflow: hidden; margin: 12px 0 6px; }}

  .banner {{ padding: 12px 14px; border-radius: 12px; font-weight: 600;
             font-size: 13.5px; margin-bottom: 14px; }}
  .banner.ok {{ background: rgba(33,208,122,.12); color: {C['risk_lo']};
                border: 1px solid rgba(33,208,122,.25); }}

  .actions td:first-child {{ color: {C['ink_faint']}; width: 110px;
                             text-transform: uppercase; font-size: 11px;
                             letter-spacing: .06em; }}
  .foot {{ color: {C['ink_faint']}; font-size: 11.5px; text-align: center;
           margin-top: 22px; line-height: 1.6; }}
</style>
</head>
<body>
<div class="wrap">

  <div class="top">
    <div class="brand">Regime <span class="mono">Monitor</span></div>
    <div class="asof">As of {as_of}</div>
  </div>

  <!-- 1 · THE risk dial (hero, big + center) -->
  <section class="card hero">
    <div class="eyebrow">Probability of a bear regime &middot; the risk dial</div>
    <div class="gauge-wrap">
      {gauge_svg}
      <div class="dial-center">
        <div class="dial-num" style="color:{scolor}">{_fmt_prob(bp)}</div>
        <div class="dial-cap">probability of bear regime</div>
      </div>
    </div>
    <div class="dial-ends"><span>risk-on</span><span>risk-off</span></div>
    <div class="hero-chips">
      <span class="chip" style="background:{scolor}1f;color:{scolor}">{stance}</span>
      <span class="chip" style="background:{rcolor}14;color:{rcolor}">{html.escape(regime)} regime</span>
    </div>
    <div class="hero-sub">Size your aggressiveness on this one number — it's the
      continuous CJM bear-probability nowcast, the most finely-tuned signal in the
      system. Higher = de-risk.</div>
    {spark_prob}
  </section>

  <!-- 2 · How this works (interpretability) -->
  <section class="card">
    <h2>How this works &middot; methodology</h2>
    <div class="steps">
      <div class="step"><div class="n">1</div><div class="t"><b>Continuous Jump
        Model (CJM)</b> — Shu &amp; Mulvey (Princeton), the continuous extension
        (§2.4) of the Statistical Jump Model. Each day is described by 8
        standardized, backward-looking features (returns, realized &amp; implied
        vol, the vol risk premium, MACD/MA trend, multi-horizon momentum). The
        model fits two regime centroids and assigns each day a regime
        <b>probability</b> by minimizing fit loss <i>plus</i> a temporal jump
        penalty &lambda; on
        <span class="mono">&Sigma;&#8741;s<sub>t</sub>&minus;s<sub>t&minus;1</sub>&#8741;</span>
        via coordinate descent on the simplex.</div></div>
      <div class="step"><div class="n">2</div><div class="t"><b>Why a jump model
        over an HMM.</b> An HMM imposes a parametric generative story (a fixed
        emission distribution + Markov transition matrix) and is fit by EM, which
        is sensitive to initialization, prone to spurious rapid switching, and
        mismatched at inference (Viterbi path ≠ smoothed posteriors). The jump
        model is <b>distribution-free</b>: it explicitly penalizes regime
        switches, so persistence is a tunable cost (&lambda;), not a fragile
        by-product of a transition matrix — yielding stabler, more interpretable
        regimes with far fewer assumptions.</div></div>
      <div class="step"><div class="n">3</div><div class="t"><b>Why CONTINUOUS over
        the discrete SJM.</b> The discrete model emits hard 0/1 labels — coarse
        for sizing risk and jumpy near the boundary. The CJM returns a smooth
        regime probability (the dial), is far better <b>calibrated</b> (Brier
        ≈&nbsp;0.02), and is internally consistent: <span class="mono">argmax</span>
        of the same probabilities gives the label, with none of the
        Viterbi-vs-forward-backward inconsistency of an HMM.</div></div>
      <div class="step"><div class="n">4</div><div class="t"><b>Leak-free by
        construction.</b> A walk-forward harness refits the whole pipeline only
        on past data (online inference, monthly refit, ~5y minimum train), so
        today's reading uses <i>zero</i> future information — the paper warns that
        forward-looking labels inflate results. Measured: <b>~1.5 regime
        switches/yr</b> (vs ~4.5 for the legacy forecast), ~166-day average
        regimes. It is a <b>volatility / risk</b> detector — it sizes risk well
        and does <i>not</i> claim to predict direction.</div></div>
      <div class="step"><div class="n">5</div><div class="t">Layered on top are
        <b>display-only overlays</b> (re-entry timing &amp; the leading fragility
        score) that never touch the dial — the traded signal stays a pure CJM
        nowcast.</div></div>
    </div>
  </section>

  <!-- 3 · Why (driver attribution) -->
  <section class="card">
    <h2>Why — what's driving today's read</h2>
    <table>
      <tr><th>Feature</th><th class="num">Now</th><th class="num">vs normal</th><th>Pushing</th><th class="num">Weight</th></tr>
      {_drivers_rows(rec.get("drivers") or [])}
    </table>
    <p class="sub" style="margin-top:10px">Each feature's lean toward the bear vs
      bull centroid — a leak-free attribution of the live model's dial reading.
      <b>vs normal</b> shows today's standardized z-score with its empirical
      percentile (where the raw value sits in its own history) beneath.</p>
  </section>

  <!-- 4 · Overlays -->
  <section class="card">
    <h2>Timing overlay</h2>
    {reentry_banner}
    {reentry_tile}
    <p class="sub" style="margin-top:12px">Get-short / buy-protection timing lives
      in the leading fragility score below (the opposite loss function from
      re-entry). Both are display-only and never move the risk dial.</p>
  </section>

  <!-- 5 · Fragility -->
  {fragility_card}

  <!-- 6 · Suggested stance -->
  {stance_card}

  <!-- 7 · Recent calls (bottom) -->
  <section class="card">
    <h2>Recent calls</h2>
    <table>
      <tr><th>Date</th><th class="num">P(bear)</th><th>Regime</th><th>Stance</th><th>Signals</th></tr>
      {_history_rows(h)}
    </table>
  </section>

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
