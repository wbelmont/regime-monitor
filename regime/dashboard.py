"""Static, phone-friendly dashboard for the regime monitor.

Renders a single self-contained `index.html` (inline CSS, embedded sparkline as
a base64 PNG) into `reports/site/`. It is fully static, so it can be published
anywhere free — GitHub Pages, Netlify drop, or just opened from iCloud Drive on
your phone. No server, no JS framework.

The page shows the at-a-glance dial (stance + bear probability), the re-entry
overlay flag, a 6-month bear-probability sparkline, the per-feature "why" table
(the live CJM driver attribution), and the recent call history.

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
    "curve_slope": "Yield-curve slope",
    "hy_oas_level": "HY credit spread",
}


def _flabel(key: str) -> str:
    return _FEATURE_LABELS.get(key, key)


def _sparkline_png(history: "pd.DataFrame | None", days: int = 180) -> str | None:
    """Return a base64-encoded PNG of the recent bear_prob path, or None."""
    if history is None or history.empty or "next_bear_prob" not in history.columns:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = history.tail(days)
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.plot(pd.to_datetime(h["date"]), h["next_bear_prob"], color="#2563eb", lw=1.6)
    ax.axhline(config.BEAR_THRESHOLD, color="#dc2626", ls=":", lw=0.8)
    ax.axhline(config.BULL_THRESHOLD, color="#16a34a", ls=":", lw=0.8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1.0])
    ax.margins(x=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("P(bear) — last 6 months", fontsize=9, loc="left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _drivers_rows(drivers: list[dict]) -> str:
    rows = []
    for d in drivers[:6]:
        toward = "BEAR" if d["bear_pull"] > 0 else "BULL"
        color = "#dc2626" if d["bear_pull"] > 0 else "#16a34a"
        z = d["z"]
        zlabel = f"{z:+.1f}\u03c3"
        rows.append(
            f"<tr><td>{html.escape(_flabel(d['feature']))}</td>"
            f"<td class='num'>{d['value']:.2f}</td>"
            f"<td class='num'>{zlabel}</td>"
            f"<td style='color:{color};font-weight:600'>{toward}</td>"
            f"<td class='num'>{d['share']:.0%}</td></tr>"
        )
    return "\n".join(rows)


def _history_rows(history: "pd.DataFrame | None", n: int = 10) -> str:
    if history is None or history.empty:
        return "<tr><td colspan='3'>No history yet.</td></tr>"
    h = history.tail(n).iloc[::-1]
    rows = []
    for _, r in h.iterrows():
        stance = str(r.get("stance", ""))
        color = _STANCE_COLOR.get(stance, "#64748b")
        rows.append(
            f"<tr><td>{html.escape(str(r['date']))}</td>"
            f"<td style='color:{color};font-weight:600'>{html.escape(stance)}</td>"
            f"<td class='num'>{float(r['next_bear_prob']):.0%}</td></tr>"
        )
    return "\n".join(rows)


def render(
    rec: dict, history: pd.DataFrame | None = None, filename: str = "index.html"
) -> str:
    """Write the dashboard HTML and return its path."""
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    stance = rec["stance"]
    color = _STANCE_COLOR.get(stance, "#64748b")
    bp = rec["next_bear_prob"]
    as_of = str(rec["as_of"])[:10]
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    spark = _sparkline_png(history)
    spark_html = (
        f"<img class='spark' src='data:image/png;base64,{spark}' alt='P(bear) sparkline'/>"
        if spark
        else ""
    )

    reentry_html = ""
    if rec.get("reentry_flag"):
        reentry_html = (
            "<div class='card'><div class='banner'>\u2705 Re-entry confirmed "
            f"(overlay {rec.get('bear_prob_overlay', 0):.0%}) — consider covering "
            "shorts / re-entering longs.</div></div>"
        )

    spark_block = (
        spark_html
        if spark_html
        else "<div class='sub'>Run the monitor a few days to build the sparkline.</div>"
    )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta http-equiv="refresh" content="3600"/>
<title>Regime Monitor — {stance}</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
          margin: 0; padding: 16px; background: #0b1020; color: #e5e7eb; }}
  .wrap {{ max-width: 720px; margin: 0 auto; }}
  .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 16px;
           padding: 18px; margin-bottom: 14px; }}
  .dial {{ text-align: center; }}
  .stance {{ font-size: 40px; font-weight: 800; color: {color}; letter-spacing: 1px; }}
  .prob {{ font-size: 56px; font-weight: 800; margin: 4px 0; }}
  .sub {{ color: #9ca3af; font-size: 14px; }}
  .banner {{ background: #064e3b; color: #d1fae5; padding: 10px 12px;
             border-radius: 10px; font-weight: 600; }}
  h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: .06em;
        color: #9ca3af; margin: 0 0 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  td, th {{ padding: 7px 6px; border-bottom: 1px solid #1f2937; text-align: left; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .spark {{ width: 100%; border-radius: 10px; background: #fff; }}
  .actions td:first-child {{ color: #9ca3af; width: 96px; vertical-align: top; }}
  .foot {{ color: #6b7280; font-size: 12px; text-align: center; margin-top: 18px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="card dial">
    <div class="stance">{stance}</div>
    <div class="prob" style="color:{color}">{bp:.0%}</div>
    <div class="sub">probability of bear regime &middot; detected today: <b>{html.escape(rec["current_regime"])}</b> &middot; as of {as_of}</div>
  </div>

  {reentry_html}

  <div class="card">
    <h2>P(bear) — recent path</h2>
    {spark_block}
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
    <h2>Recent calls</h2>
    <table>
      <tr><th>Date</th><th>Stance</th><th class="num">P(bear)</th></tr>
      {_history_rows(history)}
    </table>
  </div>

  <div class="foot">Generated {generated} &middot; decision support only — not financial advice.</div>
</div>
</body>
</html>
"""
    path = SITE_DIR / filename
    path.write_text(doc, encoding="utf-8")
    # Also drop a machine-readable snapshot next to it for any other consumer.
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
