"""Daily digest + iMessage delivery (macOS only).

This is the *delivery* layer for operational use: it turns today's regime read
into a short text digest and sends it to your phone via iMessage, gated so it
only pings loudly when something actually changed (avoids alert fatigue on the
~95% of days the slow-moving signal says "no change").

Why iMessage (and why this must run on YOUR Mac): iMessage can only be sent from
a signed-in macOS account via the Messages app + AppleScript. No cloud server
can send it, so the digest job runs locally (scheduled via launchd; see
`deploy/`). The 24/7 *dashboard* is the part that's hosted independently.

Nothing here places trades. Decision support only; not financial advice.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess

from . import config

# Where we remember the last digest we sent, so we can detect "what changed"
# and decide whether to ping. Small JSON next to the signal history.
STATE_FILE = config.DATA_DIR / "notify_state.json"

# A loud ping is warranted when any of these happen day-over-day.
_BIG_JUMP = 0.15  # bear_prob moved this much in one day -> notify

# Unicode glyphs kept as module constants so we never embed escapes in f-strings
# (which isn't allowed before Python 3.12).
_UP = "\u25b2"
_DOWN = "\u25bc"


def _emoji(stance: str) -> str:
    return {"BULL": "\U0001f7e2", "NEUTRAL": "\U0001f7e1", "BEAR": "\U0001f534"}.get(
        stance, "\u26aa"
    )


def _arrow(curr: float, prev: float | None) -> str:
    if prev is None:
        return ""
    delta = curr - prev
    if abs(delta) < 0.005:
        return "  (flat)"
    glyph = _UP if delta > 0 else _DOWN
    return f"  ({glyph}{abs(delta):.0%} d/d)"


def format_digest(rec: dict, prev: dict | None) -> str:
    """Build the short, glanceable text body for the morning message."""
    stance = rec["stance"]
    bp = rec["next_bear_prob"]
    prev_bp = prev.get("next_bear_prob") if prev else None
    as_of = str(rec["as_of"])[:10]

    lines = [
        f"{_emoji(stance)} REGIME: {stance}",
        f"P(bear) = {bp:.0%}{_arrow(bp, prev_bp)}",
        f"Detected today: {rec['current_regime']}   (as of {as_of})",
    ]

    # Re-entry / cover-short overlay flag, if enabled.
    if rec.get("reentry_flag"):
        lines.append(
            f"\u2705 RE-ENTRY confirmed (overlay {rec.get('bear_prob_overlay', 0):.0%}) "
            "- consider covering shorts / re-entering."
        )

    # Top 3 drivers, compact.
    drivers = rec.get("drivers") or []
    if drivers:
        bits = []
        for d in drivers[:3]:
            toward = "bear" if d["bear_pull"] > 0 else "bull"
            bits.append(f"{d['feature']} {d['z']:+.1f}\u03c3\u2192{toward}")
        lines.append("Why: " + "; ".join(bits))

    # Account stance one-liners.
    lines.append(f"401k: {rec['fidelity_401k']}")
    lines.append(f"ToS:  {rec['thinkorswim']}")
    return "\n".join(lines)


def _load_state() -> dict | None:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return None
    return None


def _save_state(rec: dict) -> None:
    state = {
        "as_of": str(rec["as_of"])[:10],
        "stance": rec["stance"],
        "next_bear_prob": round(float(rec["next_bear_prob"]), 4),
        "reentry_flag": bool(rec.get("reentry_flag", False)),
        "sent_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))


def decide(rec: dict, prev: dict | None) -> tuple[bool, str]:
    """Decide whether today warrants a *loud* ping, and why.

    Triggers (any one): first run, stance change, bear_prob crossing a config
    threshold band, a confirmed re-entry that wasn't flagged yesterday, or a big
    one-day move in bear_prob. Returns (notify, reason).
    """
    if prev is None:
        return True, "first run"
    reasons = []
    if rec["stance"] != prev.get("stance"):
        reasons.append(f"stance {prev.get('stance')} -> {rec['stance']}")
    bp = rec["next_bear_prob"]
    pbp = prev.get("next_bear_prob")
    if pbp is not None:
        for thr, name in (
            (config.BULL_THRESHOLD, "bull"),
            (config.BEAR_THRESHOLD, "bear"),
        ):
            if (pbp - thr) * (bp - thr) < 0:  # crossed the threshold
                reasons.append(f"crossed {name} threshold {thr:.0%}")
        if abs(bp - pbp) >= _BIG_JUMP:
            reasons.append(f"big move {pbp:.0%}->{bp:.0%}")
    if rec.get("reentry_flag") and not prev.get("reentry_flag"):
        reasons.append("re-entry confirmed")
    return (bool(reasons), "; ".join(reasons) if reasons else "no change")


def send_imessage(body: str, recipient: str) -> None:
    """Send `body` to `recipient` (phone number or Apple ID email) via iMessage.

    Uses AppleScript through `osascript`. Requires macOS, the Messages app set up
    with an iMessage account, and (the first time) Automation permission for the
    terminal/launchd job to control Messages. Raises on failure so the caller can
    log it.
    """
    # AppleScript: send to the iMessage service buddy. Falls back to the first
    # available service if "iMessage" isn't found by name.
    script = (
        "on run {targetBuddy, targetMessage}\n"
        '    tell application "Messages"\n'
        "        set targetService to 1st account whose service type = iMessage\n"
        "        set targetBuddyObj to participant targetBuddy of targetService\n"
        "        send targetMessage to targetBuddyObj\n"
        "    end tell\n"
        "end run"
    )
    subprocess.run(
        ["osascript", "-", recipient, body],
        input=script,
        text=True,
        check=True,
        capture_output=True,
    )


def run_digest(
    rec: dict, *, recipient: str | None, force: bool = False, dry_run: bool = False
) -> dict:
    """Format, gate, and (optionally) send the digest. Returns a result dict.

    `recipient` defaults to config.IMESSAGE_RECIPIENT. `force` sends regardless
    of the change gate. `dry_run` formats + decides but never sends.
    """
    recipient = recipient or getattr(config, "IMESSAGE_RECIPIENT", "") or None
    prev = _load_state()
    body = format_digest(rec, prev)
    notify, reason = decide(rec, prev)
    notify = notify or force

    result = {"notify": notify, "reason": reason, "body": body, "sent": False}

    if notify and not dry_run:
        if not recipient:
            result["error"] = (
                "No iMessage recipient set. Set IMESSAGE_RECIPIENT in regime/config.py "
                "or pass --to."
            )
        else:
            try:
                send_imessage(body, recipient)
                result["sent"] = True
            except subprocess.CalledProcessError as e:
                result["error"] = (e.stderr or str(e)).strip()
            except Exception as e:  # pragma: no cover - platform/permission issues
                result["error"] = str(e)

    # Always remember today's read so tomorrow's change-gate is correct, unless
    # this was a dry run (we don't want a dry run to suppress a real ping later).
    if not dry_run:
        _save_state(rec)
    return result
