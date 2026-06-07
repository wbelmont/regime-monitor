# Operationalizing the regime monitor

Two independent pieces:

1. **Morning iMessage digest** — runs on **your Mac** at **9:00 AM ET** (iMessage
   can only be sent from a signed-in Messages app, so this must be local).
   Change-gated: it only texts you when something meaningful moves.
2. **24/7 hosted dashboard** — a static page published to **GitHub Pages**,
   rebuilt daily by GitHub Actions, independent of whether your Mac is on.

---

## Part 1 — iMessage digest on the Mac (launchd + wake)

### 1. Set your recipient

Edit `regime/config.py`:

```python
IMESSAGE_RECIPIENT = "+15551234567"   # your phone, or your Apple ID email
```

Preview the message any time (does not send):

```bash
PYTHONPATH=. .venv/bin/python -m regime.cli digest --no-refresh --dry-run
```

Send a real test (bypasses the change gate):

```bash
PYTHONPATH=. .venv/bin/python -m regime.cli digest --no-refresh --force
```

> The **first** send will trigger a macOS prompt to allow Terminal (and later
> launchd) to control **Messages**. Approve it. If it was denied, re-enable under
> *System Settings → Privacy & Security → Automation*.

### 2. Make the runner executable

```bash
chmod +x deploy/run_digest.sh
```

### 3. Install the launchd agent (fires at 9:00 AM **local** time)

```bash
cp deploy/com.regimemonitor.digest.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.regimemonitor.digest.plist 2>/dev/null
launchctl load   ~/Library/LaunchAgents/com.regimemonitor.digest.plist
```

This fires at **9:00 AM local time**. It equals 9:00 AM ET only if this Mac's
timezone is Eastern — confirm in *System Settings → General → Date & Time*.

Force a run now to confirm the whole chain works:

```bash
launchctl start com.regimemonitor.digest
cat reports/digest.log
```

### 4. Wake the Mac before 9:00 (it's on but asleep)

`launchd` can't run while the Mac is fully asleep. Schedule a daily wake a few
minutes early (one-time, needs your password):

```bash
sudo pmset repeat wakeorpoweron MTWRFSU 08:55:00
```

If a fire is ever missed (lid closed, off), launchd runs it on the next wake.

---

## Part 2 — Hosted dashboard on GitHub Pages

1. Push this repo to GitHub.
2. In the repo: **Settings → Pages → Build and deployment → Source = GitHub
   Actions**.
3. The workflow `.github/workflows/dashboard.yml` then:
   - runs daily (~9:30 AM ET) and on pushes to `main`,
   - builds the dashboard with **live data**,
   - commits the day's call to `data/signal_history.csv` (so the sparkline and
     "recent calls" accumulate on the hosted side),
   - publishes `reports/site/` to Pages.
4. Your dashboard URL appears in the Actions run summary and under Settings →
   Pages. Add it to your phone's Home Screen for one-tap access.

Trigger it manually any time from the **Actions** tab → *Publish dashboard* →
*Run workflow*.

---

## Local on-demand (anytime)

Spot-check from the Mac without any scheduling:

```bash
PYTHONPATH=. .venv/bin/python -m regime.cli update --no-refresh     # full readout
PYTHONPATH=. .venv/bin/python -m regime.cli dashboard --no-refresh  # rebuild local page
open reports/site/index.html
```

To have the local dashboard on your phone without hosting, point the `dashboard`
output at an iCloud Drive folder (or symlink `reports/site` into iCloud) and open
it from the Files app.

---

## What triggers a *loud* iMessage (the change gate)

The digest sends only when one of these is true day-over-day (else it stays
quiet to avoid alert fatigue on the slow-moving signal):

- stance flips (BULL / NEUTRAL / BEAR),
- `bear_prob` crosses the `BULL_THRESHOLD` (40%) or `BEAR_THRESHOLD` (60%) band,
- the re-entry / cover-short overlay newly confirms,
- a large one-day move in `bear_prob` (≥ 15 pts),
- the first ever run.

Use `--force` to send regardless.
