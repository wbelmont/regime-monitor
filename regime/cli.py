"""Command-line interface — the thing you actually run.

Usage (after install):
    regime update      # pull latest data + print today's regime & suggestion
    regime backtest    # show honest out-of-sample performance vs buy & hold
    regime chart       # regenerate the regime-history chart in reports/

If you don't want to remember commands, just double-click `Run Monitor.command`
on your Desktop project (created by setup) — it runs `regime update`.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from . import config, data, features, pipeline, recommend, report, backtest, tune
from . import notify, dashboard

console = Console()


# Human-readable names for the CJM regime features (display only).
FEATURE_LABELS = {
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
    return FEATURE_LABELS.get(key, key)


def _load_features(refresh: bool) -> pd.DataFrame:
    with console.status("[bold]Loading market data..."):
        raw = data.load_raw(refresh=refresh)
        feat = features.build_features(raw)
    return feat


def _log_history(rec: dict) -> None:
    """Append today's call to a CSV so you can review the track record later."""
    row = {
        "date": pd.Timestamp(rec["as_of"]).date().isoformat(),
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "stance": rec["stance"],
        "current_regime": rec["current_regime"],
        "next_bear_prob": round(rec["next_bear_prob"], 4),
    }
    f = config.SIGNAL_HISTORY_FILE
    df = pd.DataFrame([row])
    if f.exists():
        df.to_csv(f, mode="a", header=False, index=False)
    else:
        df.to_csv(f, index=False)


def cmd_update(args) -> None:
    feat = _load_features(refresh=not args.no_refresh)
    with console.status("[bold]Computing regime signal..."):
        sig = pipeline.latest_signal(feat)
    rec = recommend.build_recommendation(sig)
    _log_history(rec)

    color = {"BULL": "green", "NEUTRAL": "yellow", "BEAR": "red"}[rec["stance"]]
    header = (
        f"[bold {color}]{rec['stance']}[/]   "
        f"(P(next-period bear) = {rec['next_bear_prob']:.0%})\n"
        f"Today's detected regime: [bold]{rec['current_regime']}[/]   "
        f"as of {pd.Timestamp(rec['as_of']).date()}"
    )
    console.print(Panel(header, title="Market Regime Monitor", border_style=color))

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Account")
    tbl.add_column("Suggested action")
    tbl.add_row("Fidelity 401k", rec["fidelity_401k"])
    tbl.add_row("thinkorswim", rec["thinkorswim"])
    console.print(tbl)

    if rec["top_drivers"]:
        drivers = ", ".join(f"{k} ({v:.0%})" for k, v in rec["top_drivers"])
        console.print(f"[dim]Top signal drivers: {drivers}[/]")

    # WHY: per-feature attribution of today's bear/bull lean from the live CJM.
    # `bear_pull > 0` = the feature is pushing toward BEAR; `< 0` = toward BULL.
    drivers = rec.get("drivers") or []
    if drivers:
        dtbl = Table(
            title="Why — what's driving today's regime read (live CJM)",
            header_style="bold",
            title_style="bold",
        )
        dtbl.add_column("Feature")
        dtbl.add_column("Current", justify="right")
        dtbl.add_column("vs normal (z)", justify="right")
        dtbl.add_column("Pushing", justify="center")
        dtbl.add_column("Weight", justify="right")
        for d in drivers[:6]:
            toward = "[red]BEAR[/]" if d["bear_pull"] > 0 else "[green]BULL[/]"
            z = d["z"]
            zlabel = f"{z:+.1f}σ" + (" high" if z > 0 else " low" if z < 0 else "")
            dtbl.add_row(
                _flabel(d["feature"]),
                f"{d['value']:.2f}",
                zlabel,
                toward,
                f"{d['share']:.0%}",
            )
        console.print(dtbl)
        console.print(
            "[dim]Read: each feature's distance to the bear vs bull centroid. "
            "'Pushing' = which regime it leans toward today; 'Weight' = its share "
            "of the total lean. Leak-free attribution of the live model.[/]"
        )

    # Opt-in re-entry / cover-short overlay (config.REENTRY_OVERLAY).
    if "reentry_flag" in rec:
        if rec["reentry_flag"]:
            console.print(
                "[bold cyan]Re-entry overlay: CONFIRMED[/] — S&P has rebounded off "
                "its trailing low with VIX receding. Consider covering shorts / "
                f"re-entering longs (overlay bear reading {rec['bear_prob_overlay']:.0%})."
            )
        else:
            console.print(
                "[dim]Re-entry overlay: not triggered (no confirmed rebound yet).[/]"
            )

    console.print(
        "\n[dim]Decision-support only — review before acting. Not financial advice.[/]"
    )


def cmd_backtest(args) -> None:
    feat = _load_features(refresh=not args.no_refresh)
    console.print(
        "[bold]Running leak-free walk-forward backtest[/] "
        "[dim](refits the model across ~25 years; ~1-2 min)[/]"
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} windows"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("walk-forward", total=100)

        def on_step(done, total):
            progress.update(task, completed=done, total=total)

        signals = pipeline.walk_forward(feat, progress=on_step)
    res = backtest.run(feat, signals)

    tbl = Table(title="Out-of-sample performance (net of costs)", header_style="bold")
    tbl.add_column("Metric")
    tbl.add_column("Regime strategy", justify="right")
    tbl.add_column("Buy & hold", justify="right")
    s, b = res["strategy"], res["buy_hold"]

    def pct(x):
        return f"{x:.1%}" if x is not None else "-"

    tbl.add_row(
        "Annual return", pct(s.get("annual_return")), pct(b.get("annual_return"))
    )
    tbl.add_row("Annual vol", pct(s.get("annual_vol")), pct(b.get("annual_vol")))
    tbl.add_row(
        "Sharpe",
        f"{s.get('sharpe', float('nan')):.2f}",
        f"{b.get('sharpe', float('nan')):.2f}",
    )
    tbl.add_row("Max drawdown", pct(s.get("max_drawdown")), pct(b.get("max_drawdown")))
    tbl.add_row("Total return", pct(s.get("total_return")), pct(b.get("total_return")))
    console.print(tbl)

    path = report.equity_chart(res["equity"])
    console.print(f"[green]Saved equity chart ->[/] {path}")
    console.print(
        "\n[dim]Past performance does not guarantee future results. "
        "This compares a simple in/out switch vs. buy & hold.[/]"
    )


def cmd_chart(args) -> None:
    feat = _load_features(refresh=not args.no_refresh)
    with console.status("[bold]Labeling full history for chart..."):
        labels = pipeline.label_full_sample(feat)
    path = report.regime_chart(feat, labels)
    console.print(f"[green]Saved regime chart ->[/] {path}")


def cmd_digest(args) -> None:
    """Compute today's signal and send the change-gated iMessage digest."""
    feat = _load_features(refresh=not args.no_refresh)
    with console.status("[bold]Computing regime signal..."):
        sig = pipeline.latest_signal(feat)
    rec = recommend.build_recommendation(sig)
    _log_history(rec)

    result = notify.run_digest(
        rec, recipient=args.to, force=args.force, dry_run=args.dry_run
    )

    console.print(Panel(result["body"], title="Daily digest", border_style="cyan"))
    console.print(f"[dim]Change gate: {result['reason']}[/]")
    if result.get("error"):
        console.print(f"[red]Send failed:[/] {result['error']}")
    elif result["sent"]:
        console.print("[green]Sent via iMessage.[/]")
    elif args.dry_run:
        console.print("[yellow]Dry run — not sent.[/]")
    elif not result["notify"]:
        console.print(
            "[dim]No meaningful change — not sent (use --force to send anyway).[/]"
        )


def cmd_dashboard(args) -> None:
    """Render the static, phone-friendly dashboard into reports/site/."""
    feat = _load_features(refresh=not args.no_refresh)
    with console.status("[bold]Computing regime signal..."):
        sig = pipeline.latest_signal(feat)
    rec = recommend.build_recommendation(sig)
    if not args.no_log:
        _log_history(rec)
    history = dashboard.load_history()
    path = dashboard.render(rec, history)
    console.print(f"[green]Wrote dashboard ->[/] {path}")
    console.print(
        "[dim]Open it on your phone, sync to iCloud, or publish via GitHub Pages.[/]"
    )


def cmd_tune(args) -> None:
    feat = _load_features(refresh=not args.no_refresh)

    # Parse the lambda grid: explicit --grid wins; otherwise the module default.
    grid = None
    if args.grid:
        grid = [float(x) for x in args.grid.split(",") if x.strip()]

    fast = not args.full
    if fast:
        n_init = args.n_init if args.n_init is not None else tune.FAST_CV["n_init"]
        max_iter = (
            args.max_iter if args.max_iter is not None else tune.FAST_CV["max_iter"]
        )
        refit_every = (
            args.refit_every
            if args.refit_every is not None
            else tune.FAST_CV["refit_every"]
        )
        max_oos_days = (
            args.max_oos_days
            if args.max_oos_days is not None
            else tune.FAST_CV["max_oos_days"]
        )
    else:
        # Full confirmation: use the same effort as `regime backtest` (config
        # defaults via None) over the whole out-of-sample span.
        n_init = args.n_init if args.n_init is not None else 10
        max_iter = args.max_iter if args.max_iter is not None else 50
        refit_every = args.refit_every  # None -> config.REFIT_EVERY_DAYS
        max_oos_days = args.max_oos_days  # None -> full OOS span

    shown_grid = grid if grid is not None else tune.DEFAULT_GRID
    mode = "fast CV" if fast else "FULL confirmation"
    crit = (
        f"jumps (target {args.target_jumps:g}/yr)"
        if args.select_by == "jumps"
        else "OOS Sharpe"
    )
    console.print(
        f"[bold]Tuning jump penalty (lambda)[/] [dim]({mode}, nested CV)[/]\n"
        f"[dim]grid = {shown_grid} | select-by = {crit} | folds={args.folds} | "
        f"eval_frac={args.eval_frac:g} | n_init={n_init} max_iter={max_iter} "
        f"refit_every={refit_every or config.REFIT_EVERY_DAYS} "
        f"max_oos_days={max_oos_days or 'all'}[/]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} fits"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("sweeping lambda", total=100)

        def on_step(done, total):
            progress.update(task, completed=done, total=total)

        results, best = tune.tune_jump_penalty(
            feat,
            grid=grid,
            select_by=args.select_by,
            target_jumps=args.target_jumps,
            n_folds=args.folds,
            n_init=n_init,
            max_iter=max_iter,
            refit_every=refit_every,
            max_oos_days=max_oos_days,
            eval_frac=args.eval_frac,
            progress=on_step,
        )

    sel_label = "Jumps score" if args.select_by == "jumps" else "Sel CV Sharpe"
    tbl = Table(
        title=(
            f"Lambda sweep — selected by {args.select_by} on the earlier span, "
            "scored on the held-out later span"
        ),
        header_style="bold",
    )
    tbl.add_column("lambda", justify="right")
    tbl.add_column(sel_label, justify="right")  # SELECTION span
    tbl.add_column("±std", justify="right")  # selection-fold std
    tbl.add_column("Sel jumps/yr", justify="right")  # SELECTION span
    tbl.add_column("Eval Sharpe", justify="right")  # held-out EVAL span
    tbl.add_column("Eval Max DD", justify="right")  # held-out EVAL span
    tbl.add_column("Eval ann.", justify="right")  # held-out EVAL span

    def f2(x):
        return f"{x:.2f}" if x == x else "-"  # NaN-safe

    def pct(x):
        return f"{x:.1%}" if x == x else "-"

    for r in results:
        marker = " [green]*[/]" if r is best else ""
        sel_val = r.select_score if args.select_by == "jumps" else r.cv_sharpe
        tbl.add_row(
            f"{r.jump_penalty:g}{marker}",
            f2(sel_val),
            f2(r.cv_sharpe_std),
            f2(r.sel_jumps_per_year),
            f2(r.oos_sharpe),
            pct(r.max_drawdown),
            pct(r.annual_return),
        )
    console.print(tbl)

    csv_path, json_path = tune.save_results(results, best)
    console.print(
        Panel(
            f"Best lambda = [bold green]{best.jump_penalty:g}[/]   "
            f"(selected by {best.select_metric} on the earlier span)\n"
            f"Held-out EVAL: Sharpe = {best.oos_sharpe:.2f}, "
            f"Max DD = {best.max_drawdown:.1%}, "
            f"jumps/yr = {best.sel_jumps_per_year:.2f}",
            title="Recommended jump penalty",
            border_style="green",
        )
    )
    console.print(f"[dim]Saved sweep -> {csv_path}\n             -> {json_path}[/]")

    if args.write_config:
        path = tune.write_jump_penalty_to_config(best.jump_penalty)
        console.print(
            f"[green]Updated JUMP_PENALTY = {best.jump_penalty:g} in[/] {path}"
        )
    else:
        console.print(
            f"[dim]To apply: edit JUMP_PENALTY in regime/config.py to "
            f"{best.jump_penalty:g}, or re-run with --write-config.[/]"
        )
    if fast:
        console.print(
            "[yellow]This was a fast CV pass. Confirm the winner with a full run:[/]\n"
            f"[dim]  regime tune --no-refresh --full --grid {best.jump_penalty:g}[/]"
        )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="regime", description="Personal market-regime monitor."
    )
    p.add_argument(
        "--no-refresh", action="store_true", help="Use cached data, don't re-download."
    )

    # Shared parent so --no-refresh also works AFTER the subcommand name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--no-refresh", action="store_true", help="Use cached data, don't re-download."
    )

    sub = p.add_subparsers(dest="command")
    sub.add_parser(
        "update",
        parents=[common],
        help="Pull data and print today's regime + suggestion.",
    )
    sub.add_parser(
        "backtest", parents=[common], help="Show honest out-of-sample performance."
    )
    sub.add_parser(
        "chart", parents=[common], help="Regenerate the regime-history chart."
    )

    p_digest = sub.add_parser(
        "digest",
        parents=[common],
        help="Send today's change-gated regime digest via iMessage.",
    )
    p_digest.add_argument(
        "--to",
        type=str,
        default=None,
        help="iMessage recipient (phone like +15551234567 or Apple ID email). "
        "Defaults to config.IMESSAGE_RECIPIENT.",
    )
    p_digest.add_argument(
        "--force",
        action="store_true",
        help="Send even if nothing changed (bypass the alert-fatigue gate).",
    )
    p_digest.add_argument(
        "--dry-run",
        action="store_true",
        help="Format and decide, but do not send (preview the message).",
    )

    p_dash = sub.add_parser(
        "dashboard",
        parents=[common],
        help="Render the static phone-friendly dashboard into reports/site/.",
    )
    p_dash.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append today's call to the signal-history CSV.",
    )

    p_tune = sub.add_parser(
        "tune",
        parents=[common],
        help="Tune the CJM jump penalty (lambda) via leak-free time-series CV.",
    )
    p_tune.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Comma-separated lambdas to sweep (e.g. '12.5,25,50,100,200'). "
        "Default: a coarse log-spaced grid around 50.",
    )
    p_tune.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of contiguous CV folds (default 3).",
    )
    p_tune.add_argument(
        "--select-by",
        dest="select_by",
        choices=["sharpe", "jumps"],
        default="sharpe",
        help="Selection criterion: 'sharpe' maximizes strategy OOS Sharpe; "
        "'jumps' targets a realistic number of regime transitions per year "
        "(decouples detector smoothness from P&L). Default: sharpe.",
    )
    p_tune.add_argument(
        "--target-jumps",
        dest="target_jumps",
        type=float,
        default=tune.DEFAULT_TARGET_JUMPS_PER_YEAR,
        help="For --select-by jumps: target regime transitions per year "
        f"(default {tune.DEFAULT_TARGET_JUMPS_PER_YEAR:g}). Whipsaws above the "
        "target are penalized harder than over-smoothing.",
    )
    p_tune.add_argument(
        "--eval-frac",
        dest="eval_frac",
        type=float,
        default=tune.DEFAULT_EVAL_FRAC,
        help="Fraction of the OOS timeline held out (at the end) as the nested "
        f"evaluation span, unseen during selection (default {tune.DEFAULT_EVAL_FRAC:g}).",
    )
    p_tune.add_argument(
        "--full",
        action="store_true",
        help="Full-rigor confirmation run (slow): config-default refit cadence, "
        "10 restarts, whole OOS span. Default is a fast CV pass.",
    )
    p_tune.add_argument(
        "--n-init",
        dest="n_init",
        type=int,
        default=None,
        help="CJM k-means++ restarts.",
    )
    p_tune.add_argument(
        "--max-iter",
        dest="max_iter",
        type=int,
        default=None,
        help="CJM coordinate-descent iterations.",
    )
    p_tune.add_argument(
        "--refit-every",
        dest="refit_every",
        type=int,
        default=None,
        help="Walk-forward refit cadence in trading days (larger = faster).",
    )
    p_tune.add_argument(
        "--max-oos-days",
        dest="max_oos_days",
        type=int,
        default=None,
        help="Cap the recent out-of-sample window (trading days) used for CV.",
    )
    p_tune.add_argument(
        "--write-config",
        action="store_true",
        help="Write the best lambda into regime/config.py (JUMP_PENALTY).",
    )

    args = p.parse_args(argv)
    cmd = args.command or "update"
    {
        "update": cmd_update,
        "backtest": cmd_backtest,
        "chart": cmd_chart,
        "digest": cmd_digest,
        "dashboard": cmd_dashboard,
        "tune": cmd_tune,
    }[cmd](args)


if __name__ == "__main__":
    main()
