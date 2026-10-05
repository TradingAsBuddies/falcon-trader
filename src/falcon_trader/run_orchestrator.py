#!/usr/bin/env python3
"""
Multi-Strategy Orchestrator - Main Runner
Processes AI screener results through the complete orchestrator workflow
"""
import logging
import os
import sys
import yaml
import time

from falcon_trader import screener_feed, trading_guards
from datetime import datetime
from falcon_trader.orchestrator.execution.trade_executor import TradeExecutor
from falcon_trader.orchestrator.monitors.performance_tracker import PerformanceTracker
from falcon_core import get_db_manager, prices

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


logger = logging.getLogger(__name__)

#: Cycles that may fail in a row before the process exits and lets systemd
#: restart it. Enough to ride out a database restart, few enough that a real
#: breakage is not hidden by retrying forever.
MAX_CONSECUTIVE_CYCLE_FAILURES = 5


def eod_flatten_enabled(config) -> bool:
    """Whether to close every open position at 15:55.

    Off unless asked for: the strategies hold for max_hold_days (12-20 days),
    so flattening daily would close each position in the session it opened.
    The operator chose swing behaviour for this deployment (falcon-trader#43).
    """
    raw = os.getenv('FALCON_EOD_FLATTEN')
    if raw is None:
        raw = ((config or {}).get('session') or {}).get('eod_flatten', False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)


def print_header(title):
    """Print a formatted header"""
    print_separator()
    print(f"{title:^80}")
    print_separator()


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*20} {title} {'='*20}")


def process_screener_results(executor, tracker, db):
    """Process the screener's recommendations, read from the database.

    This used to read the relative path 'screened_stocks.json'. The screener
    writes that file into its own container's volume and the orchestrator runs
    in /app, so it was never found: every cycle printed "Screener file not
    found" and no trade was ever placed (falcon-trader#43). Both containers
    share PostgreSQL, where the screener records every run.
    """

    print_section("PROCESSING AI SCREENER RESULTS")

    recommendations, newest = screener_feed.latest_recommendations(db)

    if not recommendations:
        print("[WARN] No screener recommendations in the last 24h — nothing to process")
        return None

    age = screener_feed.utc_now_naive() - newest if newest else None
    print(f"[ORCHESTRATOR] {len(recommendations)} recommendation(s); "
          f"newest screen {newest} UTC"
          + (f" ({age.total_seconds() / 3600:.1f}h old)" if age else ""))
    summary = executor.process_recommendations(recommendations)

    print(f"\n[RESULTS]")
    print(f"  Total Stocks: {summary['total_stocks']}")
    print(f"  Processed: {summary['processed']}")
    print(f"  Trades Executed: {summary['trades_executed']}")
    print(f"  Skipped: {summary['skipped']}")

    if summary['errors'] > 0:
        print(f"  Errors: {summary['errors']}")

    # Show details
    if summary['details']:
        print(f"\n[DETAILS]")
        for detail in summary['details']:
            symbol = detail['symbol']
            success = detail['success']

            if success:
                # Quantity and price live under details['execution']; read from
                # the top level they are absent, so every fill printed as
                # "BUY 0 @ $0.00" -- today's three real trades looked like
                # nothing had happened.
                action = detail.get('action', 'N/A')
                execution = (detail.get('details') or {}).get('execution') or {}
                quantity = execution.get('quantity', 0)
                price = execution.get('price', 0.0)
                print(f"  [OK] {symbol}: {action} {quantity} @ ${prices.quantize(price)}")
            else:
                reason = detail.get('reason', 'Unknown')
                print(f"  [SKIP] {symbol}: {reason}")

    return summary


def monitor_positions(executor, tracker):
    """Monitor open positions for exit signals"""

    print_section("MONITORING POSITIONS")

    results = executor.monitor_positions()

    if not results:
        print("[INFO] No open positions to monitor")
        return

    failures = [r for r in results if r.get('action') == 'ERROR']
    print(f"[ORCHESTRATOR] Monitored {len(results)} positions"
          + (f" — {len(failures)} could not be evaluated" if failures else ""))

    for result in results:
        symbol = result['symbol']
        action = result['action']

        if action == 'HOLD':
            current_price = result.get('current_price', 0.0)
            pnl_pct = result.get('pnl_pct', 0.0)
            print(f"  [HOLD] {symbol}: ${prices.quantize(current_price)} ({pnl_pct:+.2f}%)")
        elif action == 'SELL':
            reason = result.get('reason', 'Exit signal')
            print(f"  [EXIT] {symbol}: {reason}")
        elif action == 'ERROR':
            # Named, because a position nothing could evaluate has no stop and
            # no target in force, which is the opposite of a quiet hold.
            print(f"  [UNMONITORED] {symbol}: {result.get('reason')}")


def flatten_positions(executor, tracker, reason='eod'):
    """Close all open positions at the end of the session (falcon-trader#23)."""

    print_section(f"END-OF-DAY FLATTEN (reason={reason})")

    results = executor.flatten_positions(reason=reason)

    if not results:
        print("[INFO] No open positions to flatten")
        return

    closed = [r for r in results if r.get('action') == 'SELL']
    skipped = [r for r in results if r.get('action') != 'SELL']
    print(f"[EOD] Closed {len(closed)} position(s); {len(skipped)} not closed")
    for r in skipped:
        print(f"  [OPEN] {r['symbol']}: {r.get('reason')}")


def show_performance_summary(tracker, days=7):
    """Show performance summary"""

    print_section(f"PERFORMANCE SUMMARY (Last {days} Days)")

    tracker.print_performance_summary(days=days)


def show_account_status(executor):
    """Show current account status"""

    print_section("ACCOUNT STATUS")

    db = executor.db

    # Get account info
    account = db.execute("""
        SELECT cash, total_value
        FROM account
        ORDER BY id DESC
        LIMIT 1
    """, fetch='one')

    if account:
        cash, total_value = account
        print(f"  Cash: ${cash:,.2f}")
        print(f"  Total Value: ${total_value:,.2f}")

    # Get open positions
    positions = db.execute("""
        SELECT symbol, quantity, entry_price, current_price,
               (current_price - entry_price) / entry_price * 100 as pnl_pct
        FROM positions
        WHERE quantity > 0
    """, fetch='all')

    if positions:
        print(f"\n  Open Positions: {len(positions)}")
        for symbol, qty, entry, current, pnl in positions:
            value = qty * current
            print(f"    {symbol}: {qty} @ ${entry:.2f} -> ${current:.2f} ({pnl:+.2f}%) = ${value:,.2f}")
    else:
        print(f"\n  Open Positions: 0")


def main():
    """Main orchestrator runner"""

    # Print banner
    print("\n")
    print_header("FALCON MULTI-STRATEGY ORCHESTRATOR")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()

    # Load configuration.
    #
    # This used to be the bare relative path 'orchestrator/orchestrator_config.yaml',
    # which resolves against the current working directory. In the container the
    # working directory is /app and the file installs into the package, so the
    # orchestrator exited 1 on every start. It only ever worked when something
    # happened to be run from src/falcon_trader.
    #
    # Resolution order: an explicit override, then a file relative to the working
    # directory (so an operator-supplied config still wins), then the copy that
    # ships with the package.
    print("\n[INIT] Loading configuration...")
    _packaged = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'orchestrator', 'orchestrator_config.yaml',
    )
    _cwd_relative = os.path.join('orchestrator', 'orchestrator_config.yaml')

    config_file = os.environ.get('FALCON_ORCHESTRATOR_CONFIG') or (
        _cwd_relative if os.path.exists(_cwd_relative) else _packaged
    )

    if not os.path.exists(config_file):
        print(f"[ERROR] Configuration file not found: {config_file}")
        print(f"[HINT]  Set FALCON_ORCHESTRATOR_CONFIG, or reinstall the package "
              f"(expected at {_packaged})")
        sys.exit(1)

    print(f"[OK] Using config: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    print(f"[OK] Configuration loaded")

    # Initialize database.
    #
    # This was a hardcoded SQLite file at ./paper_trading.db -- inside the
    # container, thrown away on every restart -- while the dashboard, the
    # screener and every sentinel read PostgreSQL. The orchestrator therefore
    # monitored an empty book: real positions had no stop-loss or target
    # monitoring from anything, and any trade it had placed would have been
    # invisible (falcon-trader#43). get_db_manager() honors DATABASE_URL.
    print("[INIT] Initializing database...")
    db = get_db_manager()
    print(f"[OK] Database ready ({getattr(db, 'db_type', 'unknown')})")

    # Initialize components
    print("[INIT] Initializing orchestrator components...")
    executor = TradeExecutor(config, db_manager=db)
    tracker = PerformanceTracker(config, db_manager=db)
    print(f"[OK] All components initialized")

    print_separator()

    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == '--process':
            # Process screener results
            process_screener_results(executor, tracker, db)

        elif command == '--process-file':
            # Process a screener JSON file (operator-supplied path)
            path = sys.argv[2] if len(sys.argv) > 2 else 'screened_stocks.json'
            print(f"[ORCHESTRATOR] Processing file {path}...")
            executor.process_ai_screener(path)

        elif command == '--monitor':
            # Monitor positions
            monitor_positions(executor, tracker)

        elif command == '--performance':
            # Show performance
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            show_performance_summary(tracker, days)

        elif command == '--status':
            # Show account status
            show_account_status(executor)

        elif command == '--once':
            # Full cycle once
            process_screener_results(executor, tracker, db)
            monitor_positions(executor, tracker)
            show_account_status(executor)
            show_performance_summary(tracker, days=1)

        else:
            print(f"[ERROR] Unknown command: {command}")
            print("\nUsage:")
            print("  python3 run_orchestrator.py --process      # Process screener results from the database")
            print("  python3 run_orchestrator.py --process-file F  # Process a screener JSON file")
            print("  python3 run_orchestrator.py --monitor      # Monitor open positions")
            print("  python3 run_orchestrator.py --performance  # Show performance summary")
            print("  python3 run_orchestrator.py --status       # Show account status")
            print("  python3 run_orchestrator.py --once         # Run full cycle once")
            print("  python3 run_orchestrator.py --daemon       # Run continuous monitoring")
            sys.exit(1)

    else:
        # Default: Run full cycle in daemon mode
        print("\n[MODE] Continuous monitoring (daemon mode)")
        flatten_eod = eod_flatten_enabled(config)
        print(f"[MODE] End-of-day flatten: {'on' if flatten_eod else 'off'}"
              + ("" if flatten_eod else " — exits come from stops, targets and max_hold_days"))
        print("Press Ctrl+C to stop\n")

        try:
            cycle = 0
            last_block = None
            flattened_for = None
            consecutive_failures = 0
            while True:
                # Market-hours gate. This 5-minute cycle ran around the clock,
                # and monitor_positions() landing after the close is exactly
                # what produced the 16:50 SELL burst (falcon-trader#23).
                session = trading_guards.check_market_open()
                if not session.allowed:
                    if last_block != session.reason:
                        print(f"\n[IDLE] {session.message}")
                        last_block = session.reason
                    time.sleep(300)
                    continue
                last_block = None

                cycle += 1
                print(f"\n{'='*80}")
                print(f"CYCLE {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")

                # End-of-day flatten. Deliberate, once per session, with
                # reason='eod' on the trade rows -- as opposed to positions
                # simply carrying and being closed by whichever cycle happened
                # to run last (falcon-trader#23).
                today = datetime.now().date()
                if flatten_eod and trading_guards.should_flatten():
                    if flattened_for != today:
                        print("[EOD] Flatten window reached; closing open positions")
                        flatten_positions(executor, tracker, reason='eod')
                        flattened_for = today
                    time.sleep(60)
                    continue

                # One cycle's failure must not end the daemon.
                #
                # PostgreSQL recycles every backend when one dies -- a broken
                # COPY pipe during a backup was enough -- and the next query
                # raised OperationalError, which reached main() and exited 1.
                # systemd restarted the container thirty seconds later, so the
                # session lost a cycle over a two-second database blip. A
                # trading loop that dies on a transient error is worse than one
                # that logs it and comes back on the next cycle.
                try:
                    process_screener_results(executor, tracker, db)
                    monitor_positions(executor, tracker)

                    # Show status every 10 cycles
                    if cycle % 10 == 0:
                        show_account_status(executor)
                        show_performance_summary(tracker, days=1)
                    consecutive_failures = 0
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    consecutive_failures += 1
                    logger.exception("Cycle %s failed", cycle)
                    print(f"[ERROR] Cycle {cycle} failed: {type(exc).__name__}: {exc}")
                    print(f"[ERROR] {consecutive_failures} consecutive failure(s); "
                          f"retrying on the next cycle")
                    # Exit rather than spin if it is not transient: systemd
                    # restarts the unit, which re-reads config and reconnects.
                    if consecutive_failures >= MAX_CONSECUTIVE_CYCLE_FAILURES:
                        print(f"[FATAL] {consecutive_failures} cycles failed in a row; "
                              f"exiting so the unit restarts")
                        raise

                # Wait before next cycle (5 minutes)
                print(f"\n[SLEEP] Waiting 5 minutes until next cycle...")
                time.sleep(300)

        except KeyboardInterrupt:
            print(f"\n\n[SHUTDOWN] Orchestrator stopped by user")
            show_account_status(executor)
            show_performance_summary(tracker, days=1)

    print("\n")
    print_separator()
    print(f"Stopped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    print()


if __name__ == '__main__':
    main()
