#!/usr/bin/env python3
"""
Multi-Strategy Orchestrator - Main Runner
Processes AI screener results through the complete orchestrator workflow
"""
import os
import sys
import yaml
import time
from datetime import datetime
from falcon_trader.orchestrator.execution.trade_executor import TradeExecutor
from falcon_trader.orchestrator.monitors.performance_tracker import PerformanceTracker
from falcon_core import DatabaseManager

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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


def process_screener_results(executor, tracker, screener_file='screened_stocks.json'):
    """Process AI screener results through orchestrator"""

    print_section("PROCESSING AI SCREENER RESULTS")

    if not os.path.exists(screener_file):
        print(f"[ERROR] Screener file not found: {screener_file}")
        return None

    print(f"[ORCHESTRATOR] Processing {screener_file}...")
    summary = executor.process_ai_screener(screener_file)

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
                action = detail.get('action', 'N/A')
                quantity = detail.get('quantity', 0)
                price = detail.get('price', 0.0)
                print(f"  [OK] {symbol}: {action} {quantity} @ ${price:.2f}")
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

    print(f"[ORCHESTRATOR] Monitored {len(results)} positions")

    for result in results:
        symbol = result['symbol']
        action = result['action']

        if action == 'HOLD':
            current_price = result.get('current_price', 0.0)
            pnl_pct = result.get('pnl_pct', 0.0)
            print(f"  [HOLD] {symbol}: ${current_price:.2f} ({pnl_pct:+.2f}%)")
        elif action == 'SELL':
            reason = result.get('reason', 'Exit signal')
            print(f"  [EXIT] {symbol}: {reason}")


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

    # Load configuration
    print("\n[INIT] Loading configuration...")
    config_file = 'orchestrator/orchestrator_config.yaml'

    if not os.path.exists(config_file):
        print(f"[ERROR] Configuration file not found: {config_file}")
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    print(f"[OK] Configuration loaded")

    # Initialize database
    print("[INIT] Initializing database...")
    db_config = {'db_type': 'sqlite', 'db_path': './paper_trading.db'}
    db = DatabaseManager(db_config)
    print(f"[OK] Database ready")

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
            process_screener_results(executor, tracker)

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
            process_screener_results(executor, tracker)
            monitor_positions(executor, tracker)
            show_account_status(executor)
            show_performance_summary(tracker, days=1)

        else:
            print(f"[ERROR] Unknown command: {command}")
            print("\nUsage:")
            print("  python3 run_orchestrator.py --process      # Process AI screener results")
            print("  python3 run_orchestrator.py --monitor      # Monitor open positions")
            print("  python3 run_orchestrator.py --performance  # Show performance summary")
            print("  python3 run_orchestrator.py --status       # Show account status")
            print("  python3 run_orchestrator.py --once         # Run full cycle once")
            print("  python3 run_orchestrator.py --daemon       # Run continuous monitoring")
            sys.exit(1)

    else:
        # Default: Run full cycle in daemon mode
        print("\n[MODE] Continuous monitoring (daemon mode)")
        print("Press Ctrl+C to stop\n")

        try:
            cycle = 0
            while True:
                cycle += 1
                print(f"\n{'='*80}")
                print(f"CYCLE {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")

                # Process screener results
                process_screener_results(executor, tracker)

                # Monitor positions
                monitor_positions(executor, tracker)

                # Show status every 10 cycles
                if cycle % 10 == 0:
                    show_account_status(executor)
                    show_performance_summary(tracker, days=1)

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
