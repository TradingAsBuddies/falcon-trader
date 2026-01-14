#!/usr/bin/env python3
"""
Strategy Orchestrator - Main Entry Point
Integrates all components of the automated strategy execution system
"""

import os
import sys
import time
import signal
from datetime import datetime
from paper_trading_bot import PaperTradingBot
from strategy_executor import StrategyExecutor
from strategy_optimizer import StrategyOptimizer
from strategy_analytics import StrategyAnalytics

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class StrategyOrchestrator:
    """Main orchestrator for automated strategy system"""

    def __init__(self, massive_api_key: str, claude_api_key: str,
                 db_path: str = "/var/lib/falcon/paper_trading.db",
                 symbols: list = None,
                 initial_balance: float = 10000.0,
                 update_interval: int = 60,
                 optimization_threshold: float = 0.05):
        """
        Initialize the strategy orchestrator

        Args:
            massive_api_key: Polygon.io API key
            claude_api_key: Claude API key
            db_path: Database path
            symbols: List of symbols to track
            initial_balance: Starting balance
            update_interval: Seconds between strategy evaluations
            optimization_threshold: Minimum improvement % for auto-deploy
        """
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'AAPL']

        self.massive_api_key = massive_api_key
        self.claude_api_key = claude_api_key
        self.db_path = db_path
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.update_interval = update_interval
        self.optimization_threshold = optimization_threshold

        # Components (initialized on start)
        self.bot = None
        self.executor = None
        self.optimizer = None
        self.analytics = None

        # Running state
        self.running = False
        self.start_time = None

        print(f"[ORCHESTRATOR] Initialized")
        print(f"[ORCHESTRATOR] Symbols: {', '.join(symbols)}")
        print(f"[ORCHESTRATOR] Initial balance: ${initial_balance:,.2f}")
        print(f"[ORCHESTRATOR] Update interval: {update_interval}s")
        print(f"[ORCHESTRATOR] Optimization threshold: {optimization_threshold:.1%}")

    def initialize_components(self):
        """Initialize all system components"""
        print(f"\n[ORCHESTRATOR] Initializing components...")

        # Initialize paper trading bot
        print(f"[ORCHESTRATOR] Initializing paper trading bot...")
        self.bot = PaperTradingBot(
            symbols=self.symbols,
            massive_api_key=self.massive_api_key,
            claude_api_key=self.claude_api_key,
            initial_balance=self.initial_balance,
            update_interval=self.update_interval
        )

        # Initialize strategy executor
        print(f"[ORCHESTRATOR] Initializing strategy executor...")
        self.executor = StrategyExecutor(
            paper_trading_bot=self.bot,
            db_path=self.db_path,
            update_interval=self.update_interval
        )

        # Initialize strategy optimizer
        print(f"[ORCHESTRATOR] Initializing strategy optimizer...")
        self.optimizer = StrategyOptimizer(
            claude_api_key=self.claude_api_key,
            db_path=self.db_path,
            improvement_threshold=self.optimization_threshold
        )

        # Initialize strategy analytics
        print(f"[ORCHESTRATOR] Initializing strategy analytics...")
        self.analytics = StrategyAnalytics(db_path=self.db_path)

        print(f"[ORCHESTRATOR] All components initialized")

    def start(self):
        """Start all components"""
        if self.running:
            print(f"[ORCHESTRATOR] System already running")
            return

        print(f"\n" + "=" * 80)
        print(f"Starting Falcon Automated Strategy Execution System")
        print(f"=" * 80)

        try:
            # Initialize if not already done
            if self.bot is None:
                self.initialize_components()

            # Start paper trading bot
            print(f"[ORCHESTRATOR] Starting paper trading bot...")
            self.bot.start()

            # Start strategy executor
            print(f"[ORCHESTRATOR] Starting strategy executor...")
            self.executor.start()

            self.running = True
            self.start_time = datetime.now()

            print(f"\n" + "=" * 80)
            print(f"[ORCHESTRATOR] System running")
            print(f"=" * 80)
            print(f"Start time: {self.start_time.isoformat()}")
            print(f"Database: {self.db_path}")
            print(f"\nComponents:")
            print(f"  - Paper Trading Bot: Running")
            print(f"  - Strategy Executor: Running")
            print(f"  - Strategy Optimizer: Ready")
            print(f"  - Strategy Analytics: Ready")
            print(f"\nPress Ctrl+C to stop")
            print(f"=" * 80 + "\n")

        except Exception as e:
            print(f"[ORCHESTRATOR] Error starting system: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
            raise

    def stop(self):
        """Stop all components"""
        if not self.running:
            print(f"[ORCHESTRATOR] System not running")
            return

        print(f"\n[ORCHESTRATOR] Stopping system...")

        # Stop executor first
        if self.executor:
            print(f"[ORCHESTRATOR] Stopping strategy executor...")
            self.executor.stop()

        # Stop bot
        if self.bot:
            print(f"[ORCHESTRATOR] Stopping paper trading bot...")
            self.bot.stop()

        self.running = False

        # Print summary
        if self.start_time:
            runtime = datetime.now() - self.start_time
            print(f"\n[ORCHESTRATOR] System stopped")
            print(f"Runtime: {runtime}")

            # Get final stats
            try:
                account = self.bot.get_account()
                print(f"\nFinal Account Value: ${account['total_value']:,.2f}")
                print(f"Cash: ${account['cash']:,.2f}")
                print(f"Initial Balance: ${self.initial_balance:,.2f}")
                pnl = account['total_value'] - self.initial_balance
                pnl_pct = (pnl / self.initial_balance) * 100
                print(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            except Exception as e:
                print(f"[ORCHESTRATOR] Could not retrieve final stats: {e}")

        print(f"[ORCHESTRATOR] Goodbye")

    def get_status(self) -> dict:
        """Get current system status"""
        status = {
            "running": self.running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "components": {
                "bot": self.bot.running if self.bot else False,
                "executor": self.executor.running if self.executor else False,
                "optimizer": self.optimizer is not None,
                "analytics": self.analytics is not None
            }
        }

        if self.bot and self.running:
            account = self.bot.get_account()
            status["account"] = {
                "total_value": account['total_value'],
                "cash": account['cash'],
                "pnl": account['total_value'] - self.initial_balance,
                "pnl_pct": ((account['total_value'] - self.initial_balance) / self.initial_balance) * 100
            }

        if self.executor and self.running:
            status["active_strategies"] = len(self.executor.active_strategies)

        return status

    def run_optimization_cycle(self):
        """Run a single optimization cycle (can be called manually or scheduled)"""
        if not self.running:
            print(f"[ORCHESTRATOR] System not running")
            return

        print(f"\n[ORCHESTRATOR] Running optimization cycle...")
        self.optimizer.monitor_and_optimize()
        print(f"[ORCHESTRATOR] Optimization cycle complete\n")

    def print_status(self):
        """Print current system status"""
        status = self.get_status()

        print(f"\n" + "=" * 80)
        print(f"SYSTEM STATUS")
        print(f"=" * 80)
        print(f"Running: {status['running']}")

        if status['start_time']:
            print(f"Start time: {status['start_time']}")
            print(f"Uptime: {status['uptime_seconds']:.0f} seconds")

        print(f"\nComponents:")
        for component, running in status['components'].items():
            print(f"  {component}: {'Running' if running else 'Stopped'}")

        if 'account' in status:
            print(f"\nAccount:")
            print(f"  Total value: ${status['account']['total_value']:,.2f}")
            print(f"  Cash: ${status['account']['cash']:,.2f}")
            print(f"  P&L: ${status['account']['pnl']:,.2f} ({status['account']['pnl_pct']:+.2f}%)")

        if 'active_strategies' in status:
            print(f"\nActive strategies: {status['active_strategies']}")

        print(f"=" * 80 + "\n")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n[ORCHESTRATOR] Received interrupt signal")
    if orchestrator:
        orchestrator.stop()
    sys.exit(0)


def main():
    """Main entry point"""
    global orchestrator

    # Get API keys from environment
    massive_key = os.getenv('MASSIVE_API_KEY')
    claude_key = os.getenv('CLAUDE_API_KEY')

    if not massive_key:
        print("ERROR: MASSIVE_API_KEY not set")
        print("Please set MASSIVE_API_KEY in environment or .env file")
        sys.exit(1)

    if not claude_key:
        print("ERROR: CLAUDE_API_KEY not set")
        print("Please set CLAUDE_API_KEY in environment or .env file")
        sys.exit(1)

    # Configuration
    db_path = "/var/lib/falcon/paper_trading.db"
    if not os.path.exists('/var/lib/falcon'):
        # Development mode - use local database
        db_path = "paper_trading.db"
        print(f"[ORCHESTRATOR] Running in development mode")

    symbols = os.getenv('TRADING_SYMBOLS', 'SPY,QQQ,AAPL').split(',')
    initial_balance = float(os.getenv('INITIAL_BALANCE', '10000'))
    update_interval = int(os.getenv('UPDATE_INTERVAL', '60'))

    # Initialize orchestrator
    orchestrator = StrategyOrchestrator(
        massive_api_key=massive_key,
        claude_api_key=claude_key,
        db_path=db_path,
        symbols=symbols,
        initial_balance=initial_balance,
        update_interval=update_interval
    )

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the system
    orchestrator.start()

    # Main loop - print status every 5 minutes and check for optimization triggers
    try:
        last_status_time = time.time()
        last_optimization_time = time.time()
        status_interval = 300  # 5 minutes
        optimization_interval = 3600  # 1 hour

        while True:
            time.sleep(10)

            current_time = time.time()

            # Print status every 5 minutes
            if current_time - last_status_time >= status_interval:
                orchestrator.print_status()
                last_status_time = current_time

            # Run optimization cycle every hour
            if current_time - last_optimization_time >= optimization_interval:
                orchestrator.run_optimization_cycle()
                last_optimization_time = current_time

    except KeyboardInterrupt:
        print(f"\n[ORCHESTRATOR] Keyboard interrupt received")
        orchestrator.stop()
    except Exception as e:
        print(f"\n[ORCHESTRATOR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        orchestrator.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
