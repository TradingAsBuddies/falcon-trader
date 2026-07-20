"""
Trade Executor

Orchestrates the full multi-strategy trading workflow:
1. Route stocks to optimal strategies
2. Validate entry conditions
3. Fetch market data
4. Generate signals from engines
5. Execute trades
6. Monitor positions for exits
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

from falcon_trader.orchestrator.utils.timezone import now_et

from falcon_core import DatabaseManager, get_db_manager
from falcon_trader.orchestrator.utils.cents import to_cents, to_dollars, calc_cost, calc_pnl
from falcon_trader.orchestrator.routers.strategy_router import StrategyRouter
from falcon_trader.orchestrator.validators.entry_validator import EntryValidator
from falcon_trader.orchestrator.engines import RSIEngine, MomentumEngine, BollingerEngine, RosterStrategyEngine
from falcon_trader.orchestrator.execution.market_data_fetcher import MarketDataFetcher


class TradeExecutor:
    """
    Main trade execution orchestrator

    Integrates:
    - Strategy Router (Phase 1)
    - Entry Validator (Phase 2)
    - Strategy Engines (Phase 3)
    - Market Data Fetcher (Phase 4)
    """

    def __init__(self, config: dict, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize trade executor

        Args:
            config: Configuration dictionary
            db_manager: Optional DatabaseManager instance
        """
        self.config = config
        self.running = False

        # Initialize database manager
        if db_manager:
            self.db = db_manager
        else:
            self.db = get_db_manager()

        # Initialize components
        self.router = StrategyRouter(config)
        self.validator = EntryValidator(config)
        self.data_fetcher = MarketDataFetcher(config)

        # Initialize strategy engines
        self.engines = {
            'rsi_mean_reversion': RSIEngine(config, self.db),
            'momentum_breakout': MomentumEngine(config, self.db),
            'bollinger_mean_reversion': BollingerEngine(config, self.db)
        }

        # Roster (paper_trading) strategies loaded from strategy_roster. These are
        # falcon_core BaseStrategy plugins promoted through the backtest lifecycle,
        # wrapped so they drive the same paper broker as the hardcoded engines.
        # See falcon-core#8.
        self.roster_engines = {}          # strategy_name -> RosterStrategyEngine
        self.roster_symbols = {}          # strategy_name -> [symbols]
        exec_config = config.get('execution', {})
        self.use_roster_strategies = exec_config.get('use_roster_strategies', True)
        if self.use_roster_strategies:
            self.load_roster_strategies()

        # Get monitoring config
        self.monitoring_config = config.get('monitoring', {})
        self.check_interval = self.monitoring_config.get('check_interval_seconds', 60)

        print("[EXECUTOR] Trade Executor initialized")
        print(f"[EXECUTOR] Strategies: {list(self.engines.keys())}")
        if self.roster_engines:
            print(f"[EXECUTOR] Roster strategies (paper_trading): {list(self.roster_engines.keys())}")

    def load_roster_strategies(self) -> None:
        """Load status='paper_trading' strategies from strategy_roster.

        Each row's strategy_code is validated + loaded into a BaseStrategy class
        (via falcon_core), instantiated with its roster params, and wrapped in a
        RosterStrategyEngine. The engine is registered under the strategy_name in
        both self.roster_engines (for entry generation) and self.engines (so
        monitor_positions can find it to run stop/target exits on its positions).
        """
        try:
            from falcon_core.backtesting.strategy_loader import (
                validate_strategy_code, load_strategy_from_code,
            )
        except Exception as e:
            print(f"[EXECUTOR] Roster strategies unavailable (falcon_core import failed): {e}")
            return

        try:
            rows = self.db.execute(
                "SELECT strategy_name, symbols, interval, params, strategy_code "
                "FROM strategy_roster WHERE status = %s AND strategy_code IS NOT NULL",
                ('paper_trading',),
                fetch='all'
            )
        except Exception as e:
            print(f"[EXECUTOR] Could not query strategy_roster: {e}")
            return

        for row in (rows or []):
            name = row['strategy_name']
            code = row['strategy_code']
            if not code or not code.strip():
                continue

            is_valid, err = validate_strategy_code(code)
            if not is_valid:
                print(f"[EXECUTOR] Roster strategy '{name}' failed validation: {err}")
                continue

            cls = load_strategy_from_code(code, name)
            if cls is None:
                print(f"[EXECUTOR] Roster strategy '{name}' could not be loaded")
                continue

            # Parse roster params (JSON/JSONB) and symbols.
            params_raw = row.get('params') if isinstance(row, dict) else None
            symbols = row.get('symbols') if isinstance(row, dict) else None
            interval = (row.get('interval') if isinstance(row, dict) else None) or '5m'
            params_dict = self._as_dict(params_raw)
            symbols = self._as_list(symbols)

            try:
                # Start from the strategy's OWN default params (a subclass may add
                # fields like orb_minutes), then overlay any roster-provided values
                # onto fields that exist — never downcast to the base StrategyParams.
                params = cls.default_params()
                for pk, pv in (params_dict or {}).items():
                    if hasattr(params, pk):
                        setattr(params, pk, pv)
                instance = cls(params)
            except Exception as e:
                print(f"[EXECUTOR] Roster strategy '{name}' init failed: {e}")
                continue

            engine = RosterStrategyEngine(self.config, self.db, instance, name, interval)
            self.roster_engines[name] = engine
            self.roster_symbols[name] = symbols
            # Register so monitor_positions resolves positions tagged with this name.
            self.engines[name] = engine

    @staticmethod
    def _as_dict(value) -> dict:
        """Coerce a JSON/JSONB column value into a dict."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except (ValueError, TypeError):
                return {}
        return {}

    @staticmethod
    def _as_list(value) -> list:
        """Coerce a JSON/JSONB column value into a list of symbols."""
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except (ValueError, TypeError):
                return []
        return []

    def process_roster_strategies(self) -> Dict:
        """Run each paper_trading roster strategy over its symbols for entries/exits.

        For every (strategy, symbol) pair this fetches bars at the strategy's
        interval, generates a signal via the wrapped BaseStrategy, and executes
        BUY/SELL through the inherited paper broker. Risk-based exits (stop/target)
        are still handled separately by monitor_positions.
        """
        summary = {
            'strategies': len(self.roster_engines),
            'symbols_processed': 0,
            'trades_executed': 0,
            'details': []
        }
        if not self.roster_engines:
            return summary

        # Global position cap shared with the rest of the executor.
        max_positions = self.config.get('risk_management', {}).get('max_positions', 10)

        print(f"\n[ROSTER] Running {len(self.roster_engines)} paper_trading strategies")
        print("=" * 60)

        for name, engine in self.roster_engines.items():
            symbols = self.roster_symbols.get(name, [])
            interval = engine.interval
            for symbol in symbols:
                summary['symbols_processed'] += 1
                detail = {'strategy': name, 'symbol': symbol, 'action': 'NONE', 'reason': ''}
                try:
                    market_data = self.data_fetcher.fetch_market_data(
                        symbol, lookback_days=5, interval=interval,
                    )
                    if not market_data or market_data.get('error'):
                        detail['reason'] = f"data unavailable: {market_data.get('error', 'unknown')}"
                        summary['details'].append(detail)
                        continue

                    is_valid, reason = self.data_fetcher.validate_data_quality(market_data, min_periods=20)
                    if not is_valid:
                        detail['reason'] = f"data quality: {reason}"
                        summary['details'].append(detail)
                        continue

                    signal = engine.generate_signal(symbol, market_data)
                    detail['action'] = signal.action
                    detail['reason'] = signal.reason

                    # Enforce the global position cap on new entries only.
                    if signal.action == 'BUY':
                        open_positions = self.db.execute(
                            "SELECT COUNT(*) AS c FROM positions WHERE quantity > 0",
                            fetch='one'
                        )
                        count = int(open_positions['c']) if open_positions else 0
                        if count >= max_positions:
                            detail['action'] = 'SKIP'
                            detail['reason'] = f"position cap reached ({count}/{max_positions})"
                            summary['details'].append(detail)
                            continue

                    if signal.action in ('BUY', 'SELL'):
                        print(f"[ROSTER] {name}/{symbol}: {signal.action} — {signal.reason}")
                        result = engine.execute_signal(signal)
                        if result.success:
                            summary['trades_executed'] += 1
                            detail['executed'] = True
                        else:
                            detail['executed'] = False
                            detail['reason'] = result.error or signal.reason
                    summary['details'].append(detail)

                except Exception as e:
                    detail['reason'] = f"error: {e}"
                    summary['details'].append(detail)
                    continue

        print(f"[ROSTER] Symbols processed: {summary['symbols_processed']}, "
              f"trades executed: {summary['trades_executed']}")
        return summary

    def process_stock(self, symbol: str, ai_recommendation: Optional[Dict] = None) -> Dict:
        """
        Process a stock through the full workflow

        Args:
            symbol: Stock symbol
            ai_recommendation: Optional AI screener recommendation

        Returns:
            Dict with processing results
        """
        result = {
            'symbol': symbol,
            'timestamp': now_et().isoformat(),
            'success': False,
            'action': 'NONE',
            'reason': '',
            'details': {}
        }

        try:
            print(f"\n[EXECUTOR] Processing {symbol}")
            print("=" * 60)

            # Step 1: Quick daily fetch for routing classification
            print(f"[STEP 1] Classifying stock...")
            daily_data = self.data_fetcher.fetch_market_data(
                symbol, lookback_days=30, interval='1d',
            )

            if not daily_data or daily_data.get('error'):
                result['reason'] = f"Failed to fetch market data: {daily_data.get('error', 'Unknown error')}"
                print(f"  [ERROR] {result['reason']}")
                return result

            # Route to strategy using daily data (price + volatility classification)
            routing_decision = self.router.route_with_market_data(symbol, daily_data)

            print(f"  Strategy: {routing_decision.selected_strategy}")
            print(f"  Classification: {routing_decision.classification}")
            print(f"  Confidence: {routing_decision.confidence:.1%}")
            print(f"  Volatility: {routing_decision.profile.volatility:.1%}")

            result['details']['routing'] = {
                'strategy': routing_decision.selected_strategy,
                'classification': routing_decision.classification,
                'confidence': routing_decision.confidence,
                'reason': routing_decision.reason
            }

            # Step 2: Fetch strategy-specific market data at the right interval
            strategy_name = routing_decision.selected_strategy
            strategy_config = self.config.get('strategies', {}).get(strategy_name, {})
            interval = strategy_config.get('interval', '1m')

            print(f"[STEP 2] Fetching {interval} bars for {strategy_name}...")
            market_data = self.data_fetcher.fetch_market_data(
                symbol, lookback_days=5, interval=interval,
            )

            if not market_data or market_data.get('error'):
                result['reason'] = f"Failed to fetch {interval} data: {market_data.get('error', 'Unknown error')}"
                print(f"  [ERROR] {result['reason']}")
                return result

            print(f"  Price: ${market_data['price']:.2f}")
            print(f"  Bars: {len(market_data['prices'])} ({interval})")
            print(f"  Source: {market_data['source']}")

            # Validate data quality
            is_valid, reason = self.data_fetcher.validate_data_quality(market_data, min_periods=20)
            if not is_valid:
                result['reason'] = f"Data quality check failed: {reason}"
                print(f"  [ERROR] {result['reason']}")
                return result

            result['details']['market_data'] = {
                'price': market_data['price'],
                'volume': market_data.get('volume', 0),
                'bars': len(market_data['prices']),
                'interval': interval,
                'source': market_data['source']
            }

            # Step 3: Validate entry
            print(f"[STEP 3] Validating entry...")

            # Get recommended stop-loss
            stop_loss = self.validator.get_recommended_stop_loss(symbol, market_data['price'])

            validation_result = self.validator.validate_entry(
                symbol,
                market_data['price'],
                stop_loss
            )

            print(f"  Valid: {validation_result.is_valid}")
            print(f"  Reason: {validation_result.reason}")

            if not validation_result.is_valid:
                result['reason'] = f"Entry validation failed: {validation_result.reason}"
                result['details']['validation'] = {
                    'is_valid': False,
                    'reason': validation_result.reason
                }
                print(f"  [SKIP] Entry not valid")
                return result

            result['details']['validation'] = {
                'is_valid': True,
                'reason': validation_result.reason
            }

            # Step 4: Generate signal from engine
            print(f"[STEP 4] Generating signal from {routing_decision.selected_strategy} engine...")

            engine = self.engines[routing_decision.selected_strategy]
            signal = engine.generate_signal(symbol, market_data)

            print(f"  Signal: {signal.action}")
            print(f"  Reason: {signal.reason}")
            print(f"  Confidence: {signal.confidence:.1%}")

            result['details']['signal'] = {
                'action': signal.action,
                'reason': signal.reason,
                'confidence': signal.confidence
            }

            if signal.action == 'HOLD':
                result['reason'] = f"No entry signal: {signal.reason}"
                result['action'] = 'HOLD'
                print(f"  [HOLD] No entry signal")
                return result

            # Step 5: Execute trade
            if signal.action == 'BUY':
                print(f"[STEP 5] Executing BUY order...")
                print(f"  Quantity: {signal.quantity}")
                print(f"  Price: ${signal.price:.2f}")
                print(f"  Stop Loss: ${signal.stop_loss:.2f}")
                print(f"  Profit Target: ${signal.profit_target:.2f}")

                execution_result = engine.execute_signal(signal)

                if execution_result.success:
                    print(f"  [SUCCESS] Trade executed")
                    result['success'] = True
                    result['action'] = 'BUY'
                    result['reason'] = signal.reason
                    result['details']['execution'] = {
                        'success': True,
                        'quantity': execution_result.quantity,
                        'price': execution_result.price,
                        'timestamp': execution_result.timestamp.isoformat()
                    }
                else:
                    print(f"  [ERROR] Trade failed: {execution_result.error}")
                    result['reason'] = f"Execution failed: {execution_result.error}"
                    result['details']['execution'] = {
                        'success': False,
                        'error': execution_result.error
                    }

        except Exception as e:
            result['reason'] = f"Error processing stock: {str(e)}"
            print(f"[ERROR] {result['reason']}")
            import traceback
            traceback.print_exc()

        return result

    def monitor_positions(self) -> List[Dict]:
        """
        Monitor all open positions for exit signals

        Returns:
            List of actions taken
        """
        actions = []

        # Map short strategy names to engine keys
        strategy_map = {
            'rsi': 'rsi_mean_reversion',
            'momentum': 'momentum_breakout',
            'bollinger': 'bollinger_mean_reversion'
        }

        try:
            # Get all positions
            positions_data = self.db.execute(
                "SELECT * FROM positions WHERE quantity > 0",
                fetch='all'
            )

            if not positions_data:
                return actions

            print(f"\n[MONITOR] Checking {len(positions_data)} positions")
            print("=" * 60)

            for pos_data in positions_data:
                symbol = pos_data['symbol']
                # sqlite3.Row doesn't have .get() method, use dict() or try/except
                try:
                    strategy = pos_data['strategy'] if pos_data['strategy'] else 'unknown'
                except (KeyError, IndexError):
                    strategy = 'unknown'

                try:
                    # Fetch current market data at the strategy's interval
                    engine_key = strategy_map.get(strategy, strategy)
                    strat_config = self.config.get('strategies', {}).get(engine_key, {})
                    interval = strat_config.get('interval', '1m')
                    market_data = self.data_fetcher.fetch_market_data(
                        symbol, lookback_days=5, interval=interval,
                    )

                    if not market_data or market_data.get('error'):
                        print(f"[WARNING] Could not fetch data for {symbol}")
                        continue

                    current_price = market_data['price']

                    # Update current price in database
                    self.db.execute("""
                        UPDATE positions
                        SET current_price = %s, last_updated = %s
                        WHERE symbol = %s
                    """, (current_price, now_et().isoformat(), symbol))

                    print(f"\n{symbol} ({strategy}):")
                    print(f"  Entry: ${pos_data['entry_price']:.2f}")
                    print(f"  Current: ${current_price:.2f}")

                    # Calculate P&L (cents arithmetic)
                    entry_price = float(pos_data['entry_price'])
                    entry_cents = to_cents(entry_price)
                    pnl_pct = (to_cents(current_price) - entry_cents) / entry_cents if entry_cents else 0.0
                    print(f"  P&L: {pnl_pct:+.1%}")

                    # Get engine for this strategy (map short name to full key)
                    engine_key = strategy_map.get(strategy, strategy)
                    if engine_key in self.engines:
                        engine = self.engines[engine_key]

                        # Check for exit signal
                        signal = engine.monitor_position(symbol, current_price)

                        if signal and signal.action == 'SELL':
                            print(f"  [EXIT SIGNAL] {signal.reason}")

                            # Execute sell
                            result = engine.execute_signal(signal)

                            if result.success:
                                print(f"  [SUCCESS] Position closed")
                                actions.append({
                                    'symbol': symbol,
                                    'action': 'SELL',
                                    'price': result.price,
                                    'reason': signal.reason,
                                    'pnl_pct': pnl_pct
                                })
                            else:
                                print(f"  [ERROR] Sell failed: {result.error}")
                        else:
                            print(f"  [HOLD] No exit signal")

                except Exception as e:
                    print(f"[ERROR] Error monitoring {symbol}: {e}")
                    continue

        except Exception as e:
            print(f"[ERROR] Error in monitor_positions: {e}")

        return actions

    def process_ai_screener(self, screener_file: str = 'screened_stocks.json') -> Dict:
        """
        Process stocks from AI screener file

        Args:
            screener_file: Path to screener JSON file

        Returns:
            Dict with processing summary
        """
        summary = {
            'total_stocks': 0,
            'processed': 0,
            'trades_executed': 0,
            'skipped': 0,
            'errors': 0,
            'details': []
        }

        try:
            # Load screener data
            if not os.path.exists(screener_file):
                print(f"[ERROR] Screener file not found: {screener_file}")
                return summary

            with open(screener_file, 'r') as f:
                screener_data = json.load(f)

            # Handle array format (multiple screening sessions)
            if isinstance(screener_data, list):
                if len(screener_data) > 0:
                    latest_screen = screener_data[-1]  # Most recent (last in array)
                    recommendations = latest_screen.get('recommendations', [])
                else:
                    recommendations = []
            else:
                # Object format
                recommendations = screener_data.get('stocks', [])

            summary['total_stocks'] = len(recommendations)

            print(f"\n[SCREENER] Processing {len(recommendations)} stocks from AI screener")
            print("=" * 60)

            for rec in recommendations:
                symbol = rec.get('ticker', rec.get('symbol', ''))
                if not symbol:
                    continue

                summary['processed'] += 1

                # Process the stock
                result = self.process_stock(symbol, rec)

                if result['success'] and result['action'] == 'BUY':
                    summary['trades_executed'] += 1
                elif result.get('reason'):
                    if 'error' in result['reason'].lower():
                        summary['errors'] += 1
                    else:
                        summary['skipped'] += 1

                summary['details'].append(result)

                # Small delay between stocks
                time.sleep(0.5)

        except Exception as e:
            print(f"[ERROR] Error processing screener: {e}")

        return summary

    def get_portfolio_status(self) -> Dict:
        """
        Get current portfolio status

        Returns:
            Dict with portfolio metrics
        """
        try:
            # Get account balance
            account = self.db.execute("SELECT * FROM account LIMIT 1", fetch='one')
            cash = float(account['cash']) if account else 0.0

            # Get positions
            positions_data = self.db.execute(
                "SELECT * FROM positions WHERE quantity > 0",
                fetch='all'
            )

            positions = []
            total_position_cents = 0
            total_pnl_cents = 0

            if positions_data:
                for pos_data in positions_data:
                    symbol = pos_data['symbol']
                    qty = int(pos_data['quantity'])
                    entry = float(pos_data['entry_price'])

                    # Fetch current price
                    current_price = self.data_fetcher.get_current_price(symbol)

                    # Cents arithmetic
                    pos_value_cents = to_cents(current_price) * qty
                    pnl_cents = (to_cents(current_price) - to_cents(entry)) * qty
                    entry_cents = to_cents(entry)
                    pnl_pct = (to_cents(current_price) - entry_cents) / entry_cents if entry_cents else 0.0

                    total_position_cents += pos_value_cents
                    total_pnl_cents += pnl_cents

                    positions.append({
                        'symbol': symbol,
                        'quantity': qty,
                        'entry_price': entry,
                        'current_price': current_price,
                        'position_value': to_dollars(pos_value_cents),
                        'unrealized_pnl': to_dollars(pnl_cents),
                        'unrealized_pnl_pct': pnl_pct,
                        'strategy': pos_data.get('strategy', 'unknown')
                    })

            total_value = to_dollars(to_cents(cash) + total_position_cents)

            return {
                'cash': cash,
                'position_value': to_dollars(total_position_cents),
                'total_value': total_value,
                'unrealized_pnl': to_dollars(total_pnl_cents),
                'unrealized_pnl_pct': (total_pnl_cents / (to_cents(cash) + total_position_cents)) if (to_cents(cash) + total_position_cents) > 0 else 0.0,
                'positions_count': len(positions),
                'positions': positions
            }

        except Exception as e:
            print(f"[ERROR] Error getting portfolio status: {e}")
            return {
                'cash': 0.0,
                'position_value': 0.0,
                'total_value': 0.0,
                'error': str(e)
            }

    def run_monitoring_loop(self, interval_seconds: Optional[int] = None):
        """
        Run continuous monitoring loop

        Args:
            interval_seconds: Override check interval
        """
        interval = interval_seconds or self.check_interval
        self.running = True

        print(f"\n[EXECUTOR] Starting monitoring loop (interval: {interval}s)")
        print("[EXECUTOR] Press Ctrl+C to stop")

        try:
            while self.running:
                # Monitor positions
                actions = self.monitor_positions()

                if actions:
                    print(f"\n[MONITOR] Actions taken: {len(actions)}")
                    for action in actions:
                        print(f"  {action['symbol']}: {action['action']} at ${action['price']:.2f} ({action['reason']})")

                # Sleep
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[EXECUTOR] Monitoring loop stopped by user")
            self.running = False

    def stop(self):
        """Stop the executor"""
        self.running = False
        print("[EXECUTOR] Executor stopped")
