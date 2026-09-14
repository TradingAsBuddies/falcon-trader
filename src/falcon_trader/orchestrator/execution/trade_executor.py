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

from falcon_core import DatabaseManager
from falcon_trader import trading_guards
from falcon_trader.orchestrator.routers.strategy_router import StrategyRouter
from falcon_trader.orchestrator.validators.entry_validator import EntryValidator
from falcon_trader.orchestrator.engines import RSIEngine, MomentumEngine, BollingerEngine
from falcon_trader.orchestrator.engines.base_engine import TradeSignal
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
            db_config = {'db_type': 'sqlite', 'db_path': 'paper_trading.db'}
            self.db = DatabaseManager(db_config)

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

        # Get monitoring config
        self.monitoring_config = config.get('monitoring', {})
        self.check_interval = self.monitoring_config.get('check_interval_seconds', 60)

        print("[EXECUTOR] Trade Executor initialized")
        print(f"[EXECUTOR] Strategies: {list(self.engines.keys())}")

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
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'action': 'NONE',
            'reason': '',
            'details': {}
        }

        try:
            print(f"\n[EXECUTOR] Processing {symbol}")
            print("=" * 60)

            # Step 1: Route to strategy
            print(f"[STEP 1] Routing to strategy...")
            routing_decision = self.router.route(symbol, use_yfinance=False)

            print(f"  Strategy: {routing_decision.selected_strategy}")
            print(f"  Classification: {routing_decision.classification}")
            print(f"  Confidence: {routing_decision.confidence:.1%}")

            result['details']['routing'] = {
                'strategy': routing_decision.selected_strategy,
                'classification': routing_decision.classification,
                'confidence': routing_decision.confidence,
                'reason': routing_decision.reason
            }

            # Step 2: Fetch market data
            print(f"[STEP 2] Fetching market data...")
            market_data = self.data_fetcher.fetch_market_data(symbol, lookback_days=30)

            if not market_data or market_data.get('error'):
                result['reason'] = f"Failed to fetch market data: {market_data.get('error', 'Unknown error')}"
                print(f"  [ERROR] {result['reason']}")
                return result

            print(f"  Current Price: ${market_data['price']:.2f}")
            print(f"  Data Points: {len(market_data['prices'])}")
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
                'data_points': len(market_data['prices']),
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
                    # Fetch current market data
                    market_data = self.data_fetcher.fetch_market_data(symbol, lookback_days=30)

                    if not market_data or market_data.get('error'):
                        print(f"[WARNING] Could not fetch data for {symbol}")
                        continue

                    current_price = market_data['price']

                    # Update current price in database
                    self.db.execute("""
                        UPDATE positions
                        SET current_price = %s, last_updated = %s
                        WHERE symbol = %s
                    """, (current_price, datetime.now().isoformat(), symbol))

                    print(f"\n{symbol} ({strategy}):")
                    print(f"  Entry: ${pos_data['entry_price']:.2f}")
                    print(f"  Current: ${current_price:.2f}")

                    # Calculate P&L
                    pnl_pct = (current_price - pos_data['entry_price']) / pos_data['entry_price']
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

    def flatten_positions(self, reason: str = 'eod') -> List[Dict]:
        """Close every open position deliberately (falcon-trader#23).

        There was no EOD flatten anywhere in the repo. Positions simply carried,
        and the 16:50 SELL burst in the live book was the 5-minute monitor cycle
        happening to land after the close -- an accident, not an exit policy.

        Every resulting trade row carries `reason`, so an end-of-day exit is
        distinguishable from a stop-loss or a signal exit after the fact.
        """
        actions = []

        try:
            positions_data = self.db.execute(
                "SELECT * FROM positions WHERE quantity > 0", fetch='all',
            ) or []
        except Exception as e:
            print(f"[ERROR] flatten_positions could not read positions: {e}")
            return actions

        if not positions_data:
            print("[EOD] No open positions to flatten")
            return actions

        print(f"[EOD] Flattening {len(positions_data)} position(s), reason={reason}")

        for pos_data in positions_data:
            symbol = pos_data['symbol']
            try:
                market_data = self.data_fetcher.fetch_market_data(
                    symbol, lookback_days=5,
                )
                current_price = (
                    market_data.get('price')
                    if market_data and not market_data.get('error') else None
                )
                if not current_price:
                    # Refuse to invent a price. An unflattened position that is
                    # reported is safer than one closed at a made-up mark.
                    print(f"  [SKIP] {symbol}: no current price; left open")
                    actions.append({
                        'symbol': symbol, 'action': 'SKIP',
                        'reason': 'no_price', 'flatten_reason': reason,
                    })
                    continue

                strategy = pos_data.get('strategy') or ''
                engine_key = {
                    'rsi': 'rsi_mean_reversion',
                    'momentum': 'momentum_breakout',
                    'bollinger': 'bollinger_mean_reversion',
                }.get(strategy, strategy)

                engine = self.engines.get(engine_key) or next(
                    iter(self.engines.values()), None,
                )
                if engine is None:
                    print(f"  [SKIP] {symbol}: no engine available")
                    continue

                signal = TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=int(pos_data['quantity']),
                    price=current_price,
                    reason=reason,
                )
                result = engine.execute_signal(signal)

                if result.success:
                    print(f"  [FLAT] {symbol}: {pos_data['quantity']} @ ${current_price:.4f}")
                    actions.append({
                        'symbol': symbol, 'action': 'SELL',
                        'price': result.price, 'reason': reason,
                        'quantity': int(pos_data['quantity']),
                    })
                else:
                    print(f"  [ERROR] {symbol}: flatten failed: {result.error}")
                    actions.append({
                        'symbol': symbol, 'action': 'ERROR',
                        'reason': result.error, 'flatten_reason': reason,
                    })

            except Exception as e:
                print(f"[ERROR] Error flattening {symbol}: {e}")
                continue

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
            total_position_value = 0.0
            total_unrealized_pnl = 0.0

            if positions_data:
                for pos_data in positions_data:
                    symbol = pos_data['symbol']

                    # Fetch current price
                    current_price = self.data_fetcher.get_current_price(symbol)

                    position_value = current_price * pos_data['quantity']
                    unrealized_pnl = (current_price - pos_data['entry_price']) * pos_data['quantity']
                    unrealized_pnl_pct = (current_price - pos_data['entry_price']) / pos_data['entry_price']

                    total_position_value += position_value
                    total_unrealized_pnl += unrealized_pnl

                    positions.append({
                        'symbol': symbol,
                        'quantity': pos_data['quantity'],
                        'entry_price': pos_data['entry_price'],
                        'current_price': current_price,
                        'position_value': position_value,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': unrealized_pnl_pct,
                        'strategy': pos_data.get('strategy', 'unknown')
                    })

            total_value = cash + total_position_value

            return {
                'cash': cash,
                'position_value': total_position_value,
                'total_value': total_value,
                'unrealized_pnl': total_unrealized_pnl,
                'unrealized_pnl_pct': (total_unrealized_pnl / total_value) if total_value > 0 else 0.0,
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

        last_block = None
        flattened_for = None
        try:
            while self.running:
                # Market-hours gate (falcon-trader#23). This loop ran 24/7/365.
                session = trading_guards.check_market_open()
                if not session.allowed:
                    if last_block != session.reason:
                        print(f"[EXECUTOR] Idle: {session.message}")
                        last_block = session.reason
                    time.sleep(interval)
                    continue
                last_block = None

                today = datetime.now().date()
                if trading_guards.should_flatten():
                    if flattened_for != today:
                        self.flatten_positions(reason='eod')
                        flattened_for = today
                    time.sleep(interval)
                    continue

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
