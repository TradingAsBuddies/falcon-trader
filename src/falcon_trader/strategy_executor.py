#!/usr/bin/env python3
"""
Strategy Executor - Multi-Strategy Execution Engine
Runs multiple trading strategies in parallel with real-time market data
"""

import json
import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from db_manager import DatabaseManager
from paper_trading_bot import PaperTradingBot


@dataclass
class Signal:
    """Trading signal from a strategy"""
    strategy_id: int
    symbol: str
    action: str  # 'buy' or 'sell'
    quantity: int
    confidence: float
    reason: str
    price: float


class StrategyInstance:
    """Represents a single active strategy"""

    def __init__(self, strategy_id: int, strategy_name: str, code: str,
                 symbols: List[str], parameters: Dict, allocation_pct: float):
        """
        Initialize strategy instance

        Args:
            strategy_id: Database ID
            strategy_name: Strategy name
            code: Strategy code
            symbols: List of symbols to monitor
            parameters: Strategy parameters
            allocation_pct: Portfolio allocation percentage
        """
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name
        self.code = code
        self.symbols = symbols
        self.parameters = parameters
        self.allocation_pct = allocation_pct
        self.positions = {}  # symbol -> position info

    def evaluate_signals(self, symbol: str, market_data: Dict,
                        historical_data: pd.DataFrame) -> Optional[Signal]:
        """
        Execute strategy logic and generate signal

        Args:
            symbol: Stock symbol
            market_data: Current market data dict
            historical_data: Historical OHLCV data

        Returns:
            Signal object if conditions met, None otherwise
        """
        try:
            # Extract current price
            if not market_data or 'price' not in market_data:
                return None

            current_price = market_data['price']

            # Calculate indicators from historical data
            if historical_data is None or len(historical_data) < 20:
                return None

            # Calculate RSI
            rsi = self._calculate_rsi(historical_data['close'], period=14)
            if rsi is None:
                return None

            # Get parameters
            entry_threshold = self.parameters.get('entry_threshold', 30)
            exit_threshold = self.parameters.get('exit_threshold', 70)

            # Check if we have a position
            has_position = symbol in self.positions and self.positions[symbol]['quantity'] > 0

            # Entry logic
            if not has_position:
                if rsi < entry_threshold:
                    # Generate BUY signal
                    confidence = (entry_threshold - rsi) / entry_threshold  # Higher confidence when more oversold
                    return Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        action='buy',
                        quantity=0,  # Will be calculated by executor based on allocation
                        confidence=min(confidence, 1.0),
                        reason=f"RSI {rsi:.1f} < {entry_threshold} (oversold)",
                        price=current_price
                    )

            # Exit logic
            else:
                position = self.positions[symbol]
                bars_held = position.get('bars_held', 0)
                hold_days = self.parameters.get('hold_days', 5)

                if rsi > exit_threshold or bars_held >= hold_days:
                    # Generate SELL signal
                    confidence = 0.8
                    reason = f"RSI {rsi:.1f} > {exit_threshold}" if rsi > exit_threshold else f"Held {bars_held} days"
                    return Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        action='sell',
                        quantity=position['quantity'],
                        confidence=confidence,
                        reason=reason,
                        price=current_price
                    )

            return None

        except Exception as e:
            print(f"[STRATEGY {self.strategy_id}] Error evaluating {symbol}: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return None

            # Calculate price changes
            deltas = prices.diff()

            # Separate gains and losses
            gains = deltas.copy()
            losses = deltas.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)

            # Calculate average gains and losses
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.iloc[-1]

        except Exception as e:
            print(f"[RSI] Calculation error: {e}")
            return None

    def update_position_state(self, symbol: str, action: str, quantity: int, price: float):
        """Update position tracking"""
        if action == 'buy':
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': price,
                'bars_held': 0,
                'entry_time': datetime.now()
            }
        elif action == 'sell':
            if symbol in self.positions:
                del self.positions[symbol]

    def increment_bars_held(self):
        """Increment bars held counter for all positions"""
        for symbol in self.positions:
            self.positions[symbol]['bars_held'] += 1


class StrategyExecutor:
    """Manages execution of multiple trading strategies"""

    def __init__(self, paper_trading_bot: PaperTradingBot,
                 db_path: str = "/var/lib/falcon/paper_trading.db",
                 update_interval: int = 60):
        """
        Initialize strategy executor

        Args:
            paper_trading_bot: PaperTradingBot instance
            db_path: Database path
            update_interval: Seconds between strategy evaluations
        """
        self.bot = paper_trading_bot
        self.db = DatabaseManager({'db_path': db_path, 'db_type': 'sqlite'})
        self.active_strategies = {}  # strategy_id -> StrategyInstance
        self.update_interval = update_interval
        self.running = False
        self.thread = None

        print(f"[EXECUTOR] Initialized with update interval: {update_interval}s")

    def load_active_strategies(self):
        """Load all active strategies from database"""
        strategies = self.db.execute(
            "SELECT * FROM active_strategies WHERE status = 'active'",
            fetch='all'
        )

        if not strategies:
            print("[EXECUTOR] No active strategies found")
            return

        for strat in strategies:
            try:
                self.active_strategies[strat['id']] = StrategyInstance(
                    strategy_id=strat['id'],
                    strategy_name=strat['strategy_name'],
                    code=strat['strategy_code'],
                    symbols=json.loads(strat['symbols']),
                    parameters=json.loads(strat['parameters']),
                    allocation_pct=strat['allocation_pct']
                )
                print(f"[EXECUTOR] Loaded strategy {strat['id']}: {strat['strategy_name']}")
            except Exception as e:
                print(f"[EXECUTOR] Error loading strategy {strat['id']}: {e}")

        print(f"[EXECUTOR] Loaded {len(self.active_strategies)} active strategies")

    def _get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical data for indicator calculation

        Args:
            symbol: Stock symbol
            days: Number of days of history

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # For now, create synthetic data from current price
            # In production, this would fetch from massive_flat_files or API
            current_price = self.bot.market_data.get(symbol, {}).get('price', 100)

            # Generate simple historical data
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

            # Simple random walk around current price
            import numpy as np
            np.random.seed(hash(symbol) % 2**32)
            prices = current_price * (1 + np.random.randn(days) * 0.02).cumprod()

            df = pd.DataFrame({
                'date': dates,
                'close': prices,
                'open': prices * 0.99,
                'high': prices * 1.01,
                'low': prices * 0.98,
                'volume': np.random.randint(1000000, 10000000, days)
            })

            return df

        except Exception as e:
            print(f"[EXECUTOR] Error getting historical data for {symbol}: {e}")
            return None

    def _run_loop(self):
        """Main execution loop (runs in background thread)"""
        print("[EXECUTOR] Starting execution loop")

        while self.running:
            try:
                # Reload strategies in case new ones were added
                self.load_active_strategies()

                # Get current market data from bot
                market_data = self.bot.get_market_data()

                if not market_data:
                    print("[EXECUTOR] No market data available, waiting...")
                    time.sleep(self.update_interval)
                    continue

                # Evaluate all strategies
                all_signals = []
                for strategy in self.active_strategies.values():
                    # Increment bars held for all positions
                    strategy.increment_bars_held()

                    for symbol in strategy.symbols:
                        # Get historical data for indicators
                        hist_data = self._get_historical_data(symbol, days=30)

                        # Evaluate strategy
                        signal = strategy.evaluate_signals(
                            symbol,
                            market_data.get(symbol),
                            hist_data
                        )

                        if signal:
                            all_signals.append(signal)
                            print(f"[EXECUTOR] Signal: {signal.action.upper()} {signal.symbol} "
                                  f"from strategy {signal.strategy_id} "
                                  f"(confidence: {signal.confidence:.2f})")

                # Process signals with allocation
                if all_signals:
                    self.process_signals(all_signals)

                time.sleep(self.update_interval)

            except Exception as e:
                print(f"[EXECUTOR] Error in loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(self.update_interval)

        print("[EXECUTOR] Execution loop stopped")

    def process_signals(self, signals: List[Signal]):
        """
        Process signals with performance-weighted allocation

        Args:
            signals: List of Signal objects
        """
        account = self.bot.get_account()
        available_cash = account['cash']

        print(f"[EXECUTOR] Processing {len(signals)} signals with ${available_cash:,.2f} available")

        # Get performance weights for all strategies
        signal_weights = {}
        for signal in signals:
            perf = self._get_strategy_performance(signal.strategy_id)

            # Performance weight = win_rate * signal_confidence
            # Default win_rate to 0.5 for new strategies
            win_rate = perf.get('win_rate', 0.5) if perf else 0.5
            weight = win_rate * signal.confidence
            signal_weights[signal] = weight

            print(f"[EXECUTOR] Strategy {signal.strategy_id} weight: {weight:.3f} "
                  f"(win_rate: {win_rate:.2f}, confidence: {signal.confidence:.2f})")

        # Calculate allocation for each signal
        total_weight = sum(signal_weights.values())
        if total_weight == 0:
            print("[EXECUTOR] Total weight is zero, skipping signals")
            return

        for signal in signals:
            try:
                # Calculate allocation
                weight_fraction = signal_weights[signal] / total_weight
                allocation = weight_fraction * available_cash

                # Calculate quantity
                if signal.action == 'buy':
                    quantity = int(allocation / signal.price)

                    if quantity > 0:
                        result = self.bot.place_order_with_strategy(
                            strategy_id=signal.strategy_id,
                            symbol=signal.symbol,
                            side='buy',
                            quantity=quantity,
                            signal_reason=signal.reason,
                            confidence=signal.confidence
                        )

                        if result['status'] == 'success':
                            print(f"[EXECUTOR] BUY executed: {quantity} {signal.symbol} @ ${signal.price:.2f}")

                            # Update strategy position state
                            strategy = self.active_strategies.get(signal.strategy_id)
                            if strategy:
                                strategy.update_position_state(
                                    signal.symbol, 'buy', quantity, signal.price
                                )
                    else:
                        print(f"[EXECUTOR] Insufficient funds for {signal.symbol} (need ${signal.price:.2f})")

                elif signal.action == 'sell':
                    # For sell signals, use the quantity from the signal
                    result = self.bot.place_order_with_strategy(
                        strategy_id=signal.strategy_id,
                        symbol=signal.symbol,
                        side='sell',
                        quantity=signal.quantity,
                        signal_reason=signal.reason,
                        confidence=signal.confidence
                    )

                    if result['status'] == 'success':
                        print(f"[EXECUTOR] SELL executed: {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")

                        # Update strategy position state
                        strategy = self.active_strategies.get(signal.strategy_id)
                        if strategy:
                            strategy.update_position_state(
                                signal.symbol, 'sell', signal.quantity, signal.price
                            )

            except Exception as e:
                print(f"[EXECUTOR] Error processing signal for {signal.symbol}: {e}")

    def _get_strategy_performance(self, strategy_id: int) -> Optional[Dict]:
        """Get performance metrics for a strategy"""
        try:
            perf = self.db.execute(
                "SELECT * FROM strategy_performance WHERE strategy_id = %s",
                (strategy_id,),
                fetch='one'
            )
            return dict(perf) if perf else None
        except Exception as e:
            print(f"[EXECUTOR] Error getting performance for strategy {strategy_id}: {e}")
            return None

    def start(self):
        """Start executor background thread"""
        if not self.running:
            self.running = True
            self.load_active_strategies()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("[EXECUTOR] Started")

    def stop(self):
        """Stop executor"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=5)
            print("[EXECUTOR] Stopped")


def main():
    """Test the strategy executor"""
    import os
    import sys

    massive_key = os.getenv('MASSIVE_API_KEY')
    if not massive_key:
        print("ERROR: MASSIVE_API_KEY not set")
        sys.exit(1)

    # Initialize bot
    bot = PaperTradingBot(
        symbols=['SPY', 'QQQ', 'AAPL'],
        massive_api_key=massive_key,
        initial_balance=10000.0,
        update_interval=60
    )
    bot.start()

    # Initialize executor
    executor = StrategyExecutor(bot, update_interval=60)

    print("\n" + "=" * 60)
    print("Strategy Executor Test")
    print("=" * 60)
    print("\nStarting executor...")

    executor.start()

    # Run for a bit
    try:
        print("\nExecutor running... Press Ctrl+C to stop")
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        executor.stop()
        bot.stop()
        print("Stopped")


if __name__ == '__main__':
    main()
