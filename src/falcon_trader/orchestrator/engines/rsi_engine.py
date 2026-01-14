"""
RSI Mean Reversion Strategy Engine

Best for:
- ETFs (SPY, QQQ, IWM)
- Stable large-cap stocks
- Low-volatility stocks

Strategy:
- Buy when RSI < oversold threshold (mean reversion opportunity)
- Sell when RSI > overbought OR profit target hit OR max hold time exceeded
- Stop-loss for risk management
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta

from .base_engine import BaseStrategyEngine, TradeSignal


class RSIEngine(BaseStrategyEngine):
    """
    RSI Mean Reversion Strategy Engine

    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - rsi_oversold: Buy threshold (default: 45)
    - rsi_overbought: Sell threshold (default: 55)
    - profit_target: Profit target % (default: 0.025 = 2.5%)
    - max_hold_days: Maximum holding period (default: 12)
    - position_size_pct: % of portfolio per trade (default: 0.25 = 25%)
    """

    def __init__(self, config: dict, db_manager=None):
        super().__init__(config, db_manager)

        # Get RSI-specific config
        rsi_config = config.get('strategies', {}).get('rsi_mean_reversion', {})

        self.rsi_period = rsi_config.get('rsi_period', 14)
        self.rsi_oversold = rsi_config.get('rsi_oversold', 45)
        self.rsi_overbought = rsi_config.get('rsi_overbought', 55)
        self.profit_target = rsi_config.get('profit_target', 0.025)
        self.max_hold_days = rsi_config.get('max_hold_days', 12)
        self.position_size_pct = rsi_config.get('position_size_pct', 0.25)

    def calculate_rsi(self, prices: list, period: int = 14) -> float:
        """
        Calculate RSI indicator

        Args:
            prices: List of closing prices (most recent last)
            period: RSI period

        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data

        # Convert to numpy array
        prices = np.array(prices)

        # Calculate price changes
        deltas = np.diff(prices)

        # Separate gains and losses
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0  # Prevent division by zero

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signal(self, symbol: str, market_data: Dict) -> TradeSignal:
        """
        Generate trade signal based on RSI

        Args:
            symbol: Stock symbol
            market_data: Dict with:
                - 'price': Current price
                - 'prices': Historical prices (list, most recent last)
                - 'volume': Current volume (optional)

        Returns:
            TradeSignal object
        """
        current_price = market_data.get('price', 0.0)
        prices = market_data.get('prices', [])

        if not current_price or len(prices) < self.rsi_period + 1:
            return TradeSignal(
                symbol=symbol,
                action='HOLD',
                quantity=0,
                price=current_price,
                reason="Insufficient data for RSI calculation"
            )

        # Calculate RSI
        rsi = self.calculate_rsi(prices, self.rsi_period)

        # Check if we have a position
        position = self.get_position(symbol)

        if position:
            # We have a position - check for exit signals
            position.update_current_price(current_price)

            # Calculate days held
            days_held = (datetime.now() - position.entry_timestamp).days

            # Check profit
            profit_pct = position.unrealized_pnl_pct

            # Exit conditions
            if rsi > self.rsi_overbought:
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason=f"RSI overbought: {rsi:.1f} > {self.rsi_overbought}",
                    confidence=0.8,
                    metadata={'rsi': rsi, 'profit_pct': profit_pct}
                )

            if profit_pct >= self.profit_target:
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason=f"Profit target reached: {profit_pct:.1%} >= {self.profit_target:.1%}",
                    confidence=0.9,
                    metadata={'rsi': rsi, 'profit_pct': profit_pct}
                )

            if days_held >= self.max_hold_days:
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason=f"Max hold time: {days_held} days >= {self.max_hold_days} days",
                    confidence=0.7,
                    metadata={'rsi': rsi, 'profit_pct': profit_pct}
                )

            # Hold position
            return TradeSignal(
                symbol=symbol,
                action='HOLD',
                quantity=0,
                price=current_price,
                reason=f"Holding position (RSI: {rsi:.1f}, P&L: {profit_pct:.1%})",
                confidence=0.5,
                metadata={'rsi': rsi, 'profit_pct': profit_pct}
            )

        else:
            # No position - check for entry signals
            if rsi < self.rsi_oversold:
                # Calculate position size
                quantity = self.calculate_position_size(
                    symbol,
                    current_price,
                    self.position_size_pct
                )

                if quantity > 0:
                    # Calculate stop-loss and profit target
                    stop_loss = current_price * (1 - self.min_stop_buffer)
                    profit_target = current_price * (1 + self.profit_target)

                    return TradeSignal(
                        symbol=symbol,
                        action='BUY',
                        quantity=quantity,
                        price=current_price,
                        reason=f"RSI oversold: {rsi:.1f} < {self.rsi_oversold}",
                        stop_loss=stop_loss,
                        profit_target=profit_target,
                        confidence=0.8,
                        metadata={'rsi': rsi}
                    )

            # No entry signal
            return TradeSignal(
                symbol=symbol,
                action='HOLD',
                quantity=0,
                price=current_price,
                reason=f"RSI neutral: {rsi:.1f}",
                confidence=0.3,
                metadata={'rsi': rsi}
            )

    def backtest_signal(self, symbol: str, historical_data: pd.DataFrame) -> Dict:
        """
        Backtest strategy on historical data

        Args:
            symbol: Stock symbol
            historical_data: DataFrame with columns: date, close, volume

        Returns:
            Dict with backtest results
        """
        if len(historical_data) < self.rsi_period + 1:
            return {'error': 'Insufficient data for backtest'}

        # Calculate RSI for all periods
        closes = historical_data['close'].values
        rsis = []

        for i in range(self.rsi_period, len(closes)):
            window = closes[i - self.rsi_period:i + 1]
            rsi = self.calculate_rsi(window, self.rsi_period)
            rsis.append(rsi)

        # Simulate trades
        trades = []
        position = None
        entry_price = 0
        entry_date = None
        entry_rsi = 0

        for i in range(len(rsis)):
            idx = i + self.rsi_period
            date = historical_data.iloc[idx]['date']
            price = historical_data.iloc[idx]['close']
            rsi = rsis[i]

            if position is None:
                # Check for entry
                if rsi < self.rsi_oversold:
                    position = 'LONG'
                    entry_price = price
                    entry_date = date
                    entry_rsi = rsi
            else:
                # Check for exit
                days_held = (date - entry_date).days if hasattr(date - entry_date, 'days') else 0
                profit_pct = (price - entry_price) / entry_price

                exit_signal = False
                exit_reason = ""

                if rsi > self.rsi_overbought:
                    exit_signal = True
                    exit_reason = "RSI overbought"
                elif profit_pct >= self.profit_target:
                    exit_signal = True
                    exit_reason = "Profit target"
                elif days_held >= self.max_hold_days:
                    exit_signal = True
                    exit_reason = "Max hold time"

                if exit_signal:
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'entry_rsi': entry_rsi,
                        'exit_rsi': rsi,
                        'profit_pct': profit_pct,
                        'days_held': days_held,
                        'reason': exit_reason
                    })
                    position = None

        # Calculate statistics
        if not trades:
            return {
                'total_trades': 0,
                'message': 'No trades generated'
            }

        profits = [t['profit_pct'] for t in trades]
        wins = [p for p in profits if p > 0]

        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'win_rate': len(wins) / len(trades),
            'avg_profit': np.mean(profits),
            'total_return': sum(profits),
            'avg_hold_days': np.mean([t['days_held'] for t in trades]),
            'trades': trades
        }
