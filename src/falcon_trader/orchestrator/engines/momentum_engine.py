"""
Momentum Breakout Strategy Engine

Best for:
- Penny stocks (<$5)
- High-volatility stocks (>30% volatility)
- Momentum plays with volume confirmation

Strategy:
- Buy on breakout above resistance with high volume
- Sell on momentum loss (MA cross) OR profit target OR trailing stop
- Aggressive profit targets (5-10%)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta

from .base_engine import BaseStrategyEngine, TradeSignal


class MomentumEngine(BaseStrategyEngine):
    """
    Momentum Breakout Strategy Engine

    Parameters:
    - breakout_period: Period for resistance level (default: 20)
    - volume_multiplier: Volume must be X times average (default: 1.5)
    - ma_fast: Fast moving average period (default: 5)
    - ma_slow: Slow moving average period (default: 20)
    - profit_target: Profit target % (default: 0.08 = 8%)
    - trailing_stop_pct: Trailing stop % (default: 0.10 = 10%)
    - max_hold_days: Maximum holding period (default: 20)
    - position_size_pct: % of portfolio per trade (default: 0.20 = 20%)
    """

    def __init__(self, config: dict, db_manager=None):
        super().__init__(config, db_manager)

        # Get momentum-specific config
        momentum_config = config.get('strategies', {}).get('momentum_breakout', {})

        self.breakout_period = momentum_config.get('breakout_period', 20)
        self.volume_multiplier = momentum_config.get('volume_multiplier', 1.5)
        self.ma_fast = momentum_config.get('ma_fast', 5)
        self.ma_slow = momentum_config.get('ma_slow', 20)
        self.profit_target = momentum_config.get('profit_target', 0.08)
        self.trailing_stop_pct = momentum_config.get('trailing_stop_pct', 0.10)
        self.max_hold_days = momentum_config.get('max_hold_days', 20)
        self.position_size_pct = momentum_config.get('position_size_pct', 0.20)

    def calculate_moving_average(self, prices: list, period: int) -> float:
        """
        Calculate simple moving average

        Args:
            prices: List of closing prices
            period: MA period

        Returns:
            Moving average value
        """
        if len(prices) < period:
            return prices[-1] if prices else 0.0

        return np.mean(prices[-period:])

    def calculate_resistance(self, prices: list, period: int) -> float:
        """
        Calculate resistance level (highest high in period)

        Args:
            prices: List of closing prices
            period: Lookback period

        Returns:
            Resistance level
        """
        if len(prices) < period:
            return max(prices) if prices else 0.0

        return max(prices[-period:])

    def check_breakout(self, current_price: float, prices: list,
                       current_volume: float, volumes: list) -> bool:
        """
        Check if breakout conditions are met

        Args:
            current_price: Current price
            prices: Historical prices
            current_volume: Current volume
            volumes: Historical volumes

        Returns:
            True if breakout confirmed
        """
        # Check price breakout
        resistance = self.calculate_resistance(prices[:-1], self.breakout_period)
        if current_price <= resistance:
            return False

        # Check volume confirmation
        if volumes and current_volume:
            avg_volume = np.mean(volumes[-self.breakout_period:])
            if current_volume < avg_volume * self.volume_multiplier:
                return False

        return True

    def check_momentum_loss(self, prices: list) -> bool:
        """
        Check if momentum is lost (fast MA crosses below slow MA)

        Args:
            prices: List of closing prices

        Returns:
            True if momentum lost
        """
        if len(prices) < self.ma_slow:
            return False

        ma_fast = self.calculate_moving_average(prices, self.ma_fast)
        ma_slow = self.calculate_moving_average(prices, self.ma_slow)

        return ma_fast < ma_slow

    def generate_signal(self, symbol: str, market_data: Dict) -> TradeSignal:
        """
        Generate trade signal based on momentum

        Args:
            symbol: Stock symbol
            market_data: Dict with:
                - 'price': Current price
                - 'prices': Historical prices (list)
                - 'volume': Current volume
                - 'volumes': Historical volumes (list)

        Returns:
            TradeSignal object
        """
        current_price = market_data.get('price', 0.0)
        prices = market_data.get('prices', [])
        current_volume = market_data.get('volume', 0)
        volumes = market_data.get('volumes', [])

        if not current_price or len(prices) < self.breakout_period:
            return TradeSignal(
                symbol=symbol,
                action='HOLD',
                quantity=0,
                price=current_price,
                reason="Insufficient data for momentum calculation"
            )

        # Check if we have a position
        position = self.get_position(symbol)

        if position:
            # We have a position - check for exit signals
            position.update_current_price(current_price)

            # Calculate days held
            days_held = (datetime.now() - position.entry_timestamp).days

            # Check profit
            profit_pct = position.unrealized_pnl_pct

            # Calculate highest price since entry (for trailing stop)
            highest_price = max(prices[-days_held:]) if days_held > 0 else current_price
            trailing_stop = highest_price * (1 - self.trailing_stop_pct)

            # Exit conditions
            if profit_pct >= self.profit_target:
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason=f"Profit target reached: {profit_pct:.1%} >= {self.profit_target:.1%}",
                    confidence=0.9,
                    metadata={'profit_pct': profit_pct}
                )

            if current_price <= trailing_stop:
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason=f"Trailing stop hit: ${current_price:.2f} <= ${trailing_stop:.2f}",
                    confidence=0.85,
                    metadata={'profit_pct': profit_pct, 'trailing_stop': trailing_stop}
                )

            if self.check_momentum_loss(prices):
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason="Momentum lost (MA cross)",
                    confidence=0.75,
                    metadata={'profit_pct': profit_pct}
                )

            if days_held >= self.max_hold_days:
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason=f"Max hold time: {days_held} days >= {self.max_hold_days} days",
                    confidence=0.7,
                    metadata={'profit_pct': profit_pct}
                )

            # Hold position
            return TradeSignal(
                symbol=symbol,
                action='HOLD',
                quantity=0,
                price=current_price,
                reason=f"Holding position (P&L: {profit_pct:.1%})",
                confidence=0.6,
                metadata={'profit_pct': profit_pct}
            )

        else:
            # No position - check for entry signals
            if self.check_breakout(current_price, prices, current_volume, volumes):
                # Calculate position size
                quantity = self.calculate_position_size(
                    symbol,
                    current_price,
                    self.position_size_pct
                )

                if quantity > 0:
                    # Calculate stop-loss and profit target
                    resistance = self.calculate_resistance(prices[:-1], self.breakout_period)
                    stop_loss = max(
                        current_price * (1 - self.min_stop_buffer),
                        resistance * 0.95  # 5% below resistance
                    )
                    profit_target = current_price * (1 + self.profit_target)

                    return TradeSignal(
                        symbol=symbol,
                        action='BUY',
                        quantity=quantity,
                        price=current_price,
                        reason=f"Breakout confirmed with volume",
                        stop_loss=stop_loss,
                        profit_target=profit_target,
                        confidence=0.85,
                        metadata={
                            'resistance': resistance,
                            'volume_ratio': current_volume / np.mean(volumes[-self.breakout_period:]) if volumes else 0
                        }
                    )

            # No entry signal
            ma_fast = self.calculate_moving_average(prices, self.ma_fast)
            ma_slow = self.calculate_moving_average(prices, self.ma_slow)

            return TradeSignal(
                symbol=symbol,
                action='HOLD',
                quantity=0,
                price=current_price,
                reason=f"No breakout (MA Fast: ${ma_fast:.2f}, MA Slow: ${ma_slow:.2f})",
                confidence=0.3,
                metadata={'ma_fast': ma_fast, 'ma_slow': ma_slow}
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
        if len(historical_data) < self.breakout_period + 1:
            return {'error': 'Insufficient data for backtest'}

        closes = historical_data['close'].values
        volumes = historical_data['volume'].values if 'volume' in historical_data.columns else None

        # Simulate trades
        trades = []
        position = None
        entry_price = 0
        entry_date = None
        highest_price = 0

        for i in range(self.breakout_period, len(closes)):
            date = historical_data.iloc[i]['date']
            price = closes[i]
            volume = volumes[i] if volumes is not None else 0

            if position is None:
                # Check for entry
                prices_window = closes[:i + 1]
                volumes_window = volumes[:i + 1] if volumes is not None else []

                if self.check_breakout(price, prices_window, volume, volumes_window):
                    position = 'LONG'
                    entry_price = price
                    entry_date = date
                    highest_price = price
            else:
                # Update highest price
                highest_price = max(highest_price, price)

                # Check for exit
                days_held = (date - entry_date).days if hasattr(date - entry_date, 'days') else 0
                profit_pct = (price - entry_price) / entry_price
                trailing_stop = highest_price * (1 - self.trailing_stop_pct)

                exit_signal = False
                exit_reason = ""

                if profit_pct >= self.profit_target:
                    exit_signal = True
                    exit_reason = "Profit target"
                elif price <= trailing_stop:
                    exit_signal = True
                    exit_reason = "Trailing stop"
                elif self.check_momentum_loss(closes[:i + 1]):
                    exit_signal = True
                    exit_reason = "Momentum loss"
                elif days_held >= self.max_hold_days:
                    exit_signal = True
                    exit_reason = "Max hold time"

                if exit_signal:
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'highest_price': highest_price,
                        'profit_pct': profit_pct,
                        'days_held': days_held,
                        'reason': exit_reason
                    })
                    position = None
                    highest_price = 0

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
