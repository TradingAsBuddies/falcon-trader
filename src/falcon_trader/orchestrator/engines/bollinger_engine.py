"""
Bollinger Bands Mean Reversion Strategy Engine

Best for:
- Range-bound stocks
- Mid-cap stocks with moderate volatility
- Stable sector stocks (utilities, consumer staples)

Strategy:
- Buy at lower Bollinger Band (oversold, mean reversion expected)
- Sell at middle band (return to mean) OR upper band (overbought)
- Stop-loss below lower band for failed reversions
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from .base_engine import BaseStrategyEngine, TradeSignal


class BollingerEngine(BaseStrategyEngine):
    """
    Bollinger Bands Mean Reversion Strategy Engine

    Parameters:
    - bb_period: Bollinger Bands period (default: 20)
    - bb_std: Standard deviations for bands (default: 2.0)
    - profit_target: Profit target % (default: 0.04 = 4%)
    - max_hold_days: Maximum holding period (default: 15)
    - position_size_pct: % of portfolio per trade (default: 0.25 = 25%)
    - exit_at_middle: Exit at middle band vs upper band (default: True)
    """

    def __init__(self, config: dict, db_manager=None):
        super().__init__(config, db_manager)

        # Get Bollinger-specific config
        bb_config = config.get('strategies', {}).get('bollinger_mean_reversion', {})

        self.bb_period = bb_config.get('bb_period', 20)
        self.bb_std = bb_config.get('bb_std', 2.0)
        self.profit_target = bb_config.get('profit_target', 0.04)
        self.max_hold_days = bb_config.get('max_hold_days', 15)
        self.position_size_pct = bb_config.get('position_size_pct', 0.25)
        self.exit_at_middle = bb_config.get('exit_at_middle', True)

    def calculate_bollinger_bands(self, prices: list) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands

        Args:
            prices: List of closing prices

        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        if len(prices) < self.bb_period:
            # Not enough data - return current price as all bands
            current = prices[-1] if prices else 0.0
            return (current, current, current)

        # Calculate middle band (SMA)
        prices_array = np.array(prices[-self.bb_period:])
        middle = np.mean(prices_array)

        # Calculate standard deviation
        std = np.std(prices_array)

        # Calculate upper and lower bands
        upper = middle + (self.bb_std * std)
        lower = middle - (self.bb_std * std)

        return (middle, upper, lower)

    def calculate_bandwidth(self, prices: list) -> float:
        """
        Calculate Bollinger Band Width (volatility indicator)

        Args:
            prices: List of closing prices

        Returns:
            Band width as percentage of middle band
        """
        middle, upper, lower = self.calculate_bollinger_bands(prices)

        if middle == 0:
            return 0.0

        bandwidth = (upper - lower) / middle
        return bandwidth

    def check_at_lower_band(self, current_price: float, prices: list,
                           tolerance: float = 0.02) -> bool:
        """
        Check if price is at or below lower Bollinger Band

        Args:
            current_price: Current price
            prices: Historical prices
            tolerance: Allow entry slightly above lower band (default 2%)

        Returns:
            True if at lower band
        """
        middle, upper, lower = self.calculate_bollinger_bands(prices)

        # Allow entry slightly above lower band
        entry_threshold = lower * (1 + tolerance)

        return current_price <= entry_threshold

    def check_at_middle_band(self, current_price: float, prices: list,
                            tolerance: float = 0.02) -> bool:
        """
        Check if price is at or above middle Bollinger Band

        Args:
            current_price: Current price
            prices: Historical prices
            tolerance: Allow exit slightly below middle band (default 2%)

        Returns:
            True if at middle band
        """
        middle, upper, lower = self.calculate_bollinger_bands(prices)

        # Allow exit slightly below middle band
        exit_threshold = middle * (1 - tolerance)

        return current_price >= exit_threshold

    def check_at_upper_band(self, current_price: float, prices: list,
                           tolerance: float = 0.02) -> bool:
        """
        Check if price is at or above upper Bollinger Band

        Args:
            current_price: Current price
            prices: Historical prices
            tolerance: Allow exit slightly below upper band (default 2%)

        Returns:
            True if at upper band
        """
        middle, upper, lower = self.calculate_bollinger_bands(prices)

        # Allow exit slightly below upper band
        exit_threshold = upper * (1 - tolerance)

        return current_price >= exit_threshold

    def generate_signal(self, symbol: str, market_data: Dict) -> TradeSignal:
        """
        Generate trade signal based on Bollinger Bands

        Args:
            symbol: Stock symbol
            market_data: Dict with:
                - 'price': Current price
                - 'prices': Historical prices (list)
                - 'volume': Current volume (optional)

        Returns:
            TradeSignal object
        """
        current_price = market_data.get('price', 0.0)
        prices = market_data.get('prices', [])

        if not current_price or len(prices) < self.bb_period:
            return TradeSignal(
                symbol=symbol,
                action='HOLD',
                quantity=0,
                price=current_price,
                reason="Insufficient data for Bollinger Bands calculation"
            )

        # Calculate Bollinger Bands
        middle, upper, lower = self.calculate_bollinger_bands(prices)
        bandwidth = self.calculate_bandwidth(prices)

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
            if self.exit_at_middle and self.check_at_middle_band(current_price, prices):
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason=f"Reached middle band: ${current_price:.2f} >= ${middle:.2f}",
                    confidence=0.8,
                    metadata={
                        'middle_band': middle,
                        'upper_band': upper,
                        'lower_band': lower,
                        'bandwidth': bandwidth,
                        'profit_pct': profit_pct
                    }
                )

            if self.check_at_upper_band(current_price, prices):
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason=f"Reached upper band: ${current_price:.2f} >= ${upper:.2f}",
                    confidence=0.9,
                    metadata={
                        'middle_band': middle,
                        'upper_band': upper,
                        'lower_band': lower,
                        'bandwidth': bandwidth,
                        'profit_pct': profit_pct
                    }
                )

            if profit_pct >= self.profit_target:
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason=f"Profit target reached: {profit_pct:.1%} >= {self.profit_target:.1%}",
                    confidence=0.85,
                    metadata={
                        'middle_band': middle,
                        'bandwidth': bandwidth,
                        'profit_pct': profit_pct
                    }
                )

            if days_held >= self.max_hold_days:
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=current_price,
                    reason=f"Max hold time: {days_held} days >= {self.max_hold_days} days",
                    confidence=0.7,
                    metadata={
                        'bandwidth': bandwidth,
                        'profit_pct': profit_pct
                    }
                )

            # Hold position
            band_position = "lower"
            if current_price >= middle:
                band_position = "above middle"
            elif current_price >= lower:
                band_position = "between bands"

            return TradeSignal(
                symbol=symbol,
                action='HOLD',
                quantity=0,
                price=current_price,
                reason=f"Holding position ({band_position}, P&L: {profit_pct:.1%})",
                confidence=0.5,
                metadata={
                    'middle_band': middle,
                    'bandwidth': bandwidth,
                    'profit_pct': profit_pct
                }
            )

        else:
            # No position - check for entry signals
            if self.check_at_lower_band(current_price, prices):
                # Calculate position size
                quantity = self.calculate_position_size(
                    symbol,
                    current_price,
                    self.position_size_pct
                )

                if quantity > 0:
                    # Calculate stop-loss and profit target
                    # Stop below lower band
                    stop_loss = max(
                        current_price * (1 - self.min_stop_buffer),
                        lower * 0.95  # 5% below lower band
                    )

                    # Target at middle band (or upper band)
                    if self.exit_at_middle:
                        profit_target = middle
                    else:
                        profit_target = upper

                    return TradeSignal(
                        symbol=symbol,
                        action='BUY',
                        quantity=quantity,
                        price=current_price,
                        reason=f"At lower band: ${current_price:.2f} <= ${lower:.2f}",
                        stop_loss=stop_loss,
                        profit_target=profit_target,
                        confidence=0.8,
                        metadata={
                            'middle_band': middle,
                            'upper_band': upper,
                            'lower_band': lower,
                            'bandwidth': bandwidth
                        }
                    )

            # No entry signal
            if current_price > upper:
                reason = f"Above upper band (${current_price:.2f} > ${upper:.2f})"
            elif current_price > middle:
                reason = f"Above middle band (${current_price:.2f} > ${middle:.2f})"
            else:
                reason = f"Between bands (Middle: ${middle:.2f}, Lower: ${lower:.2f})"

            return TradeSignal(
                symbol=symbol,
                action='HOLD',
                quantity=0,
                price=current_price,
                reason=reason,
                confidence=0.3,
                metadata={
                    'middle_band': middle,
                    'upper_band': upper,
                    'lower_band': lower,
                    'bandwidth': bandwidth
                }
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
        if len(historical_data) < self.bb_period + 1:
            return {'error': 'Insufficient data for backtest'}

        closes = historical_data['close'].values

        # Simulate trades
        trades = []
        position = None
        entry_price = 0
        entry_date = None
        entry_lower_band = 0

        for i in range(self.bb_period, len(closes)):
            date = historical_data.iloc[i]['date']
            price = closes[i]
            prices_window = closes[:i + 1]

            middle, upper, lower = self.calculate_bollinger_bands(prices_window)

            if position is None:
                # Check for entry at lower band
                if self.check_at_lower_band(price, prices_window):
                    position = 'LONG'
                    entry_price = price
                    entry_date = date
                    entry_lower_band = lower
            else:
                # Check for exit
                days_held = (date - entry_date).days if hasattr(date - entry_date, 'days') else 0
                profit_pct = (price - entry_price) / entry_price

                exit_signal = False
                exit_reason = ""

                if self.exit_at_middle and self.check_at_middle_band(price, prices_window):
                    exit_signal = True
                    exit_reason = "Middle band"
                elif self.check_at_upper_band(price, prices_window):
                    exit_signal = True
                    exit_reason = "Upper band"
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
                        'entry_lower_band': entry_lower_band,
                        'exit_middle_band': middle,
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
