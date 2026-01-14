#!/usr/bin/env python3
"""
One Candle Rule Strategy by Scarface Trades (Chart Fanatics)

Strategy Overview:
A price action-based strategy that focuses on trading breakouts and retests
of key support/resistance levels using single candle confirmation.

Entry Rules:
1. Identify clear support/resistance level (swing highs/lows)
2. Wait for price to break above resistance (or below support for shorts)
3. Wait for price to retest the broken level
4. Look for candlestick confirmation (hammer/bullish engulfing)
5. Enter on confirmation candle that holds above key level

Exit Rules:
- Take profit at 1:2 risk-reward ratio
- Stop loss below the retest support level

Performance (from creator):
- Win Rate: 60-80%
- Risk-Reward: 1:2 average
- Total Profit: $3M to date

YouTube: https://www.youtube.com/watch?v=ZwV-xkXoeuA
"""

import backtrader as bt
from datetime import time


class OneCandleStrategy(bt.Strategy):
    """
    One Candle Rule - Breakout/Retest with Confirmation

    Quantitative Implementation:
    - Uses swing highs/lows to identify support/resistance
    - Detects breakouts above resistance
    - Waits for retest (pullback to broken level)
    - Confirms with bullish candle patterns
    - Enters with 1:2 risk-reward and stop loss
    """

    params = (
        # Strategy parameters
        ('lookback_period', 20),           # Period to identify S/R levels
        ('breakout_threshold', 0.001),     # 0.1% above high to confirm breakout
        ('retest_tolerance', 0.003),       # 0.3% range for retest zone
        ('risk_reward_ratio', 2.0),        # Target profit / risk (1:2 = 2.0)
        ('position_size_pct', 0.20),       # 20% of portfolio per trade

        # Confirmation patterns
        ('require_confirmation', True),    # Require bullish candle pattern
        ('min_body_size', 0.003),         # Minimum 0.3% candle body

        # Time filters (9:30 AM - 11:00 AM trading window)
        ('trade_start_hour', 9),
        ('trade_start_minute', 30),
        ('trade_end_hour', 11),
        ('trade_end_minute', 0),

        # Risk management
        ('max_positions', 1),              # One position at a time
        ('stop_loss_pct', 0.02),          # 2% stop loss from entry

        # Debug mode
        ('debug', False),
    )

    def __init__(self):
        """Initialize strategy indicators and state"""
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

        # Track swing highs/lows for S/R levels
        self.swing_highs = []
        self.swing_lows = []

        # Track breakout state
        self.breakout_level = None
        self.breakout_bar = None
        self.waiting_for_retest = False

        # Performance tracking
        self.trades = []
        self.entry_bar = None

    def log(self, txt, dt=None):
        """Logging function"""
        if self.params.debug:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')

    def is_trading_hours(self):
        """Check if current time is within trading window (9:30-11 AM)"""
        if len(self.data) == 0:
            return False

        current_time = self.data.datetime.time(0)
        if current_time is None:
            return True  # If no intraday time data, trade all day

        start_time = time(self.params.trade_start_hour, self.params.trade_start_minute)
        end_time = time(self.params.trade_end_hour, self.params.trade_end_minute)

        return start_time <= current_time <= end_time

    def identify_swing_levels(self):
        """Identify swing highs and lows for S/R levels"""
        if len(self.data) < self.params.lookback_period:
            return

        # Get recent highs and lows
        highs = [self.data.high[-i] for i in range(self.params.lookback_period)]
        lows = [self.data.low[-i] for i in range(self.params.lookback_period)]

        # Find local maxima (swing highs)
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                self.swing_highs.append(highs[i])

        # Find local minima (swing lows)
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                self.swing_lows.append(lows[i])

        # Keep only recent levels (last 10)
        self.swing_highs = self.swing_highs[-10:]
        self.swing_lows = self.swing_lows[-10:]

    def detect_breakout(self):
        """Detect breakout above resistance level"""
        if not self.swing_highs or self.waiting_for_retest:
            return False

        # Get most recent resistance level (highest swing high)
        resistance = max(self.swing_highs)
        current_close = self.data.close[0]

        # Check if price closed above resistance with threshold
        breakout_price = resistance * (1 + self.params.breakout_threshold)

        if current_close > breakout_price:
            self.breakout_level = resistance
            self.breakout_bar = len(self.data)
            self.waiting_for_retest = True
            self.log(f'BREAKOUT detected at {current_close:.2f}, resistance {resistance:.2f}')
            return True

        return False

    def detect_retest(self):
        """Detect retest of broken resistance (now support)"""
        if not self.waiting_for_retest or not self.breakout_level:
            return False

        current_low = self.data.low[0]
        current_close = self.data.close[0]

        # Define retest zone (resistance level +/- tolerance)
        retest_upper = self.breakout_level * (1 + self.params.retest_tolerance)
        retest_lower = self.breakout_level * (1 - self.params.retest_tolerance)

        # Check if price touched the retest zone
        if retest_lower <= current_low <= retest_upper:
            self.log(f'RETEST detected at {current_low:.2f}, level {self.breakout_level:.2f}')
            return True

        return False

    def is_bullish_confirmation(self):
        """
        Check for bullish confirmation candle patterns:
        1. Hammer candle (long lower wick, small body near top)
        2. Bullish engulfing
        3. Strong close above support
        """
        if not self.params.require_confirmation:
            return True

        open_price = self.data.open[0]
        high_price = self.data.high[0]
        low_price = self.data.low[0]
        close_price = self.data.close[0]

        # Calculate candle properties
        body = abs(close_price - open_price)
        total_range = high_price - low_price

        if total_range == 0:
            return False

        # Check for bullish close
        is_bullish = close_price > open_price

        # Check body size (must be significant)
        body_ratio = body / close_price
        if body_ratio < self.params.min_body_size:
            return False

        # Pattern 1: Hammer (long lower wick, close near high)
        lower_wick = min(open_price, close_price) - low_price
        upper_wick = high_price - max(open_price, close_price)

        is_hammer = (lower_wick > 2 * body) and (upper_wick < body) and is_bullish

        # Pattern 2: Bullish engulfing (compare to previous candle)
        if len(self.data) > 1:
            prev_open = self.data.open[-1]
            prev_close = self.data.close[-1]
            prev_was_bearish = prev_close < prev_open

            engulfs_previous = (close_price > prev_open and
                              open_price < prev_close and
                              is_bullish and prev_was_bearish)
        else:
            engulfs_previous = False

        # Pattern 3: Strong bullish close (close in top 25% of range)
        close_position = (close_price - low_price) / total_range if total_range > 0 else 0
        strong_close = close_position > 0.75 and is_bullish

        confirmation = is_hammer or engulfs_previous or strong_close

        if confirmation:
            self.log(f'CONFIRMATION: Hammer={is_hammer}, Engulfing={engulfs_previous}, StrongClose={strong_close}')

        return confirmation

    def calculate_position_size(self):
        """Calculate position size based on portfolio percentage"""
        portfolio_value = self.broker.getvalue()
        position_value = portfolio_value * self.params.position_size_pct
        shares = int(position_value / self.data.close[0])
        return max(shares, 1)  # At least 1 share

    def next(self):
        """Execute strategy logic on each bar"""
        # Skip if we have a pending order
        if self.order:
            return

        # Skip if outside trading hours
        if not self.is_trading_hours():
            return

        # Update swing levels
        self.identify_swing_levels()

        # If no position, look for entry signal
        if not self.position:
            # Step 1: Detect breakout
            self.detect_breakout()

            # Step 2: Detect retest
            if self.detect_retest():
                # Step 3: Check for confirmation
                if self.is_bullish_confirmation():
                    # Step 4: Enter trade
                    size = self.calculate_position_size()

                    # Calculate stop loss and target
                    entry_price = self.data.close[0]
                    stop_distance = entry_price * self.params.stop_loss_pct
                    stop_price = entry_price - stop_distance
                    target_price = entry_price + (stop_distance * self.params.risk_reward_ratio)

                    # Place buy order
                    self.entry_price = entry_price
                    self.stop_price = stop_price
                    self.target_price = target_price
                    self.entry_bar = len(self.data)

                    self.order = self.buy(size=size)

                    self.log(f'BUY CREATE, Size: {size}, Price: {entry_price:.2f}, '
                           f'Stop: {stop_price:.2f}, Target: {target_price:.2f}')

                    # Reset breakout tracking
                    self.waiting_for_retest = False

        # If we have a position, manage exits
        else:
            current_price = self.data.close[0]

            # Exit 1: Stop loss hit
            if current_price <= self.stop_price:
                self.order = self.close()
                self.log(f'STOP LOSS triggered at {current_price:.2f}')

            # Exit 2: Target hit
            elif current_price >= self.target_price:
                self.order = self.close()
                self.log(f'TARGET HIT at {current_price:.2f}')

            # Exit 3: End of trading window
            elif not self.is_trading_hours():
                self.order = self.close()
                self.log(f'CLOSE at end of trading window at {current_price:.2f}')

    def notify_order(self, order):
        """Receive order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                       f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                       f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

                # Calculate trade result
                if self.entry_price:
                    pnl = (order.executed.price - self.entry_price) * order.executed.size
                    pnl_pct = ((order.executed.price - self.entry_price) / self.entry_price) * 100

                    self.trades.append({
                        'entry': self.entry_price,
                        'exit': order.executed.price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'bars_held': len(self.data) - self.entry_bar if self.entry_bar else 0
                    })

                    self.log(f'TRADE RESULT: P&L ${pnl:.2f} ({pnl_pct:.2f}%)')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        """Receive trade notifications"""
        if not trade.isclosed:
            return

        self.log(f'TRADE PROFIT, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

    def stop(self):
        """Called when backtest ends"""
        if self.params.debug:
            print('\n' + '='*80)
            print('STRATEGY PERFORMANCE SUMMARY')
            print('='*80)
            print(f'Final Portfolio Value: ${self.broker.getvalue():,.2f}')
            print(f'Total Trades: {len(self.trades)}')

            if self.trades:
                winning_trades = [t for t in self.trades if t['pnl'] > 0]
                losing_trades = [t for t in self.trades if t['pnl'] <= 0]

                win_rate = (len(winning_trades) / len(self.trades)) * 100 if self.trades else 0
                avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
                avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

                print(f'Winning Trades: {len(winning_trades)}')
                print(f'Losing Trades: {len(losing_trades)}')
                print(f'Win Rate: {win_rate:.1f}%')
                print(f'Avg Win: ${avg_win:.2f}')
                print(f'Avg Loss: ${avg_loss:.2f}')

                if avg_loss != 0:
                    profit_factor = abs(avg_win / avg_loss) if avg_loss else 0
                    print(f'Profit Factor: {profit_factor:.2f}')

            print('='*80)


if __name__ == '__main__':
    print("One Candle Rule Strategy")
    print("="*50)
    print("This strategy should be run via strategy_manager.py or massive_flat_files.py")
    print("\nExample usage:")
    print("  python3 strategy_manager.py backtest -f strategies/one_candle_strategy.py")
    print("  python3 massive_flat_files.py --backtest SPY --strategy one_candle")
