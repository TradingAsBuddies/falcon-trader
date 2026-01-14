"""
Momentum Breakout Strategy
Buys on strong upward momentum with volume confirmation
"""
import backtrader as bt


class MomentumBreakoutStrategy(bt.Strategy):
    """
    Momentum strategy based on price breakouts and volume
    - Enters when price breaks above 20-day high with volume surge
    - Exits on 10-day low or profit target
    """
    params = (
        ('breakout_period', 20),      # Period for high/low breakout
        ('volume_factor', 1.5),        # Volume must be 1.5x average
        ('exit_period', 10),           # Exit on 10-day low
        ('profit_target', 0.15),       # 15% profit target
        ('stop_loss', 0.08),           # 8% stop loss
        ('position_size', 0.95),       # Use 95% of capital
    )

    def __init__(self):
        # Price indicators
        self.highest = bt.indicators.Highest(self.data.close, period=self.params.breakout_period)
        self.lowest = bt.indicators.Lowest(self.data.close, period=self.params.exit_period)

        # Volume indicator
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=20)

        # Track order and entry
        self.order = None
        self.buy_price = None
        self.buy_bar = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Entry conditions:
            # 1. Price breaks above 20-day high
            # 2. Volume is 1.5x the 20-day average
            price_breakout = self.data.close[0] > self.highest[-1]
            volume_surge = self.data.volume[0] > (self.volume_sma[0] * self.params.volume_factor)

            if price_breakout and volume_surge:
                cash = self.broker.getcash()
                size = int((cash * self.params.position_size) / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Exit conditions:
            # 1. Price drops below 10-day low
            # 2. Hit profit target
            # 3. Hit stop loss
            current_profit = (self.data.close[0] - self.buy_price) / self.buy_price if self.buy_price else 0

            exit_low = self.data.close[0] < self.lowest[-1]
            hit_target = current_profit >= self.params.profit_target
            hit_stop = current_profit <= -self.params.stop_loss

            if exit_low or hit_target or hit_stop:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_bar = len(self)
        self.order = None
