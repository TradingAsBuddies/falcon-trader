"""
MACD Momentum Strategy
Uses MACD indicator for trend-following entries
"""
import backtrader as bt


class MACDMomentumStrategy(bt.Strategy):
    """
    MACD-based momentum strategy
    - Enters on MACD crossover with histogram confirmation
    - Exits on opposite crossover or trailing stop
    """
    params = (
        ('macd_fast', 12),             # Fast EMA period
        ('macd_slow', 26),             # Slow EMA period
        ('macd_signal', 9),            # Signal line period
        ('atr_period', 14),            # ATR for stop loss
        ('atr_multiplier', 2.0),       # ATR multiplier for trailing stop
        ('position_size', 0.95),       # Use 95% of capital
        ('min_histogram', 0.0),        # Min histogram value for entry
    )

    def __init__(self):
        # MACD indicator
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )

        # Calculate histogram manually (MACD line - Signal line)
        self.macd_histogram = self.macd.macd - self.macd.signal

        # ATR for trailing stop
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)

        # Track order and highest price
        self.order = None
        self.buy_price = None
        self.highest_price = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Entry: MACD line crosses above signal line with positive histogram
            macd_crossover = self.macd.macd[0] > self.macd.signal[0] and \
                           self.macd.macd[-1] <= self.macd.signal[-1]
            histogram_positive = self.macd_histogram[0] > self.params.min_histogram

            if macd_crossover and histogram_positive:
                cash = self.broker.getcash()
                size = int((cash * self.params.position_size) / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Update highest price for trailing stop
            if self.highest_price is None or self.data.close[0] > self.highest_price:
                self.highest_price = self.data.close[0]

            # Exit conditions:
            # 1. MACD crosses below signal (trend reversal)
            # 2. Trailing stop based on ATR
            macd_crossunder = self.macd.macd[0] < self.macd.signal[0] and \
                            self.macd.macd[-1] >= self.macd.signal[-1]

            trailing_stop = self.highest_price - (self.atr[0] * self.params.atr_multiplier)
            hit_trailing_stop = self.data.close[0] < trailing_stop

            if macd_crossunder or hit_trailing_stop:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.highest_price = self.buy_price
            elif order.issell():
                self.highest_price = None
        self.order = None
