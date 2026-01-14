"""
Moving Average Crossover Strategy
Classic dual moving average crossover system
"""
import backtrader as bt


class MovingAverageCrossoverStrategy(bt.Strategy):
    """
    Dual moving average crossover strategy
    - Buys when fast MA crosses above slow MA
    - Sells when fast MA crosses below slow MA
    - Includes trend filter using longer MA
    """
    params = (
        ('fast_period', 10),           # Fast MA period
        ('slow_period', 30),           # Slow MA period
        ('trend_period', 100),         # Long-term trend filter
        ('position_size', 0.95),       # Use 95% of capital
        ('use_trend_filter', True),    # Only trade in direction of trend
    )

    def __init__(self):
        # Moving averages
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.trend_ma = bt.indicators.SMA(self.data.close, period=self.params.trend_period)

        # Crossover signal
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

        # Track order
        self.order = None
        self.buy_price = None

    def next(self):
        if self.order:
            return

        # Determine trend direction
        uptrend = self.data.close[0] > self.trend_ma[0] if self.params.use_trend_filter else True

        if not self.position:
            # Buy signal: fast MA crosses above slow MA (and in uptrend)
            if self.crossover[0] > 0 and uptrend:
                cash = self.broker.getcash()
                size = int((cash * self.params.position_size) / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Sell signal: fast MA crosses below slow MA
            if self.crossover[0] < 0:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
        self.order = None
