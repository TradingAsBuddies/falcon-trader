"""
Bollinger Band Mean Reversion Strategy
Buys oversold conditions, sells overbought
"""
import backtrader as bt


class BollingerMeanReversionStrategy(bt.Strategy):
    """
    Mean reversion using Bollinger Bands
    - Buys when price touches lower band (oversold)
    - Sells when price reaches middle band or upper band
    - Includes volume confirmation
    """
    params = (
        ('bb_period', 20),             # Bollinger Band period
        ('bb_dev', 2.0),               # Standard deviations
        ('volume_period', 20),         # Volume MA period
        ('volume_threshold', 0.8),     # Min volume (80% of average)
        ('profit_target', 0.05),       # 5% profit target
        ('stop_loss', 0.03),           # 3% stop loss
        ('position_size', 0.90),       # Use 90% of capital
        ('max_hold_days', 15),         # Max holding period
    )

    def __init__(self):
        # Bollinger Bands
        self.boll = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_dev
        )

        # Volume filter
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.volume_period)

        # Track order and entry
        self.order = None
        self.buy_price = None
        self.buy_bar = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Entry: Price touches or breaks below lower band with decent volume
            price_oversold = self.data.close[0] <= self.boll.lines.bot[0]
            volume_ok = self.data.volume[0] >= (self.volume_sma[0] * self.params.volume_threshold)

            if price_oversold and volume_ok:
                cash = self.broker.getcash()
                size = int((cash * self.params.position_size) / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Exit conditions
            bars_held = len(self) - self.buy_bar if self.buy_bar else 0
            current_profit = (self.data.close[0] - self.buy_price) / self.buy_price if self.buy_price else 0

            # Exit on: middle band, upper band, profit target, stop loss, or max hold
            at_middle = self.data.close[0] >= self.boll.lines.mid[0]
            at_upper = self.data.close[0] >= self.boll.lines.top[0]
            hit_target = current_profit >= self.params.profit_target
            hit_stop = current_profit <= -self.params.stop_loss
            max_hold = bars_held >= self.params.max_hold_days

            if at_middle or at_upper or hit_target or hit_stop or max_hold:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_bar = len(self)
        self.order = None
