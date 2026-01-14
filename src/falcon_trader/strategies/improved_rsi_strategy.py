"""
Improved RSI Mean Reversion Strategy
Optimized exit conditions for better profit capture
"""
import backtrader as bt

class ImprovedRSIStrategy(bt.Strategy):
    """
    Improved RSI mean reversion strategy with optimized exit conditions

    Key Improvements:
    - Wider RSI exit band (65 vs 55) for longer holds
    - Higher profit target (8% vs 2.5%) aligned with AI screener targets
    - Longer max hold period (20 vs 12 days)
    - Better risk management
    """
    params = (
        ('rsi_period', 14),
        ('rsi_buy', 45),        # Entry when oversold
        ('rsi_sell', 65),       # Exit when overbought (was 55)
        ('hold_days', 20),      # Max hold period (was 12)
        ('profit_target', 0.08),  # 8% target (was 2.5%)
        ('position_size', 0.92),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None
        self.buy_price = None
        self.buy_bar = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Entry: RSI oversold condition
            if self.rsi[0] < self.params.rsi_buy:
                cash = self.broker.getcash()
                size = int((cash * self.params.position_size) / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Calculate position metrics
            bars_held = len(self) - self.buy_bar if self.buy_bar else 0
            profit_pct = (self.data.close[0] - self.buy_price) / self.buy_price if self.buy_price else 0

            # Exit conditions (ANY triggers sell)
            if (self.rsi[0] > self.params.rsi_sell or         # RSI overbought
                bars_held >= self.params.hold_days or         # Max hold reached
                profit_pct >= self.params.profit_target):     # Profit target hit
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_bar = len(self)
        self.order = None
