"""
Balanced RSI Mean Reversion Strategy
Optimized for better profit capture while maintaining quick exits
"""
import backtrader as bt

class BalancedRSIStrategy(bt.Strategy):
    """
    Balanced RSI mean reversion strategy

    Key Improvements:
    - Moderate RSI exit (60 vs 55) - compromise between quick exits and profit capture
    - Higher profit target (5% vs 2.5%) - more realistic targets
    - Moderate max hold period (15 vs 12 days)
    - Better balance of returns and consistency
    """
    params = (
        ('rsi_period', 14),
        ('rsi_buy', 45),        # Entry when oversold
        ('rsi_sell', 60),       # Exit when overbought (was 55, compromise at 60)
        ('hold_days', 15),      # Max hold period (was 12)
        ('profit_target', 0.05),  # 5% target (was 2.5%, compromise at 5%)
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
            if (self.rsi[0] > self.params.rsi_sell or         # RSI overbought (60)
                bars_held >= self.params.hold_days or         # Max hold reached (15)
                profit_pct >= self.params.profit_target):     # Profit target hit (5%)
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_bar = len(self)
        self.order = None
