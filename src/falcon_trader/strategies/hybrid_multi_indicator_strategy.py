"""
Hybrid Multi-Indicator Strategy
Combines multiple signals for robust entries
"""
import backtrader as bt


class HybridMultiIndicatorStrategy(bt.Strategy):
    """
    Multi-indicator hybrid strategy combining:
    - RSI for oversold/overbought
    - MACD for trend confirmation
    - Volume for strength confirmation
    - ATR for dynamic position sizing and stops
    """
    params = (
        ('rsi_period', 14),            # RSI period
        ('rsi_oversold', 45),          # RSI oversold threshold (relaxed)
        ('rsi_overbought', 70),        # RSI overbought threshold
        ('macd_fast', 12),             # MACD fast period
        ('macd_slow', 26),             # MACD slow period
        ('macd_signal', 9),            # MACD signal period
        ('volume_period', 20),         # Volume MA period
        ('volume_threshold', 1.1),     # Volume threshold (110% of avg, relaxed)
        ('atr_period', 14),            # ATR period
        ('atr_stop_mult', 2.0),        # ATR multiplier for stop
        ('max_position_size', 0.90),   # Max position size
        ('profit_target', 0.08),       # 8% profit target (more realistic)
        ('min_signals', 2),            # Minimum signals required (2 out of 3)
    )

    def __init__(self):
        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # MACD
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )

        # Volume
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.volume_period)

        # ATR for stops and position sizing
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)

        # Track state
        self.order = None
        self.buy_price = None
        self.stop_price = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Multi-indicator entry signal:
            # 1. RSI shows oversold
            # 2. MACD is positive (above signal line)
            # 3. Volume confirms with surge
            rsi_oversold = self.rsi[0] < self.params.rsi_oversold
            macd_bullish = self.macd.macd[0] > self.macd.signal[0]
            volume_surge = self.data.volume[0] > (self.volume_sma[0] * self.params.volume_threshold)

            # Count how many signals are active
            signal_count = sum([rsi_oversold, macd_bullish, volume_surge])

            # Require at least min_signals (default 2 out of 3)
            if signal_count >= self.params.min_signals:
                # Dynamic position sizing based on ATR (volatility)
                # Lower volatility = larger position, higher volatility = smaller position
                atr_pct = self.atr[0] / self.data.close[0]
                position_mult = min(1.0, 0.02 / max(atr_pct, 0.01))  # Cap at 100%

                cash = self.broker.getcash()
                size = int((cash * self.params.max_position_size * position_mult) / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Exit conditions (multi-factor):
            # 1. RSI becomes overbought
            # 2. MACD crosses negative
            # 3. Hit profit target
            # 4. ATR-based stop loss
            current_profit = (self.data.close[0] - self.buy_price) / self.buy_price if self.buy_price else 0

            rsi_overbought = self.rsi[0] > self.params.rsi_overbought
            macd_bearish = self.macd.macd[0] < self.macd.signal[0]
            hit_target = current_profit >= self.params.profit_target
            hit_stop = self.data.close[0] < self.stop_price if self.stop_price else False

            if rsi_overbought or macd_bearish or hit_target or hit_stop:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                # Set ATR-based stop loss
                self.stop_price = self.buy_price - (self.atr[0] * self.params.atr_stop_mult)
            elif order.issell():
                self.stop_price = None
        self.order = None
