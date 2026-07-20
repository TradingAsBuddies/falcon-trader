"""
Roster Strategy Engine

Adapter that lets a strategy promoted in ``strategy_roster`` (a falcon_core
``BaseStrategy`` loaded from ``strategy_code``) drive the live paper-trading
executor. It subclasses ``BaseStrategyEngine`` so all execution, position
tracking, cash accounting, and DB writes are inherited unchanged — only signal
generation is replaced.

This is what makes "submit a strategy" mean it actually trades: promoting a
strategy to ``status='paper_trading'`` in the roster causes the executor to run
its ``generate_signals()`` on live bars and act on the result, instead of only
the three hardcoded engines. See falcon-core#8.

Signal mapping (the paper broker is long-only):
    LONG        -> BUY   (open a long if flat)
    EXIT_LONG   -> SELL  (close the long)
    SHORT       -> SELL  (flatten a long if held; no short is opened)
    EXIT_SHORT  -> HOLD  (no short positions exist)
    HOLD/none   -> HOLD

Only a signal stamped at the most recent bar is acted on, so historical signals
from the lookback window never fire as live orders.
"""
from typing import List, Optional

import pandas as pd

from falcon_core.backtesting.strategies.base import Signal, SignalType
from falcon_trader.orchestrator.engines.base_engine import BaseStrategyEngine, TradeSignal


class RosterStrategyEngine(BaseStrategyEngine):
    """Wrap a loaded ``BaseStrategy`` instance as an executor engine."""

    def __init__(self, config, db_manager, strategy, strategy_name, interval="5m"):
        super().__init__(config, db_manager)
        self.strategy = strategy
        # Override the class-name-derived default so orders/positions are tagged
        # with the roster strategy_name (also how monitor_positions finds us).
        self.strategy_name = strategy_name
        self.interval = interval

    # --------------------------------------------------------------- data prep
    def _build_dataframe(self, market_data: dict) -> pd.DataFrame:
        """Build an OHLCV DataFrame with a DatetimeIndex from a market_data dict."""
        prices = list(market_data.get("prices") or [])
        n = len(prices)
        opens = list(market_data.get("opens") or prices)[:n]
        highs = list(market_data.get("highs") or prices)[:n]
        lows = list(market_data.get("lows") or prices)[:n]
        volumes = list(market_data.get("volumes") or [0] * n)[:n]
        timestamps = list(market_data.get("timestamps") or [])

        # Pad any short OHLC lists with close so column lengths always match.
        while len(opens) < n:
            opens.append(prices[len(opens)])
        while len(highs) < n:
            highs.append(prices[len(highs)])
        while len(lows) < n:
            lows.append(prices[len(lows)])
        while len(volumes) < n:
            volumes.append(0)

        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes}
        )

        if timestamps and len(timestamps) == n:
            idx = pd.to_datetime(pd.Series(timestamps), unit="ms", utc=True)
            df.index = pd.DatetimeIndex(idx)
        else:
            # No bar timestamps (e.g. daily flat files): synthesize an ascending
            # index ending now so preprocess_data has a DatetimeIndex to work with.
            df.index = pd.date_range(
                end=pd.Timestamp.utcnow().floor("min"), periods=n, freq="min"
            )
        return df

    # ------------------------------------------------------------ signal logic
    def _latest_actionable(self, signals: List[Signal], last_ts) -> Optional[Signal]:
        """Return the last signal stamped at the final bar, else None.

        Only acting on a signal whose timestamp matches the most recent bar keeps
        historical entries/exits from the lookback window from firing as live
        orders.
        """
        actionable = {
            SignalType.LONG,
            SignalType.SHORT,
            SignalType.EXIT_LONG,
            SignalType.EXIT_SHORT,
        }
        for sig in reversed(signals):
            if sig.signal_type not in actionable:
                continue
            ts = getattr(sig, "timestamp", None)
            if ts is None:
                continue
            try:
                ts = pd.Timestamp(ts)
                if ts.tzinfo is None and last_ts.tzinfo is not None:
                    ts = ts.tz_localize("UTC")
            except Exception:
                continue
            if ts == last_ts:
                return sig
        return None

    def _to_trade_signal(self, symbol: str, sig: Signal, current_price: float) -> TradeSignal:
        price = float(sig.price) if sig.price else current_price
        params = self.strategy.params

        if sig.signal_type == SignalType.LONG:
            # Position sizing: signal's dollar size (if any), capped by cash * max fraction.
            size_dollars = float(sig.position_size) if sig.position_size else float(params.position_size)
            cash = self.get_account_balance()
            max_frac = self.config.get("risk_management", {}).get("max_position_size", 0.95)
            budget = min(size_dollars, cash * max_frac)
            quantity = int(budget // price) if price > 0 else 0
            if quantity <= 0:
                return TradeSignal(symbol, "HOLD", 0, price,
                                   reason=f"{self.strategy_name}: LONG but position size 0 (cash ${cash:.2f})")
            stop_loss = float(sig.stop_loss) if sig.stop_loss else price * (1 - params.default_stop_loss_pct)
            profit_target = float(sig.take_profit) if sig.take_profit else price * (1 + params.default_take_profit_pct)
            return TradeSignal(
                symbol=symbol, action="BUY", quantity=quantity, price=price,
                stop_loss=stop_loss, profit_target=profit_target,
                confidence=sig.confidence,
                reason=sig.reason or f"{self.strategy_name} LONG",
            )

        if sig.signal_type in (SignalType.EXIT_LONG, SignalType.SHORT):
            # Close an existing long; no short is opened (long-only paper book).
            position = self.get_position(symbol)
            if not position or position.quantity <= 0:
                return TradeSignal(symbol, "HOLD", 0, price,
                                   reason=f"{self.strategy_name}: exit/short but no long position")
            note = "EXIT_LONG" if sig.signal_type == SignalType.EXIT_LONG else "SHORT (flatten long)"
            return TradeSignal(
                symbol=symbol, action="SELL", quantity=position.quantity, price=price,
                confidence=sig.confidence,
                reason=sig.reason or f"{self.strategy_name} {note}",
            )

        # EXIT_SHORT / HOLD / anything else
        return TradeSignal(symbol, "HOLD", 0, price, reason=sig.reason or f"{self.strategy_name} HOLD")

    def generate_signal(self, symbol: str, market_data: dict) -> TradeSignal:
        """Run the wrapped roster strategy on live bars and map its output."""
        prices = market_data.get("prices") or []
        current_price = market_data.get("price") or (prices[-1] if prices else 0.0)

        if len(prices) < 20:
            return TradeSignal(symbol, "HOLD", 0, current_price,
                               reason=f"{self.strategy_name}: insufficient bars ({len(prices)})")

        try:
            df = self._build_dataframe(market_data)
            last_ts = df.index[-1]
            signals = self.strategy.run(df, symbol)
        except Exception as e:
            return TradeSignal(symbol, "HOLD", 0, current_price,
                               reason=f"{self.strategy_name}: run error: {e}")

        sig = self._latest_actionable(signals, last_ts)
        if sig is None:
            return TradeSignal(symbol, "HOLD", 0, current_price,
                               reason=f"{self.strategy_name}: no fresh signal at latest bar")
        return self._to_trade_signal(symbol, sig, current_price)
