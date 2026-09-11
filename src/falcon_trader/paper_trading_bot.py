#!/usr/bin/env python3
"""
Falcon Paper Trading Bot
Real-time paper trading with Polygon.io market data
"""

import os
import time
import requests
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from falcon_core import DatabaseManager, FalconConfig

from falcon_trader import portfolio, trading_guards
from falcon_trader.risk_limits import KillSwitch
from falcon_trader.symbol_state import load_symbol_state

try:
    falcon_config = FalconConfig()
except Exception:
    falcon_config = None


class PaperTradingBot:
    """
    Paper trading bot with real-time market data from Polygon.io
    Integrates with FHS-compliant database using DatabaseManager
    """

    def __init__(self, symbols: List[str], massive_api_key: str,
                 claude_api_key: str = None, initial_balance: float = 10000.0,
                 update_interval: int = 60):
        """
        Initialize paper trading bot

        Args:
            symbols: List of stock symbols to track
            massive_api_key: Polygon.io API key
            claude_api_key: Claude API key (optional, for AI analysis)
            initial_balance: Starting cash balance
            update_interval: Seconds between market data updates
        """
        self.symbols = symbols
        self.massive_api_key = massive_api_key
        self.claude_api_key = claude_api_key
        self.update_interval = update_interval
        self.running = False
        self.thread = None

        # Initialize database manager
        if falcon_config:
            db_config = falcon_config.get_db_config()
        else:
            db_config = {'db_type': 'sqlite', 'db_path': 'paper_trading.db'}

        self.db = DatabaseManager(db_config)

        # Churn limits and the kill switch. Both are consulted in place_order,
        # which is the single chokepoint every entry path goes through.
        self.cooldown_policy = trading_guards.CooldownPolicy.from_config(
            (falcon_config.get('cooldown') if falcon_config
             and hasattr(falcon_config, 'get') else None)
        )
        self.kill_switch = KillSwitch()

        # Initialize account if needed
        self._initialize_account(initial_balance)

        # Current market data cache
        self.market_data = {}

        print(f"[BOT] Initialized with symbols: {symbols}")
        print(f"[BOT] Database: {db_config.get('db_type')} at {db_config.get('db_path', 'N/A')}")

    def _initialize_account(self, initial_balance: float):
        """Initialize account if it doesn't exist"""
        try:
            account = self.db.execute("SELECT * FROM account LIMIT 1", fetch='one')
            if not account:
                from datetime import datetime
                self.db.execute(
                    "INSERT INTO account (cash, last_updated) VALUES (%s, %s)",
                    (initial_balance, datetime.now().isoformat())
                )
                print(f"[BOT] Account initialized with ${initial_balance:,.2f}")
        except Exception as e:
            print(f"[BOT] Error initializing account: {e}")

    def get_account(self) -> Dict:
        """Get current account information"""
        account = self.db.execute("SELECT * FROM account LIMIT 1", fetch='one')
        if account:
            cash = float(account['cash'])

            # Calculate total value (cash + positions)
            positions = self.get_positions()
            positions_value = 0.0
            for pos in positions:
                quote = self.get_quote(pos['symbol'])
                if quote:
                    positions_value += float(quote['price']) * float(pos['quantity'])

            total_value = cash + positions_value

            return {
                'cash': cash,
                'total_value': total_value,
                'updated_at': account['last_updated'] if 'last_updated' in account.keys() else datetime.now().isoformat()
            }
        return {'cash': 0.0, 'total_value': 0.0}

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        positions = self.db.execute(
            "SELECT * FROM positions WHERE quantity > 0",
            fetch='all'
        )
        result = []
        if positions:
            for pos in positions:
                result.append({
                    'symbol': pos['symbol'],
                    'quantity': pos['quantity'],
                    'average_price': pos['entry_price'],  # Map entry_price to average_price
                    'entry_date': pos['entry_date'] if 'entry_date' in pos.keys() else '',
                    'last_updated': pos['last_updated'] if 'last_updated' in pos.keys() else ''
                })
        return result

    def get_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades"""
        trades = self.db.execute(
            "SELECT * FROM orders ORDER BY timestamp DESC LIMIT %s",
            (limit,),
            fetch='all'
        )
        return [dict(trade) for trade in trades] if trades else []

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get a current quote, preferring the latest minute bar.

        The previous-close endpoint (``/v2/aggs/ticker/{sym}/prev``) returns
        *yesterday's* daily bar. Pricing fills from it is what let orders
        "fill" at 16:50 at the prior session's close (falcon-trader#23) -- and
        once the stale-bar guard was added, it made every fill fail instead,
        because a previous-session bar can never be from the current session.

        Order of preference:

        1. ``/v2/snapshot`` -- last trade, the real current price.
        2. ``/v2/aggs/.../range/1/minute`` -- the most recent minute bar.
        3. ``/prev`` -- previous close, returned with ``stale=True`` so the
           caller can price marks from it but refuse to fill on it.

        A DELAYED-tier key is accepted throughout: it returns ``status``
        ``'DELAYED'`` with good results, and gating on ``'OK'`` alone made this
        return ``None`` for every symbol (falcon-core fixed the same bug in
        ``c491253``).
        """
        for fetch in (self._quote_from_snapshot,
                      self._quote_from_minute_agg,
                      self._quote_from_prev_close):
            try:
                quote = fetch(symbol)
            except Exception as e:
                print(f"[BOT] {fetch.__name__} failed for {symbol}: {e}")
                continue
            if quote and quote.get('price'):
                return quote
        return None

    def _get_json(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """GET returning parsed JSON when the tier's status is usable."""
        params = dict(params or {})
        params['apiKey'] = self.massive_api_key
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"[BOT] Quote error {response.status_code} for {url.rsplit('/', 1)[-1]}")
            return None
        data = response.json()
        if data.get('status') in ('OK', 'DELAYED') or 'ticker' in data:
            return data
        return None

    def _quote_from_snapshot(self, symbol: str) -> Optional[Dict]:
        """Last trade from the snapshot endpoint -- the true current price."""
        data = self._get_json(
            "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/"
            f"tickers/{symbol}"
        )
        ticker = (data or {}).get('ticker') or {}
        last = ticker.get('lastTrade') or {}
        price = last.get('p')
        if not price:
            return None
        # lastTrade timestamps are nanoseconds.
        ts_ns = last.get('t') or 0
        return {
            'symbol': symbol,
            'price': price,
            'open': (ticker.get('day') or {}).get('o', 0),
            'high': (ticker.get('day') or {}).get('h', 0),
            'low': (ticker.get('day') or {}).get('l', 0),
            'volume': (ticker.get('day') or {}).get('v', 0),
            'timestamp': int(ts_ns / 1_000_000) if ts_ns else 0,
            'source': 'snapshot',
            'stale': False,
        }

    def _quote_from_minute_agg(self, symbol: str) -> Optional[Dict]:
        """Most recent minute bar from the last two calendar days."""
        today = datetime.now(trading_guards.EASTERN).date()
        start = today - timedelta(days=4)
        data = self._get_json(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
            f"{start.isoformat()}/{today.isoformat()}",
            {'adjusted': 'true', 'sort': 'desc', 'limit': 1},
        )
        results = (data or {}).get('results') or []
        if not results:
            return None
        bar = results[0]
        return {
            'symbol': symbol,
            'price': bar.get('c', 0),
            'open': bar.get('o', 0),
            'high': bar.get('h', 0),
            'low': bar.get('l', 0),
            'volume': bar.get('v', 0),
            'timestamp': bar.get('t', 0),
            'source': 'minute_agg',
            'stale': False,
        }

    def _quote_from_prev_close(self, symbol: str) -> Optional[Dict]:
        """Previous session's daily bar. Usable as a mark, never as a fill."""
        data = self._get_json(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev",
            {'adjusted': 'true'},
        )
        results = (data or {}).get('results') or []
        if not results:
            return None
        bar = results[0]
        return {
            'symbol': symbol,
            'price': bar.get('c', 0),
            'open': bar.get('o', 0),
            'high': bar.get('h', 0),
            'low': bar.get('l', 0),
            'volume': bar.get('v', 0),
            'timestamp': bar.get('t', 0),
            'source': 'prev_close',
            'stale': True,
        }

    def place_order(self, symbol: str, side: str, quantity: int,
                   order_type: str = 'market', price: float = None) -> Dict:
        """
        Place a paper trade order

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            order_type: 'market' or 'limit'
            price: Limit price (for limit orders)

        Returns:
            Dict with order result
        """
        try:
            # Market-hours gate (falcon-trader#23). Last line of defence: the
            # loops are gated too, but /api/order reaches here directly.
            # FALCON_ALLOW_EXTENDED_HOURS=1 is the deliberate override.
            if os.getenv('FALCON_ALLOW_EXTENDED_HOURS') != '1':
                session = trading_guards.check_market_open()
                if not session.allowed:
                    print(f"[BOT] Rejected {side} {quantity} {symbol}: {session.message}")
                    return {
                        'status': 'error',
                        'reason': session.reason,
                        'message': session.message,
                    }

            # Get current market price
            quote = self.get_quote(symbol)
            if not quote:
                return {'status': 'error', 'message': f'Could not get quote for {symbol}'}

            # Refuse to fill off a stale bar.
            #
            # get_quote prefers the snapshot / latest minute bar and falls back
            # to the previous close, which it marks stale. A stale quote is a
            # usable *mark* but never a fill price -- filling from it is how the
            # 16:50 orders succeeded at the prior session's close
            # (falcon-trader#23).
            if quote.get('stale') and os.getenv('FALCON_ALLOW_STALE_FILLS') != '1':
                msg = (
                    f"Only a stale {quote.get('source', 'unknown')} quote is "
                    f"available for {symbol}; refusing to fill"
                )
                print(f"[BOT] Rejected {side} {quantity} {symbol}: {msg}")
                return {
                    'status': 'error',
                    'reason': 'stale_price',
                    'message': msg,
                    'quoteSource': quote.get('source'),
                }

            bar_ts = quote.get('timestamp')
            if bar_ts and os.getenv('FALCON_ALLOW_STALE_FILLS') != '1':
                bar_dt = datetime.fromtimestamp(
                    bar_ts / 1000 if bar_ts > 1e11 else bar_ts,
                    tz=trading_guards.EASTERN,
                )
                fresh = trading_guards.check_fill_price_freshness(bar_dt)
                if not fresh.allowed:
                    print(f"[BOT] Rejected {side} {quantity} {symbol}: {fresh.message}")
                    return {
                        'status': 'error',
                        'reason': fresh.reason,
                        'message': fresh.message,
                        'barTimestamp': bar_dt.isoformat(),
                    }

            execution_price = price if order_type == 'limit' and price else quote['price']
            total_cost = execution_price * quantity

            # Check account balance for buys
            if side == 'buy':
                account = self.get_account()
                if account['cash'] < total_cost:
                    return {
                        'status': 'error',
                        'message': f'Insufficient funds. Need ${total_cost:,.2f}, have ${account["cash"]:,.2f}'
                    }

            # Churn + risk gates on BUYs. These live here, not only in
            # base_engine.execute_buy, because strategy_executor reaches the
            # book through place_order_with_strategy -> place_order and so had
            # no cooldown at all (falcon-trader#24), and risk_limits was
            # imported by nothing but its own tests (#26).
            if side == 'buy':
                if not self.kill_switch.is_trading_enabled():
                    reason = self.kill_switch.reason() or 'trading halted'
                    print(f"[BOT] Rejected {side} {quantity} {symbol}: {reason}")
                    return {'status': 'error', 'reason': 'trading_halted',
                            'message': reason}

                state = load_symbol_state(self.db, symbol)
                cooled = trading_guards.check_cooldown(
                    symbol, self.cooldown_policy,
                    last_exit_at=state.last_exit_at,
                    loss_streak=state.loss_streak,
                    round_trips_today=state.round_trips_today,
                )
                if not cooled.allowed:
                    print(f"[BOT] Rejected {side} {quantity} {symbol}: {cooled.message}")
                    return {'status': 'error', 'reason': cooled.reason,
                            'message': cooled.message}

            # Validate the sell BEFORE touching anything.
            #
            # This used to happen the other way round: cash was credited for
            # every sell unconditionally, while _update_position silently did
            # nothing when there was no position and deleted the row (still
            # crediting the full notional) when the quantity exceeded holdings.
            # Both minted cash from nothing (falcon-trader#22).
            if side == 'sell':
                held = self.db.execute(
                    "SELECT quantity FROM positions WHERE symbol = %s",
                    (symbol,),
                    fetch='one',
                )
                check = portfolio.validate_sell(
                    held['quantity'] if held else 0, quantity,
                )
                if not check.ok:
                    print(f"[BOT] Rejected sell {quantity} {symbol}: {check.message}")
                    return {
                        'status': 'error',
                        'reason': check.reason,
                        'message': check.message,
                    }

            # Execute trade
            timestamp = datetime.now().isoformat()
            pnl = 0.0

            # Update positions and calculate P&L for sells
            if side == 'buy':
                if not self._update_account_cash(-total_cost):
                    return {
                        'status': 'error',
                        'reason': 'insufficient_funds',
                        'message': (
                            f'Insufficient funds for {quantity} {symbol} '
                            f'at ${execution_price:,.4f}'
                        ),
                    }
                self._update_position(symbol, quantity, execution_price, 'buy')
            else:  # sell
                pnl = self._update_position(symbol, quantity, execution_price, 'sell')
                self._update_account_cash(total_cost)

            # Record order with P&L
            self.db.execute(
                """INSERT INTO orders (symbol, side, quantity, price, timestamp, pnl)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (symbol, side, quantity, execution_price, timestamp, pnl)
            )

            return {
                'status': 'success',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': execution_price,
                'total': total_cost,
                'pnl': pnl,
                'timestamp': timestamp
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _update_position(self, symbol: str, quantity: int, price: float, side: str) -> float:
        """
        Update position in database and calculate P&L for sells

        Returns:
            P&L for the trade (0.0 for buys, calculated value for sells)
        """
        timestamp = datetime.now().isoformat()
        pnl = 0.0

        # Get current position
        position = self.db.execute(
            "SELECT * FROM positions WHERE symbol = %s",
            (symbol,),
            fetch='one'
        )

        if position:
            current_qty = position['quantity']
            current_avg = position['entry_price']

            if side == 'buy':
                # Add to position
                new_qty = current_qty + quantity
                new_avg = ((current_avg * current_qty) + (price * quantity)) / new_qty

                self.db.execute(
                    "UPDATE positions SET quantity = %s, entry_price = %s, last_updated = %s WHERE symbol = %s",
                    (new_qty, new_avg, timestamp, symbol)
                )
            else:  # sell
                # Calculate P&L: (sell_price - entry_price) * quantity
                pnl = (price - current_avg) * quantity

                # Reduce position
                new_qty = current_qty - quantity
                if new_qty <= 0:
                    self.db.execute("DELETE FROM positions WHERE symbol = %s", (symbol,))
                else:
                    self.db.execute(
                        "UPDATE positions SET quantity = %s, last_updated = %s WHERE symbol = %s",
                        (new_qty, timestamp, symbol)
                    )

                print(f"[P&L] {symbol}: sold {quantity} @ ${price:.4f}, entry @ ${current_avg:.4f}, P&L: ${pnl:.2f}")
        else:
            # New position (only for buys)
            if side == 'buy':
                self.db.execute(
                    "INSERT INTO positions (symbol, quantity, entry_price, entry_date, last_updated) VALUES (%s, %s, %s, %s, %s)",
                    (symbol, quantity, price, timestamp, timestamp)
                )

        return pnl

    def _update_account_cash(self, amount: float) -> bool:
        """Apply a cash delta atomically. Returns False if a debit was refused.

        This was a read-modify-write -- SELECT the cash, add, then
        ``UPDATE account SET cash = %s`` with **no WHERE clause** -- and three
        different code paths did it concurrently (this bot, the Flask thread,
        and the orchestrator engines). Interleaved debits were simply lost
        (falcon-trader#22).

        Now the arithmetic happens in the database, and a debit carries its own
        sufficient-funds condition, so an overdraft loses the race instead of
        going negative.
        """
        timestamp = datetime.now().isoformat()

        if amount < 0:
            required = -amount
            # RETURNING, not a row count: DatabaseManager.execute gives
            # cursor.lastrowid on SQLite and cursor.rowcount on Postgres
            # (db_manager.py:165), so `rows == 0` silently never fires on
            # SQLite. A row comes back only if the WHERE matched, which makes
            # the refusal definitive on both backends and still atomic.
            row = self.db.execute(
                """UPDATE account
                      SET cash = cash - %s, last_updated = %s
                    WHERE id = (SELECT id FROM account ORDER BY id LIMIT 1)
                      AND cash >= %s
                RETURNING cash""",
                (required, timestamp, required),
                fetch='one',
            )
            if not row:
                print(f"[BOT] Cash debit of ${required:,.2f} refused (insufficient funds)")
                return False
            return True

        self.db.execute(
            """UPDATE account
                  SET cash = cash + %s, last_updated = %s
                WHERE id = (SELECT id FROM account ORDER BY id LIMIT 1)""",
            (amount, timestamp),
        )
        return True

    def update_market_data(self):
        """Refresh quotes for the watchlist AND mark every open position.

        The watchlist loop alone is why held symbols outside
        FALCON_DASHBOARD_SYMBOLS never got a price, so every position reported
        currentPrice == avgPrice (falcon-trader#21). mark_positions() existed
        but nothing called it; this is the call site.
        """
        for symbol in self.symbols:
            quote = self.get_quote(symbol)
            if quote:
                self.market_data[symbol] = quote

        try:
            self.mark_positions()
        except Exception as e:
            # A marking failure must not take down the market-data thread.
            print(f"[BOT] mark_positions failed: {e}")

    def _run_loop(self):
        """Background loop for market data updates"""
        print(f"[BOT] Starting market data loop (update every {self.update_interval}s)")

        while self.running:
            try:
                self.update_market_data()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"[BOT] Error in update loop: {e}")
                time.sleep(self.update_interval)

        print("[BOT] Market data loop stopped")

    def start(self):
        """Start the trading bot background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("[BOT] Started")

    def stop(self):
        """Stop the trading bot"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=5)
            print("[BOT] Stopped")

    def place_order_with_strategy(self, strategy_id: int, symbol: str,
                                  side: str, quantity: int,
                                  signal_reason: str, confidence: float) -> Dict:
        """
        Place order and link to strategy for attribution

        Args:
            strategy_id: ID from active_strategies table
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            signal_reason: Why the strategy triggered
            confidence: 0.0 to 1.0

        Returns:
            Order result dict
        """
        # Execute trade using existing place_order
        result = self.place_order(symbol, side, quantity)

        if result['status'] == 'success':
            # Get the order ID (last inserted row)
            order_id = self.db.execute(
                "SELECT id FROM orders ORDER BY id DESC LIMIT 1",
                fetch='one'
            )

            if order_id:
                # Record in strategy_trades table
                self.db.execute(
                    """INSERT INTO strategy_trades
                       (strategy_id, order_id, symbol, side, quantity, price,
                        signal_reason, signal_confidence, timestamp, pnl)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (strategy_id, order_id['id'], symbol, side, quantity,
                     result['price'], signal_reason, confidence,
                     result['timestamp'], 0)
                )

                # Log signal
                self._log_strategy_signal(
                    strategy_id, symbol, side, signal_reason,
                    confidence, result['price'], 'executed'
                )

                print(f"[BOT] Order attributed to strategy {strategy_id}")

        return result

    def _log_strategy_signal(self, strategy_id: int, symbol: str,
                            signal_type: str, reason: str,
                            confidence: float, price: float,
                            action_taken: str):
        """
        Log signal to strategy_signals table for debugging

        Args:
            strategy_id: Strategy that generated signal
            symbol: Stock symbol
            signal_type: 'buy', 'sell', or 'hold'
            reason: Signal reasoning
            confidence: Signal confidence
            price: Market price at signal time
            action_taken: 'executed', 'ignored', 'insufficient_funds'
        """
        try:
            self.db.execute(
                """INSERT INTO strategy_signals
                   (strategy_id, symbol, signal_type, signal_reason,
                    confidence, market_price, action_taken, timestamp)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (strategy_id, symbol, signal_type, reason, confidence,
                 price, action_taken, datetime.now().isoformat())
            )
        except Exception as e:
            print(f"[BOT] Warning: Could not log signal: {e}")

    def get_market_data(self) -> Dict:
        """Get cached market data"""
        return self.market_data.copy()

    def get_current_prices(self, symbols=None) -> Dict[str, float]:
        """Get current prices.

        `symbols` defaults to the static watchlist, but callers that need to
        mark a book must pass the symbols actually held. Iterating only
        ``self.symbols`` is why every open position outside the three-symbol
        FALCON_DASHBOARD_SYMBOLS list reported currentPrice == avgPrice
        (falcon-trader#21).

        A symbol with no obtainable quote is **absent** from the returned map --
        not present with a fallback value. Callers use that absence to mark the
        position stale.
        """
        wanted = list(self.symbols if symbols is None else symbols)
        prices = {}
        for symbol in wanted:
            if symbol in self.market_data:
                prices[symbol] = self.market_data[symbol]['price']
                continue
            quote = self.get_quote(symbol)
            if quote and quote.get('price'):
                prices[symbol] = quote['price']
        return prices

    def get_position_symbols(self):
        """Symbols with an open position."""
        rows = self.db.execute(
            "SELECT symbol FROM positions WHERE quantity > 0", fetch='all',
        ) or []
        return [r['symbol'] for r in rows]

    def mark_positions(self):
        """Refresh positions.current_price for every open position.

        One path owns this column. It was written by
        orchestrator/execution/trade_executor but never read by /api/positions,
        while the dashboard computed its own marks from a different source that
        fell back to cost (falcon-trader#21).
        """
        symbols = self.get_position_symbols()
        if not symbols:
            return {}

        prices = self.get_current_prices(symbols)
        timestamp = datetime.now().isoformat()
        for symbol, price in prices.items():
            self.db.execute(
                """UPDATE positions
                      SET current_price = %s, last_updated = %s
                    WHERE symbol = %s""",
                (price, timestamp, symbol),
            )

        missing = [s for s in symbols if s not in prices]
        if missing:
            print(f"[BOT] No quote for {len(missing)} held symbol(s): {missing}")
        return prices


# For standalone testing
if __name__ == '__main__':
    import sys

    # Environment only. A key passed on the command line is visible in `ps` to
    # every user on the host (falcon-trader#25).
    api_key = os.getenv('MASSIVE_API_KEY', '') or os.getenv('POLYGON_API_KEY', '')

    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        print(
            "Error: the API key is no longer accepted as a command-line "
            "argument -- it is visible in `ps` to every user on the host.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not api_key or api_key == 'your_polygon_api_key_here':
        print("Error: MASSIVE_API_KEY not set")
        print("Usage: MASSIVE_API_KEY=... python3 paper_trading_bot.py")
        sys.exit(1)

    # Initialize bot
    bot = PaperTradingBot(
        symbols=['SPY', 'QQQ', 'AAPL'],
        massive_api_key=api_key,
        initial_balance=10000.0,
        update_interval=60
    )

    # Start bot
    bot.start()

    print("\nBot is running. Commands:")
    print("  account  - Show account info")
    print("  positions - Show positions")
    print("  trades   - Show recent trades")
    print("  quote <symbol> - Get quote")
    print("  buy <symbol> <quantity> - Buy shares")
    print("  sell <symbol> <quantity> - Sell shares")
    print("  quit     - Exit\n")

    try:
        while True:
            cmd = input("> ").strip().split()
            if not cmd:
                continue

            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'account':
                print(bot.get_account())
            elif cmd[0] == 'positions':
                positions = bot.get_positions()
                if positions:
                    for pos in positions:
                        print(f"{pos['symbol']}: {pos['quantity']} @ ${pos['average_price']:.2f}")
                else:
                    print("No positions")
            elif cmd[0] == 'trades':
                trades = bot.get_trades(10)
                for trade in trades:
                    print(f"{trade['timestamp']}: {trade['side']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
            elif cmd[0] == 'quote' and len(cmd) > 1:
                quote = bot.get_quote(cmd[1].upper())
                if quote:
                    print(f"{quote['symbol']}: ${quote['price']:.2f}")
            elif cmd[0] == 'buy' and len(cmd) > 2:
                result = bot.place_order(cmd[1].upper(), 'buy', int(cmd[2]))
                print(result)
            elif cmd[0] == 'sell' and len(cmd) > 2:
                result = bot.place_order(cmd[1].upper(), 'sell', int(cmd[2]))
                print(result)
            else:
                print("Unknown command")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bot.stop()
