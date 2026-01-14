#!/usr/bin/env python3
"""
Falcon Paper Trading Bot
Real-time paper trading with Polygon.io market data
"""

import os
import time
import requests
import threading
from datetime import datetime
from typing import List, Dict, Optional
from db_manager import DatabaseManager

try:
    from config import FalconConfig
    falcon_config = FalconConfig()
except ImportError:
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
        """
        Get quote from Polygon.io (uses previous close for free tier)

        Args:
            symbol: Stock symbol

        Returns:
            Dict with quote data or None
        """
        try:
            # Use previous close endpoint (available on free tier)
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            params = {'adjusted': 'true', 'apiKey': self.massive_api_key}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK' and data.get('results'):
                    result = data['results'][0]
                    return {
                        'symbol': symbol,
                        'price': result.get('c', 0),  # Close price
                        'open': result.get('o', 0),
                        'high': result.get('h', 0),
                        'low': result.get('l', 0),
                        'volume': result.get('v', 0),
                        'timestamp': result.get('t', 0)
                    }
            else:
                print(f"[BOT] Quote error for {symbol}: {response.status_code}")
        except Exception as e:
            print(f"[BOT] Error fetching quote for {symbol}: {e}")

        return None

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
            # Get current market price
            quote = self.get_quote(symbol)
            if not quote:
                return {'status': 'error', 'message': f'Could not get quote for {symbol}'}

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

            # Execute trade
            timestamp = datetime.now().isoformat()
            pnl = 0.0

            # Update positions and calculate P&L for sells
            if side == 'buy':
                self._update_position(symbol, quantity, execution_price, 'buy')
                self._update_account_cash(-total_cost)
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

    def _update_account_cash(self, amount: float):
        """Update account cash balance"""
        account = self.get_account()
        new_cash = account['cash'] + amount

        self.db.execute(
            "UPDATE account SET cash = %s, last_updated = %s",
            (new_cash, datetime.now().isoformat())
        )

    def update_market_data(self):
        """Fetch latest market data for tracked symbols"""
        for symbol in self.symbols:
            quote = self.get_quote(symbol)
            if quote:
                self.market_data[symbol] = quote

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

    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for tracked symbols"""
        prices = {}
        for symbol in self.symbols:
            if symbol in self.market_data:
                prices[symbol] = self.market_data[symbol]['price']
            else:
                quote = self.get_quote(symbol)
                if quote:
                    prices[symbol] = quote['price']
        return prices


# For standalone testing
if __name__ == '__main__':
    import sys

    # Get API key from environment or command line
    api_key = os.getenv('MASSIVE_API_KEY', '')
    if len(sys.argv) > 1:
        api_key = sys.argv[1]

    if not api_key or api_key == 'your_polygon_api_key_here':
        print("Error: MASSIVE_API_KEY not set")
        print("Usage: python3 paper_trading_bot.py <polygon_api_key>")
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
