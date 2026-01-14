"""
Base Strategy Engine

Provides common functionality for all strategy engines:
- Position management
- Order execution
- Stop-loss and profit target monitoring
- Database integration
"""
import sys
import os
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from db_manager import DatabaseManager
from orchestrator.utils.data_structures import Position


@dataclass
class TradeSignal:
    """Trade signal from strategy engine"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: int
    price: float
    reason: str
    stop_loss: Optional[float] = None
    profit_target: Optional[float] = None
    confidence: float = 0.0
    metadata: Optional[Dict] = None


@dataclass
class ExecutionResult:
    """Result of trade execution"""
    success: bool
    trade_id: Optional[int] = None
    symbol: str = ""
    action: str = ""
    quantity: int = 0
    price: float = 0.0
    pnl: float = 0.0
    reason: str = ""
    error: Optional[str] = None
    timestamp: datetime = None


class BaseStrategyEngine:
    """
    Base class for all strategy engines

    Provides common functionality:
    - Database interaction
    - Position tracking
    - Order execution
    - Stop-loss and profit target monitoring
    """

    def __init__(self, config: dict, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize strategy engine

        Args:
            config: Configuration dictionary
            db_manager: Optional DatabaseManager instance (creates one if not provided)
        """
        self.config = config
        self.strategy_name = self.__class__.__name__.replace('Engine', '').lower()

        # Initialize database manager
        if db_manager:
            self.db = db_manager
        else:
            db_config = {'db_type': 'sqlite', 'db_path': 'paper_trading.db'}
            self.db = DatabaseManager(db_config)

        # Get routing config
        self.routing_config = config.get('routing', {})
        self.min_stop_buffer = self.routing_config.get('min_stop_loss_buffer', 0.05)

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for symbol

        Args:
            symbol: Stock symbol

        Returns:
            Position object or None
        """
        result = self.db.execute(
            "SELECT * FROM positions WHERE symbol = %s AND quantity > 0",
            (symbol,),
            fetch='one'
        )

        if not result:
            return None

        # Handle sqlite3.Row which doesn't have .get() method
        try:
            stop_loss = result['stop_loss'] if result['stop_loss'] else 0.0
        except (KeyError, IndexError):
            stop_loss = 0.0

        try:
            profit_target = result['profit_target'] if result['profit_target'] else 0.0
        except (KeyError, IndexError):
            profit_target = 0.0

        try:
            strategy = result['strategy'] if result['strategy'] else self.strategy_name
        except (KeyError, IndexError):
            strategy = self.strategy_name

        return Position(
            symbol=result['symbol'],
            quantity=result['quantity'],
            entry_price=result['entry_price'],
            current_price=result['entry_price'],  # Will be updated
            stop_loss=stop_loss,
            profit_target=profit_target,
            strategy=strategy,
            entry_timestamp=datetime.fromisoformat(result['entry_date'])
        )

    def get_account_balance(self) -> float:
        """Get current cash balance"""
        result = self.db.execute(
            "SELECT cash FROM account LIMIT 1",
            fetch='one'
        )
        return float(result['cash']) if result else 0.0

    def calculate_position_size(self, symbol: str, price: float,
                               max_allocation: float = 0.25) -> int:
        """
        Calculate position size based on available cash

        Args:
            symbol: Stock symbol
            price: Current price
            max_allocation: Maximum % of portfolio to allocate (default 25%)

        Returns:
            Number of shares to buy
        """
        cash = self.get_account_balance()
        max_investment = cash * max_allocation
        quantity = int(max_investment / price)
        return max(quantity, 0)

    def execute_buy(self, symbol: str, quantity: int, price: float,
                   stop_loss: float, profit_target: float,
                   reason: str = "") -> ExecutionResult:
        """
        Execute a buy order

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Entry price
            stop_loss: Stop-loss price
            profit_target: Profit target price
            reason: Reason for trade

        Returns:
            ExecutionResult with trade details
        """
        try:
            # Check if we already have a position
            existing = self.get_position(symbol)
            if existing:
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    error=f"Already have position in {symbol}"
                )

            # Check if we have enough cash
            cost = quantity * price
            cash = self.get_account_balance()

            if cost > cash:
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    error=f"Insufficient funds: need ${cost:.2f}, have ${cash:.2f}"
                )

            # Insert order record
            self.db.execute("""
                INSERT INTO orders (
                    symbol, side, quantity, price, timestamp,
                    strategy, reason
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol, 'BUY', quantity, price,
                datetime.now().isoformat(),
                self.strategy_name, reason
            ))

            # Insert or update position
            self.db.execute("""
                INSERT INTO positions (
                    symbol, quantity, entry_price, entry_date,
                    stop_loss, profit_target, strategy, last_updated
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(symbol) DO UPDATE SET
                    quantity = quantity + %s,
                    last_updated = %s
            """, (
                symbol, quantity, price, datetime.now().isoformat(),
                stop_loss, profit_target, self.strategy_name,
                datetime.now().isoformat(),
                quantity, datetime.now().isoformat()
            ))

            # Update cash balance
            new_cash = cash - cost
            self.db.execute(
                "UPDATE account SET cash = %s, last_updated = %s",
                (new_cash, datetime.now().isoformat())
            )

            return ExecutionResult(
                success=True,
                symbol=symbol,
                action='BUY',
                quantity=quantity,
                price=price,
                reason=reason,
                timestamp=datetime.now()
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                action='BUY',
                error=str(e)
            )

    def execute_sell(self, symbol: str, quantity: int, price: float,
                    reason: str = "") -> ExecutionResult:
        """
        Execute a sell order

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Exit price
            reason: Reason for trade

        Returns:
            ExecutionResult with trade details
        """
        try:
            # Check if we have a position
            position = self.get_position(symbol)
            if not position:
                return ExecutionResult(
                    success=False,
                    symbol=symbol,
                    error=f"No position in {symbol}"
                )

            # Adjust quantity if needed
            if quantity > position.quantity:
                quantity = position.quantity

            # Calculate P&L
            pnl = (price - position.entry_price) * quantity
            print(f"[P&L] {symbol}: sold {quantity} @ ${price:.4f}, entry @ ${position.entry_price:.4f}, P&L: ${pnl:.2f}")

            # Insert order record with P&L
            self.db.execute("""
                INSERT INTO orders (
                    symbol, side, quantity, price, timestamp,
                    strategy, reason, pnl
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol, 'SELL', quantity, price,
                datetime.now().isoformat(),
                self.strategy_name, reason, pnl
            ))

            # Update position
            new_quantity = position.quantity - quantity
            if new_quantity == 0:
                # Close position
                self.db.execute(
                    "DELETE FROM positions WHERE symbol = %s",
                    (symbol,)
                )
            else:
                # Reduce position
                self.db.execute(
                    "UPDATE positions SET quantity = %s, last_updated = %s WHERE symbol = %s",
                    (new_quantity, datetime.now().isoformat(), symbol)
                )

            # Update cash balance
            proceeds = quantity * price
            cash = self.get_account_balance()
            new_cash = cash + proceeds
            self.db.execute(
                "UPDATE account SET cash = %s, last_updated = %s",
                (new_cash, datetime.now().isoformat())
            )

            return ExecutionResult(
                success=True,
                symbol=symbol,
                action='SELL',
                quantity=quantity,
                price=price,
                pnl=pnl,
                reason=reason,
                timestamp=datetime.now()
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                symbol=symbol,
                action='SELL',
                error=str(e)
            )

    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        """
        Check if stop-loss should trigger

        Args:
            position: Position object
            current_price: Current market price

        Returns:
            True if stop-loss triggered
        """
        if position.stop_loss > 0 and current_price <= position.stop_loss:
            return True
        return False

    def check_profit_target(self, position: Position, current_price: float) -> bool:
        """
        Check if profit target reached

        Args:
            position: Position object
            current_price: Current market price

        Returns:
            True if profit target reached
        """
        if position.profit_target > 0 and current_price >= position.profit_target:
            return True
        return False

    def generate_signal(self, symbol: str, market_data: Dict) -> TradeSignal:
        """
        Generate trade signal (must be implemented by subclass)

        Args:
            symbol: Stock symbol
            market_data: Market data dict with price, volume, indicators

        Returns:
            TradeSignal object
        """
        raise NotImplementedError("Subclass must implement generate_signal()")

    def execute_signal(self, signal: TradeSignal) -> ExecutionResult:
        """
        Execute a trade signal

        Args:
            signal: TradeSignal object

        Returns:
            ExecutionResult
        """
        if signal.action == 'BUY':
            return self.execute_buy(
                signal.symbol,
                signal.quantity,
                signal.price,
                signal.stop_loss or 0.0,
                signal.profit_target or 0.0,
                signal.reason
            )
        elif signal.action == 'SELL':
            return self.execute_sell(
                signal.symbol,
                signal.quantity,
                signal.price,
                signal.reason
            )
        else:  # HOLD
            return ExecutionResult(
                success=True,
                symbol=signal.symbol,
                action='HOLD',
                reason=signal.reason
            )

    def monitor_position(self, symbol: str, current_price: float) -> Optional[TradeSignal]:
        """
        Monitor position for stop-loss or profit target

        Args:
            symbol: Stock symbol
            current_price: Current market price

        Returns:
            TradeSignal if action needed, None otherwise
        """
        position = self.get_position(symbol)
        if not position:
            return None

        position.update_current_price(current_price)

        # Check stop-loss
        if self.check_stop_loss(position, current_price):
            return TradeSignal(
                symbol=symbol,
                action='SELL',
                quantity=position.quantity,
                price=current_price,
                reason=f"Stop-loss triggered at ${current_price:.2f}"
            )

        # Check profit target
        if self.check_profit_target(position, current_price):
            return TradeSignal(
                symbol=symbol,
                action='SELL',
                quantity=position.quantity,
                price=current_price,
                reason=f"Profit target reached at ${current_price:.2f}"
            )

        return None

    def get_strategy_name(self) -> str:
        """Get strategy name"""
        return self.strategy_name
