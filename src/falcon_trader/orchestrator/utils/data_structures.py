"""
Data structures for Multi-Strategy Orchestrator
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class StockProfile:
    """Stock classification profile"""
    symbol: str
    price: float
    volatility: float = 0.0
    market_cap: float = 0.0
    sector: str = "UNKNOWN"
    is_etf: bool = False
    avg_volume: int = 0
    classification: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volatility': self.volatility,
            'market_cap': self.market_cap,
            'sector': self.sector,
            'is_etf': self.is_etf,
            'avg_volume': self.avg_volume,
            'classification': self.classification
        }


@dataclass
class RoutingDecision:
    """Strategy routing decision"""
    symbol: str
    selected_strategy: str
    classification: str
    reason: str
    confidence: float
    timestamp: datetime
    profile: StockProfile
    alternatives: Optional[List[Dict]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'selected_strategy': self.selected_strategy,
            'classification': self.classification,
            'reason': self.reason,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'profile': self.profile.to_dict(),
            'alternatives': self.alternatives or []
        }


@dataclass
class Position:
    """Active position tracking"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: float
    profit_target: float
    strategy: str
    entry_timestamp: datetime
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update_current_price(self, price: float):
        """Update current price and recalculate P&L"""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity
        self.unrealized_pnl_pct = (price - self.entry_price) / self.entry_price

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'profit_target': self.profit_target,
            'strategy': self.strategy,
            'entry_timestamp': self.entry_timestamp.isoformat(),
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct
        }


@dataclass
class ValidationResult:
    """Result of entry validation"""
    is_valid: bool
    reason: str
    details: Optional[Dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'is_valid': self.is_valid,
            'reason': self.reason,
            'details': self.details or {}
        }
