"""
Strategy routing logic
"""
import sys
import os
from datetime import datetime
from falcon_trader.orchestrator.routers.stock_classifier import StockClassifier
from falcon_trader.orchestrator.utils.data_structures import RoutingDecision, StockProfile


class StrategyRouter:
    """Route stocks to optimal strategies"""

    def __init__(self, config: dict):
        self.config = config
        self.classifier = StockClassifier(config)
        self.strategy_mapping = config['strategy_mapping']
        self.sector_routing = config.get('sector_routing', {})

    def route(self, symbol: str, use_yfinance: bool = False) -> RoutingDecision:
        """
        Route stock to optimal strategy

        Args:
            symbol: Stock ticker symbol
            use_yfinance: Whether to use yfinance for data (False = mock data for testing)

        Returns:
            RoutingDecision with strategy, confidence, and reasoning
        """
        # Get stock profile
        profile = self.classifier.get_stock_profile(symbol, use_yfinance=use_yfinance)

        if profile.price == 0.0:
            # Failed to get data
            return RoutingDecision(
                symbol=symbol,
                selected_strategy='rsi_mean_reversion',  # Default
                classification='unknown',
                reason='Failed to fetch stock data, using default strategy',
                confidence=0.50,
                timestamp=datetime.now(),
                profile=profile
            )

        # Select strategy based on profile
        strategy, alternatives = self._select_strategy_with_alternatives(profile)

        # Determine reason and confidence
        reason = self._get_routing_reason(profile, strategy)
        confidence = self._calculate_confidence(profile, strategy)

        return RoutingDecision(
            symbol=symbol,
            selected_strategy=strategy,
            classification=profile.classification,
            reason=reason,
            confidence=confidence,
            timestamp=datetime.now(),
            profile=profile,
            alternatives=alternatives
        )

    def _select_strategy_with_alternatives(self, profile: StockProfile) -> tuple:
        """
        Select strategy and calculate alternatives

        Returns: (selected_strategy, list_of_alternatives)
        """
        scores = {}

        # Score each strategy
        for strategy in ['rsi_mean_reversion', 'momentum_breakout', 'bollinger_mean_reversion']:
            scores[strategy] = self._score_strategy_for_profile(strategy, profile)

        # Select best strategy
        best_strategy = max(scores, key=scores.get)

        # Create alternatives list
        alternatives = [
            {'strategy': strategy, 'score': score}
            for strategy, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return best_strategy, alternatives

    def _score_strategy_for_profile(self, strategy: str, profile: StockProfile) -> float:
        """
        Score how well a strategy fits a stock profile

        Returns: Score from 0.0 to 1.0
        """
        score = 0.5  # Base score

        # RSI Mean Reversion scoring
        if strategy == 'rsi_mean_reversion':
            if profile.is_etf:
                score = 0.95  # Excellent for ETFs
            elif profile.classification == 'large_cap' and profile.volatility < 0.25:
                score = 0.85  # Good for stable large caps
            elif profile.classification in ['mid_cap', 'small_cap']:
                score = 0.70  # Decent for mid/small caps
            elif profile.classification == 'penny_stock':
                score = 0.30  # Poor for penny stocks

        # Momentum Breakout scoring
        elif strategy == 'momentum_breakout':
            if profile.classification == 'penny_stock':
                score = 0.90  # Excellent for penny stocks
            elif profile.volatility > 0.30:
                score = 0.85  # Good for high volatility
            elif profile.sector == 'Technology' and profile.volatility > 0.25:
                score = 0.80  # Good for volatile tech
            elif profile.is_etf:
                score = 0.25  # Poor for ETFs
            else:
                score = 0.60  # Decent default

        # Bollinger Mean Reversion scoring
        elif strategy == 'bollinger_mean_reversion':
            if profile.classification == 'etf':
                score = 0.75  # Good for ETFs
            elif profile.classification == 'large_cap' and profile.volatility < 0.20:
                score = 0.70  # Good for very stable stocks
            else:
                score = 0.50  # Neutral

        return score

    def _select_strategy(self, profile: StockProfile) -> str:
        """
        Select strategy based on profile

        This is the core routing logic
        """
        # Priority 1: Sector-specific routing
        if profile.sector in self.sector_routing:
            return self.sector_routing[profile.sector]

        # Priority 2: ETFs always use RSI Mean Reversion
        if profile.is_etf:
            return self.strategy_mapping.get('etfs', 'rsi_mean_reversion')

        # Priority 3: Penny stocks use Momentum Breakout
        if profile.classification == 'penny_stock':
            return self.strategy_mapping.get('penny_stocks', 'momentum_breakout')

        # Priority 4: High volatility uses Momentum Breakout
        if self.classifier.is_high_volatility(profile.volatility):
            return self.strategy_mapping.get('high_volatility', 'momentum_breakout')

        # Priority 5: Stable large caps use RSI Mean Reversion
        if self.classifier.is_stable_large_cap(profile.market_cap, profile.volatility):
            return self.strategy_mapping.get('large_cap_stable', 'rsi_mean_reversion')

        # Default: RSI Mean Reversion
        return self.strategy_mapping.get('default', 'rsi_mean_reversion')

    def _get_routing_reason(self, profile: StockProfile, strategy: str) -> str:
        """Generate human-readable routing reason"""
        if profile.is_etf:
            return f"ETF classification - proven {strategy} performer"

        elif profile.classification == 'penny_stock':
            return f"Penny stock (${profile.price:.2f}) - using {strategy} for volatile small caps"

        elif self.classifier.is_high_volatility(profile.volatility):
            return f"High volatility ({profile.volatility:.1%}) - {strategy} captures momentum"

        elif self.classifier.is_stable_large_cap(profile.market_cap, profile.volatility):
            return f"Stable large-cap (${profile.market_cap/1e9:.1f}B cap, {profile.volatility:.1%} vol) - {strategy} optimal"

        elif profile.sector in self.sector_routing:
            return f"{profile.sector} sector - specialized {strategy} routing"

        else:
            return f"{profile.classification} stock - default {strategy} strategy"

    def _calculate_confidence(self, profile: StockProfile, strategy: str) -> float:
        """
        Calculate confidence score for routing decision

        Returns: Confidence from 0.0 to 1.0
        """
        # Get score for selected strategy
        score = self._score_strategy_for_profile(strategy, profile)

        # Adjust based on data quality
        if profile.price == 0.0:
            score *= 0.5  # Low confidence if no price data

        if profile.volatility == 0.0:
            score *= 0.9  # Slightly lower if no volatility data

        if profile.market_cap == 0.0:
            score *= 0.9  # Slightly lower if no market cap

        return min(max(score, 0.10), 0.99)  # Clamp between 10% and 99%

    def get_strategy_description(self, strategy: str) -> dict:
        """Get description of a strategy"""
        descriptions = {
            'rsi_mean_reversion': {
                'name': 'RSI Mean Reversion',
                'best_for': 'ETFs, stable large-caps',
                'expected_return': '+15-20% annual',
                'win_rate': '80-90%',
                'characteristics': 'Quick exits (RSI 55), tight targets (2.5%)'
            },
            'momentum_breakout': {
                'name': 'Momentum Breakout',
                'best_for': 'Penny stocks, volatile stocks, semiconductors',
                'expected_return': '+50-100% on winners',
                'win_rate': '70-80%',
                'characteristics': 'Volume-confirmed entries, 15% targets, 8% stops'
            },
            'bollinger_mean_reversion': {
                'name': 'Bollinger Mean Reversion',
                'best_for': 'Stable stocks, ETFs',
                'expected_return': '+10-15% annual',
                'win_rate': '85-90%',
                'characteristics': 'Oversold/overbought bands, 5% targets'
            }
        }

        return descriptions.get(strategy, {'name': strategy})
