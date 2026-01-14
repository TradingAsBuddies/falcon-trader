"""
Stock classification for strategy routing
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from orchestrator.utils.data_structures import StockProfile


class StockClassifier:
    """Classify stocks for strategy routing"""

    def __init__(self, config: dict):
        self.config = config
        self.etf_list = config.get('etf_symbols', [])
        self.penny_threshold = config['routing']['penny_stock_threshold']
        self.high_vol_threshold = config['routing']['high_volatility_threshold']
        self.large_cap_threshold = config['routing']['large_cap_threshold']

    def get_stock_profile(self, symbol: str, use_yfinance: bool = True) -> StockProfile:
        """
        Get complete stock profile

        Args:
            symbol: Stock ticker symbol
            use_yfinance: Whether to fetch data from yfinance (False = use mock data)
        """
        if use_yfinance:
            return self._get_profile_from_yfinance(symbol)
        else:
            return self._get_mock_profile(symbol)

    def _get_profile_from_yfinance(self, symbol: str) -> StockProfile:
        """Get profile from yfinance API"""
        try:
            import yfinance as yf
            import numpy as np

            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='1mo')

            # Get current price
            if len(hist) > 0:
                price = float(hist['Close'].iloc[-1])
            else:
                price = float(info.get('currentPrice', 0))

            # Calculate 30-day volatility (annualized)
            if len(hist) > 1:
                returns = hist['Close'].pct_change().dropna()
                volatility = float(returns.std() * (252 ** 0.5))  # Annualized
            else:
                volatility = 0.0

            # Get market cap
            market_cap = float(info.get('marketCap', 0))

            # Get average volume
            avg_volume = int(info.get('averageVolume', 0))

            # Check if ETF
            is_etf = symbol in self.etf_list or info.get('quoteType') == 'ETF'

            # Get sector
            sector = info.get('sector', 'UNKNOWN')

            # Classify
            classification = self._classify(price, volatility, market_cap, is_etf)

            return StockProfile(
                symbol=symbol,
                price=price,
                volatility=volatility,
                market_cap=market_cap,
                sector=sector,
                is_etf=is_etf,
                avg_volume=avg_volume,
                classification=classification
            )

        except Exception as e:
            print(f"[ERROR] Failed to classify {symbol} with yfinance: {e}")
            print(f"[INFO] Falling back to mock data for {symbol}")
            return self._get_mock_profile(symbol)

    def _get_mock_profile(self, symbol: str) -> StockProfile:
        """Get mock profile for testing without API calls"""
        # Mock data for common test symbols
        mock_data = {
            'SPY': {
                'price': 475.50,
                'volatility': 0.15,
                'market_cap': 400e9,
                'sector': 'ETF',
                'is_etf': True,
                'avg_volume': 50000000
            },
            'QQQ': {
                'price': 395.25,
                'volatility': 0.18,
                'market_cap': 200e9,
                'sector': 'ETF',
                'is_etf': True,
                'avg_volume': 40000000
            },
            'MU': {
                'price': 95.50,
                'volatility': 0.35,
                'market_cap': 105e9,
                'sector': 'Technology',
                'is_etf': False,
                'avg_volume': 15000000
            },
            'NVDA': {
                'price': 495.25,
                'volatility': 0.38,
                'market_cap': 1200e9,
                'sector': 'Technology',
                'is_etf': False,
                'avg_volume': 40000000
            },
            'ABTC': {
                'price': 1.91,
                'volatility': 0.52,
                'market_cap': 50e6,
                'sector': 'Technology',
                'is_etf': False,
                'avg_volume': 500000
            },
            'AAPL': {
                'price': 185.50,
                'volatility': 0.22,
                'market_cap': 2900e9,
                'sector': 'Technology',
                'is_etf': False,
                'avg_volume': 55000000
            },
            'MSFT': {
                'price': 375.00,
                'volatility': 0.20,
                'market_cap': 2800e9,
                'sector': 'Technology',
                'is_etf': False,
                'avg_volume': 25000000
            },
            'TSLA': {
                'price': 245.00,
                'volatility': 0.45,
                'market_cap': 780e9,
                'sector': 'Consumer Cyclical',
                'is_etf': False,
                'avg_volume': 95000000
            }
        }

        if symbol in mock_data:
            data = mock_data[symbol]
            classification = self._classify(
                data['price'],
                data['volatility'],
                data['market_cap'],
                data['is_etf']
            )

            return StockProfile(
                symbol=symbol,
                price=data['price'],
                volatility=data['volatility'],
                market_cap=data['market_cap'],
                sector=data['sector'],
                is_etf=data['is_etf'],
                avg_volume=data['avg_volume'],
                classification=classification
            )
        else:
            # Unknown symbol - return minimal profile
            return StockProfile(
                symbol=symbol,
                price=0.0,
                classification='unknown'
            )

    def _classify(self, price: float, volatility: float, market_cap: float, is_etf: bool) -> str:
        """Classify stock type"""
        if is_etf:
            return "etf"
        elif price < self.penny_threshold:
            return "penny_stock"
        elif market_cap > self.large_cap_threshold:
            return "large_cap"
        elif market_cap > 10e9:
            return "mid_cap"
        else:
            return "small_cap"

    def is_high_volatility(self, volatility: float) -> bool:
        """Check if stock has high volatility"""
        return volatility > self.high_vol_threshold

    def is_stable_large_cap(self, market_cap: float, volatility: float) -> bool:
        """Check if stock is stable large cap"""
        return market_cap > self.large_cap_threshold and volatility < 0.25
