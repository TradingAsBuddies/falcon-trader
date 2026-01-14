"""
Market Data Fetcher

Fetches historical prices and volumes for strategy engines
Supports multiple data sources:
- Polygon.io (real-time and historical)
- Flat files (local CSV storage)
- yfinance (fallback)
"""
import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class MarketDataFetcher:
    """
    Fetches market data from various sources

    Priority:
    1. Polygon.io (if API key available)
    2. Flat files (if available)
    3. yfinance (fallback)
    """

    def __init__(self, config: dict):
        """
        Initialize market data fetcher

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data_sources', {})
        self.source = self.data_config.get('market_data', 'polygon')

        # Get API keys
        self.polygon_api_key = os.getenv('MASSIVE_API_KEY', '')

        # Flat files path
        self.flat_files_path = 'market_data/daily_bars'

    def fetch_market_data(self, symbol: str, lookback_days: int = 30) -> Dict:
        """
        Fetch market data for a symbol

        Args:
            symbol: Stock symbol
            lookback_days: Number of days of historical data

        Returns:
            Dict with:
                - 'price': Current price
                - 'prices': Historical prices (list, most recent last)
                - 'volume': Current volume
                - 'volumes': Historical volumes (list, most recent last)
                - 'source': Data source used
        """
        # Try primary source
        if self.source == 'polygon' and self.polygon_api_key:
            data = self._fetch_from_polygon(symbol, lookback_days)
            if data:
                return data

        # Try flat files
        if self.source == 'flatfiles' or not self.polygon_api_key:
            data = self._fetch_from_flatfiles(symbol, lookback_days)
            if data:
                return data

        # Try yfinance fallback
        if self.data_config.get('use_yfinance_fallback', True):
            data = self._fetch_from_yfinance(symbol, lookback_days)
            if data:
                return data

        # No data available
        return {
            'price': 0.0,
            'prices': [],
            'volume': 0,
            'volumes': [],
            'source': 'none',
            'error': 'No data source available'
        }

    def _fetch_from_polygon(self, symbol: str, lookback_days: int) -> Optional[Dict]:
        """
        Fetch data from Polygon.io

        Args:
            symbol: Stock symbol
            lookback_days: Number of days

        Returns:
            Market data dict or None
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 5)  # Extra buffer

            # Format dates for Polygon API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')

            # Fetch aggregates (daily bars)
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': self.polygon_api_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('status') == 'OK' and data.get('results'):
                    results = data['results']

                    # Extract prices and volumes
                    prices = [r['c'] for r in results]  # Close prices
                    volumes = [r['v'] for r in results]  # Volumes

                    # Get current (most recent) values
                    current_price = prices[-1] if prices else 0.0
                    current_volume = volumes[-1] if volumes else 0

                    return {
                        'price': current_price,
                        'prices': prices,
                        'volume': current_volume,
                        'volumes': volumes,
                        'source': 'polygon'
                    }

        except Exception as e:
            print(f"[WARNING] Polygon fetch failed for {symbol}: {e}")

        return None

    def _fetch_from_flatfiles(self, symbol: str, lookback_days: int) -> Optional[Dict]:
        """
        Fetch data from local flat files

        Args:
            symbol: Stock symbol
            lookback_days: Number of days

        Returns:
            Market data dict or None
        """
        try:
            # Look for recent flat files
            if not os.path.exists(self.flat_files_path):
                return None

            # Get list of files sorted by date (most recent first)
            files = sorted(
                [f for f in os.listdir(self.flat_files_path) if f.endswith('.csv.gz')],
                reverse=True
            )

            if not files:
                return None

            # Read files until we have enough data
            all_data = []
            for filename in files[:lookback_days + 5]:
                filepath = os.path.join(self.flat_files_path, filename)
                try:
                    df = pd.read_csv(filepath, compression='gzip')
                    # Filter for symbol
                    symbol_data = df[df['symbol'] == symbol]
                    if not symbol_data.empty:
                        all_data.append(symbol_data.iloc[0])
                except:
                    continue

                if len(all_data) >= lookback_days:
                    break

            if all_data:
                # Convert to DataFrame
                df = pd.DataFrame(all_data)
                df = df.sort_values('date')  # Oldest first

                prices = df['close'].tolist()
                volumes = df['volume'].tolist()

                return {
                    'price': prices[-1] if prices else 0.0,
                    'prices': prices,
                    'volume': volumes[-1] if volumes else 0,
                    'volumes': volumes,
                    'source': 'flatfiles'
                }

        except Exception as e:
            print(f"[WARNING] Flatfiles fetch failed for {symbol}: {e}")

        return None

    def _fetch_from_yfinance(self, symbol: str, lookback_days: int) -> Optional[Dict]:
        """
        Fetch data from yfinance (fallback)

        Args:
            symbol: Stock symbol
            lookback_days: Number of days

        Returns:
            Market data dict or None
        """
        try:
            import yfinance as yf

            # Fetch data
            ticker = yf.Ticker(symbol)

            # Get current/intraday price first (most recent data)
            current_df = ticker.history(period='1d')
            current_price = None
            current_volume = None

            if not current_df.empty:
                current_price = current_df['Close'].iloc[-1]
                current_volume = current_df['Volume'].iloc[-1]

            # Get historical daily data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 5)
            hist_df = ticker.history(start=start_date, end=end_date)

            if not hist_df.empty:
                prices = hist_df['Close'].tolist()
                volumes = hist_df['Volume'].tolist()

                # Use intraday price if available, otherwise use last historical
                final_price = current_price if current_price else (prices[-1] if prices else 0.0)
                final_volume = current_volume if current_volume else (volumes[-1] if volumes else 0)

                return {
                    'price': final_price,
                    'prices': prices,
                    'volume': final_volume,
                    'volumes': volumes,
                    'source': 'yfinance'
                }

        except ImportError:
            print("[WARNING] yfinance not installed (optional fallback)")
        except Exception as e:
            print(f"[WARNING] yfinance fetch failed for {symbol}: {e}")

        return None

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Current price or 0.0 if unavailable
        """
        data = self.fetch_market_data(symbol, lookback_days=1)
        return data.get('price', 0.0)

    def get_quote(self, symbol: str) -> Dict:
        """
        Get current quote for a symbol (compatible with paper_trading_bot)

        Args:
            symbol: Stock symbol

        Returns:
            Dict with quote data
        """
        data = self.fetch_market_data(symbol, lookback_days=1)

        return {
            'symbol': symbol,
            'price': data.get('price', 0.0),
            'volume': data.get('volume', 0),
            'source': data.get('source', 'none')
        }

    def validate_data_quality(self, market_data: Dict, min_periods: int = 20) -> Tuple[bool, str]:
        """
        Validate market data quality

        Args:
            market_data: Market data dict
            min_periods: Minimum required data points

        Returns:
            Tuple of (is_valid, reason)
        """
        if not market_data.get('prices'):
            return False, "No price data available"

        if len(market_data['prices']) < min_periods:
            return False, f"Insufficient data: {len(market_data['prices'])} < {min_periods} required"

        if market_data.get('price', 0.0) <= 0:
            return False, "Invalid current price"

        # Check for data gaps (more than 30% zeros)
        prices = market_data['prices']
        zero_count = sum(1 for p in prices if p <= 0)
        if zero_count / len(prices) > 0.3:
            return False, f"Too many invalid prices: {zero_count}/{len(prices)}"

        return True, "Data quality OK"
