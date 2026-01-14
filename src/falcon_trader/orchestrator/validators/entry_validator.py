"""
Entry validation against AI screener recommendations
"""
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict
from falcon_trader.orchestrator.utils.data_structures import ValidationResult


class EntryValidator:
    """
    Validates trade entries against AI screener recommendations

    Checks:
    - Entry price within AI recommended range
    - Stop-loss buffer >= minimum threshold
    - AI confidence meets minimum requirements
    - Screener data is recent (not stale)
    """

    def __init__(self, config: dict):
        self.config = config
        self.validation_config = config.get('entry_validation', {})
        self.min_stop_buffer = config['routing']['min_stop_loss_buffer']
        self.screener_file = config['data_sources']['ai_screener']
        self.screener_data = None
        self.screener_loaded_at = None

    def load_screener_data(self, force_reload: bool = False) -> bool:
        """
        Load AI screener data from JSON file

        Args:
            force_reload: Force reload even if recently loaded

        Returns:
            True if loaded successfully, False otherwise
        """
        # Cache screener data for 5 minutes
        if not force_reload and self.screener_data and self.screener_loaded_at:
            elapsed = (datetime.now() - self.screener_loaded_at).total_seconds()
            if elapsed < 300:  # 5 minutes
                return True

        try:
            with open(self.screener_file, 'r') as f:
                self.screener_data = json.load(f)
                self.screener_loaded_at = datetime.now()
                return True
        except FileNotFoundError:
            print(f"[WARNING] AI screener file not found: {self.screener_file}")
            self.screener_data = None
            return False
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse AI screener JSON: {e}")
            self.screener_data = None
            return False

    def get_ai_recommendation(self, symbol: str) -> Optional[Dict]:
        """
        Get AI screener recommendation for a symbol

        Returns:
            Dict with recommendation data or None if not found
        """
        from falcon_trader.orchestrator.validators.screener_parser import ScreenerParser

        if not self.load_screener_data():
            return None

        # Handle both array format and object format
        if isinstance(self.screener_data, list):
            # Array of screening sessions - get most recent
            if len(self.screener_data) > 0:
                latest_screen = self.screener_data[0]  # Most recent
                recommendations = latest_screen.get('recommendations', [])

                for rec in recommendations:
                    ticker = rec.get('ticker', rec.get('symbol', ''))
                    if ticker == symbol:
                        # Normalize the recommendation
                        return ScreenerParser.normalize_recommendation(rec)

        elif isinstance(self.screener_data, dict):
            # Object format with stocks array
            stocks = self.screener_data.get('stocks', [])

            for stock in stocks:
                if stock.get('symbol', stock.get('ticker', '')) == symbol:
                    return ScreenerParser.normalize_recommendation(stock)

        return None

    def validate_entry(self, symbol: str, current_price: float,
                      proposed_stop_loss: Optional[float] = None) -> ValidationResult:
        """
        Validate a proposed trade entry

        Args:
            symbol: Stock ticker
            current_price: Current market price
            proposed_stop_loss: Proposed stop-loss price (optional)

        Returns:
            ValidationResult with is_valid, reason, and details
        """
        checks_enabled = self.validation_config.get('check_ai_screener', True)

        if not checks_enabled:
            return ValidationResult(
                is_valid=True,
                reason="Validation disabled in configuration",
                details={'validation_enabled': False}
            )

        # Get AI recommendation
        recommendation = self.get_ai_recommendation(symbol)

        if not recommendation:
            # No recommendation found - allow by default (backwards compatible)
            return ValidationResult(
                is_valid=True,
                reason=f"No AI screener data found for {symbol} - allowing entry",
                details={'has_recommendation': False}
            )

        # Perform validation checks
        checks = []

        # 1. Check entry price range
        price_check = self._validate_price_range(current_price, recommendation)
        checks.append(price_check)

        # 2. Check stop-loss buffer (if provided)
        if proposed_stop_loss is not None:
            stop_check = self._validate_stop_loss_buffer(
                current_price,
                proposed_stop_loss,
                recommendation
            )
            checks.append(stop_check)

        # 3. Check AI confidence
        confidence_check = self._validate_confidence(recommendation)
        checks.append(confidence_check)

        # 4. Check data freshness
        freshness_check = self._validate_data_freshness(recommendation)
        checks.append(freshness_check)

        # Determine overall result
        failed_checks = [c for c in checks if not c['passed']]

        if failed_checks:
            # At least one check failed
            reasons = [c['reason'] for c in failed_checks]
            return ValidationResult(
                is_valid=False,
                reason="; ".join(reasons),
                details={
                    'recommendation': recommendation,
                    'current_price': current_price,
                    'checks': checks,
                    'failed_checks': failed_checks
                }
            )
        else:
            # All checks passed
            return ValidationResult(
                is_valid=True,
                reason="All validation checks passed",
                details={
                    'recommendation': recommendation,
                    'current_price': current_price,
                    'checks': checks
                }
            )

    def _validate_price_range(self, current_price: float, recommendation: Dict) -> Dict:
        """Check if current price is within AI recommended entry range"""
        entry_range = recommendation.get('entry_range', '')

        if not entry_range or '-' not in entry_range:
            return {
                'check': 'price_range',
                'passed': True,
                'reason': 'No entry range specified'
            }

        # Parse entry range like "$2.00-$2.05"
        try:
            parts = entry_range.replace('$', '').split('-')
            min_price = float(parts[0].strip())
            max_price = float(parts[1].strip())

            if min_price <= current_price <= max_price:
                return {
                    'check': 'price_range',
                    'passed': True,
                    'reason': f'Price ${current_price:.2f} within range ${min_price:.2f}-${max_price:.2f}'
                }
            else:
                if current_price < min_price:
                    return {
                        'check': 'price_range',
                        'passed': False,
                        'reason': f'Price ${current_price:.2f} below entry range ${min_price:.2f}-${max_price:.2f}'
                    }
                else:
                    return {
                        'check': 'price_range',
                        'passed': False,
                        'reason': f'Price ${current_price:.2f} above entry range ${min_price:.2f}-${max_price:.2f}'
                    }

        except (ValueError, IndexError) as e:
            return {
                'check': 'price_range',
                'passed': True,
                'reason': f'Could not parse entry range: {entry_range}'
            }

    def _validate_stop_loss_buffer(self, entry_price: float, stop_loss: float,
                                   recommendation: Dict) -> Dict:
        """Check if stop-loss has adequate buffer from entry"""
        # Calculate buffer percentage
        buffer = (entry_price - stop_loss) / entry_price

        if buffer < 0:
            return {
                'check': 'stop_loss_buffer',
                'passed': False,
                'reason': f'Stop-loss ${stop_loss:.2f} is ABOVE entry ${entry_price:.2f}'
            }

        if buffer < self.min_stop_buffer:
            return {
                'check': 'stop_loss_buffer',
                'passed': False,
                'reason': f'Stop-loss buffer {buffer:.1%} below minimum {self.min_stop_buffer:.1%}'
            }

        return {
            'check': 'stop_loss_buffer',
            'passed': True,
            'reason': f'Stop-loss buffer {buffer:.1%} meets minimum {self.min_stop_buffer:.1%}'
        }

    def _validate_confidence(self, recommendation: Dict) -> Dict:
        """Check if AI confidence meets minimum requirements"""
        confidence = recommendation.get('confidence', 'UNKNOWN').upper()
        min_confidence = self.validation_config.get('min_confidence', 'MEDIUM').upper()

        confidence_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'UNKNOWN': 0}

        current_level = confidence_levels.get(confidence, 0)
        required_level = confidence_levels.get(min_confidence, 2)

        if current_level >= required_level:
            return {
                'check': 'ai_confidence',
                'passed': True,
                'reason': f'AI confidence {confidence} meets minimum {min_confidence}'
            }
        else:
            return {
                'check': 'ai_confidence',
                'passed': False,
                'reason': f'AI confidence {confidence} below minimum {min_confidence}'
            }

    def _validate_data_freshness(self, recommendation: Dict) -> Dict:
        """Check if screener data is recent"""
        if not self.screener_data:
            return {
                'check': 'data_freshness',
                'passed': True,
                'reason': 'No timestamp available'
            }

        # Get timestamp based on format
        timestamp_str = None

        if isinstance(self.screener_data, list):
            # Array format - get timestamp from first (most recent) screening session
            if len(self.screener_data) > 0:
                timestamp_str = self.screener_data[0].get('timestamp', '')
        elif isinstance(self.screener_data, dict):
            # Dict format - get timestamp from top level
            timestamp_str = self.screener_data.get('timestamp', '')

        if not timestamp_str:
            return {
                'check': 'data_freshness',
                'passed': True,
                'reason': 'No timestamp in screener data'
            }

        try:
            # Try multiple timestamp formats
            # Format 1: ISO 8601 with timezone (e.g., "2026-01-05T18:11:24.043177-05:00")
            try:
                from dateutil import parser
                screener_time = parser.parse(timestamp_str)
                # Make naive for comparison
                if screener_time.tzinfo is not None:
                    screener_time = screener_time.replace(tzinfo=None)
            except:
                # Format 2: Simple format (e.g., "2026-01-08 05:01:23")
                screener_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

            age = datetime.now() - screener_time

            # Warn if data is > 24 hours old
            if age.total_seconds() > 86400:  # 24 hours
                return {
                    'check': 'data_freshness',
                    'passed': False,
                    'reason': f'Screener data is {age.days} days old'
                }

            return {
                'check': 'data_freshness',
                'passed': True,
                'reason': f'Screener data is {age.seconds // 3600} hours old'
            }

        except Exception as e:
            return {
                'check': 'data_freshness',
                'passed': True,
                'reason': f'Could not parse timestamp: {str(e)}'
            }

    def get_recommended_stop_loss(self, symbol: str, entry_price: float) -> Optional[float]:
        """
        Calculate recommended stop-loss with minimum buffer enforcement

        Args:
            symbol: Stock ticker
            entry_price: Entry price

        Returns:
            Recommended stop-loss price or None
        """
        recommendation = self.get_ai_recommendation(symbol)

        if not recommendation:
            # No recommendation - use minimum buffer
            return entry_price * (1 - self.min_stop_buffer)

        # Get AI stop-loss
        ai_stop_str = recommendation.get('stop_loss', '')

        if not ai_stop_str:
            # No AI stop - use minimum buffer
            return entry_price * (1 - self.min_stop_buffer)

        try:
            ai_stop = float(ai_stop_str.replace('$', '').strip())

            # Check if AI stop meets minimum buffer
            buffer = (entry_price - ai_stop) / entry_price

            if buffer >= self.min_stop_buffer:
                # AI stop is good
                return ai_stop
            else:
                # AI stop too close - use minimum buffer
                adjusted_stop = entry_price * (1 - self.min_stop_buffer)
                print(f"[WARNING] AI stop ${ai_stop:.2f} too close to entry ${entry_price:.2f}")
                print(f"[WARNING] Adjusting to ${adjusted_stop:.2f} ({self.min_stop_buffer:.1%} buffer)")
                return adjusted_stop

        except ValueError:
            # Could not parse AI stop - use minimum buffer
            return entry_price * (1 - self.min_stop_buffer)

    def should_wait_for_better_entry(self, symbol: str, current_price: float) -> Dict:
        """
        Determine if we should wait for a better entry price

        Returns:
            Dict with 'should_wait', 'reason', 'target_range'
        """
        recommendation = self.get_ai_recommendation(symbol)

        if not recommendation:
            return {
                'should_wait': False,
                'reason': 'No AI recommendation available',
                'target_range': None
            }

        entry_range = recommendation.get('entry_range', '')

        if not entry_range or '-' not in entry_range:
            return {
                'should_wait': False,
                'reason': 'No entry range specified',
                'target_range': None
            }

        try:
            parts = entry_range.replace('$', '').split('-')
            min_price = float(parts[0].strip())
            max_price = float(parts[1].strip())

            if current_price < min_price:
                # Price below range - wait for it to enter range
                return {
                    'should_wait': True,
                    'reason': f'Price ${current_price:.2f} below entry range, wait for ${min_price:.2f}+',
                    'target_range': f'${min_price:.2f}-${max_price:.2f}'
                }
            elif current_price > max_price:
                # Price above range - missed opportunity or wait for pullback
                return {
                    'should_wait': True,
                    'reason': f'Price ${current_price:.2f} above entry range ${max_price:.2f}, wait for pullback',
                    'target_range': f'${min_price:.2f}-${max_price:.2f}'
                }
            else:
                # Price in range - good to enter
                return {
                    'should_wait': False,
                    'reason': f'Price ${current_price:.2f} in entry range',
                    'target_range': f'${min_price:.2f}-${max_price:.2f}'
                }

        except (ValueError, IndexError):
            return {
                'should_wait': False,
                'reason': 'Could not parse entry range',
                'target_range': None
            }
