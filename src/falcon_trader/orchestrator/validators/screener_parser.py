"""
Parser for AI screener data with multiple format support
"""
from typing import Optional, Dict


class ScreenerParser:
    """
    Parse AI screener recommendations which can have different formats

    Supports various field name variations:
    - entry_price_range / entry_range / entry
    - target_price / target
    - stop_loss / stop
    - confidence_score / confidence
    """

    @staticmethod
    def parse_entry_range(recommendation: Dict) -> Optional[str]:
        """
        Extract entry range from recommendation

        Returns: String like "$2.00-$2.05" or "2.00-2.05" or None
        """
        # Try various field names
        for field in ['entry_price_range', 'entry_range', 'entry', 'entry_price']:
            value = recommendation.get(field)
            if value and '-' in str(value):
                return str(value)

        return None

    @staticmethod
    def parse_target(recommendation: Dict) -> Optional[str]:
        """
        Extract target price from recommendation

        Returns: String like "$2.25" or "2.25-2.40" or None
        """
        for field in ['target_price', 'target']:
            value = recommendation.get(field)
            if value:
                return str(value)

        return None

    @staticmethod
    def parse_stop_loss(recommendation: Dict) -> Optional[str]:
        """
        Extract stop-loss from recommendation

        Returns: String like "$1.90" or "1.90" or None
        """
        for field in ['stop_loss', 'stop', 'Stop_loss']:  # Note: Some have capital S
            value = recommendation.get(field)
            if value:
                return str(value)

        return None

    @staticmethod
    def parse_confidence(recommendation: Dict) -> str:
        """
        Extract confidence level from recommendation

        Returns: String like "HIGH", "MEDIUM", "LOW" or "UNKNOWN"
        """
        # Try confidence field (string)
        confidence_str = recommendation.get('confidence', '').upper()
        if confidence_str in ['HIGH', 'MEDIUM', 'LOW', 'MEDIUM-HIGH', 'LOW-MEDIUM']:
            return confidence_str

        # Try confidence_score field (number)
        confidence_score = recommendation.get('confidence_score')
        if confidence_score is not None:
            try:
                score = int(confidence_score)
                if score >= 7:
                    return 'HIGH'
                elif score >= 5:
                    return 'MEDIUM'
                else:
                    return 'LOW'
            except (ValueError, TypeError):
                pass

        return 'UNKNOWN'

    @staticmethod
    def normalize_recommendation(recommendation: Dict) -> Dict:
        """
        Normalize recommendation to standardformat

        Returns dict with:
        - symbol
        - entry_range
        - target
        - stop_loss
        - confidence
        """
        return {
            'symbol': recommendation.get('ticker', recommendation.get('symbol', '')),
            'entry_range': ScreenerParser.parse_entry_range(recommendation),
            'target': ScreenerParser.parse_target(recommendation),
            'stop_loss': ScreenerParser.parse_stop_loss(recommendation),
            'confidence': ScreenerParser.parse_confidence(recommendation),
            'original': recommendation  # Keep original for reference
        }
