"""
Volatility Surfer Strategy (Phase 2 - STUB)
=============================================
Detection: ATR spike >1.5x 30-day avg (same signal goalkeeper PAUSES on)
Entry: First pullback after volatility spike
Stop: 3x ATR (wider for elevated volatility)
Target: Trailing stop at 2x ATR
Position size: HALF normal (self-adjusts via ATR)

NOT IMPLEMENTED YET - Phase 2
"""

from typing import Dict, Any


class VolatilitySurfer:
    """Volatility spike strategy. Phase 2 implementation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = config.get('enabled', False)

    def evaluate_entry(self, **kwargs: Any) -> Dict[str, Any]:
        """Stub: Not implemented in Phase 1."""
        return {'should_enter': False, 'reason': 'phase_2_not_implemented'}

    def check_exit(self, **kwargs: Any) -> Dict[str, Any]:
        """Stub: Not implemented in Phase 1."""
        return {'should_exit': False, 'reason': 'phase_2_not_implemented'}
