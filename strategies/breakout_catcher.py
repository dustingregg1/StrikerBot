"""
Breakout Catcher Strategy (Phase 2 - STUB)
============================================
Detection: Bollinger Band squeeze + volume spike on break
Entry: Market order at breakout confirmation
Stop: Just below breakout level (0.5-1x ATR)
Target: 1:3 risk/reward minimum
Filter: Volume must be >1.5x 20-period average

NOT IMPLEMENTED YET - Phase 2
"""

from typing import Dict, Any


class BreakoutCatcher:
    """Breakout detection strategy. Phase 2 implementation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = config.get('enabled', False)

    def evaluate_entry(self, **kwargs: Any) -> Dict[str, Any]:
        """Stub: Not implemented in Phase 1."""
        return {'should_enter': False, 'reason': 'phase_2_not_implemented'}

    def check_exit(self, **kwargs: Any) -> Dict[str, Any]:
        """Stub: Not implemented in Phase 1."""
        return {'should_exit': False, 'reason': 'phase_2_not_implemented'}
