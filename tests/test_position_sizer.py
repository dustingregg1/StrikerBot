"""
Tests for Position Sizer
==========================
Tests ATR calculation, Half-Kelly, and position sizing.
"""

import pytest
import numpy as np
from typing import List

from risk.position_sizer import calculate_atr, half_kelly, size_position


class TestCalculateATR:
    """Tests for ATR calculation."""

    def test_basic_atr(self) -> None:
        """ATR should be positive for valid price data."""
        highs = [100.0 + i for i in range(20)]
        lows = [98.0 + i for i in range(20)]
        closes = [99.0 + i for i in range(20)]
        atr = calculate_atr(highs, lows, closes, period=14)
        assert atr > 0

    def test_atr_increases_with_volatility(self) -> None:
        """ATR should be higher for more volatile data."""
        # Low volatility
        h_low = [100.0 + i * 0.1 for i in range(20)]
        l_low = [99.5 + i * 0.1 for i in range(20)]
        c_low = [99.8 + i * 0.1 for i in range(20)]
        atr_low = calculate_atr(h_low, l_low, c_low, period=14)

        # High volatility
        h_high = [100.0 + i * 0.1 + 5 for i in range(20)]
        l_high = [100.0 + i * 0.1 - 5 for i in range(20)]
        c_high = [100.0 + i * 0.1 for i in range(20)]
        atr_high = calculate_atr(h_high, l_high, c_high, period=14)

        assert atr_high > atr_low

    def test_atr_insufficient_data(self) -> None:
        """Should handle insufficient data gracefully."""
        atr = calculate_atr([100.0], [99.0], [99.5], period=14)
        assert atr >= 0


class TestHalfKelly:
    """Tests for Half-Kelly fraction calculation."""

    def test_standard_scenario(self) -> None:
        """40% win rate, 3:1 ratio should give 20% Full Kelly, 10% Half."""
        fraction = half_kelly(win_rate=0.40, avg_win=3.0, avg_loss=1.0)
        # Full Kelly = (0.4*3 - 0.6)/3 = 0.6/3 = 0.20
        # Half Kelly = 0.10
        assert fraction == pytest.approx(0.10, abs=0.01)

    def test_zero_win_rate(self) -> None:
        """Zero win rate should give minimum fraction."""
        fraction = half_kelly(win_rate=0.0, avg_win=3.0, avg_loss=1.0)
        assert fraction == pytest.approx(0.01)

    def test_high_win_rate(self) -> None:
        """High win rate with good ratio should give higher fraction."""
        fraction = half_kelly(win_rate=0.60, avg_win=2.0, avg_loss=1.0)
        assert fraction > half_kelly(win_rate=0.40, avg_win=2.0, avg_loss=1.0)

    def test_capped_at_50pct(self) -> None:
        """Should never exceed 50% even with extreme inputs."""
        fraction = half_kelly(win_rate=0.95, avg_win=10.0, avg_loss=1.0)
        assert fraction <= 0.50

    def test_negative_expectancy(self) -> None:
        """Negative expectancy should give minimum fraction."""
        fraction = half_kelly(win_rate=0.20, avg_win=1.0, avg_loss=1.0)
        # Kelly = (0.2*1 - 0.8)/1 = -0.6, clamped to 0
        # Half Kelly = 0, but minimum is 0.01
        assert fraction >= 0.0


class TestSizePosition:
    """Tests for position sizing."""

    def test_basic_sizing(self) -> None:
        """Should return valid position size."""
        result = size_position(
            equity_usd=500.0,
            entry_price=95000.0,
            stop_price=93000.0,  # ~2.1% stop
            max_risk_pct=2.0,
        )
        assert result['position_usd'] > 0
        assert result['quantity'] > 0
        assert result['risk_pct'] <= 2.0
        assert result['reason'] == 'sized'

    def test_zero_equity(self) -> None:
        """Should return zero position for zero equity."""
        result = size_position(
            equity_usd=0.0,
            entry_price=95000.0,
            stop_price=93000.0,
        )
        assert result['position_usd'] == 0.0
        assert result['reason'] == 'invalid_inputs'

    def test_stop_too_tight(self) -> None:
        """Should reject when stop is essentially at entry."""
        result = size_position(
            equity_usd=500.0,
            entry_price=95000.0,
            stop_price=94999.0,  # 0.001% - way too tight
        )
        assert result['position_usd'] == 0.0
        assert result['reason'] == 'stop_too_tight'

    def test_max_position_cap(self) -> None:
        """Position should never exceed max_position_pct."""
        result = size_position(
            equity_usd=500.0,
            entry_price=95000.0,
            stop_price=94500.0,  # Very tight stop -> large position
            max_risk_pct=5.0,  # High risk tolerance
            max_position_pct=30.0,
        )
        assert result['position_usd'] <= 500.0 * 0.30 + 0.01  # Small float tolerance

    def test_below_minimum(self) -> None:
        """Should reject positions below minimum size."""
        result = size_position(
            equity_usd=10.0,  # Very small equity
            entry_price=95000.0,
            stop_price=85000.0,  # 10% stop
            min_position_usd=10.0,
        )
        # With $10 equity, 2% risk = $0.20. Position = $0.20/0.10 = $2.
        # Below $10 minimum.
        assert result['position_usd'] == 0.0

    def test_risk_never_exceeds_max(self) -> None:
        """Risk percentage should never exceed max_risk_pct."""
        result = size_position(
            equity_usd=500.0,
            entry_price=95000.0,
            stop_price=93000.0,
            max_risk_pct=1.0,
        )
        assert result['risk_pct'] <= 1.0
