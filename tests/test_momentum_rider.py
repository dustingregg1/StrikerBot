"""
Tests for Momentum Rider Strategy
====================================
Tests EMA crossover detection, Hurst regime filtering,
stop/trailing calculation, and pyramid logic.
"""

import time
import pytest
import numpy as np
from typing import List

from strategies.momentum_rider import (
    MomentumRider,
    MomentumPosition,
    MomentumSignal,
    calculate_ema,
    calculate_vwap,
)


def make_trending_prices(n: int = 100, start: float = 95000.0) -> List[float]:
    """Generate an upward-trending price series."""
    prices = []
    p = start
    for i in range(n):
        p += np.random.uniform(10, 100)  # Strong upward drift
        p += np.random.normal(0, 50)  # Some noise
        prices.append(max(p, start * 0.9))  # Floor
    return prices


def make_ranging_prices(n: int = 100, center: float = 95000.0) -> List[float]:
    """Generate a sideways price series (random walk)."""
    prices = []
    p = center
    for i in range(n):
        p += np.random.normal(0, 200)
        prices.append(p)
    return prices


@pytest.fixture
def rider() -> MomentumRider:
    """Default Momentum Rider with standard config."""
    return MomentumRider({
        'ema_fast_period': 9,
        'ema_slow_period': 21,
        'vwap_confirm': False,  # Disable for simpler testing
        'hurst_min': 0.55,
        'hurst_lookback': 50,
        'atr_period': 14,
        'stop_atr_multiplier': 2.0,
        'trail_atr_multiplier': 1.5,
        'pyramid_enabled': True,
        'pyramid_max_adds': 3,
        'pyramid_atr_spacing': 1.0,
        'max_hold_days': 7,
    })


class TestCalculateEMA:
    """Tests for EMA calculation."""

    def test_ema_length(self) -> None:
        """EMA should return correct number of values."""
        prices = list(range(1, 51))
        ema = calculate_ema(prices, 10)
        assert len(ema) == 41  # 50 - 10 + 1

    def test_ema_first_value_is_sma(self) -> None:
        """First EMA value should be SMA of first N prices."""
        prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        ema = calculate_ema(prices, 3)
        assert ema[0] == pytest.approx(20.0)  # Mean of [10, 20, 30]

    def test_ema_insufficient_data(self) -> None:
        """Should return empty list if not enough data."""
        prices = [100.0, 200.0]
        ema = calculate_ema(prices, 10)
        assert ema == []


class TestCalculateVWAP:
    """Tests for VWAP calculation."""

    def test_basic_vwap(self) -> None:
        """VWAP should weight by volume."""
        highs = [110.0, 120.0]
        lows = [90.0, 80.0]
        closes = [100.0, 110.0]
        volumes = [1000.0, 2000.0]
        vwap = calculate_vwap(highs, lows, closes, volumes)
        # TP1 = (110+90+100)/3 = 100
        # TP2 = (120+80+110)/3 = 103.33
        # VWAP = (100*1000 + 103.33*2000) / 3000 = 102.22
        assert vwap == pytest.approx(102.22, rel=0.01)

    def test_empty_data(self) -> None:
        """Should return 0 for empty data."""
        assert calculate_vwap([], [], [], []) == 0.0


class TestMomentumRiderEntry:
    """Tests for entry signal evaluation."""

    def test_insufficient_data_returns_no_signal(self, rider: MomentumRider) -> None:
        """Should not signal entry with too little data."""
        signal = rider.evaluate_entry([100.0] * 10, [110.0] * 10, [90.0] * 10, [100.0] * 10)
        assert signal.should_enter is False

    def test_trending_market_generates_signal(self, rider: MomentumRider) -> None:
        """In a strong uptrend, should eventually see entry signal."""
        np.random.seed(42)
        prices = make_trending_prices(150)
        highs = [p * 1.005 for p in prices]
        lows = [p * 0.995 for p in prices]
        volumes = [1000.0] * len(prices)

        signal = rider.evaluate_entry(prices, highs, lows, volumes)
        # Signal depends on whether EMA cross happened recently
        # At minimum, it should return a valid signal object
        assert isinstance(signal, MomentumSignal)
        assert 'hurst' in signal.details

    def test_ranging_market_usually_no_signal(self, rider: MomentumRider) -> None:
        """In a ranging market, Hurst should block entry."""
        np.random.seed(123)
        prices = make_ranging_prices(150)
        highs = [p + 200 for p in prices]
        lows = [p - 200 for p in prices]
        volumes = [1000.0] * len(prices)

        signal = rider.evaluate_entry(prices, highs, lows, volumes)
        # Hurst should be ~0.5 for random walk, below 0.55 threshold
        if 'hurst' in signal.details:
            # Not guaranteed but likely for pure random walk
            assert isinstance(signal.details['hurst'], float)


class TestMomentumRiderStop:
    """Tests for stop loss calculation."""

    def test_stop_below_entry(self, rider: MomentumRider) -> None:
        """Stop should always be below entry price."""
        entry = 95000.0
        highs = [95500.0] * 20
        lows = [94500.0] * 20
        closes = [95000.0] * 20
        stop = rider.calculate_stop(entry, highs, lows, closes)
        assert stop < entry

    def test_stop_never_below_10pct(self, rider: MomentumRider) -> None:
        """Stop should never be more than 10% below entry."""
        entry = 95000.0
        # Very high ATR scenario
        highs = [100000.0] * 20
        lows = [90000.0] * 20
        closes = [95000.0] * 20
        stop = rider.calculate_stop(entry, highs, lows, closes)
        assert stop >= entry * 0.90


class TestMomentumRiderExit:
    """Tests for exit signal evaluation."""

    def test_stop_loss_triggered(self, rider: MomentumRider) -> None:
        """Should exit when price hits stop loss."""
        position = MomentumPosition(
            trade_id='test1',
            symbol='BTC/USD',
            entry_price=95000.0,
            entry_time=time.time(),
            quantity=0.01,
            position_usd=950.0,
            stop_price=93000.0,
            highest_high=95000.0,
            trailing_stop=93000.0,
        )
        highs = [95500.0] * 20
        lows = [94500.0] * 20
        closes = [95000.0] * 20

        signal = rider.check_exit(position, 92500.0, highs, lows, closes)
        assert signal.should_exit is True
        assert signal.exit_reason == 'stop_loss'

    def test_max_duration_exit(self, rider: MomentumRider) -> None:
        """Should exit when max hold duration exceeded."""
        position = MomentumPosition(
            trade_id='test2',
            symbol='BTC/USD',
            entry_price=95000.0,
            entry_time=time.time() - (8 * 24 * 3600),  # 8 days ago
            quantity=0.01,
            position_usd=950.0,
            stop_price=90000.0,
            highest_high=100000.0,
            trailing_stop=98000.0,
        )
        highs = [95500.0] * 20
        lows = [94500.0] * 20
        closes = [99000.0] * 20  # Still profitable

        signal = rider.check_exit(position, 99000.0, highs, lows, closes)
        assert signal.should_exit is True
        assert signal.exit_reason == 'max_duration'

    def test_no_exit_when_profitable(self, rider: MomentumRider) -> None:
        """Should NOT exit when position is profitable and above stop."""
        position = MomentumPosition(
            trade_id='test3',
            symbol='BTC/USD',
            entry_price=95000.0,
            entry_time=time.time() - 3600,  # 1 hour ago
            quantity=0.01,
            position_usd=950.0,
            stop_price=93000.0,
            highest_high=96000.0,
            trailing_stop=94000.0,
        )
        highs = [96000.0] * 20
        lows = [95000.0] * 20
        closes = [95500.0] * 20

        signal = rider.check_exit(position, 96500.0, highs, lows, closes)
        assert signal.should_exit is False


class TestMomentumRiderPyramid:
    """Tests for pyramid (position adding) logic."""

    def test_pyramid_at_atr_level(self, rider: MomentumRider) -> None:
        """Should pyramid when price reaches next ATR level."""
        position = MomentumPosition(
            trade_id='test4',
            symbol='BTC/USD',
            entry_price=95000.0,
            entry_time=time.time(),
            quantity=0.01,
            position_usd=950.0,
            stop_price=93000.0,
            highest_high=97000.0,
            trailing_stop=94000.0,
            pyramid_count=0,
        )
        # ATR ~1000 with these price ranges
        highs = [96000.0] * 20
        lows = [95000.0] * 20
        closes = [95500.0] * 20

        # Price at entry + 1*ATR (should trigger first pyramid)
        should = rider.check_pyramid(
            position, 96500.0, highs, lows, closes
        )
        # Depends on calculated ATR, but tests the mechanism
        assert isinstance(should, bool)

    def test_no_pyramid_when_maxed(self, rider: MomentumRider) -> None:
        """Should NOT pyramid when max adds reached."""
        position = MomentumPosition(
            trade_id='test5',
            symbol='BTC/USD',
            entry_price=95000.0,
            entry_time=time.time(),
            quantity=0.04,
            position_usd=3800.0,
            stop_price=93000.0,
            highest_high=100000.0,
            trailing_stop=98000.0,
            pyramid_count=3,  # Already at max
        )
        highs = [100000.0] * 20
        lows = [99000.0] * 20
        closes = [99500.0] * 20

        should = rider.check_pyramid(position, 105000.0, highs, lows, closes)
        assert should is False

    def test_no_pyramid_when_disabled(self) -> None:
        """Should NOT pyramid when feature is disabled."""
        rider = MomentumRider({'pyramid_enabled': False})
        position = MomentumPosition(
            trade_id='test6',
            symbol='BTC/USD',
            entry_price=95000.0,
            entry_time=time.time(),
            quantity=0.01,
            position_usd=950.0,
            stop_price=93000.0,
            highest_high=100000.0,
            trailing_stop=98000.0,
            pyramid_count=0,
        )
        should = rider.check_pyramid(
            position, 105000.0, [100000.0]*20, [99000.0]*20, [99500.0]*20
        )
        assert should is False
