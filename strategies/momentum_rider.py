"""
Momentum Rider Strategy (Phase 1 MVP)
=======================================
Entry: EMA9 crosses above EMA21 + price above VWAP + Hurst > 0.55
Stop: 2x ATR below entry (adaptive to volatility)
Exit: Trailing stop at 1.5x ATR from highest high
Pyramid: Up to 3 adds at +1, +2, +3 ATR above entry
Max duration: 7 days
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from signals.hurst_regime import compute_hurst_exponent, is_trending
from risk.position_sizer import calculate_atr, size_position


@dataclass
class MomentumPosition:
    """Active momentum position state."""
    trade_id: str
    symbol: str
    entry_price: float
    entry_time: float
    quantity: float
    position_usd: float
    stop_price: float
    highest_high: float
    trailing_stop: float
    pyramid_count: int = 0
    pyramid_prices: list = field(default_factory=list)


@dataclass
class MomentumSignal:
    """Result of signal evaluation."""
    should_enter: bool = False
    should_exit: bool = False
    exit_reason: str = ""
    entry_direction: str = "long"  # Only long for Phase 1
    signal_strength: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


def calculate_ema(prices: list[float], period: int) -> list[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return []

    multiplier = 2.0 / (period + 1)
    ema = [float(np.mean(prices[:period]))]

    for price in prices[period:]:
        ema.append(price * multiplier + ema[-1] * (1 - multiplier))

    return ema


def calculate_vwap(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> float:
    """Calculate Volume-Weighted Average Price for the session."""
    if not highs or not volumes:
        return 0.0

    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    cum_tp_vol = sum(tp * v for tp, v in zip(typical_prices, volumes))
    cum_vol = sum(volumes)

    return cum_tp_vol / cum_vol if cum_vol > 0 else 0.0


class MomentumRider:
    """
    Momentum Rider strategy.
    Catches trending moves with EMA crossover + regime confirmation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.ema_fast = config.get('ema_fast_period', 9)
        self.ema_slow = config.get('ema_slow_period', 21)
        self.vwap_confirm = config.get('vwap_confirm', True)
        self.hurst_min = config.get('hurst_min', 0.55)
        self.hurst_lookback = config.get('hurst_lookback', 100)
        self.atr_period = config.get('atr_period', 14)
        self.stop_atr_mult = config.get('stop_atr_multiplier', 2.0)
        self.trail_atr_mult = config.get('trail_atr_multiplier', 1.5)
        self.pyramid_enabled = config.get('pyramid_enabled', True)
        self.pyramid_max_adds = config.get('pyramid_max_adds', 3)
        self.pyramid_atr_spacing = config.get('pyramid_atr_spacing', 1.0)
        self.max_hold_days = config.get('max_hold_days', 7)

    def evaluate_entry(
        self,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        volumes: list[float],
    ) -> MomentumSignal:
        """
        Evaluate entry signals.

        Requirements:
        1. EMA9 crossed above EMA21 (within last 2 bars)
        2. Price above VWAP (if enabled)
        3. Hurst > threshold (trending regime)
        """
        signal = MomentumSignal()
        details: Dict[str, Any] = {}

        if len(closes) < max(self.ema_slow + 2, self.hurst_lookback):
            signal.details = {'error': 'insufficient_data'}
            return signal

        # Calculate indicators
        ema_fast = calculate_ema(closes, self.ema_fast)
        ema_slow = calculate_ema(closes, self.ema_slow)

        if len(ema_fast) < 3 or len(ema_slow) < 3:
            signal.details = {'error': 'ema_too_short'}
            return signal

        # Align EMAs (they start at different offsets)
        offset = self.ema_slow - self.ema_fast
        fast_aligned = ema_fast[offset:]

        if len(fast_aligned) < 3 or len(ema_slow) < 3:
            signal.details = {'error': 'alignment_issue'}
            return signal

        # Check 1: EMA crossover (fast crossed above slow within last 2 bars)
        cross_now = fast_aligned[-1] > ema_slow[-1]
        cross_prev = fast_aligned[-2] <= ema_slow[-2]
        cross_prev2 = fast_aligned[-3] <= ema_slow[-3] if len(fast_aligned) >= 3 else False

        ema_cross = cross_now and (cross_prev or cross_prev2)
        details['ema_cross'] = ema_cross
        details['ema_fast'] = round(fast_aligned[-1], 2)
        details['ema_slow'] = round(ema_slow[-1], 2)

        # Check 2: Price above VWAP
        if self.vwap_confirm:
            vwap = calculate_vwap(highs[-24:], lows[-24:], closes[-24:], volumes[-24:])
            above_vwap = closes[-1] > vwap
            details['vwap'] = round(vwap, 2)
            details['above_vwap'] = above_vwap
        else:
            above_vwap = True

        # Check 3: Hurst exponent (trending regime)
        hurst = compute_hurst_exponent(closes[-self.hurst_lookback:])
        trending = is_trending(hurst, self.hurst_min)
        details['hurst'] = round(hurst, 3)
        details['trending'] = trending

        # All three must be true
        signal.should_enter = ema_cross and above_vwap and trending
        signal.entry_direction = 'long'

        # Signal strength for logging (0-1)
        strength = 0.0
        if ema_cross:
            strength += 0.4
        if above_vwap:
            strength += 0.3
        if trending:
            strength += 0.3
        signal.signal_strength = strength
        signal.details = details

        return signal

    def calculate_stop(
        self,
        entry_price: float,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> float:
        """Calculate initial stop price: entry - (ATR * multiplier)."""
        atr = calculate_atr(highs, lows, closes, self.atr_period)
        if atr <= 0:
            # Fallback: 3% below entry
            return entry_price * 0.97
        stop = entry_price - (atr * self.stop_atr_mult)
        return max(stop, entry_price * 0.90)  # Never more than 10% below entry

    def calculate_trailing_stop(
        self,
        highest_high: float,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> float:
        """Calculate trailing stop: highest_high - (ATR * trail_mult)."""
        atr = calculate_atr(highs, lows, closes, self.atr_period)
        if atr <= 0:
            return highest_high * 0.97
        return highest_high - (atr * self.trail_atr_mult)

    def check_exit(
        self,
        position: MomentumPosition,
        current_price: float,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> MomentumSignal:
        """
        Check if position should be exited.

        Exit conditions:
        1. Price hit stop loss
        2. Price hit trailing stop
        3. Max hold duration exceeded
        """
        signal = MomentumSignal()

        # Update highest high
        new_high = max(position.highest_high, current_price)

        # Calculate trailing stop
        trailing = self.calculate_trailing_stop(new_high, highs, lows, closes)
        # Use the higher of initial stop and trailing stop
        effective_stop = max(position.stop_price, trailing)

        # Check stop
        if current_price <= effective_stop:
            signal.should_exit = True
            if current_price <= position.stop_price:
                signal.exit_reason = 'stop_loss'
            else:
                signal.exit_reason = 'trailing_stop'

        # Check max hold duration
        hold_hours = (time.time() - position.entry_time) / 3600
        if hold_hours > self.max_hold_days * 24:
            signal.should_exit = True
            signal.exit_reason = 'max_duration'

        signal.details = {
            'highest_high': round(new_high, 2),
            'trailing_stop': round(trailing, 2),
            'effective_stop': round(effective_stop, 2),
            'hold_hours': round(hold_hours, 1),
        }

        return signal

    def check_pyramid(
        self,
        position: MomentumPosition,
        current_price: float,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> bool:
        """Check if we should add to the position (pyramid)."""
        if not self.pyramid_enabled:
            return False
        if position.pyramid_count >= self.pyramid_max_adds:
            return False

        atr = calculate_atr(highs, lows, closes, self.atr_period)
        if atr <= 0:
            return False

        # Next pyramid level: entry + (pyramid_count + 1) * atr_spacing * ATR
        next_level = position.entry_price + (
            (position.pyramid_count + 1) * self.pyramid_atr_spacing * atr
        )

        return current_price >= next_level
