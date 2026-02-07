"""
ATR-Based Half-Kelly Position Sizer
=====================================
Sizes positions based on:
1. ATR (Average True Range) for stop distance
2. Half-Kelly criterion for optimal fraction
3. Max risk per trade (hard limit from PortfolioGuard)
"""

import numpy as np
from typing import Optional


def calculate_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> float:
    """
    Calculate Average True Range.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ATR lookback period

    Returns:
        Current ATR value
    """
    if len(highs) < period + 1:
        # Not enough data, use simple range
        if highs and lows:
            return float(np.mean(np.array(highs[-period:]) - np.array(lows[-period:])))
        return 0.0

    h = np.array(highs, dtype=np.float64)
    l = np.array(lows, dtype=np.float64)
    c = np.array(closes, dtype=np.float64)

    # True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    tr1 = h[1:] - l[1:]
    tr2 = np.abs(h[1:] - c[:-1])
    tr3 = np.abs(l[1:] - c[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # Exponential moving average of TR
    atr_values = [float(np.mean(tr[:period]))]
    for i in range(period, len(tr)):
        atr_values.append(
            (atr_values[-1] * (period - 1) + tr[i]) / period
        )

    return float(atr_values[-1]) if atr_values else 0.0


def half_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate Half-Kelly fraction.

    Full Kelly: f = (p * b - q) / b
    where p = win_rate, q = 1-p, b = avg_win/avg_loss

    Half-Kelly = Kelly / 2 (recommended for crypto volatility)

    Returns:
        Fraction of capital to risk (0.0 to 0.5)
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.01  # Minimum fraction

    p = win_rate
    q = 1.0 - p
    b = avg_win / avg_loss

    kelly = (p * b - q) / b
    kelly = max(kelly, 0.0)  # Never negative

    # Half-Kelly, capped at 50%
    return min(kelly / 2.0, 0.50)


def size_position(
    equity_usd: float,
    entry_price: float,
    stop_price: float,
    max_risk_pct: float = 2.0,
    win_rate: float = 0.40,
    avg_win_loss_ratio: float = 3.0,
    min_position_usd: float = 10.0,
    max_position_pct: float = 30.0,
) -> dict:
    """
    Calculate position size using ATR-based Half-Kelly.

    Args:
        equity_usd: Current account equity
        entry_price: Planned entry price
        stop_price: Stop loss price
        max_risk_pct: Maximum risk per trade (from PortfolioGuard)
        win_rate: Historical win rate (default 40%)
        avg_win_loss_ratio: Average win / average loss ratio
        min_position_usd: Minimum position size
        max_position_pct: Maximum position as % of equity

    Returns:
        Dict with position_usd, quantity, risk_usd, risk_pct
    """
    if entry_price <= 0 or stop_price <= 0 or equity_usd <= 0:
        return {
            'position_usd': 0.0,
            'quantity': 0.0,
            'risk_usd': 0.0,
            'risk_pct': 0.0,
            'reason': 'invalid_inputs',
        }

    # Distance to stop as percentage
    stop_distance_pct = abs(entry_price - stop_price) / entry_price * 100
    if stop_distance_pct < 0.1:
        return {
            'position_usd': 0.0,
            'quantity': 0.0,
            'risk_usd': 0.0,
            'risk_pct': 0.0,
            'reason': 'stop_too_tight',
        }

    # Half-Kelly determines what fraction of equity to risk
    kelly_fraction = half_kelly(
        win_rate=win_rate,
        avg_win=avg_win_loss_ratio,  # Normalized to 1 loss
        avg_loss=1.0,
    )

    # Risk in USD = min(Kelly fraction, max_risk_pct) * equity
    risk_pct = min(kelly_fraction * 100, max_risk_pct)
    risk_usd = equity_usd * (risk_pct / 100.0)

    # Position size = risk / stop_distance
    position_usd = risk_usd / (stop_distance_pct / 100.0)

    # Apply max position cap
    max_position_usd = equity_usd * (max_position_pct / 100.0)
    position_usd = min(position_usd, max_position_usd)

    # Apply minimum
    if position_usd < min_position_usd:
        return {
            'position_usd': 0.0,
            'quantity': 0.0,
            'risk_usd': 0.0,
            'risk_pct': 0.0,
            'reason': f'below_minimum_{min_position_usd}',
        }

    quantity = position_usd / entry_price

    return {
        'position_usd': round(position_usd, 2),
        'quantity': round(quantity, 8),
        'risk_usd': round(risk_usd, 2),
        'risk_pct': round(risk_pct, 2),
        'stop_distance_pct': round(stop_distance_pct, 2),
        'kelly_fraction': round(kelly_fraction, 4),
        'reason': 'sized',
    }
