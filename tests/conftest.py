"""Shared fixtures for StrikerBot tests."""

import pytest
from typing import Dict, Any


@pytest.fixture
def sample_ohlcv_data() -> list[list[float]]:
    """Generate sample OHLCV data for testing strategies."""
    # 50 candles of fake 1h BTC data trending upward
    base_price = 95000.0
    data = []
    for i in range(50):
        drift = i * 50  # Upward trend
        noise = (i % 7 - 3) * 100  # Some noise
        o = base_price + drift + noise
        h = o + abs(noise) + 200
        l = o - abs(noise) - 200
        c = o + (50 if i % 3 else -30)
        v = 100 + (i % 10) * 20
        data.append([1700000000 + i * 3600, o, h, l, c, v])
    return data


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Minimal StrikerBot config for testing."""
    return {
        'paper_trading': True,
        'exchange': 'kraken',
        'symbol': 'BTC/USD',
        'timeframe': '1h',
        'capital_usd': 500.0,
        'max_risk_per_trade_pct': 2.0,
        'daily_loss_limit_usd': 50.0,
        'max_concurrent_positions': 3,
        'max_deployment_pct': 95.0,
        'consecutive_loss_pause_count': 3,
        'consecutive_loss_pause_hours': 4,
    }
