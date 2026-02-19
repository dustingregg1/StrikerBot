"""
StrikerBot Configuration
=========================
Lean config (~150 lines, not 1100). Every dollar works.
Phase 1: Paper trading, BTC/USD only, Momentum Rider strategy.
"""

import os
from typing import Dict, Any

# =============================================================================
# VERSION
# =============================================================================
BOT_VERSION = "0.1.0"

# =============================================================================
# MODE
# =============================================================================
PAPER_TRADING = True  # Phase 1: ALWAYS paper trade
DRY_RUN = True  # Alias for compatibility
LOG_LEVEL = "INFO"

# =============================================================================
# EXCHANGE
# =============================================================================
EXCHANGE = "kraken"
SYMBOL = "BTC/USD"
TIMEFRAME = "1h"  # Primary timeframe for signals
TIMEFRAME_FAST = "15m"  # Fast timeframe for entry timing

# =============================================================================
# CAPITAL
# =============================================================================
INITIAL_CAPITAL_USD = 500.0  # Phase 1: $500 paper trading
MAX_DEPLOYMENT_PCT = 95.0  # Keep 5% reserve for fees/slippage
MIN_EQUITY_USD = 100.0  # Circuit breaker: stop if equity < $100

# =============================================================================
# PORTFOLIO GUARD - THE ONLY SAFETY LAYER (5 rules)
# =============================================================================
MAX_RISK_PER_TRADE_PCT = 2.0  # Never risk >2% on single trade
DAILY_LOSS_LIMIT_USD = 50.0  # Stop NEW entries if -$50 in a day (10%)
MAX_CONCURRENT_POSITIONS = 3  # Max 3 open positions
CONSECUTIVE_LOSS_PAUSE_COUNT = 3  # Pause after 3 consecutive losses
CONSECUTIVE_LOSS_PAUSE_HOURS = 4  # Cool down for 4 hours

# =============================================================================
# MOMENTUM RIDER STRATEGY (Phase 1 MVP)
# =============================================================================
MOMENTUM_RIDER = {
    'enabled': True,
    # Entry signals
    'ema_fast_period': 9,
    'ema_slow_period': 21,
    'vwap_confirm': True,  # Price must be above VWAP
    'hurst_min': 0.55,  # Hurst > 0.55 = trending regime
    'hurst_lookback': 100,  # R/S analysis window
    # Stop loss
    'atr_period': 14,
    'stop_atr_multiplier': 2.0,  # 2x ATR below entry
    # Trailing exit
    'trail_atr_multiplier': 1.5,  # Trail at 1.5x ATR from highest high
    # Pyramiding
    'pyramid_enabled': True,
    'pyramid_max_adds': 3,  # Up to 3 additional entries
    'pyramid_atr_spacing': 1.0,  # Add at +1, +2, +3 ATR above entry
    # Time limit
    'max_hold_days': 7,
}

# =============================================================================
# BREAKOUT CATCHER STRATEGY (Phase 2 - STUB)
# =============================================================================
BREAKOUT_CATCHER = {
    'enabled': False,  # Phase 2
    'bb_period': 20,
    'bb_std': 2.0,
    'volume_spike_multiplier': 1.5,
    'stop_atr_multiplier': 1.0,
    'min_reward_risk': 3.0,
}

# =============================================================================
# VOLATILITY SURFER STRATEGY (Phase 2 - STUB)
# =============================================================================
VOLATILITY_SURFER = {
    'enabled': False,  # Phase 2
    'atr_spike_threshold': 1.5,  # ATR > 1.5x 30-day avg
    'stop_atr_multiplier': 3.0,
    'trail_atr_multiplier': 2.0,
    'position_size_reduction': 0.5,  # Half normal size
}

# =============================================================================
# POSITION SIZING
# =============================================================================
POSITION_SIZING = {
    'method': 'atr_kelly',  # ATR-based Half-Kelly
    'kelly_fraction': 0.5,  # Half-Kelly (conservative)
    'atr_period': 14,
    'min_position_usd': 10.0,
    'max_position_pct': 30.0,  # Never >30% in single position
}

# =============================================================================
# DEAD MAN'S SWITCH
# =============================================================================
DEAD_MAN_SWITCH_ENABLED = True
DEAD_MAN_SWITCH_TIMEOUT_SECONDS = 120  # 2x cycle for striker (faster cycles)

# =============================================================================
# FEES & SLIPPAGE (Paper Trading Simulation)
# Verified via Kraken /private/TradeVolume API, 2026-02-18
# =============================================================================
MAKER_FEE_PCT = 0.25  # $0-volume tier
TAKER_FEE_PCT = 0.40  # $0-volume tier
SLIPPAGE_BPS = 10     # 10 basis points simulated slippage
ASSUME_TAKER = True   # Momentum entries are usually taker (crossing spread)

# =============================================================================
# CYCLE TIMING
# =============================================================================
CYCLE_INTERVAL_SECONDS = 60  # 1 minute cycles (striker is fast)
SIGNAL_CHECK_INTERVAL_SECONDS = 15  # Check for entries every 15s

# =============================================================================
# TRADE JOURNAL
# =============================================================================
JOURNAL_FILE = "trade_journal.jsonl"
STATE_FILE = "state.json"
LOG_FILE = "striker.log"

# =============================================================================
# NOTIFICATIONS (Phase 2)
# =============================================================================
TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# =============================================================================
# CREDENTIALS (from environment, never hardcoded)
# =============================================================================
def get_exchange_credentials() -> Dict[str, str]:
    """Get exchange credentials from environment variables."""
    api_key = os.environ.get('KRAKEN_API_KEY', '')
    api_secret = os.environ.get('KRAKEN_API_SECRET', '')
    if not PAPER_TRADING and (not api_key or not api_secret):
        raise EnvironmentError(
            "Kraken credentials required for live trading. "
            "Set KRAKEN_API_KEY and KRAKEN_API_SECRET."
        )
    return {'apiKey': api_key, 'secret': api_secret}


# =============================================================================
# WEEKLY SCORECARD TARGETS
# =============================================================================
TARGETS = {
    'win_rate_min': 0.35,
    'win_rate_max': 0.45,
    'win_loss_ratio_min': 2.5,
    'profit_factor_min': 1.5,
    'fee_drag_max_pct': 20.0,
    'max_drawdown_pct': 20.0,
    'sharpe_min': 1.0,
}
