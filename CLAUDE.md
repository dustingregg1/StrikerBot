# StrikerBot Project Instructions
# Location: C:\Users\dusti\OneDrive\Desktop\StrikerBot\
# Owner: dustingregg1

## Project Overview
Momentum/breakout cryptocurrency trading bot. The aggressive counterpart to BitcoinBot (the goalkeeper).
Uses EMA crossovers, Hurst Exponent regime detection, and ATR-based risk management.

## Philosophy: Striker, Not Goalkeeper
- **Enter fast**: 1-2 signal confirmations, not 27
- **Risk more per trade**: 2% per trade, 8% daily drawdown tolerance
- **Win less, win bigger**: 35-45% win rate target, 3:1 reward/risk
- **Every dollar works**: 95% max deployment, no passive allocation
- **5 safety rules maximum**: Don't bolt goalkeeper logic onto this

## SECURITY - READ FIRST
Same rules as BitcoinBot. NEVER hardcode credentials. Use `os.environ.get()`.

## Architecture
```
StrikerBot/
    striker.py              # Main loop (~200 lines)
    config.py               # <200 lines
    strategies/
        momentum_rider.py   # Phase 1 MVP
        breakout_catcher.py # Phase 2
        volatility_surfer.py # Phase 2
    signals/
        hurst_regime.py     # Hurst Exponent (R/S analysis)
        rsi_divergence.py   # Phase 2
        volume_spike.py     # Phase 2
    risk/
        position_sizer.py   # ATR-based Half-Kelly
        portfolio_guard.py  # 5 rules ONLY
        trade_journal.py    # JSONL append-only
    tests/
    scripts/
```

## Tech Stack
- Python 3.11+, CCXT, pandas, numpy, ta
- pytest for testing
- flake8 for linting

## Key Differences from BitcoinBot
| Aspect | BitcoinBot (Goalkeeper) | StrikerBot |
|--------|------------------------|------------|
| Safety gates | 27 | 5 |
| Win rate target | 50%+ | 35-45% |
| Position sizing | Fixed USD | ATR-based Kelly |
| Cycle time | 5 minutes | 15-60 seconds |
| Strategy | Grid + DCA | Momentum + Breakout |

## Phase 1 Scope (MVP)
- Momentum Rider strategy ONLY
- Paper trading mode (no real orders)
- BTC/USD only
- Local execution (not PythonAnywhere yet)
- Trade journal logging all paper trades
