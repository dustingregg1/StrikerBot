#!/usr/bin/env python3
"""
StrikerBot - Momentum/Breakout Crypto Trading Bot
===================================================
The aggressive counterpart to BitcoinBot (the goalkeeper).

Phase 1 MVP: Paper trading, BTC/USD only, Momentum Rider strategy.
Target ~200 lines. Every dollar works.
"""

import json
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

import config as cfg
from strategies.momentum_rider import MomentumRider, MomentumPosition
from risk.portfolio_guard import PortfolioGuard
from risk.position_sizer import size_position, calculate_atr
from risk.trade_journal import TradeJournal, TradeRecord

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_FILE, mode='a'),
    ]
)
log = logging.getLogger('striker')

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state() -> Dict[str, Any]:
    """Load bot state from disk."""
    state_path = Path(cfg.STATE_FILE)
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except (json.JSONDecodeError, IOError):
            log.warning("State file corrupt, starting fresh")
    return {
        'cycle_count': 0,
        'positions': {},
        'guard_state': {},
        'equity_usd': cfg.INITIAL_CAPITAL_USD,
        'version': cfg.BOT_VERSION,
    }


def save_state(state: Dict[str, Any]) -> None:
    """Save bot state to disk."""
    Path(cfg.STATE_FILE).write_text(json.dumps(state, indent=2, default=str))


# =============================================================================
# MARKET DATA (Paper Trading Implementation)
# =============================================================================

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 200) -> list:
    """
    Fetch OHLCV candles from exchange.
    In paper trading mode, uses CCXT without auth.
    """
    try:
        import ccxt
        exchange = ccxt.kraken({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return ohlcv
    except Exception as e:
        log.error(f"Failed to fetch OHLCV: {e}")
        return []


def simulate_fill_price(price: float, side: str) -> float:
    """Apply slippage to paper trade fill price."""
    slippage = price * (cfg.SLIPPAGE_BPS / 10000)
    if side == 'buy':
        return price + slippage  # Buy slightly higher
    return price - slippage      # Sell slightly lower


def calculate_trade_fee(notional_usd: float, is_maker: bool = False) -> float:
    """Calculate simulated fee for a paper trade."""
    fee_pct = cfg.MAKER_FEE_PCT if is_maker else cfg.TAKER_FEE_PCT
    return notional_usd * (fee_pct / 100)


def extract_price_arrays(ohlcv: list) -> Dict[str, list]:
    """Extract price arrays from OHLCV data."""
    if not ohlcv:
        return {'opens': [], 'highs': [], 'lows': [], 'closes': [], 'volumes': []}
    return {
        'opens': [c[1] for c in ohlcv],
        'highs': [c[2] for c in ohlcv],
        'lows': [c[3] for c in ohlcv],
        'closes': [c[4] for c in ohlcv],
        'volumes': [c[5] for c in ohlcv],
    }


# =============================================================================
# MAIN CYCLE
# =============================================================================

def run_cycle(
    state: Dict[str, Any],
    strategy: MomentumRider,
    guard: PortfolioGuard,
    journal: TradeJournal,
) -> bool:
    """Run a single trading cycle. Returns True on success."""
    cycle_num = state['cycle_count'] + 1
    log.info(f"{'='*40}")
    log.info(f"Cycle #{cycle_num} | Equity: ${state['equity_usd']:.2f}")

    # Fetch market data
    ohlcv = fetch_ohlcv(cfg.SYMBOL, cfg.TIMEFRAME)
    if not ohlcv or len(ohlcv) < 50:
        log.warning("Insufficient market data, skipping cycle")
        return False

    prices = extract_price_arrays(ohlcv)
    current_price = prices['closes'][-1]
    log.info(f"{cfg.SYMBOL}: ${current_price:,.2f}")

    # Check existing positions for exits
    for trade_id, pos_data in list(state.get('positions', {}).items()):
        position = MomentumPosition(**pos_data)
        exit_signal = strategy.check_exit(
            position, current_price,
            prices['highs'], prices['lows'], prices['closes']
        )

        if exit_signal.should_exit:
            # Paper trade: simulate exit with slippage + fees
            fill_price = simulate_fill_price(current_price, 'sell')
            gross_pnl = (fill_price - position.entry_price) * position.quantity
            notional = fill_price * position.quantity
            exit_fee = calculate_trade_fee(notional, is_maker=False)
            # Entry fee was already deducted at entry time
            net_pnl = gross_pnl - exit_fee
            log.info(
                f"EXIT {position.symbol} @ ${fill_price:,.2f} (mid ${current_price:,.2f}) | "
                f"Reason: {exit_signal.exit_reason} | "
                f"Gross: ${gross_pnl:.2f} | Fee: ${exit_fee:.2f} | Net: ${net_pnl:.2f}"
            )

            journal.record_exit(
                trade_id=trade_id,
                exit_price=fill_price,
                exit_reason=exit_signal.exit_reason,
                fees_usd=exit_fee,
            )

            state['equity_usd'] += net_pnl
            guard.record_trade_result(net_pnl)
            del state['positions'][trade_id]
        else:
            # Update highest high for trailing stop
            position.highest_high = max(position.highest_high, current_price)
            state['positions'][trade_id] = position.__dict__

            # Check pyramid
            if strategy.check_pyramid(
                position, current_price,
                prices['highs'], prices['lows'], prices['closes']
            ):
                log.info(
                    f"PYRAMID #{position.pyramid_count + 1} for {position.symbol} "
                    f"@ ${current_price:,.2f}"
                )
                position.pyramid_count += 1
                position.pyramid_prices.append(current_price)
                state['positions'][trade_id] = position.__dict__

    # Evaluate new entry signals
    entry_signal = strategy.evaluate_entry(
        prices['closes'], prices['highs'], prices['lows'], prices['volumes']
    )

    if entry_signal.should_enter:
        # Check portfolio guard
        deployment_pct = sum(
            p.get('position_usd', 0) for p in state.get('positions', {}).values()
        ) / max(state['equity_usd'], 1) * 100
        guard.update_positions(len(state.get('positions', {})), deployment_pct)

        stop_price = strategy.calculate_stop(
            current_price, prices['highs'], prices['lows'], prices['closes']
        )

        # Size the position
        sizing = size_position(
            equity_usd=state['equity_usd'],
            entry_price=current_price,
            stop_price=stop_price,
            max_risk_pct=cfg.MAX_RISK_PER_TRADE_PCT,
        )

        risk_pct = sizing.get('risk_pct', 0.0)
        guard_check = guard.check_entry(risk_pct, state['equity_usd'])

        if guard_check.allowed and sizing['position_usd'] > 0:
            trade_id = str(uuid.uuid4())[:8]
            # Simulate entry fill with slippage + fee
            fill_price = simulate_fill_price(current_price, 'buy')
            entry_fee = calculate_trade_fee(sizing['position_usd'], is_maker=False)
            state['equity_usd'] -= entry_fee  # Deduct entry fee immediately
            log.info(
                f"ENTRY {cfg.SYMBOL} LONG @ ${fill_price:,.2f} (mid ${current_price:,.2f}) | "
                f"Size: ${sizing['position_usd']:.2f} | Fee: ${entry_fee:.2f} | "
                f"Stop: ${stop_price:,.2f} | "
                f"Risk: {risk_pct:.1f}% | "
                f"Signal: {entry_signal.details}"
            )

            # Record in journal
            trade = TradeRecord(
                trade_id=trade_id,
                symbol=cfg.SYMBOL,
                side='long',
                strategy='momentum_rider',
                entry_price=fill_price,
                entry_time=time.time(),
                entry_reason=str(entry_signal.details),
                position_usd=sizing['position_usd'],
                quantity=sizing['quantity'],
                stop_price=stop_price,
                hurst_at_entry=entry_signal.details.get('hurst', 0),
                atr_at_entry=calculate_atr(
                    prices['highs'], prices['lows'], prices['closes']
                ),
                fees_usd=entry_fee,
                paper_trade=cfg.PAPER_TRADING,
            )
            journal.record_entry(trade)

            # Store position in state
            state['positions'][trade_id] = MomentumPosition(
                trade_id=trade_id,
                symbol=cfg.SYMBOL,
                entry_price=fill_price,
                entry_time=time.time(),
                quantity=sizing['quantity'],
                position_usd=sizing['position_usd'],
                stop_price=stop_price,
                highest_high=fill_price,
                trailing_stop=stop_price,
            ).__dict__
        elif not guard_check.allowed:
            log.info(f"Entry blocked by guard: {guard_check.reason}")
    else:
        log.debug(f"No entry signal: {entry_signal.details}")

    state['cycle_count'] = cycle_num
    state['guard_state'] = guard.to_dict()
    return True


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main entry point."""
    log.info(f"StrikerBot v{cfg.BOT_VERSION} starting")
    log.info(f"Mode: {'PAPER' if cfg.PAPER_TRADING else 'LIVE'}")
    log.info(f"Symbol: {cfg.SYMBOL} | Timeframe: {cfg.TIMEFRAME}")
    log.info(f"Capital: ${cfg.INITIAL_CAPITAL_USD:.2f}")

    state = load_state()
    strategy = MomentumRider(cfg.MOMENTUM_RIDER)
    guard = PortfolioGuard(
        max_risk_per_trade_pct=cfg.MAX_RISK_PER_TRADE_PCT,
        daily_loss_limit_usd=cfg.DAILY_LOSS_LIMIT_USD,
        max_concurrent_positions=cfg.MAX_CONCURRENT_POSITIONS,
        max_deployment_pct=cfg.MAX_DEPLOYMENT_PCT,
        consecutive_loss_pause_count=cfg.CONSECUTIVE_LOSS_PAUSE_COUNT,
        consecutive_loss_pause_hours=cfg.CONSECUTIVE_LOSS_PAUSE_HOURS,
    )

    # Restore guard state if available
    if state.get('guard_state'):
        guard = PortfolioGuard.from_dict(
            state['guard_state'],
            max_risk_per_trade_pct=cfg.MAX_RISK_PER_TRADE_PCT,
            daily_loss_limit_usd=cfg.DAILY_LOSS_LIMIT_USD,
            max_concurrent_positions=cfg.MAX_CONCURRENT_POSITIONS,
            max_deployment_pct=cfg.MAX_DEPLOYMENT_PCT,
            consecutive_loss_pause_count=cfg.CONSECUTIVE_LOSS_PAUSE_COUNT,
            consecutive_loss_pause_hours=cfg.CONSECUTIVE_LOSS_PAUSE_HOURS,
        )

    journal = TradeJournal(cfg.JOURNAL_FILE)

    consecutive_errors = 0

    while True:
        try:
            success = run_cycle(state, strategy, guard, journal)
            if success:
                consecutive_errors = 0
                save_state(state)
            else:
                consecutive_errors += 1
        except KeyboardInterrupt:
            log.info("Shutting down gracefully...")
            save_state(state)
            # Print scorecard on exit
            print("\n" + journal.format_scorecard())
            break
        except Exception as e:
            log.error(f"Cycle error: {e}", exc_info=True)
            consecutive_errors += 1

        if consecutive_errors >= 5:
            log.error("Too many consecutive errors, exiting")
            save_state(state)
            break

        log.debug(f"Sleeping {cfg.CYCLE_INTERVAL_SECONDS}s...")
        time.sleep(cfg.CYCLE_INTERVAL_SECONDS)


if __name__ == '__main__':
    main()
