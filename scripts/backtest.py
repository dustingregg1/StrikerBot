#!/usr/bin/env python3
"""
StrikerBot Backtester
======================
Historical backtest of Momentum Rider strategy against OHLCV data.
Reuses all existing components (strategy, guard, journal, sizer).

Execution model (per bar):
  Phase 1 (bar open):   Fill pending orders from previous bar
  Phase 2 (intrabar):   Check hard stops against high/low
  Phase 3 (bar close):  Evaluate indicators, schedule new orders
  Phase 4 (after close): Record mark-to-market equity
"""

import sys
import os
import csv
import time
import uuid
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from strategies.momentum_rider import (
    MomentumRider, MomentumPosition, MomentumSignal,
)
from risk.portfolio_guard import PortfolioGuard
from risk.position_sizer import size_position, calculate_atr
from risk.trade_journal import TradeJournal, TradeRecord

# Reuse pure functions from striker.py
from striker import simulate_fill_price, calculate_trade_fee, extract_price_arrays

log = logging.getLogger('backtest')

# Timeframe durations in milliseconds and seconds
TIMEFRAME_MS = {
    '1m': 60_000, '5m': 300_000, '15m': 900_000, '30m': 1_800_000,
    '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000,
}
TIMEFRAME_SEC = {k: v // 1000 for k, v in TIMEFRAME_MS.items()}


# =============================================================================
# DATA FETCHING & CACHING
# =============================================================================

def fetch_historical_data(
    exchange_id: str, symbol: str, timeframe: str,
    since_ms: int, until_ms: int,
) -> List[list]:
    """Fetch OHLCV data with pagination. Returns list of [ts, o, h, l, c, v]."""
    import ccxt
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f"Unknown exchange: {exchange_id}")

    exchange = exchange_cls({'enableRateLimit': True})

    if not exchange.has.get('fetchOHLCV'):
        raise ValueError(
            f"{exchange_id} doesn't support fetchOHLCV. "
            f"Try: binance, coinbase, bitstamp"
        )

    exchange.load_markets()
    if symbol not in exchange.symbols:
        raise ValueError(
            f"{symbol} not found on {exchange_id}. "
            f"Available BTC pairs: {[s for s in exchange.symbols if 'BTC' in s][:10]}"
        )

    tf_ms = TIMEFRAME_MS.get(timeframe)
    if tf_ms is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    all_ohlcv = []
    current_since = since_ms

    while current_since < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
        if not batch:
            break

        all_ohlcv.extend(batch)
        last_ts = batch[-1][0]

        # Advance past last bar
        current_since = last_ts + tf_ms

        # Rate limit
        time.sleep(exchange.rateLimit / 1000)

        log.info(
            f"  Fetched {len(batch)} bars, total {len(all_ohlcv)}, "
            f"last: {datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}"
        )

        if last_ts >= until_ms:
            break

    # Deduplicate by timestamp
    seen = set()
    result = []
    for bar in all_ohlcv:
        if bar[0] not in seen:
            seen.add(bar[0])
            result.append(bar)

    result.sort(key=lambda x: x[0])

    # Filter to requested range
    result = [b for b in result if since_ms <= b[0] <= until_ms]

    # Drop incomplete last candle if it's in-progress
    now_ms = int(time.time() * 1000)
    if result and (now_ms - result[-1][0] < tf_ms):
        result.pop()

    return result


def cache_data(ohlcv: List[list], filepath: Path) -> None:
    """Save OHLCV to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for bar in ohlcv:
            writer.writerow(bar)


def load_cached(filepath: Path) -> Optional[List[list]]:
    """Load OHLCV from CSV. Returns None if file doesn't exist."""
    if not filepath.exists():
        return None
    result = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            result.append([
                int(float(row[0])),
                float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), float(row[5]),
            ])
    return result


def get_cache_path(cache_dir: str, exchange_id: str, symbol: str,
                   timeframe: str, since_ms: int, until_ms: int) -> Path:
    """Generate cache file path."""
    sym = symbol.replace('/', '_')
    start = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).strftime('%Y%m%d')
    end = datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc).strftime('%Y%m%d')
    return Path(cache_dir) / f"{exchange_id}_{sym}_{timeframe}_{start}_{end}.csv"


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

@dataclass
class BacktestPosition:
    """Augmented position for backtest tracking."""
    trade_id: str
    symbol: str
    entry_price: float
    entry_time: float  # seconds
    quantity: float
    position_usd: float
    stop_price: float
    highest_high: float
    trailing_stop: float  # Active trailing stop (used in Phase 2)
    trailing_stop_pending: float = 0.0  # Pending (promoted to active next bar)
    pyramid_count: int = 0
    pyramid_prices: list = field(default_factory=list)
    entry_fee: float = 0.0  # Saved for round-trip fee reporting


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    equity_curve: List[Tuple[float, float]]
    initial_capital: float
    final_equity: float
    total_bars: int
    traded_bars: int
    max_drawdown_pct: float
    sharpe_ratio: float
    scorecard: Dict[str, Any]
    scorecard_regular: Dict[str, Any]  # Excluding backtest_end trades


class BacktestEngine:
    """Bar-by-bar backtest engine mirroring striker.py run_cycle."""

    def __init__(
        self,
        ohlcv: List[list],
        initial_capital: float = 500.0,
        timeframe: str = '1h',
        start_ts: float = 0.0,
        journal_path: str = 'data/backtest_journal.jsonl',
    ):
        if not ohlcv:
            raise ValueError("Empty OHLCV data")

        self.ohlcv = ohlcv
        self.initial_capital = initial_capital
        self.timeframe = timeframe
        self.timeframe_sec = TIMEFRAME_SEC.get(timeframe, 3600)
        self.start_ts = start_ts  # Trading gate (seconds)
        self.journal_path = journal_path

        # Initialize components
        self.strategy = MomentumRider(cfg.MOMENTUM_RIDER)
        self.guard = PortfolioGuard(
            max_risk_per_trade_pct=cfg.MAX_RISK_PER_TRADE_PCT,
            daily_loss_limit_usd=cfg.DAILY_LOSS_LIMIT_USD,
            max_concurrent_positions=cfg.MAX_CONCURRENT_POSITIONS,
            max_deployment_pct=cfg.MAX_DEPLOYMENT_PCT,
            consecutive_loss_pause_count=cfg.CONSECUTIVE_LOSS_PAUSE_COUNT,
            consecutive_loss_pause_hours=cfg.CONSECUTIVE_LOSS_PAUSE_HOURS,
        )

        # Delete old backtest journal if exists
        journal_file = Path(journal_path)
        if journal_file.exists():
            journal_file.unlink()
        journal_file.parent.mkdir(parents=True, exist_ok=True)
        self.journal = TradeJournal(journal_path)

        # Derived warmup (must have 720 bars for EMA burn-in parity with live)
        self.warmup = max(
            719,
            cfg.MOMENTUM_RIDER.get('hurst_lookback', 100),
            cfg.MOMENTUM_RIDER.get('ema_slow_period', 21) + 2,
            cfg.MOMENTUM_RIDER.get('atr_period', 14) + 1,
        )

    def run(self) -> BacktestResult:
        """Run the backtest. Returns BacktestResult."""
        state = {
            'equity_usd': self.initial_capital,
            'positions': {},  # trade_id -> BacktestPosition
        }
        equity_curve: List[Tuple[float, float]] = []

        # Pending signals from previous bar
        pending_entry = None  # (signal_details, atr_at_signal)
        pending_exits: Dict[str, MomentumSignal] = {}  # trade_id -> exit signal

        total_bars = len(self.ohlcv)

        # Find start index by timestamp (not array index)
        start_idx = self.warmup
        if self.start_ts > 0:
            for i, bar in enumerate(self.ohlcv):
                if bar[0] / 1000.0 >= self.start_ts:
                    start_idx = max(i, self.warmup)
                    break

        traded_bars = 0

        for i in range(start_idx, total_bars):
            bar = self.ohlcv[i]
            open_ts = bar[0] / 1000.0  # Convert ms -> seconds
            bar_open = bar[1]
            bar_high = bar[2]
            bar_low = bar[3]
            bar_close = bar[4]
            close_ts = open_ts + self.timeframe_sec

            is_last_bar = (i == total_bars - 1)
            is_in_trading_window = (open_ts >= self.start_ts) if self.start_ts > 0 else True
            traded_bars += 1

            # Build 720-bar rolling window for indicators
            window_start = max(0, i - 719)
            window = self.ohlcv[window_start: i + 1]
            prices = extract_price_arrays(window)

            # ================================================================
            # PHASE 1 — AT BAR OPEN: Fill pending orders
            # ================================================================
            if is_in_trading_window:
                # Phase 1a: Fill pending strategy exits
                for trade_id, exit_signal in list(pending_exits.items()):
                    pos = state['positions'].get(trade_id)
                    if pos is None:
                        continue  # Already closed by stop

                    fill_price = simulate_fill_price(bar_open, 'sell')
                    self._execute_exit(
                        state, pos, fill_price, exit_signal.exit_reason, open_ts
                    )

                pending_exits.clear()

                # Phase 1b: Fill pending entry
                if pending_entry is not None:
                    signal_data, atr_at_signal = pending_entry
                    pending_entry = None

                    # Recompute stop using ATR from signal bar (not current bar)
                    stop_price = bar_open - (atr_at_signal * self.strategy.stop_atr_mult)
                    stop_price = max(stop_price, bar_open * 0.90)

                    # Gap-down invalidation: skip if fill would be at/below stop
                    if bar_open > stop_price:
                        fill_price = simulate_fill_price(bar_open, 'buy')

                        # Compute MTM equity for guard check
                        mtm_equity = self._compute_mtm(state, bar_open)

                        # Size position at fill time (gap protection)
                        sizing = size_position(
                            equity_usd=mtm_equity,
                            entry_price=fill_price,
                            stop_price=stop_price,
                            max_risk_pct=cfg.MAX_RISK_PER_TRADE_PCT,
                        )

                        if sizing['position_usd'] > 0:
                            # Update deployment for guard
                            deployment_pct = sum(
                                p.position_usd for p in state['positions'].values()
                            ) / max(mtm_equity, 1) * 100
                            self.guard.update_positions(
                                len(state['positions']), deployment_pct
                            )

                            risk_pct = sizing.get('risk_pct', 0.0)
                            guard_check = self.guard.check_entry(
                                risk_pct, mtm_equity, now=open_ts
                            )

                            if guard_check.allowed:
                                self._execute_entry(
                                    state, fill_price, stop_price,
                                    sizing, open_ts, signal_data
                                )

            # ================================================================
            # PHASE 2 — DURING BAR: Check hard stops (intrabar)
            # ================================================================
            for trade_id in list(state['positions'].keys()):
                pos = state['positions'].get(trade_id)
                if pos is None:
                    continue

                # Use effective stop: max of initial stop and active trailing stop
                effective_stop = max(pos.stop_price, pos.trailing_stop or 0)

                if bar_low <= effective_stop:
                    # Stop triggered — fill at min(effective_stop, bar_open)
                    trigger_price = min(effective_stop, bar_open)
                    fill_price = simulate_fill_price(trigger_price, 'sell')

                    if fill_price <= pos.entry_price:
                        exit_reason = 'stop_loss'
                    else:
                        exit_reason = 'trailing_stop'

                    self._execute_exit(
                        state, pos, fill_price, exit_reason, open_ts
                    )

                    # Remove from pending exits if it was scheduled
                    pending_exits.pop(trade_id, None)
                else:
                    # Surviving: update highest_high and compute pending trailing stop
                    pos.highest_high = max(pos.highest_high, bar_high)

                    # Compute new trailing stop from updated highest_high
                    new_trail = self.strategy.calculate_trailing_stop(
                        pos.highest_high,
                        prices['highs'], prices['lows'], prices['closes']
                    )
                    # Pending: will be promoted to active at next bar
                    pos.trailing_stop_pending = max(new_trail, pos.trailing_stop_pending)

            # Promote pending trailing stops to active (for next bar's Phase 2)
            for pos in state['positions'].values():
                if pos.trailing_stop_pending > 0:
                    pos.trailing_stop = max(pos.trailing_stop, pos.trailing_stop_pending)
                    pos.trailing_stop_pending = 0.0

            # ================================================================
            # PHASE 3 — AT BAR CLOSE: Evaluate signals (skip final bar)
            # ================================================================
            if not is_last_bar and is_in_trading_window:
                # Evaluate strategy exits on remaining positions
                for trade_id, pos in list(state['positions'].items()):
                    exit_signal = self.strategy.check_exit(
                        MomentumPosition(
                            trade_id=pos.trade_id,
                            symbol=pos.symbol,
                            entry_price=pos.entry_price,
                            entry_time=pos.entry_time,
                            quantity=pos.quantity,
                            position_usd=pos.position_usd,
                            stop_price=pos.stop_price,
                            highest_high=pos.highest_high,
                            trailing_stop=pos.trailing_stop,
                            pyramid_count=pos.pyramid_count,
                            pyramid_prices=pos.pyramid_prices,
                        ),
                        bar_close,
                        prices['highs'], prices['lows'], prices['closes'],
                        now=close_ts,
                    )
                    if exit_signal.should_exit:
                        pending_exits[trade_id] = exit_signal

                # Evaluate new entry (only if no pending entry/exits)
                if pending_entry is None and not pending_exits:
                    entry_signal = self.strategy.evaluate_entry(
                        prices['closes'], prices['highs'],
                        prices['lows'], prices['volumes']
                    )
                    if entry_signal.should_enter:
                        # Store signal + ATR snapshot for fill-time sizing
                        atr_at_signal = calculate_atr(
                            prices['highs'], prices['lows'], prices['closes']
                        )
                        pending_entry = (entry_signal.details, atr_at_signal)

            # ================================================================
            # PHASE 4 — Record mark-to-market equity
            # ================================================================
            mtm = self._compute_mtm(state, bar_close)
            equity_curve.append((close_ts, mtm))

        # END: Force-close remaining positions at final bar close
        if self.ohlcv:
            last_bar = self.ohlcv[-1]
            last_close = last_bar[4]
            last_close_ts = last_bar[0] / 1000.0 + self.timeframe_sec

            for trade_id in list(state['positions'].keys()):
                pos = state['positions'].get(trade_id)
                if pos is None:
                    continue
                fill_price = simulate_fill_price(last_close, 'sell')
                self._execute_exit(
                    state, pos, fill_price, 'backtest_end', last_close_ts
                )

            # Overwrite final equity point with post-close settled equity
            if equity_curve:
                equity_curve[-1] = (last_close_ts, state['equity_usd'])

        # Compute metrics
        max_dd = compute_max_drawdown(equity_curve) if equity_curve else 0.0
        sharpe = compute_sharpe(
            equity_curve, annual_rfr=0.05,
            timeframe_sec=self.timeframe_sec,
        ) if equity_curve else 0.0

        final_close_ts = equity_curve[-1][0] if equity_curve else 0.0

        # Section 1: All trades
        scorecard = self.journal.generate_scorecard(
            days=99999, as_of=final_close_ts
        )

        # Section 2: Regular trades only (exclude backtest_end)
        original_trades = self.journal._trades.copy()
        self.journal._trades = [
            t for t in original_trades if t.exit_reason != 'backtest_end'
        ]
        scorecard_regular = self.journal.generate_scorecard(
            days=99999, as_of=final_close_ts
        )
        self.journal._trades = original_trades

        return BacktestResult(
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            final_equity=state['equity_usd'],
            total_bars=total_bars,
            traded_bars=traded_bars,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            scorecard=scorecard,
            scorecard_regular=scorecard_regular,
        )

    def _compute_mtm(self, state: Dict, current_price: float) -> float:
        """Mark-to-market equity: settled equity + unrealized PnL."""
        unrealized = sum(
            pos.quantity * (current_price - pos.entry_price)
            for pos in state['positions'].values()
        )
        return state['equity_usd'] + unrealized

    def _execute_entry(
        self, state: Dict, fill_price: float, stop_price: float,
        sizing: Dict, bar_ts: float, signal_details: Dict,
    ) -> None:
        """Execute entry at fill price."""
        trade_id = str(uuid.uuid4())[:8]
        entry_fee = calculate_trade_fee(
            sizing['quantity'] * fill_price, is_maker=False
        )
        state['equity_usd'] -= entry_fee

        # Create journal record
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=cfg.SYMBOL,
            side='long',
            strategy='momentum_rider',
            entry_price=fill_price,
            entry_time=bar_ts,
            entry_reason=str(signal_details),
            position_usd=sizing['position_usd'],
            quantity=sizing['quantity'],
            stop_price=stop_price,
            hurst_at_entry=signal_details.get('hurst', 0),
            atr_at_entry=calculate_atr([], [], []),  # Not critical for journal
            fees_usd=entry_fee,
            paper_trade=True,
        )
        self.journal.record_entry(trade)

        # Create backtest position
        pos = BacktestPosition(
            trade_id=trade_id,
            symbol=cfg.SYMBOL,
            entry_price=fill_price,
            entry_time=bar_ts,
            quantity=sizing['quantity'],
            position_usd=sizing['position_usd'],
            stop_price=stop_price,
            highest_high=fill_price,
            trailing_stop=stop_price,
            entry_fee=entry_fee,
        )
        state['positions'][trade_id] = pos

        log.debug(
            f"ENTRY {cfg.SYMBOL} @ ${fill_price:,.2f} | "
            f"Size: ${sizing['position_usd']:.2f} | Fee: ${entry_fee:.2f}"
        )

    def _execute_exit(
        self, state: Dict, pos: BacktestPosition,
        fill_price: float, exit_reason: str, bar_ts: float,
    ) -> None:
        """Execute exit at fill price."""
        gross_pnl = (fill_price - pos.entry_price) * pos.quantity
        notional = fill_price * pos.quantity
        exit_fee = calculate_trade_fee(notional, is_maker=False)
        net_pnl = gross_pnl - exit_fee

        # Settle into equity
        state['equity_usd'] += net_pnl

        # Record in journal with round-trip fees
        self.journal.record_exit(
            trade_id=pos.trade_id,
            exit_price=fill_price,
            exit_reason=exit_reason,
            fees_usd=pos.entry_fee + exit_fee,
            now=bar_ts,
        )

        # Update guard
        self.guard.record_trade_result(net_pnl, now=bar_ts)

        # Remove position
        state['positions'].pop(pos.trade_id, None)

        log.debug(
            f"EXIT {pos.symbol} @ ${fill_price:,.2f} | "
            f"Reason: {exit_reason} | Net: ${net_pnl:.2f}"
        )


# =============================================================================
# METRICS
# =============================================================================

def compute_max_drawdown(equity_curve: List[Tuple[float, float]]) -> float:
    """Maximum drawdown as a percentage."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0][1]
    max_dd = 0.0
    for _, equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
    return max_dd


def compute_sharpe(
    equity_curve: List[Tuple[float, float]],
    annual_rfr: float = 0.05,
    timeframe_sec: int = 3600,
) -> float:
    """Annualized Sharpe ratio from equity curve."""
    if len(equity_curve) < 2:
        return 0.0

    equities = [e[1] for e in equity_curve]
    returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

    if len(returns) < 2:
        return 0.0

    bars_per_year = (365 * 24 * 3600) / timeframe_sec
    bar_rfr = (1 + annual_rfr) ** (1 / bars_per_year) - 1

    excess = [r - bar_rfr for r in returns]
    stdev = float(np.std(excess, ddof=1))

    if stdev == 0:
        return 0.0

    return float(np.sqrt(bars_per_year) * np.mean(excess) / stdev)


# =============================================================================
# REPORT
# =============================================================================

def format_report(result: BacktestResult, targets: Dict) -> str:
    """Format backtest results as human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("STRIKERBOT BACKTEST REPORT")
    lines.append("=" * 60)

    # Equity summary
    pnl = result.final_equity - result.initial_capital
    pnl_pct = (pnl / result.initial_capital * 100) if result.initial_capital > 0 else 0
    lines.append(f"\nCapital:  ${result.initial_capital:.2f} -> ${result.final_equity:.2f}")
    lines.append(f"PnL:      ${pnl:.2f} ({pnl_pct:+.1f}%)")
    lines.append(f"Bars:     {result.total_bars} total, {result.traded_bars} traded")
    lines.append(f"Max DD:   {result.max_drawdown_pct:.1f}%  (target: <{targets.get('max_drawdown_pct', 20)}%)")
    lines.append(f"Sharpe:   {result.sharpe_ratio:.2f}  (target: >{targets.get('sharpe_min', 1.0)})")

    # Section 1: All trades
    sc = result.scorecard
    lines.append(f"\n{'='*60}")
    lines.append("SECTION 1: ALL TRADES (including forced closes)")
    lines.append(f"{'='*60}")
    if sc.get('total_trades', 0) > 0:
        lines.append(f"Total trades: {sc['total_trades']}")
        lines.append(f"Winners: {sc.get('winners', 0)} | Losers: {sc.get('losers', 0)}")
        lines.append(f"Win rate:     {sc.get('win_rate', 0):.0%}")
        lines.append(f"Total PnL:    ${sc.get('total_pnl_usd', 0):.2f}")
        lines.append(f"Total fees:   ${sc.get('total_fees_usd', 0):.2f}")
        lines.append(f"Avg hold:     {sc.get('avg_hold_hours', 0):.1f}h")
    else:
        lines.append("No trades executed.")

    # Section 2: Regular trades only
    sc2 = result.scorecard_regular
    lines.append(f"\n{'='*60}")
    lines.append("SECTION 2: REGULAR TRADES (excluding forced closes)")
    lines.append(f"{'='*60}")
    if sc2.get('total_trades', 0) > 0:
        lines.append(f"Total trades: {sc2['total_trades']}")
        lines.append(f"Winners: {sc2.get('winners', 0)} | Losers: {sc2.get('losers', 0)}")

        wr = sc2.get('win_rate', 0)
        wr_min = targets.get('win_rate_min', 0.35)
        wr_max = targets.get('win_rate_max', 0.45)
        wr_status = "OK" if wr_min <= wr <= wr_max else "MISS"
        lines.append(f"Win rate:     {wr:.0%}  (target: {wr_min:.0%}-{wr_max:.0%}) [{wr_status}]")

        wlr = sc2.get('win_loss_ratio', 0)
        wlr_min = targets.get('win_loss_ratio_min', 2.5)
        wlr_status = "OK" if wlr >= wlr_min else "MISS"
        lines.append(f"W/L ratio:    {wlr:.1f}:1  (target: >{wlr_min}:1) [{wlr_status}]")

        pf = sc2.get('profit_factor', 0)
        pf_min = targets.get('profit_factor_min', 1.5)
        pf_status = "OK" if pf >= pf_min else "MISS"
        lines.append(f"Profit factor: {pf:.2f}  (target: >{pf_min}) [{pf_status}]")

        fd = sc2.get('fee_drag_pct', 0)
        fd_max = targets.get('fee_drag_max_pct', 20)
        fd_status = "OK" if fd <= fd_max else "MISS"
        lines.append(f"Fee drag:     {fd:.1f}%  (target: <{fd_max}%) [{fd_status}]")

        lines.append(f"Avg win:      ${sc2.get('avg_win_usd', 0):.2f}")
        lines.append(f"Avg loss:     ${sc2.get('avg_loss_usd', 0):.2f}")
        lines.append(f"Total PnL:    ${sc2.get('total_pnl_usd', 0):.2f}")
    else:
        lines.append("No regular trades (all were forced closes or no trades at all).")

    # Overall assessment
    lines.append(f"\n{'='*60}")
    dd_ok = result.max_drawdown_pct <= targets.get('max_drawdown_pct', 20)
    sh_ok = result.sharpe_ratio >= targets.get('sharpe_min', 1.0)
    lines.append(f"Max DD:   {'PASS' if dd_ok else 'FAIL'}")
    lines.append(f"Sharpe:   {'PASS' if sh_ok else 'FAIL'}")
    lines.append("=" * 60)

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='StrikerBot Backtester')
    parser.add_argument('--start', default=None,
                        help='Start date YYYY-MM-DD (default: 6 months ago)')
    parser.add_argument('--end', default=None,
                        help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--capital', type=float, default=cfg.INITIAL_CAPITAL_USD,
                        help=f'Initial capital (default: {cfg.INITIAL_CAPITAL_USD})')
    parser.add_argument('--exchange', default='binance',
                        help='Exchange for data (default: binance)')
    parser.add_argument('--symbol', default='BTC/USDT',
                        help='Trading pair (default: BTC/USDT)')
    parser.add_argument('--timeframe', default=cfg.TIMEFRAME,
                        help=f'Timeframe (default: {cfg.TIMEFRAME})')
    parser.add_argument('--cache-dir', default='data',
                        help='Cache directory (default: data)')
    parser.add_argument('--fetch-fresh', action='store_true',
                        help='Force re-download from exchange')
    parser.add_argument('--verbose', action='store_true',
                        help='Debug logging')
    return parser.parse_args()


def main():
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()],
    )

    # Parse dates (UTC)
    if args.end:
        end_dt = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    if args.start:
        start_dt = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        # Default: 6 months before end
        import calendar
        month = end_dt.month - 6
        year = end_dt.year
        while month <= 0:
            month += 12
            year -= 1
        day = min(end_dt.day, calendar.monthrange(year, month)[1])
        start_dt = end_dt.replace(year=year, month=month, day=day)

    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()
    start_ms = int(start_ts * 1000)
    end_ms = int(end_ts * 1000)

    tf_sec = TIMEFRAME_SEC.get(args.timeframe, 3600)
    tf_ms = TIMEFRAME_MS.get(args.timeframe, 3_600_000)

    # Compute padding for warmup (720 bars minimum for EMA burn-in)
    padding_bars = max(
        719,
        cfg.MOMENTUM_RIDER.get('hurst_lookback', 100),
        cfg.MOMENTUM_RIDER.get('ema_slow_period', 21) + 2,
    )
    fetch_since_ms = start_ms - (padding_bars * tf_ms)

    log.info(f"Backtest: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    log.info(f"Exchange: {args.exchange} | Symbol: {args.symbol} | TF: {args.timeframe}")
    log.info(f"Capital: ${args.capital:.2f}")
    log.info(f"Padding: {padding_bars} bars before start")

    # Fetch or load cached data
    cache_path = get_cache_path(
        args.cache_dir, args.exchange, args.symbol,
        args.timeframe, fetch_since_ms, end_ms
    )

    ohlcv = None
    if not args.fetch_fresh:
        ohlcv = load_cached(cache_path)
        if ohlcv:
            log.info(f"Loaded {len(ohlcv)} bars from cache: {cache_path}")

    if ohlcv is None:
        log.info(f"Fetching data from {args.exchange}...")
        ohlcv = fetch_historical_data(
            args.exchange, args.symbol, args.timeframe,
            fetch_since_ms, end_ms,
        )
        log.info(f"Fetched {len(ohlcv)} bars total")
        if ohlcv:
            cache_data(ohlcv, cache_path)
            log.info(f"Cached to: {cache_path}")

    if not ohlcv:
        log.error("No data fetched. Check exchange/symbol/dates.")
        sys.exit(1)

    log.info(
        f"Data range: "
        f"{datetime.fromtimestamp(ohlcv[0][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} to "
        f"{datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}"
    )

    # Run backtest
    journal_path = str(Path(args.cache_dir) / 'backtest_journal.jsonl')
    engine = BacktestEngine(
        ohlcv=ohlcv,
        initial_capital=args.capital,
        timeframe=args.timeframe,
        start_ts=start_ts,
        journal_path=journal_path,
    )

    log.info("Running backtest...")
    result = engine.run()

    # Print report
    report = format_report(result, cfg.TARGETS)
    print("\n" + report)


if __name__ == '__main__':
    main()
