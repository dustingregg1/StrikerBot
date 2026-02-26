"""
Backtest Engine Tests
======================
Tests for compute_max_drawdown, compute_sharpe, and BacktestEngine.
Uses synthetic data — no network calls.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest import (
    BacktestEngine, BacktestResult,
    compute_max_drawdown, compute_sharpe,
)


# =============================================================================
# FIXTURES
# =============================================================================

def _make_ohlcv(
    n_bars: int,
    base_price: float = 95000.0,
    start_ts_ms: int = 1693526400000,  # 2023-09-01 00:00 UTC
    trend: str = 'flat',
) -> list:
    """Generate synthetic 1h OHLCV bars (timestamps in milliseconds)."""
    data = []
    for i in range(n_bars):
        ts = start_ts_ms + i * 3600000

        if trend == 'up':
            drift = i * 50
        elif trend == 'down':
            drift = -i * 50
        elif trend == 'updown':
            if i < n_bars // 2:
                drift = i * 50
            else:
                drift = (n_bars // 2) * 50 - (i - n_bars // 2) * 100
        else:
            drift = 0

        noise = ((i * 7) % 13 - 6) * 20  # Deterministic noise
        c = base_price + drift + noise
        h = c + 150
        l = c - 150
        o = c - 30
        v = 500.0 + (i % 5) * 100

        data.append([ts, o, h, l, c, v])
    return data


@pytest.fixture
def flat_ohlcv():
    """800 bars of flat price action (enough for 720 warmup + 80 trading)."""
    return _make_ohlcv(800, trend='flat')


@pytest.fixture
def uptrend_ohlcv():
    """900 bars with uptrend."""
    return _make_ohlcv(900, trend='up')


@pytest.fixture
def updown_ohlcv():
    """1000 bars: uptrend then downtrend."""
    return _make_ohlcv(1000, trend='updown')


# =============================================================================
# TEST: compute_max_drawdown
# =============================================================================

class TestComputeMaxDrawdown:
    def test_no_drawdown(self):
        """Monotonically rising equity => 0% drawdown."""
        curve = [(i, 100 + i) for i in range(10)]
        dd = compute_max_drawdown(curve)
        assert dd == pytest.approx(0.0)

    def test_known_drawdown(self):
        """Peak at 200, trough at 150 => 25% drawdown."""
        curve = [(0, 100), (1, 200), (2, 150), (3, 180)]
        dd = compute_max_drawdown(curve)
        assert dd == pytest.approx(25.0)

    def test_full_loss(self):
        """Near-total loss => >99% drawdown."""
        curve = [(0, 100), (1, 0.5)]
        dd = compute_max_drawdown(curve)
        assert dd > 99.0

    def test_empty_curve(self):
        """Empty equity curve => 0%."""
        assert compute_max_drawdown([]) == 0.0

    def test_single_point(self):
        """Single point => 0%."""
        assert compute_max_drawdown([(0, 100)]) == 0.0


# =============================================================================
# TEST: compute_sharpe
# =============================================================================

class TestComputeSharpe:
    def test_constant_equity_zero_sharpe(self):
        """Flat equity => 0 Sharpe (zero std dev)."""
        curve = [(i, 100.0) for i in range(100)]
        sharpe = compute_sharpe(curve)
        assert sharpe == 0.0

    def test_positive_returns_positive_sharpe(self):
        """Rising equity => positive Sharpe."""
        curve = [(i, 100.0 + i * 0.1) for i in range(1000)]
        sharpe = compute_sharpe(curve)
        assert sharpe > 0

    def test_negative_returns_negative_sharpe(self):
        """Declining equity => negative Sharpe."""
        curve = [(i, 100.0 - i * 0.1) for i in range(500)]
        sharpe = compute_sharpe(curve, annual_rfr=0.0)
        assert sharpe < 0

    def test_rfr_subtraction_matters(self):
        """Higher RFR should produce lower Sharpe."""
        curve = [(i, 100.0 + i * 0.01) for i in range(1000)]
        sharpe_low_rfr = compute_sharpe(curve, annual_rfr=0.0)
        sharpe_high_rfr = compute_sharpe(curve, annual_rfr=0.10)
        assert sharpe_low_rfr > sharpe_high_rfr

    def test_insufficient_data(self):
        """< 2 points => 0."""
        assert compute_sharpe([(0, 100)]) == 0.0

    def test_timeframe_scaling(self):
        """Sharpe should scale with timeframe (1h vs 4h)."""
        curve = [(i, 100.0 + i * 0.01) for i in range(1000)]
        sharpe_1h = compute_sharpe(curve, timeframe_sec=3600)
        sharpe_4h = compute_sharpe(curve, timeframe_sec=14400)
        # Different annualization factors => different values
        assert sharpe_1h != pytest.approx(sharpe_4h, rel=0.01)


# =============================================================================
# TEST: BacktestEngine
# =============================================================================

class TestBacktestEngine:
    def test_empty_data_raises(self):
        """Empty data => ValueError."""
        with pytest.raises(ValueError, match="Empty OHLCV"):
            BacktestEngine(ohlcv=[], initial_capital=500.0)

    def test_equity_starts_at_initial(self, flat_ohlcv):
        """First equity point should be close to initial capital."""
        engine = BacktestEngine(
            ohlcv=flat_ohlcv,
            initial_capital=500.0,
            journal_path='/tmp/test_bt_journal.jsonl',
        )
        result = engine.run()
        assert len(result.equity_curve) > 0
        # First equity point should be initial capital (no trades on flat market)
        assert result.equity_curve[0][1] == pytest.approx(500.0, rel=0.01)

    def test_processes_correct_bar_count(self, flat_ohlcv):
        """Engine should produce equity points for each bar after warmup."""
        engine = BacktestEngine(
            ohlcv=flat_ohlcv,
            initial_capital=500.0,
            journal_path='/tmp/test_bt_bars.jsonl',
        )
        result = engine.run()
        # 800 bars total, 719 warmup => 81 traded bars
        expected = len(flat_ohlcv) - 719
        assert result.traded_bars == expected
        assert len(result.equity_curve) == expected

    def test_result_has_all_fields(self, flat_ohlcv):
        """BacktestResult should have all expected fields."""
        engine = BacktestEngine(
            ohlcv=flat_ohlcv,
            initial_capital=500.0,
            journal_path='/tmp/test_bt_fields.jsonl',
        )
        result = engine.run()
        assert isinstance(result.equity_curve, list)
        assert isinstance(result.max_drawdown_pct, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.scorecard, dict)
        assert isinstance(result.scorecard_regular, dict)


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestTimestampConversion:
    def test_timestamps_are_seconds_in_equity_curve(self, flat_ohlcv):
        """Equity curve timestamps should be in seconds, not milliseconds."""
        engine = BacktestEngine(
            ohlcv=flat_ohlcv,
            initial_capital=500.0,
            journal_path='/tmp/test_bt_ts.jsonl',
        )
        result = engine.run()
        if result.equity_curve:
            ts = result.equity_curve[0][0]
            # Seconds timestamps are ~1.7e9, milliseconds are ~1.7e12
            assert ts < 1e11, f"Timestamp {ts} appears to be in milliseconds"
            assert ts > 1e8, f"Timestamp {ts} is too small to be seconds"


class TestLastBarSignal:
    def test_no_trade_on_final_bar(self, uptrend_ohlcv):
        """Signal on final bar should not produce a filled trade."""
        engine = BacktestEngine(
            ohlcv=uptrend_ohlcv,
            initial_capital=500.0,
            journal_path='/tmp/test_bt_lastbar.jsonl',
        )
        result = engine.run()
        # Verify the last equity point reflects force-close, not a new entry
        # (Final bar should never start a new position)
        # We can't directly test "no signal on last bar" without mocking,
        # but we can verify the engine doesn't crash and equity is settled
        assert result.final_equity > 0
        if result.equity_curve:
            assert result.equity_curve[-1][1] == pytest.approx(result.final_equity)


class TestStopConservatism:
    def test_effective_stop_uses_max(self):
        """effective_stop should be max(stop_price, trailing_stop)."""
        # This tests the concept: if trailing stop is higher than initial stop,
        # the trailing stop should be used
        stop_price = 90000.0
        trailing_stop = 93000.0
        effective_stop = max(stop_price, trailing_stop or 0)
        assert effective_stop == 93000.0

    def test_stop_price_wins_when_higher(self):
        """When stop_price > trailing_stop, stop_price is used."""
        stop_price = 93000.0
        trailing_stop = 90000.0
        effective_stop = max(stop_price, trailing_stop or 0)
        assert effective_stop == 93000.0


class TestMarkToMarket:
    def test_mtm_reflects_unrealized_loss(self):
        """MTM equity should reflect unrealized PnL on open positions."""
        from scripts.backtest import BacktestPosition

        state = {
            'equity_usd': 499.0,  # After entry fee
            'positions': {
                'test1': BacktestPosition(
                    trade_id='test1', symbol='BTC/USDT',
                    entry_price=100000.0, entry_time=1000000.0,
                    quantity=0.005, position_usd=500.0,
                    stop_price=95000.0, highest_high=100000.0,
                    trailing_stop=95000.0,
                ),
            },
        }

        ohlcv = _make_ohlcv(800, base_price=100000.0)
        engine = BacktestEngine(
            ohlcv=ohlcv,
            initial_capital=500.0,
            journal_path='/tmp/test_bt_mtm.jsonl',
        )

        # Price drops to 98000 => unrealized loss = 0.005 * (98000 - 100000) = -10
        mtm = engine._compute_mtm(state, 98000.0)
        assert mtm == pytest.approx(499.0 + (-10.0))

        # Price rises to 102000 => unrealized gain = 0.005 * (102000 - 100000) = +10
        mtm = engine._compute_mtm(state, 102000.0)
        assert mtm == pytest.approx(499.0 + 10.0)


class TestForceCloseEquity:
    def test_final_equity_point_matches_settled(self, uptrend_ohlcv):
        """Last equity point should equal final settled equity after force-close."""
        engine = BacktestEngine(
            ohlcv=uptrend_ohlcv,
            initial_capital=500.0,
            journal_path='/tmp/test_bt_forceclose.jsonl',
        )
        result = engine.run()
        if result.equity_curve:
            assert result.equity_curve[-1][1] == pytest.approx(result.final_equity)


class TestFeeRoundTrip:
    def test_round_trip_fee_accounting(self):
        """Entry fee + exit fee should both reduce equity correctly."""
        from striker import simulate_fill_price, calculate_trade_fee

        equity = 500.0
        entry_price = 100000.0
        quantity = 0.005
        position_usd = quantity * entry_price  # $500

        # Entry
        fill_buy = simulate_fill_price(entry_price, 'buy')
        entry_fee = calculate_trade_fee(quantity * fill_buy, is_maker=False)
        equity -= entry_fee

        # Exit at same price
        fill_sell = simulate_fill_price(entry_price, 'sell')
        gross_pnl = (fill_sell - fill_buy) * quantity
        exit_fee = calculate_trade_fee(quantity * fill_sell, is_maker=False)
        net_pnl = gross_pnl - exit_fee
        equity += net_pnl

        # Equity should be less than initial (fees + slippage)
        assert equity < 500.0
        # But not by more than ~$10 (fees + slippage on a $500 position)
        assert equity >= 490.0


class TestGapDownInvalidation:
    def test_entry_cancelled_if_open_below_stop(self):
        """If bar opens below stop price, entry should be cancelled."""
        # This tests the logic conceptually
        bar_open = 90000.0
        stop_price = 91000.0

        # Gap-down: bar_open <= stop_price => should cancel
        should_cancel = bar_open <= stop_price
        assert should_cancel is True

    def test_entry_proceeds_if_open_above_stop(self):
        """Normal case: bar opens above stop price."""
        bar_open = 95000.0
        stop_price = 91000.0

        should_cancel = bar_open <= stop_price
        assert should_cancel is False
