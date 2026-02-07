"""
Tests for Portfolio Guard
===========================
Tests all 5 safety rules.
"""

import time
import pytest

from risk.portfolio_guard import PortfolioGuard, GuardResult


@pytest.fixture
def guard() -> PortfolioGuard:
    """Default guard with standard limits."""
    return PortfolioGuard(
        max_risk_per_trade_pct=2.0,
        daily_loss_limit_usd=50.0,
        max_concurrent_positions=3,
        max_deployment_pct=95.0,
        consecutive_loss_pause_count=3,
        consecutive_loss_pause_hours=4.0,
    )


class TestRule1MaxRiskPerTrade:
    """Rule 1: Never risk >2% on single trade."""

    def test_within_limit(self, guard: PortfolioGuard) -> None:
        result = guard.check_entry(risk_pct=1.5, equity_usd=500.0)
        assert result.allowed is True

    def test_exceeds_limit(self, guard: PortfolioGuard) -> None:
        result = guard.check_entry(risk_pct=3.0, equity_usd=500.0)
        assert result.allowed is False
        assert result.rule_violated == "max_risk_per_trade"

    def test_at_exact_limit(self, guard: PortfolioGuard) -> None:
        result = guard.check_entry(risk_pct=2.0, equity_usd=500.0)
        assert result.allowed is True


class TestRule2DailyLossLimit:
    """Rule 2: Stop new entries if daily loss >= $50."""

    def test_within_limit(self, guard: PortfolioGuard) -> None:
        guard.daily_pnl_usd = -30.0
        guard.daily_reset_timestamp = time.time()
        result = guard.check_entry(risk_pct=1.0, equity_usd=500.0)
        assert result.allowed is True

    def test_exceeds_limit(self, guard: PortfolioGuard) -> None:
        guard.daily_pnl_usd = -50.0
        guard.daily_reset_timestamp = time.time()
        result = guard.check_entry(risk_pct=1.0, equity_usd=500.0)
        assert result.allowed is False
        assert result.rule_violated == "daily_loss_limit"

    def test_resets_after_24h(self, guard: PortfolioGuard) -> None:
        guard.daily_pnl_usd = -60.0
        guard.daily_reset_timestamp = time.time() - 90000  # >24h ago
        result = guard.check_entry(risk_pct=1.0, equity_usd=500.0)
        # Should have reset
        assert result.allowed is True


class TestRule3MaxPositions:
    """Rule 3: Max 3 concurrent positions."""

    def test_within_limit(self, guard: PortfolioGuard) -> None:
        guard.open_positions = 2
        result = guard.check_entry(risk_pct=1.0, equity_usd=500.0)
        assert result.allowed is True

    def test_at_limit(self, guard: PortfolioGuard) -> None:
        guard.open_positions = 3
        result = guard.check_entry(risk_pct=1.0, equity_usd=500.0)
        assert result.allowed is False
        assert result.rule_violated == "max_concurrent_positions"


class TestRule4MaxDeployment:
    """Rule 4: Max 95% deployment."""

    def test_within_limit(self, guard: PortfolioGuard) -> None:
        guard.current_deployment_pct = 50.0
        result = guard.check_entry(risk_pct=1.0, equity_usd=500.0)
        assert result.allowed is True

    def test_at_limit(self, guard: PortfolioGuard) -> None:
        guard.current_deployment_pct = 95.0
        result = guard.check_entry(risk_pct=1.0, equity_usd=500.0)
        assert result.allowed is False
        assert result.rule_violated == "max_deployment"


class TestRule5ConsecutiveLossPause:
    """Rule 5: Pause after 3 consecutive losses."""

    def test_no_pause_after_2_losses(self, guard: PortfolioGuard) -> None:
        guard.record_trade_result(-10.0)
        guard.record_trade_result(-10.0)
        result = guard.check_entry(risk_pct=1.0, equity_usd=500.0)
        assert result.allowed is True

    def test_pause_after_3_losses(self, guard: PortfolioGuard) -> None:
        guard.record_trade_result(-10.0)
        guard.record_trade_result(-10.0)
        guard.record_trade_result(-10.0)
        result = guard.check_entry(risk_pct=1.0, equity_usd=500.0)
        assert result.allowed is False
        assert result.rule_violated == "consecutive_loss_pause"

    def test_win_resets_counter(self, guard: PortfolioGuard) -> None:
        guard.record_trade_result(-10.0)
        guard.record_trade_result(-10.0)
        guard.record_trade_result(20.0)  # Win resets
        guard.record_trade_result(-10.0)
        guard.record_trade_result(-10.0)
        result = guard.check_entry(risk_pct=1.0, equity_usd=500.0)
        assert result.allowed is True  # Only 2 consecutive losses


class TestPortfolioGuardPersistence:
    """Tests for state serialization."""

    def test_round_trip(self, guard: PortfolioGuard) -> None:
        """State should survive serialize/deserialize."""
        guard.daily_pnl_usd = -25.0
        guard.consecutive_losses = 2
        guard.open_positions = 1

        data = guard.to_dict()
        restored = PortfolioGuard.from_dict(
            data,
            max_risk_per_trade_pct=2.0,
            daily_loss_limit_usd=50.0,
            max_concurrent_positions=3,
            max_deployment_pct=95.0,
            consecutive_loss_pause_count=3,
            consecutive_loss_pause_hours=4.0,
        )

        assert restored.daily_pnl_usd == -25.0
        assert restored.consecutive_losses == 2
        assert restored.open_positions == 1


class TestGuardResult:
    """Tests for GuardResult dataclass."""

    def test_allowed_result(self) -> None:
        result = GuardResult(allowed=True, reason="All clear")
        assert result.allowed is True
        assert result.rule_violated == ""

    def test_blocked_result(self) -> None:
        result = GuardResult(allowed=False, reason="Too risky", rule_violated="max_risk")
        assert result.allowed is False
        assert result.rule_violated == "max_risk"
