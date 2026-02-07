"""
Portfolio Guard - THE ONLY SAFETY LAYER
========================================
Exactly 5 rules. No more, no less.
This is NOT a goalkeeper. Don't add more rules.

Rules:
1. Max 2% risk per trade
2. Daily loss limit ($50 on $500 capital = 10%)
3. Max 3 concurrent positions
4. Max 95% deployment
5. Pause after 3 consecutive losses (4h cooldown)
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class GuardResult:
    """Result of a portfolio guard check."""
    allowed: bool
    reason: str = ""
    rule_violated: str = ""


@dataclass
class PortfolioGuard:
    """
    The only safety layer. 5 rules.
    """
    max_risk_per_trade_pct: float = 2.0
    daily_loss_limit_usd: float = 50.0
    max_concurrent_positions: int = 3
    max_deployment_pct: float = 95.0
    consecutive_loss_pause_count: int = 3
    consecutive_loss_pause_hours: float = 4.0

    # State tracking
    daily_pnl_usd: float = 0.0
    daily_reset_timestamp: float = 0.0
    open_positions: int = 0
    current_deployment_pct: float = 0.0
    consecutive_losses: int = 0
    pause_until: float = 0.0

    def check_entry(
        self,
        risk_pct: float,
        equity_usd: float,
    ) -> GuardResult:
        """
        Check if a new entry is allowed. Returns GuardResult.
        ALL 5 rules checked in order. First violation stops.
        """
        now = time.time()

        # Reset daily PnL at midnight UTC
        self._maybe_reset_daily(now)

        # Rule 1: Max risk per trade
        if risk_pct > self.max_risk_per_trade_pct:
            return GuardResult(
                allowed=False,
                reason=f"Risk {risk_pct:.1f}% exceeds {self.max_risk_per_trade_pct}% limit",
                rule_violated="max_risk_per_trade"
            )

        # Rule 2: Daily loss limit
        if self.daily_pnl_usd <= -self.daily_loss_limit_usd:
            return GuardResult(
                allowed=False,
                reason=f"Daily loss ${abs(self.daily_pnl_usd):.2f} hit limit ${self.daily_loss_limit_usd:.2f}",
                rule_violated="daily_loss_limit"
            )

        # Rule 3: Max concurrent positions
        if self.open_positions >= self.max_concurrent_positions:
            return GuardResult(
                allowed=False,
                reason=f"{self.open_positions} positions open (max {self.max_concurrent_positions})",
                rule_violated="max_concurrent_positions"
            )

        # Rule 4: Max deployment
        if self.current_deployment_pct >= self.max_deployment_pct:
            return GuardResult(
                allowed=False,
                reason=f"Deployment {self.current_deployment_pct:.1f}% at limit {self.max_deployment_pct:.1f}%",
                rule_violated="max_deployment"
            )

        # Rule 5: Consecutive loss pause
        if now < self.pause_until:
            remaining_min = (self.pause_until - now) / 60
            return GuardResult(
                allowed=False,
                reason=f"Cooling down after {self.consecutive_loss_pause_count} losses ({remaining_min:.0f}m left)",
                rule_violated="consecutive_loss_pause"
            )

        return GuardResult(allowed=True, reason="All 5 rules passed")

    def record_trade_result(self, pnl_usd: float) -> None:
        """Record a completed trade for daily tracking."""
        self.daily_pnl_usd += pnl_usd

        if pnl_usd < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.consecutive_loss_pause_count:
                self.pause_until = time.time() + (self.consecutive_loss_pause_hours * 3600)
        else:
            self.consecutive_losses = 0

    def update_positions(self, count: int, deployment_pct: float) -> None:
        """Update current position count and deployment percentage."""
        self.open_positions = count
        self.current_deployment_pct = deployment_pct

    def _maybe_reset_daily(self, now: float) -> None:
        """Reset daily PnL tracking at midnight UTC."""
        # Simple: reset if more than 24h since last reset
        if now - self.daily_reset_timestamp > 86400:
            self.daily_pnl_usd = 0.0
            self.daily_reset_timestamp = now

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'daily_pnl_usd': self.daily_pnl_usd,
            'daily_reset_timestamp': self.daily_reset_timestamp,
            'open_positions': self.open_positions,
            'current_deployment_pct': self.current_deployment_pct,
            'consecutive_losses': self.consecutive_losses,
            'pause_until': self.pause_until,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> 'PortfolioGuard':
        """Restore state from persistence."""
        guard = cls(**kwargs)
        guard.daily_pnl_usd = data.get('daily_pnl_usd', 0.0)
        guard.daily_reset_timestamp = data.get('daily_reset_timestamp', 0.0)
        guard.open_positions = data.get('open_positions', 0)
        guard.current_deployment_pct = data.get('current_deployment_pct', 0.0)
        guard.consecutive_losses = data.get('consecutive_losses', 0)
        guard.pause_until = data.get('pause_until', 0.0)
        return guard
