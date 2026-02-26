"""
Trade Journal - JSONL Append-Only Log
======================================
Every trade logged with rationale, entry/exit, PnL.
Append-only JSONL format for easy analysis.
Weekly scorecard generation.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class TradeRecord:
    """Single trade record."""
    trade_id: str
    symbol: str
    side: str  # 'long' or 'short'
    strategy: str
    entry_price: float
    entry_time: float
    entry_reason: str
    position_usd: float
    quantity: float
    stop_price: float
    # Filled on exit
    exit_price: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees_usd: float = 0.0
    hold_duration_hours: float = 0.0
    # Metadata
    hurst_at_entry: float = 0.0
    atr_at_entry: float = 0.0
    paper_trade: bool = True


class TradeJournal:
    """Append-only JSONL trade journal with scorecard."""

    def __init__(self, filepath: str = "trade_journal.jsonl") -> None:
        self.filepath = Path(filepath)
        self._trades: List[TradeRecord] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing trades from JSONL file."""
        if self.filepath.exists():
            for line in self.filepath.read_text().strip().split('\n'):
                if line.strip():
                    try:
                        data = json.loads(line)
                        self._trades.append(TradeRecord(**data))
                    except (json.JSONDecodeError, TypeError):
                        pass  # Skip malformed lines

    def record_entry(self, trade: TradeRecord) -> None:
        """Record a new trade entry."""
        self._trades.append(trade)
        self._append_to_file(trade)

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        fees_usd: float = 0.0,
        now: float = None,
    ) -> Optional[TradeRecord]:
        """Update a trade with exit information."""
        for trade in reversed(self._trades):
            if trade.trade_id == trade_id and trade.exit_time == 0.0:
                trade.exit_price = exit_price
                trade.exit_time = now if now is not None else time.time()
                trade.exit_reason = exit_reason
                trade.fees_usd = fees_usd
                trade.hold_duration_hours = (trade.exit_time - trade.entry_time) / 3600

                if trade.side == 'long':
                    trade.pnl_usd = (exit_price - trade.entry_price) * trade.quantity - fees_usd
                else:
                    trade.pnl_usd = (trade.entry_price - exit_price) * trade.quantity - fees_usd

                trade.pnl_pct = (trade.pnl_usd / trade.position_usd * 100) if trade.position_usd > 0 else 0.0

                self._append_to_file(trade)
                return trade
        return None

    def _append_to_file(self, trade: TradeRecord) -> None:
        """Append trade record to JSONL file."""
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(asdict(trade)) + '\n')

    def get_completed_trades(self, days: int = 7, as_of: float = None) -> List[TradeRecord]:
        """Get completed trades from the last N days."""
        cutoff = (as_of if as_of is not None else time.time()) - (days * 86400)
        return [
            t for t in self._trades
            if t.exit_time > 0 and t.exit_time >= cutoff
        ]

    def generate_scorecard(self, days: int = 7, as_of: float = None) -> Dict[str, Any]:
        """
        Generate weekly scorecard.

        Target metrics:
        - Win rate: 35-45%
        - Avg win/loss ratio: >2.5:1
        - Profit factor: >1.5
        - Fee drag: <20%
        """
        trades = self.get_completed_trades(days, as_of=as_of)
        if not trades:
            return {'period_days': days, 'total_trades': 0, 'message': 'No completed trades'}

        winners = [t for t in trades if t.pnl_usd > 0]
        losers = [t for t in trades if t.pnl_usd <= 0]

        total_wins = sum(t.pnl_usd for t in winners)
        total_losses = abs(sum(t.pnl_usd for t in losers))
        total_fees = sum(t.fees_usd for t in trades)
        gross_pnl = sum(t.pnl_usd + t.fees_usd for t in trades)

        avg_win = total_wins / len(winners) if winners else 0.0
        avg_loss = total_losses / len(losers) if losers else 0.0
        win_rate = len(winners) / len(trades) if trades else 0.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        fee_drag_pct = (total_fees / gross_pnl * 100) if gross_pnl > 0 else 0.0

        return {
            'period_days': days,
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': round(win_rate, 3),
            'avg_win_usd': round(avg_win, 2),
            'avg_loss_usd': round(avg_loss, 2),
            'win_loss_ratio': round(win_loss_ratio, 2),
            'profit_factor': round(profit_factor, 2),
            'total_pnl_usd': round(sum(t.pnl_usd for t in trades), 2),
            'total_fees_usd': round(total_fees, 2),
            'fee_drag_pct': round(fee_drag_pct, 1),
            'avg_hold_hours': round(
                sum(t.hold_duration_hours for t in trades) / len(trades), 1
            ),
        }

    def format_scorecard(self, days: int = 7, as_of: float = None) -> str:
        """Format scorecard as readable string."""
        sc = self.generate_scorecard(days, as_of=as_of)
        if sc.get('total_trades', 0) == 0:
            return f"No completed trades in last {days} days"

        return (
            f"=== StrikerBot Weekly Scorecard ({sc['period_days']}d) ===\n"
            f"Winners: {sc['winners']} | Losers: {sc['losers']} | "
            f"Win rate: {sc['win_rate']:.0%} (target: 35-45%)\n"
            f"Average winner: ${sc['avg_win_usd']:.2f} | "
            f"Average loser: ${sc['avg_loss_usd']:.2f}\n"
            f"Win/Loss ratio: {sc['win_loss_ratio']:.1f}:1 (target: >2.5:1)\n"
            f"Profit factor: {sc['profit_factor']:.2f} (target: >1.5)\n"
            f"Total PnL: ${sc['total_pnl_usd']:.2f}\n"
            f"Fee drag: {sc['fee_drag_pct']:.1f}% (target: <20%)\n"
            f"Avg hold: {sc['avg_hold_hours']:.1f}h"
        )
