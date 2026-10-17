"""Persistent high-water risk reference with configurable drawdown resets."""
from __future__ import annotations
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

@dataclass
class RiskState:
    active_reference: float
    stage_peak_balance: float
    reset_count: int = 0
    last_reset_utc: str | None = None

class PersistentRiskReference:
    def __init__(
        self,
        state_path: str | os.PathLike[str],
        drawdown_reset_percent: float = 0.25,
    ):
        if not 0 < drawdown_reset_percent < 1:
            raise ValueError("drawdown_reset_percent must be between 0 and 1.")
        self.path = Path(state_path)
        self.drawdown_reset_percent = float(drawdown_reset_percent)

    def _save(self, state: RiskState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(state), indent=2)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def load_or_initialize(self, live_balance: float) -> RiskState:
        if live_balance <= 0:
            raise ValueError("Live MT5 balance must be positive.")
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                raw: dict[str, Any] = json.load(handle)
            state = RiskState(
                active_reference=float(raw["active_reference"]),
                stage_peak_balance=float(raw["stage_peak_balance"]),
                reset_count=int(raw.get("reset_count", 0)),
                last_reset_utc=raw.get("last_reset_utc"),
            )
            if state.active_reference <= 0 or state.stage_peak_balance <= 0:
                raise ValueError("Persisted risk reference is invalid.")
            return state
        state = RiskState(live_balance, live_balance)
        self._save(state)
        return state

    def update(self, live_balance: float) -> RiskState:
        """Update HWM; reset the active reference after the configured drawdown."""
        if live_balance <= 0:
            raise ValueError("Live MT5 balance must be positive.")
        state = self.load_or_initialize(live_balance)
        if live_balance >= state.active_reference:
            state.active_reference = live_balance
            state.stage_peak_balance = live_balance
        elif live_balance <= state.active_reference * (1.0 - self.drawdown_reset_percent):
            state.active_reference = live_balance
            state.stage_peak_balance = live_balance
            state.reset_count += 1
            state.last_reset_utc = datetime.now(timezone.utc).isoformat()
        self._save(state)
        return state
