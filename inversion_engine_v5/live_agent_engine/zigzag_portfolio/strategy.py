"""Pure logic for the separate M5 ZigZag portfolio strategy."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Sequence

@dataclass(frozen=True)
class Swing:
    index: int
    timestamp: int
    price: float
    kind: Literal["high", "low"]
    leg_start_price: float

@dataclass(frozen=True)
class PendingSignal:
    side: Literal["buy", "sell"]
    entry: float
    stop_loss: float
    take_profit: float
    swing_timestamp: int

def in_gold_session(timestamp: int, session_start_hour_utc: int = 23, session_end_hour_utc: int = 8) -> bool:
    """Gold session from 23:00 UTC through, but not including, 08:00 UTC."""
    hour = datetime.fromtimestamp(timestamp, tz=timezone.utc).hour
    if session_start_hour_utc > session_end_hour_utc:
        return hour >= session_start_hour_utc or hour < session_end_hour_utc
    return session_start_hour_utc <= hour < session_end_hour_utc

def detect_confirmed_swings(
    highs: Sequence[float], lows: Sequence[float], times: Sequence[int],
    depth: int = 5, deviation_points: float = 3.0, point: float = 0.01, backstep: int = 2,
) -> list[Swing]:
    """Detect trailing-window extremes; only pivots with later bars are emitted.

    This causal approximation does not claim bit-for-bit parity with the native
    MT5 ZigZag indicator. It deliberately avoids retroactive pivot visibility.
    """
    if not (len(highs) == len(lows) == len(times)):
        raise ValueError("highs, lows and times must have equal lengths.")
    if depth < 2 or backstep < 1 or point <= 0:
        raise ValueError("Invalid ZigZag parameters.")
    candidates: list[tuple[int, str, float]] = []
    for i in range(depth - 1, len(times) - backstep):
        start = i - depth + 1
        max_high = max(float(v) for v in highs[start:i + 1])
        min_low = min(float(v) for v in lows[start:i + 1])
        if float(highs[i]) >= max_high:
            candidates.append((i, "high", float(highs[i])))
        if float(lows[i]) <= min_low:
            candidates.append((i, "low", float(lows[i])))
    candidates.sort(key=lambda item: item[0])
    filtered: list[tuple[int, str, float]] = []
    for candidate in candidates:
        if filtered and candidate[0] == filtered[-1][0]:
            previous = filtered[-1]
            if candidate[1] == previous[1]:
                if (candidate[1] == "high" and candidate[2] >= previous[2]) or (candidate[1] == "low" and candidate[2] <= previous[2]):
                    filtered[-1] = candidate
            else:
                filtered.pop()
            continue
        if not filtered:
            filtered.append(candidate)
            continue
        last_i, last_kind, last_price = filtered[-1]
        i, kind, price = candidate
        if kind == last_kind:
            if (price > last_price if kind == "high" else price < last_price):
                filtered[-1] = candidate
            continue
        if abs(price - last_price) < deviation_points * point:
            continue
        filtered.append(candidate)
    swings: list[Swing] = []
    for pos, (idx, kind, price) in enumerate(filtered):
        leg_start = float(lows[idx]) if kind == "high" else float(highs[idx])
        if pos > 0:
            leg_start = filtered[pos - 1][2]
        swings.append(Swing(idx, int(times[idx]), price, kind, leg_start))
    return swings

def signal_from_swing(swing: Swing, point: float, reward_risk: float = 3.0) -> PendingSignal:
    if point <= 0 or reward_risk <= 0:
        raise ValueError("point and reward_risk must be positive.")
    distance = abs(swing.price - swing.leg_start_price) / 2.0
    if distance <= 0:
        raise ValueError("Swing leg must have a non-zero length.")
    if swing.kind == "high":
        entry = swing.price + point
        return PendingSignal("buy", entry, entry - distance, entry + distance * reward_risk, swing.timestamp)
    entry = swing.price - point
    return PendingSignal("sell", entry, entry + distance, entry - distance * reward_risk, swing.timestamp)
