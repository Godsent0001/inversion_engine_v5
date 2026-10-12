"""MT5-style M5 ZigZag event detection for the separate portfolio strategy."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Sequence


@dataclass(frozen=True)
class Swing:
    # Pivot bar/price belonging to the swing that has just ended.
    index: int
    timestamp: int
    price: float
    kind: Literal["high", "low"]
    leg_start_price: float
    # Event detection occurs at T; the opposite-leg marker is T - backstep.
    event_index: int | None = None
    event_timestamp: int | None = None
    flip_index: int | None = None
    flip_timestamp: int | None = None


@dataclass(frozen=True)
class PendingSignal:
    side: Literal["buy", "sell"]
    entry: float
    stop_loss: float
    take_profit: float
    swing_timestamp: int


def in_gold_session(timestamp: int, session_start_hour_utc: int = 23, session_end_hour_utc: int = 8) -> bool:
    """Gold fill session from session start through, but not including, end."""
    hour = datetime.fromtimestamp(timestamp, tz=timezone.utc).hour
    if session_start_hour_utc > session_end_hour_utc:
        return hour >= session_start_hour_utc or hour < session_end_hour_utc
    return session_start_hour_utc <= hour < session_end_hour_utc



def detect_gold_reopen_timestamp(
    times: Sequence[int],
    gap_minutes: int = 45,
) -> int | None:
    """Return the latest Gold reopen bar detected from a gap in M5 history.

    A reopen is the first bar after a gap of at least 45 minutes, provided
    that bar opens during 22:00 or 23:00 UTC. This mirrors the portfolio
    backtest instead of assuming a fixed daily reopen hour.
    """
    if gap_minutes <= 0:
        raise ValueError("gap_minutes must be positive.")
    gap_seconds = gap_minutes * 60
    reopen: int | None = None
    for previous, current in zip(times, times[1:]):
        if int(current) - int(previous) < gap_seconds:
            continue
        hour = datetime.fromtimestamp(int(current), tz=timezone.utc).hour
        if hour in (22, 23):
            reopen = int(current)
    return reopen


def gold_fill_window_timestamps(
    times: Sequence[int],
    current_timestamp: int,
    wait_minutes: int = 60,
    session_end_hour_utc: int = 8,
) -> tuple[int, int] | None:
    """Return the active Gold fill window as [start, end) UTC timestamps.

    The end is the first configured session-end timestamp strictly after the
    start. This avoids hour-only checks incorrectly ending a window at 23:00
    when the market reopened at 22:00 UTC.
    """
    if not times or wait_minutes < 0 or not 0 <= session_end_hour_utc <= 23:
        return None
    now = int(current_timestamp)
    reopen = detect_gold_reopen_timestamp(times)
    if reopen is None:
        return None
    start = reopen + wait_minutes * 60
    start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
    end_dt = start_dt.replace(
        hour=session_end_hour_utc, minute=0, second=0, microsecond=0
    )
    if int(end_dt.timestamp()) <= start:
        from datetime import timedelta
        end_dt += timedelta(days=1)
    end = int(end_dt.timestamp())
    if not start <= now < end:
        return None
    return start, end


def in_dynamic_gold_fill_window(
    times: Sequence[int],
    current_timestamp: int,
    wait_minutes: int = 60,
    session_end_hour_utc: int = 8,
) -> bool:
    """True only inside the timestamp-defined Gold window [reopen+wait, next 08:00)."""
    return gold_fill_window_timestamps(
        times, current_timestamp, wait_minutes, session_end_hour_utc
    ) is not None

def detect_weekend_reopen_timestamp(
    times: Sequence[int],
    gap_hours: int = 40,
) -> int | None:
    """Return the latest first bar after a weekend-sized market-data gap."""
    if gap_hours <= 0:
        raise ValueError("gap_hours must be positive.")
    threshold = gap_hours * 3600
    reopen: int | None = None
    for previous, current in zip(times, times[1:]):
        if int(current) - int(previous) >= threshold:
            reopen = int(current)
    return reopen


def detect_weekend_close_bar_timestamp(
    times: Sequence[int],
    gap_hours: int = 40,
) -> int | None:
    """Return the last bar timestamp before the latest weekend-sized gap."""
    if gap_hours <= 0:
        raise ValueError("gap_hours must be positive.")
    threshold = gap_hours * 3600
    close_bar: int | None = None
    for previous, current in zip(times, times[1:]):
        if int(current) - int(previous) >= threshold:
            close_bar = int(previous)
    return close_bar


def in_dynamic_weekly_exit_window(
    times: Sequence[int],
    current_timestamp: int,
    exit_minutes_before_close: int = 30,
    timeframe_minutes: int = 5,
    gap_hours: int = 40,
) -> bool:
    """Infer next weekly close from the latest observed weekend gap.

    MT5 bar times are candle-open times. The expected close is one timeframe
    after the last pre-gap bar. The next exit window is inferred from that
    observed weekly schedule and spans from N minutes before close until close.
    """
    if (
        not times
        or exit_minutes_before_close <= 0
        or timeframe_minutes <= 0
        or gap_hours <= 0
    ):
        return False
    last_close_bar = detect_weekend_close_bar_timestamp(times, gap_hours)
    if last_close_bar is None:
        return False

    expected_close = last_close_bar + timeframe_minutes * 60 + 7 * 24 * 3600
    exit_start = expected_close - exit_minutes_before_close * 60
    now = int(current_timestamp)
    week_seconds = 7 * 24 * 3600
    while expected_close <= now:
        expected_close += week_seconds
        exit_start += week_seconds
    return exit_start <= now < expected_close

def _lowest_idx(values: Sequence[float], depth: int, start: int) -> int:
    minimum, index = values[start], start
    for i in range(start - 1, max(start - depth, -1), -1):
        if values[i] < minimum:
            minimum, index = values[i], i
    return index


def _highest_idx(values: Sequence[float], depth: int, start: int) -> int:
    maximum, index = values[start], start
    for i in range(start - 1, max(start - depth, -1), -1):
        if values[i] > maximum:
            maximum, index = values[i], i
    return index


def detect_confirmed_swings(
    highs: Sequence[float],
    lows: Sequence[float],
    times: Sequence[int],
    depth: int = 5,
    deviation_points: float = 3.0,
    point: float = 0.001,
    backstep: int = 2,
    repainting: bool = False,
) -> list[Swing]:
    """Port portfolio.py's ZigZag state machine.

    With repainting=False, events match the reference's backstep-confirmed
    events. With repainting=True, the state machine reacts to the current
    opposite-leg candidate as soon as that bar closes, which is the mode the
    live runner uses. The live mode never schedules an order on a past bar.
    """
    if not (len(highs) == len(lows) == len(times)):
        raise ValueError("highs, lows and times must have equal lengths.")
    if depth < 2 or backstep < 1 or point <= 0 or deviation_points < 0:
        raise ValueError("Invalid ZigZag parameters.")

    high_values = [float(value) for value in highs]
    low_values = [float(value) for value in lows]
    count = len(times)
    low_map = [0.0] * count
    high_map = [0.0] * count
    last_low_1 = last_high_1 = 0.0
    state = 0
    last_low = last_high = 0.0
    last_low_pivot = last_high_pivot = 0
    events: list[tuple[int, str, float, int]] = []

    for current in range(depth, count):
        value = low_values[_lowest_idx(low_values, depth, current)]
        if value == last_low_1:
            value = 0.0
        else:
            last_low_1 = value
            if (low_values[current] - value) > deviation_points * point:
                value = 0.0
            else:
                for back in range(1, backstep + 1):
                    prior = low_map[current - back]
                    if prior != 0 and prior > value:
                        low_map[current - back] = 0.0
        low_map[current] = value if low_values[current] == value else 0.0

        value = high_values[_highest_idx(high_values, depth, current)]
        if value == last_high_1:
            value = 0.0
        else:
            last_high_1 = value
            if (value - high_values[current]) > deviation_points * point:
                value = 0.0
            else:
                for back in range(1, backstep + 1):
                    prior = high_map[current - back]
                    if prior != 0 and prior < value:
                        high_map[current - back] = 0.0
        high_map[current] = value if high_values[current] == value else 0.0

        pivot_marker = current if repainting else current - backstep
        if pivot_marker < depth:
            continue

        if state == 0:
            if last_low == 0 and last_high == 0:
                if high_map[pivot_marker] != 0:
                    last_high = high_values[pivot_marker]
                    last_high_pivot = pivot_marker
                    state = -1
                if low_map[pivot_marker] != 0:
                    last_low = low_values[pivot_marker]
                    last_low_pivot = pivot_marker
                    state = 1
        elif state == 1:
            if low_map[pivot_marker] != 0 and low_map[pivot_marker] < last_low and high_map[pivot_marker] == 0:
                last_low = low_map[pivot_marker]
                last_low_pivot = pivot_marker
            if high_map[pivot_marker] != 0 and low_map[pivot_marker] == 0:
                events.append((current, "L", last_low, last_low_pivot))
                last_high = high_map[pivot_marker]
                last_high_pivot = pivot_marker
                state = -1
        else:
            if high_map[pivot_marker] != 0 and high_map[pivot_marker] > last_high and low_map[pivot_marker] == 0:
                last_high = high_map[pivot_marker]
                last_high_pivot = pivot_marker
            if low_map[pivot_marker] != 0 and high_map[pivot_marker] == 0:
                events.append((current, "H", last_high, last_high_pivot))
                last_low = low_map[pivot_marker]
                last_low_pivot = pivot_marker
                state = 1

    swings: list[Swing] = []
    for event_position, (event_index, kind, price, pivot_index) in enumerate(events):
        # Match portfolio.py's k >= 3 eligibility and previous-event leg sizing.
        if event_position < 3:
            continue
        leg_start_price = events[event_position - 1][2]
        if abs(price - leg_start_price) <= 0:
            continue
        swings.append(Swing(
            index=pivot_index,
            timestamp=int(times[pivot_index]),
            price=float(price),
            kind="high" if kind == "H" else "low",
            leg_start_price=float(leg_start_price),
            event_index=event_index,
            event_timestamp=int(times[event_index]),
            flip_index=event_index if repainting else event_index - backstep,
            flip_timestamp=int(times[event_index if repainting else event_index - backstep]),
        ))
    return swings


def signal_from_swing(swing: Swing, point: float, reward_risk: float = 3.0) -> PendingSignal:
    if point <= 0 or reward_risk <= 0:
        raise ValueError("point and reward_risk must be positive.")
    distance = abs(swing.price - swing.leg_start_price) / 2.0
    if distance <= 0:
        raise ValueError("Swing leg must have a non-zero length.")
    # De-duplication is based on the event-detection bar, not the old pivot bar.
    event_timestamp = swing.event_timestamp or swing.timestamp
    if swing.kind == "high":
        entry = swing.price
        return PendingSignal("buy", entry, entry - distance, entry + distance * reward_risk, event_timestamp)
    entry = swing.price
    return PendingSignal("sell", entry, entry + distance, entry - distance * reward_risk, event_timestamp)
