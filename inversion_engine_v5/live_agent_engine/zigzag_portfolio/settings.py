"""Configuration for the isolated ZigZag portfolio runner."""
import os

LIVE_TRADING_ENABLED = True
# Optional full path to terminal64.exe; can also be set in the environment.
MT5_TERMINAL_PATH = os.getenv("ZIGZAG_MT5_TERMINAL_PATH")
SYMBOL_CANDIDATES = {
    "gold": ("XAUUSDc", "XAUUSDm"),
    "oil": ("UKOILm", "UKOILc"),
    "step_index": ("Step Index", "Step Index 100"),
}
TIMEFRAME = "M5"
FETCH_BARS = 2500
DEPTH = 5
DEVIATION_POINTS = 3.0
BACKSTEP = 2
REWARD_RISK = 3.0
GOLD_RISK_PERCENT = 0.01
OIL_RISK_PERCENT = 0.01
STEP_INDEX_RISK_PERCENT = 0.01
RISK_PERCENT_BY_ASSET = {
    "gold": GOLD_RISK_PERCENT,
    "oil": OIL_RISK_PERCENT,
    "step_index": STEP_INDEX_RISK_PERCENT,
}
HWM_RESET_DRAWDOWN_PERCENT = 0.25
GOLD_SESSION_START_UTC = 23
GOLD_SESSION_END_UTC = 8
# Weekly close/open are inferred from a >=40-hour gap in historical M5 bars.
WEEKEND_GAP_HOURS = 40
EXIT_MINUTES_BEFORE_WEEKLY_CLOSE = 30
MAGIC = 59261012
POLL_SECONDS = 5
STATE_FILE = "storage/zigzag_portfolio_state.json"
RISK_STATE_FILE = "storage/zigzag_risk_reference.json"
MAX_SPREAD_POINTS = 0
