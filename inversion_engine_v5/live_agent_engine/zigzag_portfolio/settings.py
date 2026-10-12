"""Configuration for the isolated ZigZag portfolio runner."""
# Explicit opt-in is required before this strategy can send live orders.
LIVE_TRADING_ENABLED = False
SYMBOL_CANDIDATES = {
    "gold": ("XAUUSDc", "XAUUSDm"),
    "oil": ("UKOILm", "UKOILc"),
}
TIMEFRAME = "M5"
FETCH_BARS = 2500
DEPTH = 5
DEVIATION_POINTS = 3.0
BACKSTEP = 2
REWARD_RISK = 3.0
GOLD_RISK_PERCENT = 0.02
OIL_RISK_PERCENT = 0.01
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
