# Global Settings
SYMBOL = "XAUUSDm"   # ✅ FIXED (must match MT5 exactly)
TIMEFRAME = "M1"     # ✅ GRU Model uses 1-minute bars

# Risk Management
RISK_PER_TRADE = 0.01  # 1% per backtest settings
STARTING_BALANCE = 10000.0
AGENT_ALLOCATION = 10000.0
FETCH_BARS = 350
ATR_PERIOD = 14
FRIDAY_CLOSE_HOUR_GMT = 19  # Close trades before Friday market close (19:00 GMT, matching research engine)

# MT5 Configuration
MT5_LOGIN = 435643605
MT5_PASSWORD = "@Vivercity1(d1)"
MT5_SERVER = "Exness-MT5Trial9"
SLIPPAGE = 1000  # Large slippage tolerance to accept all market orders without spread/slippage filter rejection
MAGIC_BASE = 1000

# Directory Paths
MODELS_DIR = "models"
LOGS_DIR = "logs"
STORAGE_DIR = "storage"