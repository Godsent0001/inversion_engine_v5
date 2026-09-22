# Global Settings
SYMBOL = "XAUUSDm"   # ✅ FIXED (must match MT5 exactly)
TIMEFRAME = "M1"     # ✅ GRU Model uses 1-minute bars

# Risk Management
RISK_PER_TRADE = 0.01  # 1% per backtest settings
STARTING_BALANCE = 50000.0
AGENT_ALLOCATION = 10000.0
FETCH_BARS = 350
ATR_PERIOD = 14
FRIDAY_CLOSE_HOUR_GMT = 20  # Close trades 1 hr before Friday market close (20:00 GMT)

# MT5 Configuration
MT5_LOGIN = 435643605
MT5_PASSWORD = "@Vivercity1(e1)"
MT5_SERVER = "Exness-MT5Trial9"
SLIPPAGE = 3
MAGIC_BASE = 1000

# Directory Paths
MODELS_DIR = "models"
LOGS_DIR = "logs"
STORAGE_DIR = "storage"