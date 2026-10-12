import os

# Read local live_agent_engine/.env without overriding OS/deployment variables.
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
if _ENV_FILE.is_file():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _value = _line.split("=", 1)
        _key, _value = _key.strip(), _value.strip()
        if len(_value) >= 2 and _value[0] == _value[-1] and _value[0] in ("'", '"'):
            _value = _value[1:-1]
        if _key and _key not in os.environ:
            os.environ[_key] = _value

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
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
SLIPPAGE = 1000  # Large slippage tolerance to accept all market orders without spread/slippage filter rejection
MAGIC_BASE = 1000

# Directory Paths
MODELS_DIR = "models"
LOGS_DIR = "logs"
STORAGE_DIR = "storage"