import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os


# -------------------------
# CONFIG
# -------------------------
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M30

DAYS = 365 * 2  # 2 years

# ✅ FULL ABSOLUTE PATH (YOUR REQUEST)
OUTPUT_PATH = r"C:\Users\USER\Documents\inversion_engine_v5\trader_sim\data\raw\xauusd_30m.csv"


# -------------------------
# INIT MT5
# -------------------------
def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed. Make sure terminal is open.")


# -------------------------
# FETCH DATA
# -------------------------
def fetch_data():
    end_time = datetime.now()
    start_time = end_time - timedelta(days=DAYS)

    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_time, end_time)

    if rates is None or len(rates) == 0:
        raise RuntimeError("Failed to fetch data from MT5")

    df = pd.DataFrame(rates)

    # Convert time
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Standard format for your engine
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]

    return df


# -------------------------
# SAVE CSV (ROBUST)
# -------------------------
def save_data(df):
    # ✅ ensure folder exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved data → {OUTPUT_PATH}")


# -------------------------
# MAIN
# -------------------------
def main():
    initialize_mt5()

    print("Fetching XAUUSD M30 data (2 years)...")
    df = fetch_data()

    print(f"Total candles: {len(df)}")

    save_data(df)

    mt5.shutdown()
    print("Done ✅")


if __name__ == "__main__":
    main()