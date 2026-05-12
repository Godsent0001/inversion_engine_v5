import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import time


# ==================================================
# CONFIG
# ==================================================
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5

# Total history wanted
TOTAL_DAYS = 365 * 2

# Fetch size per batch
BATCH_DAYS = 30

# Output
OUTPUT_PATH = r"C:\Users\USER\Documents\inversion_engine_v5\trader_sim\data\raw\xauusd_5m.csv"


# ==================================================
# INITIALIZE MT5
# ==================================================
def initialize_mt5():

    if not mt5.initialize():
        raise RuntimeError(
            "MT5 initialization failed."
        )

    symbol_info = mt5.symbol_info(SYMBOL)

    if symbol_info is None:
        raise RuntimeError(
            f"Symbol not found: {SYMBOL}"
        )

    print(f"MT5 initialized successfully ✅")
    print(f"Symbol found: {SYMBOL}")


# ==================================================
# FETCH IN BATCHES
# ==================================================
def fetch_data_batches():

    end_time = datetime.now()

    all_data = []

    total_batches = TOTAL_DAYS // BATCH_DAYS

    print(f"\nFetching {TOTAL_DAYS} days in {total_batches} batches...\n")

    for i in range(total_batches):

        batch_end = end_time - timedelta(days=i * BATCH_DAYS)

        batch_start = batch_end - timedelta(days=BATCH_DAYS)

        print(
            f"[Batch {i+1}/{total_batches}] "
            f"{batch_start} → {batch_end}"
        )

        rates = mt5.copy_rates_range(
            SYMBOL,
            TIMEFRAME,
            batch_start,
            batch_end
        )

        # ==========================================
        # ERROR HANDLING
        # ==========================================
        if rates is None:

            print(
                f"❌ MT5 returned None on batch {i+1}"
            )

            print("MT5 Error:", mt5.last_error())

            continue

        if len(rates) == 0:

            print(
                f"⚠️ Empty batch returned on batch {i+1}"
            )

            continue

        # ==========================================
        # DATAFRAME
        # ==========================================
        df = pd.DataFrame(rates)

        df["time"] = pd.to_datetime(
            df["time"],
            unit="s"
        )

        df = df[
            [
                "time",
                "open",
                "high",
                "low",
                "close",
                "tick_volume"
            ]
        ]

        all_data.append(df)

        print(f"Fetched candles: {len(df)}")

        # Small pause to avoid MT5 overload
        time.sleep(0.2)

    # ==========================================
    # COMBINE
    # ==========================================
    if len(all_data) == 0:
        raise RuntimeError("No data fetched from MT5")

    final_df = pd.concat(all_data)

    # Remove duplicates
    final_df = final_df.drop_duplicates(
        subset=["time"]
    )

    # Sort oldest → newest
    final_df = final_df.sort_values("time")

    final_df = final_df.reset_index(drop=True)

    return final_df


# ==================================================
# SAVE CSV
# ==================================================
def save_data(df):

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nSaved data → {OUTPUT_PATH}")


# ==================================================
# MAIN
# ==================================================
def main():

    initialize_mt5()

    print("\nFetching XAUUSDm M5 data...\n")

    df = fetch_data_batches()

    print("\n===================================")
    print(f"TOTAL CANDLES: {len(df)}")
    print("===================================\n")

    print(df.head())

    save_data(df)

    mt5.shutdown()

    print("\nDone ✅")


# ==================================================
# ENTRY
# ==================================================
if __name__ == "__main__":
    main()