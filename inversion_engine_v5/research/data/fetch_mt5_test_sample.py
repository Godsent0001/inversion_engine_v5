import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
import time


# =========================================
# CONFIG
# =========================================

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5

# -----------------------------------------
# FIXED TEST SAMPLE PERIOD
# 2023 + 2024 ONLY
# -----------------------------------------
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 1, 1)

# -----------------------------------------
# OUTPUT
# -----------------------------------------
OUTPUT_PATH = (
    r"C:\Users\USER\Documents\inversion_engine_v5\inversion_engine_v5\research\data\raw\mt5_test_sample_5m.csv"
)

# MT5 struggles with massive requests
# so we fetch monthly
BATCH_DAYS = 30


# =========================================
# INIT MT5
# =========================================
def initialize_mt5():

    if not mt5.initialize():
        raise RuntimeError(
            f"MT5 initialization failed: {mt5.last_error()}"
        )

    print("MT5 initialized successfully ✅")

    symbol_info = mt5.symbol_info(SYMBOL)

    if symbol_info is None:
        raise RuntimeError(f"Symbol not found: {SYMBOL}")

    print(f"Symbol found: {SYMBOL}")

    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)


# =========================================
# FETCH IN BATCHES
# =========================================
def fetch_data():

    all_data = []

    current_start = START_DATE

    total_batches = (
        (END_DATE - START_DATE).days // BATCH_DAYS
    ) + 1

    batch_num = 1

    print(f"\nFetching in {total_batches} batches...\n")

    while current_start < END_DATE:

        current_end = min(
            current_start + pd.Timedelta(days=BATCH_DAYS),
            END_DATE
        )

        print(
            f"[Batch {batch_num}/{total_batches}] "
            f"{current_start} → {current_end}"
        )

        rates = mt5.copy_rates_range(
            SYMBOL,
            TIMEFRAME,
            current_start,
            current_end
        )

        if rates is None or len(rates) == 0:

            print(
                f"⚠ No data returned "
                f"| MT5 Error: {mt5.last_error()}"
            )

        else:

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

        current_start = current_end

        batch_num += 1

        # tiny sleep to avoid MT5 stress
        time.sleep(0.2)

    if len(all_data) == 0:
        raise RuntimeError("No data fetched.")

    final_df = pd.concat(all_data)

    # =====================================
    # CLEANUP
    # =====================================

    final_df = final_df.drop_duplicates(
        subset=["time"]
    )

    final_df = final_df.sort_values("time")

    final_df.reset_index(drop=True, inplace=True)

    return final_df


# =========================================
# SAVE
# =========================================
def save_data(df):

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved data → {OUTPUT_PATH}")


# =========================================
# MAIN
# =========================================
def main():

    initialize_mt5()

    print(f"\nFetching {SYMBOL} M5 test sample...\n")

    df = fetch_data()

    print("\n===================================")
    print(f"TOTAL CANDLES: {len(df)}")
    print("===================================\n")

    print(df.head())

    print(df.tail())

    save_data(df)

    mt5.shutdown()

    print("\nDone ✅")


# =========================================
# ENTRY
# =========================================
if __name__ == "__main__":
    main()