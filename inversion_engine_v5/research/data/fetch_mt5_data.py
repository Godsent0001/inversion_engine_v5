import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import time


# ============================================================
# CONFIG
# ============================================================

# ------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------

TOTAL_DAYS = 365 * 2

# Bar batch size
BATCH_DAYS = 30

# ------------------------------------------------------------
# TIMEFRAMES
# ------------------------------------------------------------

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}

# ------------------------------------------------------------
# MARKETS
# ------------------------------------------------------------
#
# The names below are aliases.
# The code will search your MT5 terminal and find the
# actual Exness symbol automatically.
#
# For example:
#
# XAUUSD may actually be XAUUSDm
# US100 may actually be US100m
#
# ------------------------------------------------------------

SYMBOL_ALIASES = {
    "XAUUSD": [
        "XAUUSD",
        "XAUUSDm",
        "XAUUSD.",
        "XAUUSDc",
        "GOLD",
        "GOLDm",
    ],

    "XAGUSD": [
        "XAGUSD",
        "XAGUSDm",
        "XAGUSD.",
        "XAGUSDc",
        "SILVER",
        "SILVERm",
    ],

    "EURUSD": [
        "EURUSD",
        "EURUSDm",
        "EURUSD.",
        "EURUSDc",
    ],

    "GBPUSD": [
        "GBPUSD",
        "GBPUSDm",
        "GBPUSD.",
        "GBPUSDc",
    ],

    "USDJPY": [
        "USDJPY",
        "USDJPYm",
        "USDJPY.",
        "USDJPYc",
    ],

    "US100": [
        "US100",
        "US100m",
        "US100.",
        "USTEC",
        "USTECm",
        "NAS100",
        "NAS100m",
    ],

    "US500": [
        "US500",
        "US500m",
        "US500.",
        "SP500",
        "SP500m",
    ],

    "US30": [
        "US30",
        "US30m",
        "US30.",
        "DJ30",
        "DJ30m",
    ],

    "USOIL": [
        "USOIL",
        "USOILm",
        "USOIL.",
        "XTIUSD",
        "XTIUSDm",
    ],

    "UKOIL": [
        "UKOIL",
        "UKOILm",
        "UKOIL.",
        "XBRUSD",
        "XBRUSDm",
    ],
}


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_ROOT = (
    r"C:\Users\USER\Documents\inversion_engine_v5"
    r"\inversion_engine_v5\research\data\raw"
)

BARS_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "markets"
)

TICKS_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "ticks"
)


# ------------------------------------------------------------
# TICK DATA
# ------------------------------------------------------------
#
# Tick history can become HUGE.
#
# Start with False.
#
# When you are ready:
#
# FETCH_TICKS = True
#
# and set TICK_DAYS to something manageable.
#
# ------------------------------------------------------------

FETCH_TICKS = False

TICK_DAYS = 30

TICK_BATCH_DAYS = 1


# ------------------------------------------------------------
# OTHER SETTINGS
# ------------------------------------------------------------

REQUEST_PAUSE = 0.2


# ============================================================
# INITIALIZE MT5
# ============================================================

def initialize_mt5():

    print("\nInitializing MetaTrader 5...\n")

    if not mt5.initialize():

        error = mt5.last_error()

        raise RuntimeError(
            f"MT5 initialization failed: {error}"
        )

    terminal = mt5.terminal_info()

    if terminal is not None:

        print(
            f"MT5 connected successfully"
        )

        print(
            f"Terminal: {terminal.name}"
        )

    account = mt5.account_info()

    if account is not None:

        print(
            f"Account: {account.login}"
        )

        print(
            f"Server: {account.server}"
        )

    print()


# ============================================================
# GET ALL SYMBOLS FROM MT5
# ============================================================

def get_available_symbols():

    symbols = mt5.symbols_get()

    if symbols is None:

        raise RuntimeError(
            f"Could not retrieve MT5 symbols: "
            f"{mt5.last_error()}"
        )

    names = [
        symbol.name
        for symbol in symbols
    ]

    return names


# ============================================================
# RESOLVE SYMBOL
# ============================================================

def resolve_symbol(
    market_name,
    available_symbols
):

    aliases = SYMBOL_ALIASES.get(
        market_name,
        []
    )

    # --------------------------------------------------------
    # Exact alias match
    # --------------------------------------------------------

    for alias in aliases:

        if alias in available_symbols:

            return alias

    # --------------------------------------------------------
    # Case-insensitive exact match
    # --------------------------------------------------------

    available_lower = {
        symbol.lower(): symbol
        for symbol in available_symbols
    }

    for alias in aliases:

        found = available_lower.get(
            alias.lower()
        )

        if found is not None:

            return found

    # --------------------------------------------------------
    # Prefix-style fallback
    # --------------------------------------------------------

    market_upper = market_name.upper()

    possible = []

    for symbol in available_symbols:

        upper = symbol.upper()

        if upper.startswith(market_upper):

            possible.append(symbol)

    if possible:

        # Prefer shorter names.
        possible.sort(
            key=len
        )

        return possible[0]

    return None


# ============================================================
# SELECT SYMBOL IN MT5
# ============================================================

def select_symbol(symbol):

    selected = mt5.symbol_select(
        symbol,
        True
    )

    if not selected:

        print(
            f"WARNING: Could not select {symbol}"
        )

        print(
            f"MT5 error: {mt5.last_error()}"
        )

        return False

    return True


# ============================================================
# FETCH BAR DATA
# ============================================================

def fetch_bar_data(
    symbol,
    timeframe,
    timeframe_name,
    total_days
):

    end_time = datetime.now(
        timezone.utc
    )

    all_data = []

    total_batches = (
        total_days // BATCH_DAYS
    )

    print(
        f"\nFetching {symbol} "
        f"{timeframe_name}"
    )

    print(
        f"History: {total_days} days"
    )

    print(
        f"Batches: {total_batches}"
    )

    print()

    for i in range(total_batches):

        batch_end = (
            end_time
            - timedelta(
                days=i * BATCH_DAYS
            )
        )

        batch_start = (
            batch_end
            - timedelta(
                days=BATCH_DAYS
            )
        )

        print(
            f"[{timeframe_name}] "
            f"Batch {i + 1}/{total_batches}: "
            f"{batch_start.date()} → "
            f"{batch_end.date()}"
        )

        rates = mt5.copy_rates_range(
            symbol,
            timeframe,
            batch_start,
            batch_end
        )

        if rates is None:

            print(
                "  ❌ MT5 returned None"
            )

            print(
                f"  MT5 error: "
                f"{mt5.last_error()}"
            )

            continue

        if len(rates) == 0:

            print(
                "  ⚠️ Empty batch"
            )

            continue

        df = pd.DataFrame(
            rates
        )

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        df["time"] = pd.to_datetime(
            df["time"],
            unit="s",
            utc=True
        )

        # ----------------------------------------------------
        # KEEP ALL RAW MT5 BAR DATA
        # ----------------------------------------------------
        #
        # MT5 normally provides:
        #
        # time
        # open
        # high
        # low
        # close
        # tick_volume
        # spread
        # real_volume
        #
        # We keep everything.
        #
        # ----------------------------------------------------

        expected_columns = [
            "time",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]

        existing_columns = [
            column
            for column in expected_columns
            if column in df.columns
        ]

        df = df[
            existing_columns
        ]

        all_data.append(
            df
        )

        print(
            f"  ✅ Candles: {len(df):,}"
        )

        time.sleep(
            REQUEST_PAUSE
        )

    if not all_data:

        raise RuntimeError(
            f"No data fetched for "
            f"{symbol} {timeframe_name}"
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    final_df = final_df.drop_duplicates(
        subset=["time"]
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    final_df = final_df.sort_values(
        "time"
    )

    final_df = final_df.reset_index(
        drop=True
    )

    return final_df


# ============================================================
# SAVE BAR DATA
# ============================================================

def save_bar_data(
    df,
    market_name,
    actual_symbol,
    timeframe_name
):

    output_dir = os.path.join(
        BARS_OUTPUT_DIR,
        market_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    filename = (
        f"{market_name}_"
        f"{timeframe_name}.csv"
    )

    output_path = os.path.join(
        output_dir,
        filename
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n💾 Saved:"
    )

    print(
        f"   {output_path}"
    )

    print(
        f"   Symbol: {actual_symbol}"
    )

    print(
        f"   Candles: {len(df):,}"
    )

    print(
        f"   From: {df['time'].iloc[0]}"
    )

    print(
        f"   To:   {df['time'].iloc[-1]}"
    )


# ============================================================
# FETCH TICK DATA
# ============================================================

def fetch_tick_data(
    symbol,
    market_name,
    total_days
):

    end_time = datetime.now(
        timezone.utc
    )

    all_data = []

    total_batches = (
        total_days // TICK_BATCH_DAYS
    )

    print(
        f"\nFetching TICKS for {market_name}"
    )

    print(
        f"Symbol: {symbol}"
    )

    print(
        f"History: {total_days} days"
    )

    print()

    for i in range(total_batches):

        batch_end = (
            end_time
            - timedelta(
                days=i * TICK_BATCH_DAYS
            )
        )

        batch_start = (
            batch_end
            - timedelta(
                days=TICK_BATCH_DAYS
            )
        )

        print(
            f"[TICKS] "
            f"Batch {i + 1}/{total_batches}: "
            f"{batch_start.date()} → "
            f"{batch_end.date()}"
        )

        ticks = mt5.copy_ticks_range(
            symbol,
            batch_start,
            batch_end,
            mt5.COPY_TICKS_ALL
        )

        if ticks is None:

            print(
                "  ❌ MT5 returned None"
            )

            print(
                f"  MT5 error: "
                f"{mt5.last_error()}"
            )

            continue

        if len(ticks) == 0:

            print(
                "  ⚠️ Empty tick batch"
            )

            continue

        df = pd.DataFrame(
            ticks
        )

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        if "time" in df.columns:

            df["time"] = pd.to_datetime(
                df["time"],
                unit="s",
                utc=True
            )

        # ----------------------------------------------------
        # MICROSECOND TIME
        # ----------------------------------------------------

        if "time_msc" in df.columns:

            df["time_msc"] = pd.to_datetime(
                df["time_msc"],
                unit="ms",
                utc=True
            )

        all_data.append(
            df
        )

        print(
            f"  ✅ Ticks: {len(df):,}"
        )

        time.sleep(
            REQUEST_PAUSE
        )

    if not all_data:

        print(
            f"⚠️ No tick data available "
            f"for {market_name}"
        )

        return None

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    duplicate_columns = []

    if "time_msc" in final_df.columns:

        duplicate_columns.append(
            "time_msc"
        )

    elif "time" in final_df.columns:

        duplicate_columns.append(
            "time"
        )

    if duplicate_columns:

        final_df = final_df.drop_duplicates(
            subset=duplicate_columns
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    if "time_msc" in final_df.columns:

        final_df = final_df.sort_values(
            "time_msc"
        )

    elif "time" in final_df.columns:

        final_df = final_df.sort_values(
            "time"
        )

    final_df = final_df.reset_index(
        drop=True
    )

    return final_df


# ============================================================
# SAVE TICK DATA
# ============================================================

def save_tick_data(
    df,
    market_name,
    actual_symbol
):

    if df is None:

        return

    output_dir = os.path.join(
        TICKS_OUTPUT_DIR,
        market_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        f"{market_name}_ticks.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n💾 Tick data saved:"
    )

    print(
        f"   {output_path}"
    )

    print(
        f"   Symbol: {actual_symbol}"
    )

    print(
        f"   Ticks: {len(df):,}"
    )


# ============================================================
# MAIN DATA COLLECTION
# ============================================================

def main():

    initialize_mt5()

    # --------------------------------------------------------
    # FIND AVAILABLE SYMBOLS
    # --------------------------------------------------------

    print(
        "Reading symbols available "
        "from your MT5 terminal..."
    )

    available_symbols = (
        get_available_symbols()
    )

    print(
        f"Available symbols: "
        f"{len(available_symbols):,}"
    )

    print()

    # --------------------------------------------------------
    # RESOLVE ALL MARKETS
    # --------------------------------------------------------

    resolved_markets = {}

    print(
        "Resolving broker symbols...\n"
    )

    for market_name in SYMBOL_ALIASES:

        actual_symbol = resolve_symbol(
            market_name,
            available_symbols
        )

        if actual_symbol is None:

            print(
                f"❌ {market_name}: "
                f"NOT FOUND"
            )

            continue

        print(
            f"✅ {market_name} → "
            f"{actual_symbol}"
        )

        resolved_markets[
            market_name
        ] = actual_symbol

        select_symbol(
            actual_symbol
        )

    print()

    # --------------------------------------------------------
    # MAKE SURE XAUUSD EXISTS
    # --------------------------------------------------------

    if "XAUUSD" not in resolved_markets:

        raise RuntimeError(
            "\nXAUUSD could not be found "
            "in your MT5 terminal.\n\n"
            "Open Market Watch in MT5 and "
            "make sure your Gold symbol is "
            "visible."
        )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    os.makedirs(
        BARS_OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        TICKS_OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FETCH BAR DATA
    # --------------------------------------------------------

    print(
        "\n"
        "=" * 70
    )

    print(
        "STARTING HISTORICAL BAR DOWNLOAD"
    )

    print(
        "=" * 70
    )

    for market_name, actual_symbol in (
        resolved_markets.items()
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"MARKET: {market_name}"
        )

        print(
            f"MT5 SYMBOL: {actual_symbol}"
        )

        print(
            "=" * 70
        )

        for timeframe_name, timeframe in (
            TIMEFRAMES.items()
        ):

            try:

                df = fetch_bar_data(
                    symbol=actual_symbol,
                    timeframe=timeframe,
                    timeframe_name=timeframe_name,
                    total_days=TOTAL_DAYS
                )

                save_bar_data(
                    df=df,
                    market_name=market_name,
                    actual_symbol=actual_symbol,
                    timeframe_name=timeframe_name
                )

            except Exception as error:

                print(
                    f"\n❌ ERROR: "
                    f"{market_name} "
                    f"{timeframe_name}"
                )

                print(
                    f"   {error}"
                )

    # --------------------------------------------------------
    # FETCH TICKS
    # --------------------------------------------------------

    if FETCH_TICKS:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STARTING TICK DATA DOWNLOAD"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # For the first version, get ticks for XAUUSD only.
        # ----------------------------------------------------

        xau_symbol = (
            resolved_markets["XAUUSD"]
        )

        try:

            tick_df = fetch_tick_data(
                symbol=xau_symbol,
                market_name="XAUUSD",
                total_days=TICK_DAYS
            )

            save_tick_data(
                df=tick_df,
                market_name="XAUUSD",
                actual_symbol=xau_symbol
            )

        except Exception as error:

            print(
                f"\n❌ XAUUSD tick error:"
            )

            print(
                error
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATA COLLECTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nMarkets successfully resolved:"
    )

    for market_name, actual_symbol in (
        resolved_markets.items()
    ):

        print(
            f"  {market_name:<10} → "
            f"{actual_symbol}"
        )

    print(
        "\nRaw bar data:"
    )

    print(
        f"  {BARS_OUTPUT_DIR}"
    )

    if FETCH_TICKS:

        print(
            "\nTick data:"
        )

        print(
            f"  {TICKS_OUTPUT_DIR}"
        )

    print(
        "\nNext step:"
    )

    print(
        "Use these raw datasets to build the "
        "feature-engineering pipeline."
    )

    mt5.shutdown()

    print(
        "\nMT5 shutdown successfully."
    )

    print(
        "Done ✅"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()