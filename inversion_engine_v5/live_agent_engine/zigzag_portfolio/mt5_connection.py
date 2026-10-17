"""MT5 connection dedicated to the ZigZag portfolio.

Reads account credentials from the existing live_agent_engine/config/settings.py.
The optional terminal executable path is supplied through ZIGZAG_MT5_TERMINAL_PATH.
"""
from __future__ import annotations

import os
import time

import MetaTrader5 as mt5

from config import settings as account_settings
from zigzag_portfolio import settings


class ZigZagMT5Connection:
    def __init__(self) -> None:
        self.last_reconnect_time = 0.0
        self.reconnect_cooldown = 5.0

    def _account_connection_valid(self) -> bool:
        account = mt5.account_info()
        return account is not None and bool(str(getattr(account, "server", "") or ""))

    def connect(self) -> bool:
        terminal_path = (
            os.getenv("ZIGZAG_MT5_TERMINAL_PATH")
            or getattr(account_settings, "MT5_TERMINAL_PATH", None)
        )
        try:
            login = int(account_settings.MT5_LOGIN)
            password = str(account_settings.MT5_PASSWORD)
            server = str(account_settings.MT5_SERVER).strip()
        except (TypeError, ValueError):
            print("MT5 credentials are missing or invalid; check live_agent_engine/.env.")
            return False
        if login <= 0 or not password or not server:
            print("MT5 credentials are missing; set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER in live_agent_engine/.env.")
            return False
        kwargs = {"login": login, "password": password, "server": server}
        try:
            initialized = (
                mt5.initialize(terminal_path, **kwargs)
                if terminal_path
                else mt5.initialize(**kwargs)
            )
            if not initialized:
                print(f"MT5 initialize/login failed: {mt5.last_error()}")
                return False
            terminal = mt5.terminal_info()
            account = mt5.account_info()
            if terminal is None or not terminal.connected or account is None:
                print("MT5 terminal/account is not connected; refusing to start.")
                mt5.shutdown()
                return False
            print(
                f"MT5 connected | login={account.login} | server={account.server} "
                f"| balance={account.balance}"
            )
            return True
        except Exception as exc:
            print(f"MT5 connection exception: {exc}")
            try:
                mt5.shutdown()
            except Exception:
                pass
            return False

    def ensure_connection(self) -> bool:
        try:
            terminal = mt5.terminal_info()
            account = mt5.account_info()
            if terminal is not None and terminal.connected and account is not None:
                return self._account_connection_valid()
            now = time.time()
            if now - self.last_reconnect_time < self.reconnect_cooldown:
                return False
            self.last_reconnect_time = now
            mt5.shutdown()
            time.sleep(1)
            return self.connect()
        except Exception as exc:
            print(f"MT5 connection check failed: {exc}")
            return False
