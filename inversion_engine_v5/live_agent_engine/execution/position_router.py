import MetaTrader5 as mt5
import time
from monitoring.logger import execution_logger, error_logger


class PositionRouter:
    def __init__(self, cache_ttl=0.5):
        """
        cache_ttl prevents MT5 overload when called multiple times per loop
        """
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._last_update = {}

    # =========================
    # INTERNAL CACHE CHECK
    # =========================
    def _should_refresh(self, agent_id):
        last = self._last_update.get(agent_id, 0)
        return (time.time() - last) > self.cache_ttl

    # =========================
    # GET POSITIONS (SAFE)
    # =========================
    def get_agent_positions(self, agent_id, symbol="XAUUSD"):
        try:
            # 🔥 ALWAYS fetch all positions first (more reliable)
            positions = mt5.positions_get()

            if positions is None:
                error_logger.warning(
                    f"MT5 returned None positions (agent {agent_id})"
                )
                return []

            # 🔥 MANUAL FILTER (CRITICAL FIX)
            filtered = [
                p for p in positions
                if p.magic == int(agent_id) and p.symbol == symbol
            ]

            return filtered

        except Exception as e:
            error_logger.error(
                f"Error fetching positions for agent {agent_id}: {e}"
            )
            return []

    # =========================
    # CHECK OPEN POSITION
    # =========================
    def has_open_position(self, agent_id, symbol="XAUUSD"):

        # 🔥 Force refresh if agent not in cache yet
        if agent_id not in self._cache:
            positions = self.get_agent_positions(agent_id, symbol)
            self._cache[agent_id] = positions
            self._last_update[agent_id] = time.time()
            return len(positions) > 0

        # 🔥 Cache logic
        if not self._should_refresh(agent_id):
            cached = self._cache.get(agent_id, [])
            return len(cached) > 0

        # 🔥 Refresh from MT5
        positions = self.get_agent_positions(agent_id, symbol)

        # 🔥 IMPORTANT: Only overwrite cache if MT5 returned valid data
        if positions is not None:
            self._cache[agent_id] = positions
            self._last_update[agent_id] = time.time()

        return len(positions) > 0

    # =========================
    # GET POSITION COUNT
    # =========================
    def position_count(self, agent_id, symbol="XAUUSD"):
        positions = self.get_agent_positions(agent_id, symbol)
        return len(positions)

    # =========================
    # GET PENDING ORDERS (NEW)
    # =========================
    def get_agent_pending_orders(self, agent_id, symbol="XAUUSD"):
        try:
            orders = mt5.orders_get()
            if orders is None:
                return []

            filtered = [
                o for o in orders
                if o.magic == int(agent_id) and o.symbol == symbol
            ]
            return filtered
        except Exception as e:
            error_logger.error(f"Error fetching pending orders for agent {agent_id}: {e}")
            return []

    # =========================
    # GET POSITION DETAILS
    # =========================
    def get_position_details(self, agent_id, symbol="XAUUSD"):

        positions = self.get_agent_positions(agent_id, symbol)

        if not positions:
            return None

        return [
            {
                "ticket": p.ticket,
                "type": p.type,
                "volume": p.volume,
                "price_open": p.price_open,
                "profit": p.profit,
                "sl": p.sl,
                "tp": p.tp,
            }
            for p in positions
        ]