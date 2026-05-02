import MetaTrader5 as mt5
from datetime import datetime, timedelta

from monitoring.logger import trade_logger, error_logger


class AgentTracker:
    def __init__(self, portfolio_manager, agents):

        self.portfolio = portfolio_manager

        # map agent_id → config
        self.agents_dict = {a["id"]: a for a in agents}

        # prevents double processing of same deal
        self.tracked_deals = set()

        # memory safety limit (prevents infinite growth)
        self.max_tracked = 5000

    # =========================
    # SYNC CLOSED TRADES
    # =========================
    def sync_with_mt5(self, agent_ids):

        try:
            # 🔥 SHORTER WINDOW = FASTER + LESS LOAD
            from_date = datetime.now() - timedelta(hours=12)
            to_date = datetime.now()

            deals = mt5.history_deals_get(from_date, to_date)

            if deals is None:
                return

            for deal in deals:

                # =========================
                # FILTER ONLY OUR AGENTS
                # =========================
                if deal.magic not in agent_ids:
                    continue

                # =========================
                # AVOID DUPLICATES
                # =========================
                if deal.ticket in self.tracked_deals:
                    continue

                # mark immediately (important!)
                self.tracked_deals.add(deal.ticket)

                # =========================
                # ONLY CLOSED TRADES
                # =========================
                if deal.entry != mt5.DEAL_ENTRY_OUT:
                    continue

                agent_id = deal.magic

                try:
                    # =========================
                    # REAL PNL
                    # =========================
                    pnl = deal.profit + deal.commission + deal.swap

                    old_equity = self.portfolio.get_equity(agent_id)

                    if old_equity <= 0:
                        error_logger.error(
                            f"Invalid equity for agent {agent_id}"
                        )
                        continue

                    # =========================
                    # UPDATE EQUITY
                    # =========================
                    pnl_percent = pnl / old_equity
                    new_equity = old_equity * (1.0 + pnl_percent)

                    self.portfolio.update_equity(agent_id, pnl_percent)

                    # =========================
                    # APPLY COOLDOWN
                    # =========================
                    cooldown_val = self.agents_dict[agent_id].get(
                        "cooldown", 0
                    )

                    self.portfolio.set_cooldown(agent_id, cooldown_val)

                    trade_logger.info(
                        f"Agent {agent_id} CLOSED | "
                        f"Ticket={deal.ticket} | "
                        f"PnL={pnl:.2f} | "
                        f"Equity={new_equity:.2f} | "
                        f"Cooldown={cooldown_val}"
                    )

                except Exception as inner_e:
                    error_logger.error(
                        f"Deal processing error {deal.ticket}: {inner_e}"
                    )

            # =========================
            # MEMORY CLEANUP (CRITICAL)
            # =========================
            if len(self.tracked_deals) > self.max_tracked:
                # keep only latest subset
                self.tracked_deals = set(list(self.tracked_deals)[-2000:])

        except Exception as e:
            error_logger.error(f"AgentTracker sync error: {e}")

    # =========================
    # OPTIONAL STATS
    # =========================
    def get_tracked_stats(self, agent_id):

        try:
            return {
                "agent_id": agent_id,
                "tracked_deals": len(self.tracked_deals)
            }

        except Exception as e:
            error_logger.error(f"Stats error: {e}")
            return None