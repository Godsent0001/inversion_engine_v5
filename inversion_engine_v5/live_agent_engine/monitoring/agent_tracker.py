import MetaTrader5 as mt5
from datetime import datetime, timedelta
from monitoring.logger import trade_logger, error_logger

class AgentTracker:
    def __init__(self, portfolio_manager, agents):
        self.portfolio = portfolio_manager
        self.agents_dict = {a["id"]: a for a in agents}
        self.tracked_deals = set() # Set of deal tickets already processed

    def sync_with_mt5(self, agent_ids):
        """
        Check MT5 history for closed deals and update internal portfolio equity.
        """
        # Look back 1 day for deals (adjust as needed)
        from_date = datetime.now() - timedelta(days=1)
        to_date = datetime.now()

        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None:
            return

        for deal in deals:
            # We only care about deals with our magic numbers (agent_ids)
            if deal.magic in agent_ids:
                if deal.ticket not in self.tracked_deals:
                    # Check if it's a closing deal (entry == DEAL_ENTRY_OUT)
                    if deal.entry == mt5.DEAL_ENTRY_OUT:
                        pnl = deal.profit + deal.commission + deal.swap

                        # Update internal virtual equity
                        # Since risk is 5% of agent's $10k, we need to map real $ pnl to virtual % pnl
                        # Or simply update the virtual balance by the real $ amount.
                        # Given the prompt, each agent starts with $10,000 virtual balance.

                        agent_id = deal.magic
                        old_equity = self.portfolio.get_equity(agent_id)

                        # We apply the profit/loss to the virtual equity
                        # We use the deal's profit directly.
                        # Note: In a real multi-agent single account setup, you'd calculate
                        # the lot size based on the virtual balance, so the profit here
                        # is already proportional to that virtual balance.

                        new_equity = old_equity + pnl
                        # Update equity (we need a direct set method or use update_equity with a calculated %)
                        pnl_percent = pnl / old_equity
                        self.portfolio.update_equity(agent_id, pnl_percent)

                        # Set cooldown after trade close (matches backtest logic)
                        cooldown_val = self.agents_dict[agent_id].get("cooldown", 0)
                        self.portfolio.set_cooldown(agent_id, cooldown_val)

                        trade_logger.info(f"Agent {agent_id} Trade Closed: Ticket {deal.ticket}, PnL: {pnl:.2f}, New Virtual Equity: {new_equity:.2f}, Cooldown Set: {cooldown_val}")

                    self.tracked_deals.add(deal.ticket)

    def get_tracked_stats(self, agent_id):
        # Additional stats like total trades, winrate could be tracked here
        pass
