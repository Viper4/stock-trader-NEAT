from neat import nn
import time
import plot
from Agents.base_agent import Agent
from data_structures import Queue
from tqdm import tqdm


class Validation(Agent):
    def __init__(self, settings, profile, stock, finbert):
        super().__init__(settings, profile, stock)
        self.finbert = finbert

    def validate(self, bars, ma_periods,
                 genome, fractionable,
                 start_cash):
        start_time = time.time()
        net = nn.RecurrentNetwork.create(genome, self.config)
        start_date = bars.index[0].date()
        settled_cash = float(start_cash)
        start_equity = float(start_cash)
        unsettled_cash = 0.0
        pending_sales = Queue()
        num_windows = 0
        shares = 0.0
        cost = 0.0
        consecutive_days = 1
        long_sells = 0
        long_buys = 0
        min_profit = (9999999, 9999999)
        min_date = None
        max_profit = (-9999999, -9999999)
        max_date = None
        last_index = bars.index[-1]
        log = []

        prev_date = None
        for row in tqdm(bars.itertuples(), total=bars.shape[0]):
            date = row.Index.to_pydatetime()

            # Check to settle cash after each day
            if prev_date is not None and (date - prev_date).days > 1:
                consecutive_days += 1
                while not pending_sales.is_empty():
                    sale_price, sale_day = pending_sales.head.value
                    if consecutive_days - sale_day >= 1:
                        settled_cash += sale_price
                        unsettled_cash -= sale_price
                        pending_sales.dequeue()
                    else:
                        break

            inputs = self.generate_inputs(row, Agent.rel_change(cost, row.close * shares), ma_periods)

            outputs = net.activate(inputs)

            qty_percent = (outputs[1] + 1) * 0.5
            if outputs[0] > 0.5:  # Buy
                quantity = qty_percent * settled_cash * self.stock["cash_at_risk"] / row.close
                if not fractionable:
                    quantity = round(quantity)
                price = quantity * row.close
                if price >= 1:  # Alpaca doesn't allow trades under $1
                    cost += price
                    shares += quantity
                    settled_cash -= price

                    action = {"inputs": inputs, "outputs": outputs,
                              "side": "Buy", "type": "long", "quantity": quantity, "price": row.close,
                              "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                              "datetime": date}
                    log.append(action)
                    long_buys += 1
            elif outputs[0] < -0.5:  # Sell
                quantity = qty_percent * shares
                if not fractionable:
                    quantity = round(quantity)
                price = quantity * row.close * (1 - self.stock["transaction_fee"])
                if price >= 1:
                    if shares - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                        price = shares * row.close * (1 - self.stock["transaction_fee"])
                        action = {"inputs": inputs, "outputs": outputs,
                                  "side": "Sell", "type": "long", "quantity": quantity, "price": row.close,
                                  "profit": price - cost, "settled_cash": settled_cash,
                                  "unsettled_cash": unsettled_cash + price,
                                  "datetime": date}
                        log.append(action)
                        shares = 0.0
                        cost = 0.0
                    else:
                        avg_cost = cost / shares
                        shares -= quantity
                        cost = avg_cost * shares
                        action = {"inputs": inputs, "outputs": outputs,
                                  "side": "Sell", "type": "long", "quantity": quantity, "price": row.close,
                                  "profit": price - (avg_cost * quantity), "settled_cash": settled_cash,
                                  "unsettled_cash": unsettled_cash + price,
                                  "datetime": date}
                        log.append(action)
                    unsettled_cash += price
                    pending_sales.enqueue((price, consecutive_days))
                    long_sells += 1

            if row.Index == last_index or (date - start_date).days >= self.profile.profit_window:
                equity = unsettled_cash + settled_cash + row.close * shares
                profit = equity - start_equity
                if profit < min_profit[0]:
                    min_profit = (profit, 100 * (profit / start_equity))
                    min_date = start_date
                if profit > max_profit[0]:
                    max_profit = (profit, 100 * (profit / start_equity))
                    max_date = start_date
                num_windows += 1
                start_equity = equity
                start_date = date

            prev_date = date

        first_row = bars.iloc[0]
        last_row = bars.iloc[-1]
        if shares < 0:
            equity = unsettled_cash + settled_cash + shares * last_row.close - cost
        else:
            equity = unsettled_cash + settled_cash + last_row.close * shares
        total_profit = equity - float(start_cash)
        avg_profit = total_profit / num_windows
        stock_change = last_row.close - first_row.close
        print(f"{self.stock['symbol']} simulation finished in {(time.time() - start_time):.2f}s over {consecutive_days} trading days and {num_windows} profit windows"
              f"\n Stock change: ${round(stock_change, 2)} {round(100 * (stock_change / first_row.close), 4)}%"
              f"\n Total profit: ${round(total_profit, 2)} {round(100 * (total_profit / float(start_cash)), 4)}%"
              f"\n Average {self.profile.profit_window} day profit: ${round(avg_profit, 2)} {round(avg_profit / float(start_cash), 4)}%"
              f"\n Min profit: ${round(min_profit[0], 2)} {round(min_profit[1], 4)}% on {min_date}"
              f"\n Max profit: ${round(max_profit[0], 2)} {round(max_profit[1], 4)}% on {max_date}"
              f"\n Total long buys: {long_buys}"
              f"\n Total long sells: {long_sells}"
              f"\n Average actions/day: {len(log) / consecutive_days}")
        plot.plot_log(self.profile.alpaca_api, self.stock["symbol"], log, self.profile.interval)
        return log

