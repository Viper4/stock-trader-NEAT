from neat import nn
import datetime as dt
import time
import plot
from base_agent import Agent


class Validation(Agent):
    def __init__(self, settings, session, stock, finbert):
        super().__init__(settings, session, stock)
        self.finbert = finbert

    def validate(self, stock_bars, sp500_bars, nasdaq_bars,
                 genome, shorting, asset, short_limit,
                 k_period, d_period, rsi_period, start_cash):
        start_time = time.time()
        net = nn.RecurrentNetwork.create(genome, self.config)
        start_date = stock_bars[0]["timestamp"].date()
        settled_cash = start_cash
        start_equity = start_cash
        unsettled_cash = 0.0
        pending_sales = []
        profit_sum = 0.0
        num_windows = 0
        shares = 0.0
        cost = 0.0
        consecutive_days = 1
        log = []
        short_sells = 0
        short_buys = 0
        long_sells = 0
        long_buys = 0
        min_profit = (999999, 999999)
        min_date = None
        max_profit = (-999999, -999999)
        max_date = None

        prev_ema = None

        bar_k_period = self.days_to_bars(k_period, self.session["interval"])
        bar_d_period = self.days_to_bars(d_period, self.session["interval"])
        bar_rsi_period = self.days_to_bars(rsi_period, self.session["interval"])
        alpha = 2 / (bar_d_period + 1)

        # Start at 1 to have previous bar for relative change
        num_bars = len(stock_bars)
        for i in range(1, num_bars):
            stock_bar = stock_bars[i]
            sp500_bar = sp500_bars[i]
            nasdaq_bar = nasdaq_bars[i]
            prev_stock_bar = stock_bars[i - 1]
            prev_sp500_bar = sp500_bars[i - 1]
            prev_nasdaq_bar = nasdaq_bars[i - 1]
            prev_date = prev_stock_bar["timestamp"].date()
            date = stock_bar["timestamp"].date()
            if date != prev_date:  # Check pending sales to settle cash after 1 day of sale
                consecutive_days += 1
                for j in reversed(range(len(pending_sales))):
                    sale_price, sale_day = pending_sales[j]
                    if consecutive_days - sale_day >= 1:
                        settled_cash += sale_price
                        unsettled_cash -= sale_price
                        pending_sales.pop(j)

            backtest_date = stock_bar["timestamp"].to_pydatetime()
            stock_sentiment = self.finbert.get_saved_sentiment(self.stock["symbol"],
                                                               backtest_date - dt.timedelta(days=2),
                                                               backtest_date)
            sp500_sentiment = self.finbert.get_saved_sentiment("SPY",
                                                               backtest_date - dt.timedelta(days=2),
                                                               backtest_date)
            nasdaq_sentiment = self.finbert.get_saved_sentiment("QQQ",
                                                                backtest_date - dt.timedelta(days=2),
                                                                backtest_date)

            k_percent = Agent.calculate_k_percent(stock_bars[i - min(bar_k_period, i):i])

            # %D = EMA(%K, N) or SMA(%K, N)
            ema = Agent.calculate_ema(stock_bar["close"], alpha, prev_ema)
            norm_ema = 2 * ((ema - stock_bar["close"]) / stock_bar["close"])
            prev_ema = ema
            k_sma = Agent.calculate_sma(stock_bars[i - min(bar_k_period, i):i])
            norm_k_sma = 2 * ((k_sma - stock_bar["close"]) / stock_bar["close"])
            d_sma = Agent.calculate_sma(stock_bars[i - min(bar_d_period, i):i])
            norm_d_sma = 2 * ((d_sma - stock_bar["close"]) / stock_bar["close"])

            rsi = Agent.calculate_rsi(stock_bars[i - min(bar_rsi_period, i):i])

            inputs = [1,  # -1 = short, 1 = long
                      Agent.rel_change(cost, stock_bar["close"] * shares),  # plpc
                      Agent.rel_change(prev_stock_bar["open"], stock_bar["open"]),
                      Agent.rel_change(prev_stock_bar["high"], stock_bar["high"]),
                      Agent.rel_change(prev_stock_bar["low"], stock_bar["low"]),
                      Agent.rel_change(prev_stock_bar["close"], stock_bar["close"]),
                      Agent.rel_change(prev_stock_bar["volume"], stock_bar["volume"]),
                      Agent.rel_change(prev_stock_bar["vwap"], stock_bar["vwap"]),
                      stock_sentiment,  # -1 = negative, 0 = neutral, 1 = positive
                      Agent.rel_change(prev_sp500_bar["close"], sp500_bar["close"]),
                      Agent.rel_change(prev_sp500_bar["volume"], sp500_bar["volume"]),
                      sp500_sentiment,
                      Agent.rel_change(prev_nasdaq_bar["close"], nasdaq_bar["close"]),
                      Agent.rel_change(prev_nasdaq_bar["volume"], nasdaq_bar["volume"]),
                      nasdaq_sentiment,
                      k_percent,
                      norm_ema,
                      norm_k_sma,
                      norm_d_sma,
                      rsi]
            if shorting and shares < 0:
                inputs[0] = -1
                inputs[1] = Agent.rel_change(stock_bar["close"] * abs(shares), cost)

            outputs = net.activate(inputs)

            qty_percent = (outputs[1] + 1) * 0.5
            if outputs[0] > 0.5:  # Buy
                if shorting and asset.shortable and shares < 0:
                    quantity = qty_percent * abs(shares)
                    quantity = round(quantity)  # Shorts don't allow fractional qty
                    price = quantity * stock_bar["close"] * (1 - self.stock["transaction_fee"])
                    if price >= 1:
                        if abs(shares) - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                            price = abs(shares) * stock_bar["close"] * (1 - self.stock["transaction_fee"])
                            profit = cost - price
                            shares = 0.0
                            cost = 0.0
                        else:
                            avg_cost = cost / abs(shares)
                            shares += quantity
                            cost = avg_cost * abs(shares)
                            profit = (avg_cost * quantity) - price
                        settled_cash += profit

                        short_buys += 1
                        action = {"inputs": inputs, "outputs": outputs,
                                  "side": "Buy", "type": "short", "quantity": abs(quantity), "price": stock_bar["close"],
                                  "profit": profit, "settled_cash": settled_cash,
                                  "unsettled_cash": unsettled_cash,
                                  "datetime": stock_bar["timestamp"].to_pydatetime()}
                        log.append(action)
                else:
                    quantity = qty_percent * settled_cash * self.stock["cash_at_risk"] / stock_bar["close"]
                    if not asset.fractionable:
                        quantity = round(quantity)
                    price = quantity * stock_bar["close"]
                    if price >= 1:  # Alpaca doesn't allow trades under $1
                        cost += price
                        shares += quantity
                        settled_cash -= price

                        action = {"inputs": inputs, "outputs": outputs,
                                  "side": "Buy", "type": "long", "quantity": quantity, "price": stock_bar["close"],
                                  "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                  "datetime": stock_bar["timestamp"].to_pydatetime()}
                        log.append(action)
                        long_buys += 1
            elif outputs[0] < -0.5:  # Sell
                if shorting and asset.shortable and shares <= 0:
                    quantity = qty_percent * (short_limit - cost) * self.stock["cash_at_risk"] / stock_bar["close"]
                    quantity = round(quantity)  # Shorts don't allow fractional qty
                    price = quantity * stock_bar["close"] * (1 - self.stock["transaction_fee"])
                    if cost + price < short_limit:
                        if price >= 1:  # Alpaca doesn't allow trades under $1
                            cost += price
                            shares -= quantity

                            action = {"inputs": inputs, "outputs": outputs,
                                      "side": "Sell", "type": "short", "quantity": abs(quantity), "price": stock_bar["close"],
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": stock_bar["timestamp"].to_pydatetime()}
                            log.append(action)
                            short_sells += 1
                elif shares > 0:
                    quantity = qty_percent * shares
                    if not asset.fractionable:
                        quantity = round(quantity)
                    price = quantity * stock_bar["close"] * (1 - self.stock["transaction_fee"])
                    if price >= 1:
                        if shares - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                            price = shares * stock_bar["close"] * (1 - self.stock["transaction_fee"])
                            action = {"inputs": inputs, "outputs": outputs,
                                      "side": "Sell", "type": "long", "quantity": quantity, "price": stock_bar["close"],
                                      "profit": price - cost, "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash + price,
                                      "datetime": stock_bar["timestamp"].to_pydatetime()}
                            log.append(action)
                            shares = 0.0
                            cost = 0.0
                        else:
                            avg_cost = cost / shares
                            shares -= quantity
                            cost = avg_cost * shares
                            action = {"inputs": inputs, "outputs": outputs,
                                      "side": "Sell", "type": "long", "quantity": quantity, "price": stock_bar["close"],
                                      "profit": price - (avg_cost * quantity), "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash + price,
                                      "datetime": stock_bar["timestamp"].to_pydatetime()}
                            log.append(action)
                        unsettled_cash += price
                        pending_sales.append((price, consecutive_days))
                        long_sells += 1
            if i == num_bars - 1 or (date - start_date).days >= self.session["profit_window"]:
                if shares < 0:
                    equity = unsettled_cash + settled_cash + shares * stock_bar["close"] - cost
                else:
                    equity = unsettled_cash + settled_cash + stock_bar["close"] * shares
                profit = equity - start_equity
                if profit < min_profit[0]:
                    min_profit = (profit, 100 * (profit / start_equity))
                    min_date = start_date
                if profit > max_profit[0]:
                    max_profit = (profit, 100 * (profit / start_equity))
                    max_date = start_date
                profit_sum += profit
                num_windows += 1
                start_equity = equity
                start_date = date

        avg_profit = profit_sum / num_windows
        stock_change = stock_bars[-1]['close'] - stock_bars[0]['close']
        print(f"Simulation finished in {str(time.time() - start_time)} seconds over {consecutive_days} trading days and {num_windows} profit windows"
              f"\n Stock change: ${round(stock_change, 2)} {round(100 * (stock_change / stock_bars[0]['close']), 4)}%"
              f"\n Total profit: ${round(profit_sum, 2)} {round(100 * (profit_sum / start_cash), 4)}%"
              f"\n Average {self.session['profit_window']} day profit: ${round(avg_profit, 2)} {round(avg_profit / start_cash, 4)}%"
              f"\n Min profit: ${round(min_profit[0], 2)} {round(min_profit[1], 4)}% on {min_date}"
              f"\n Max profit: ${round(max_profit[0], 2)} {round(max_profit[1], 4)}% on {max_date}"
              f"\n Total short buys: {short_buys}"
              f"\n Total short sells: {short_sells}"
              f"\n Total long buys: {long_buys}"
              f"\n Total long sells: {long_sells}"
              f"\n Average actions/day: {len(log) / consecutive_days}")
        plot.plot_log(self.session["alpaca_api"], self.stock["symbol"], log, self.session["interval"])
        while True:
            user_input = input("Enter action index or exit: ")
            if user_input == "exit":
                return
            else:
                i = int(user_input)
                if len(log) > i >= 0:
                    print("Action at " + str(i))
                    action = log[i]
                    for key in action:
                        if key == "inputs":
                            print("-Inputs")
                            print(f" |Short/Long: {action[key][0]}")
                            print(f" |PLPC: {action[key][1]}")
                            print(f" |Open: {action[key][2]}")
                            print(f" |High: {action[key][3]}")
                            print(f" |Low: {action[key][4]}")
                            print(f" |Close: {action[key][5]}")
                            print(f" |Volume: {action[key][6]}")
                            print(f" |VWAP: {action[key][7]}")
                            print(f" |{stock_bars[0]['symbol']} Sentiment: {action[key][8]}")
                            print(f" |S&P 500 Close: {action[key][9]}")
                            print(f" |S&P 500 Volume: {action[key][10]}")
                            print(f" |S&P 500 Sentiment: {action[key][11]}")
                            print(f" |NASDAQ Close: {action[key][12]}")
                            print(f" |NASDAQ Volume: {action[key][13]}")
                            print(f" |NASDAQ Sentiment: {action[key][14]}")
                            print(f" |%K: {action[key][15]}")
                            print(f" |{k_period}-day EMA: {action[key][16]}")
                            print(f" |{k_period}-day SMA: {action[key][17]}")
                            print(f" |{d_period}-day SMA: {action[key][18]}")
                            print(f" |{rsi_period}-day RSI: {action[key][19]}")
                        elif key == "outputs":
                            print("-Outputs")
                            print(f" |Buy/Sell: {action[key][0]}")
                            print(f" |Quantity: {action[key][1]}")
                        else:
                            print(f"-{key}: {action[key]}")
                else:
                    print("Index not in range of log")