from neat import nn
import datetime as dt
import time
import plot
from base_agent import Agent
from data_structures import Queue


class Validation(Agent):
    def __init__(self, settings, session, stock, finbert):
        super().__init__(settings, session, stock)
        self.finbert = finbert

    def validate(self, stock_bars, sp500_bars, nasdaq_bars,
                 genome, can_short, fractionable, short_limit,
                 k_period, d_period, rsi_period, start_cash):
        start_time = time.time()
        net = nn.RecurrentNetwork.create(genome, self.config)
        start_date = stock_bars[0]["timestamp"].to_pydatetime()
        settled_cash = float(start_cash)
        start_equity = float(start_cash)
        unsettled_cash = 0.0
        pending_sales = Queue()
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

        prev_d_ema = None
        prev_k_ema = None

        bar_k_period = self.days_to_bars(k_period, self.session["interval"])
        bar_d_period = self.days_to_bars(d_period, self.session["interval"])
        bar_rsi_period = self.days_to_bars(rsi_period, self.session["interval"])
        d_alpha = 2 / (bar_d_period + 1)
        k_alpha = 2 / (bar_k_period + 1)

        gain = 0
        loss = 0

        # Start at 1 to have previous bar for relative change
        num_bars = len(stock_bars)
        sp500_index = 0
        nasdaq_index = 0

        for i in range(1, num_bars):
            if i % 1000 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / i) * (num_bars - i)
                print(f" {self.stock['symbol']}: {i}/{num_bars} ({elapsed:.2f}s elapsed {eta:.2f}s remaining)")

            stock_bar = stock_bars[i]
            prev_stock_bar = stock_bars[i - 1]
            prev_date = prev_stock_bar["timestamp"].to_pydatetime()
            date = stock_bar["timestamp"].to_pydatetime()
            if date != prev_date:  # Check pending sales to settle cash after 1 day of sale
                consecutive_days += 1
                while not pending_sales.is_empty():
                    sale_price, sale_day = pending_sales.head.value
                    if consecutive_days - sale_day > 1:
                        settled_cash += sale_price
                        unsettled_cash -= sale_price
                        pending_sales.dequeue()
                    else:
                        break

            # Dealing with mismatch in length of bars for sp500 and nasdaq
            if sp500_index + 1 < len(sp500_bars):
                sp500_date = sp500_bars[sp500_index + 1]["timestamp"].to_pydatetime()
                if sp500_date <= date:
                    sp500_index += 1

            if nasdaq_index + 1 < len(nasdaq_bars):
                nasdaq_date = nasdaq_bars[nasdaq_index + 1]["timestamp"].to_pydatetime()
                if nasdaq_date <= date:
                    nasdaq_index += 1

            sp500_bar = sp500_bars[sp500_index]
            nasdaq_bar = nasdaq_bars[nasdaq_index]
            prev_sp500_bar = sp500_bars[min(0, sp500_index - 1)]
            prev_nasdaq_bar = nasdaq_bars[min(0, nasdaq_index - 1)]

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
            d_ema = Agent.calculate_ema(stock_bar["close"], d_alpha, prev_d_ema)
            prev_d_ema = d_ema
            k_ema = Agent.calculate_ema(stock_bar["close"], k_alpha, prev_k_ema)
            prev_k_ema = k_ema

            # Calculate RSI
            change = stock_bar["close"] - prev_stock_bar["close"]
            if change > 0:
                gain += change
            else:
                loss += abs(change)

            # Remove old data
            start_rsi_index = i - min(bar_rsi_period, i)
            if (i - start_rsi_index) + 1 >= bar_rsi_period:
                start_change = stock_bars[start_rsi_index]["close"] - stock_bars[start_rsi_index - 1]["close"]
                if start_change > 0:
                    gain -= change
                else:
                    loss -= abs(change)
            rsi = Agent.calculate_rsi(gain, loss, (i - start_rsi_index) + 1)

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
                      d_ema,
                      k_ema,
                      rsi]
            if can_short and shares < 0:
                inputs[0] = -1
                inputs[1] = Agent.rel_change(stock_bar["close"] * abs(shares), cost)

            outputs = net.activate(inputs)

            qty_percent = (outputs[1] + 1) * 0.5
            if outputs[0] > 0.5:  # Buy
                if can_short and shares < 0:
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
                    if not fractionable:
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
                if can_short and shares <= 0:
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
                    if not fractionable:
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
                        pending_sales.enqueue((price, consecutive_days))
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
                num_windows += 1
                start_equity = equity
                start_date = date

        print(pending_sales.head)

        if shares < 0:
            equity = unsettled_cash + settled_cash + shares * stock_bars[-1]["close"] - cost
        else:
            equity = unsettled_cash + settled_cash + stock_bars[-1]["close"] * shares
        total_profit = equity - float(start_cash)
        avg_profit = total_profit / num_windows
        stock_change = stock_bars[-1]['close'] - stock_bars[0]['close']
        print(f"{self.stock['symbol']} simulation finished in {(time.time() - start_time):.2f}s over {consecutive_days} trading days and {num_windows} profit windows"
              f"\n Stock change: ${round(stock_change, 2)} {round(100 * (stock_change / stock_bars[0]['close']), 4)}%"
              f"\n Total profit: ${round(total_profit, 2)} {round(100 * (total_profit / float(start_cash)), 4)}%"
              f"\n Average {self.session['profit_window']} day profit: ${round(avg_profit, 2)} {round(avg_profit / float(start_cash), 4)}%"
              f"\n Min profit: ${round(min_profit[0], 2)} {round(min_profit[1], 4)}% on {min_date}"
              f"\n Max profit: ${round(max_profit[0], 2)} {round(max_profit[1], 4)}% on {max_date}"
              f"\n Total short buys: {short_buys}"
              f"\n Total short sells: {short_sells}"
              f"\n Total long buys: {long_buys}"
              f"\n Total long sells: {long_sells}"
              f"\n Average actions/day: {len(log) / consecutive_days}")
        plot.plot_log(self.session["alpaca_api"], self.stock["symbol"], log, self.session["interval"])
        return log

