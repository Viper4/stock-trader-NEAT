import datetime as dt
import pytz
import time
from base_agent import Agent


class Trading(Agent):
    def __init__(self, settings, stock, trader):
        super().__init__(settings, None, stock)
        self.trader = trader
        self.net = None

    def run(self):
        if self.running:
            return
        print(f"{self.trader.profile['name']} {self.stock['symbol']}: Starting trading")
        self.running = True
        cum_stock_price = 0
        cum_stock_vol = 0
        prev_stock_candle = None

        prev_sp500_candle = None

        prev_nasdaq_candle = None

        max_period = max(self.trader.profile["k_period"], self.trader.profile["d_period"], self.trader.profile["rsi_period"])
        k_period = self.days_to_bars(self.trader.profile["k_period"], self.trader.profile["interval"])
        d_period = self.days_to_bars(self.trader.profile["d_period"], self.trader.profile["interval"])
        rsi_period = self.days_to_bars(self.trader.profile["rsi_period"], self.trader.profile["interval"])
        d_alpha = 2 / (d_period + 1)
        prev_d_ema = None
        k_alpha = 2 / (k_period + 1)
        prev_k_ema = None

        gain = 0
        loss = 0

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status():
                # Stock candles for today
                stock_candles, prev_stock_close = self.trader.scraper.get_latest_candles(self.stock["symbol"], interval=str(self.trader.profile["interval"]) + "m")
                stock_latest = stock_candles[-1]
                cum_stock_price += stock_latest["volume"] * ((stock_latest["high"] + stock_latest["low"] + stock_latest["close"]) / 3)
                cum_stock_vol += stock_latest["volume"]
                stock_latest["vwap"] = cum_stock_price / cum_stock_vol if cum_stock_vol > 0 else 0

                if prev_stock_candle is None:
                    if len(stock_candles) >= 2:
                        prev_stock_candle = stock_candles[-2]
                    else:
                        prev_stock_candle = stock_latest
                        prev_stock_candle["close"] = prev_stock_close
                    prev_stock_candle["vwap"] = (prev_stock_candle["high"] + prev_stock_candle["low"] +
                                                 prev_stock_candle["close"]) / 3

                # SP500 candles for today
                sp500_candles, prev_sp500_close = self.trader.scraper.get_latest_candles("SPY", interval=str(self.trader.profile["interval"]) + "m")
                sp500_latest = sp500_candles[-1]

                if prev_sp500_candle is None:
                    if len(sp500_candles) >= 2:
                        prev_sp500_candle = sp500_candles[-2]
                    else:
                        prev_sp500_candle = stock_latest
                        prev_sp500_candle["close"] = prev_sp500_close

                # NASDAQ candles for today
                nasdaq_candles, prev_nasdaq_close = self.trader.scraper.get_latest_candles("QQQ", interval=str(self.trader.profile["interval"]) + "m")
                nasdaq_latest = nasdaq_candles[-1]

                if prev_nasdaq_candle is None:
                    if len(nasdaq_candles) >= 2:
                        prev_nasdaq_candle = nasdaq_candles[-2]
                    else:
                        prev_nasdaq_candle = stock_latest
                        prev_nasdaq_candle["close"] = prev_nasdaq_close

                # Get current position
                position = self.trader.schwab_api.get_position(self.stock["symbol"])
                stock_sentiment = self.trader.finbert.get_api_sentiment(self.stock["symbol"],
                                                                        now_date - dt.timedelta(days=2), now_date)
                sp500_sentiment = self.trader.finbert.get_api_sentiment("SPY",
                                                                        now_date - dt.timedelta(days=2), now_date)
                nasdaq_sentiment = self.trader.finbert.get_api_sentiment("QQQ",
                                                                        now_date - dt.timedelta(days=2), now_date)

                # Get historical data for momentum indicators

                stock_bars = self.trader.get_bars(self.stock["symbol"],
                                                  self.trader.alpaca_api,
                                                  self.trader.profile["interval"],
                                                  now_date - dt.timedelta(days=max_period + 1),
                                                  now_date - dt.timedelta(days=1),
                                                  500000,
                                                  False)
                stock_bars.append(stock_candles)  # Add today's data

                last_index = len(stock_bars) - 1
                k_percent = self.calculate_k_percent(stock_bars[last_index - min(k_period, last_index):last_index])

                d_ema = self.calculate_ema(stock_latest["close"], d_alpha, prev_d_ema)
                prev_d_ema = d_ema
                k_ema = self.calculate_ema(stock_latest["close"], k_alpha, prev_k_ema)
                prev_k_ema = k_ema

                # Calculate RSI
                change = stock_latest["close"] - prev_stock_candle["close"]
                if change > 0:
                    gain += change
                else:
                    loss += abs(change)

                # Remove old data
                i = len(stock_candles) - 1
                start_rsi_index = i - min(rsi_period, i)
                if (i - start_rsi_index) + 1 >= rsi_period:
                    start_change = stock_bars[start_rsi_index]["close"] - stock_bars[start_rsi_index - 1]["close"]
                    if start_change > 0:
                        gain -= change
                    else:
                        loss -= abs(change)

                rsi = self.calculate_rsi(gain, loss, (i - start_rsi_index) + 1)

                inputs = [0,  # -1 = shorting, 1 = longing
                          0,  # profit/loss percent
                          self.rel_change(prev_stock_candle["open"], stock_latest["open"]),
                          self.rel_change(prev_stock_candle["high"], stock_latest["high"]),
                          self.rel_change(prev_stock_candle["low"], stock_latest["low"]),
                          self.rel_change(prev_stock_candle["close"], stock_latest["close"]),
                          self.rel_change(prev_stock_candle["volume"], stock_latest["volume"]),
                          self.rel_change(prev_stock_candle["vwap"], stock_latest["vwap"]),
                          stock_sentiment,  # -1 = negative, 0 = neutral, 1 = positive
                          self.rel_change(prev_sp500_candle["close"], sp500_latest["close"]),
                          self.rel_change(prev_sp500_candle["volume"], sp500_latest["volume"]),
                          sp500_sentiment,
                          self.rel_change(prev_nasdaq_candle["close"], nasdaq_latest["close"]),
                          self.rel_change(prev_nasdaq_candle["volume"], nasdaq_latest["volume"]),
                          nasdaq_sentiment,
                          k_percent,
                          d_ema,
                          k_ema,
                          rsi]

                if "shortQuantity" in position and position["shortQuantity"] > 0:
                    inputs[0] = -1
                    position_qty = -position["shortQuantity"]
                    if position["shortOpenProfitLoss"] > 0:
                        inputs[1] = position["shortOpenProfitLoss"] / position["averagePrice"]
                else:
                    inputs[0] = 1
                    position_qty = position["longQuantity"]
                    if position["longOpenProfitLoss"] > 0:
                        inputs[1] = position["longOpenProfitLoss"] / position["averagePrice"]

                outputs = self.net.activate(inputs)

                qty_percent = (outputs[1] + 1) * 0.5

                asset = self.trader.alpaca_api.get_asset(symbol=self.stock["symbol"])
                if outputs[0] > 0.5:  # Buy
                    if self.stock["shorting"] and asset.shortable and position_qty < 0:
                        account = self.trader.schwab_api.get_account()
                        unsettled_cash = account["currentBalances"]["unsettledCash"]
                        settled_cash = account["currentBalances"]["cashAvailableForTrading"] - unsettled_cash
                        quantity = abs(round(qty_percent * position_qty))
                        price = quantity * stock_latest["close"]
                        if abs(price) >= 1:
                            if abs(position_qty - quantity) < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=abs(quantity), side="BUY")
                            else:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=abs(quantity), side="BUY")

                            action = {"side": "Buy", "type": "short", "quantity": abs(quantity), "price": stock_latest["close"],
                                      "profit": price - (position["averagePrice"] * quantity),
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": now_date}
                            print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                            self.trader.logs[self.stock["symbol"]].append(action)
                    else:
                        account = self.trader.schwab_api.get_account()
                        unsettled_cash = account["currentBalances"]["unsettledCash"]
                        settled_cash = account["currentBalances"]["cashAvailableForTrading"] - unsettled_cash

                        if "longMarketValue" in account["currentBalances"]:
                            market_value = account["currentBalances"]["longMarketValue"]
                        else:
                            market_value = 0
                        used_cash = market_value + unsettled_cash
                        if used_cash < self.trader.profile["cash_limit"]:
                            quantity = min(self.trader.profile["cash_limit"], settled_cash) * qty_percent * self.stock["cash_at_risk"] / stock_latest["close"]
                            quantity = round(quantity)
                            if quantity > 0:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="BUY")

                                action = {"side": "Buy", "type": "long", "quantity": quantity, "price": stock_latest["close"],
                                          "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                          "datetime": now_date}
                                print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                                self.trader.logs[self.stock["symbol"]].append(action)
                elif outputs[0] < -0.5:  # Sell
                    account = self.trader.schwab_api.get_account()
                    unsettled_cash = account["currentBalances"]["unsettledCash"]
                    settled_cash = account["currentBalances"]["cashAvailableForTrading"] - unsettled_cash
                    if self.stock["shorting"] and asset.shortable and position_qty <= 0:
                        cost = position["averagePrice"] * position_qty

                        if abs(cost) < self.trader.profile["short_limit"]:
                            quantity = round(-qty_percent * (self.trader.profile["short_limit"] - abs(cost)) * self.stock["cash_at_risk"] / stock_latest["close"])
                            if quantity > 0:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=abs(quantity), side="SELL")

                                action = {"side": "Sell", "type": "short", "quantity": quantity, "price": stock_latest["close"],
                                          "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                          "datetime": now_date}
                                print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                                self.trader.logs[self.stock["symbol"]].append(action)
                    else:
                        quantity = round(qty_percent * position_qty)
                        price = quantity * stock_latest["close"]
                        if price >= 1:
                            if position_qty - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="SELL")
                            else:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="SELL")

                            # profit = price - cost
                            action = {"side": "Sell", "type": "long", "quantity": quantity, "price": stock_latest["close"],
                                      "profit": price - (position["averageLongPrice"] * quantity),
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": now_date}
                            print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                            self.trader.logs[self.stock["symbol"]].append(action)
                prev_stock_candle = stock_latest
                prev_sp500_candle = sp500_latest
                prev_nasdaq_candle = nasdaq_latest

                time.sleep(self.trader.profile["interval"] * 60)
            else:
                cum_stock_price = 0.0
                cum_stock_vol = 0.0

                next_open = self.trader.clock[0].next_open
                wait_time = (next_open - now_date).total_seconds()
                wait_time += self.trader.profile["interval"] * 60 + 10  # Wait for yahoo finance to update
                print(f"{self.trader.profile['name']} {self.stock['symbol']}: Pausing trading. Waiting until market opens in {wait_time / 3600} hours")
                time.sleep(wait_time)
                print(f"{self.trader.profile['name']} {self.stock['symbol']}: Resuming trading")