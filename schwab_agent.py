import datetime as dt
import os
import pytz
import time
import saving
from base_agent import Agent


class Trading(Agent):
    def __init__(self, settings, stock, trader):
        super().__init__(settings, None, stock)
        self.trader = trader
        self.net = None
        self.genome = None

    def save_memory(self):
        values_path = self.settings["save_path"] + f"/Values/{self.trader.profile['name'].replace(' ', '-')}-{self.stock['symbol']}.gz"
        saving.SaveSystem.save_data((self.net.values, self.net.active), values_path)

    def load_memory(self):
        values_path = self.settings["save_path"] + f"/Values/{self.trader.profile['name'].replace(' ', '-')}-{self.stock['symbol']}.gz"

        if os.path.exists(values_path):
            self.net.values, self.net.active = saving.SaveSystem.load_data(values_path)
            return True
        return False

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

        max_sma_period = max(self.trader.profile["sma_periods"])
        max_period = max(self.trader.profile["k_period"], self.trader.profile["d_period"], self.trader.profile["rsi_period"], self.trader.profile["atr_period"], max_sma_period)

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status():
                # TODO: Refactor to use pandas dataframe
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
                stock_sentiment = self.trader.finbert.get_api_sentiment(self.stock["symbol"], now_date - dt.timedelta(days=3), now_date)
                sp500_sentiment = self.trader.finbert.get_api_sentiment("SPY", now_date - dt.timedelta(days=3), now_date)
                nasdaq_sentiment = self.trader.finbert.get_api_sentiment("QQQ", now_date - dt.timedelta(days=3), now_date)

                # Get historical data for indicators

                stock_bars = self.trader.get_bars(self.stock["symbol"],
                                                  self.trader.alpaca_api,
                                                  self.trader.profile["interval"],
                                                  now_date - dt.timedelta(days=max_period + 1),
                                                  now_date - dt.timedelta(days=1),
                                                  500000)
                stock_bars.append(stock_candles)  # Add today's data

                # TODO: Calculate indicators using TA-Lib

                inputs = [position["longOpenProfitLoss"] / position["averagePrice"],  # profit/loss percent
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
                          ]

                outputs = self.net.activate(inputs)

                qty_percent = (outputs[1] + 1) * 0.5

                asset = self.trader.alpaca_api.get_asset(symbol=self.stock["symbol"])
                if outputs[0] > 0.5:  # Buy
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
                    quantity = round(qty_percent * position["longQuantity"])
                    price = quantity * stock_latest["close"]
                    if price >= 1:
                        if position["longQuantity"] - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
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

                self.save_memory()

                time.sleep(wait_time)
                print(f"{self.trader.profile['name']} {self.stock['symbol']}: Resuming trading")