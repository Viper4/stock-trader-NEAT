import datetime as dt
import os
import talib
import pandas as pd
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

        max_sma_period = max(self.trader.profile["sma_periods"])
        max_period = max(self.trader.profile["k_period"], self.trader.profile["d_period"], self.trader.profile["rsi_period"], self.trader.profile["atr_period"], max_sma_period)

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status():
                # Get today's candles
                candles, prev_stock_close = self.trader.scraper.get_latest_candles(self.stock["symbol"], interval=str(self.trader.profile["interval"]) + "m")
                spy_candles, prev_spy_close = self.trader.scraper.get_latest_candles("SPY", interval=str(self.trader.profile["interval"]) + "m")
                qqq_candles, prev_qqq_close = self.trader.scraper.get_latest_candles("QQQ", interval=str(self.trader.profile["interval"]) + "m")

                # Get current position
                position = self.trader.schwab_api.get_position(self.stock["symbol"])

                # Get sentiments
                stock_sentiment = self.trader.finbert.get_api_sentiment(self.stock["symbol"], now_date - dt.timedelta(days=3), now_date)
                spy_sentiment = self.trader.finbert.get_api_sentiment("SPY", now_date - dt.timedelta(days=3), now_date)
                qqq_sentiment = self.trader.finbert.get_api_sentiment("QQQ", now_date - dt.timedelta(days=3), now_date)

                # Get historical data on symbol
                bars = self.trader.get_bars(self.stock["symbol"],
                                            self.trader.alpaca_api,
                                            self.trader.profile["interval"],
                                            now_date - dt.timedelta(days=max_period + 1),
                                            now_date - dt.timedelta(days=1),
                                            500000)
                bars = pd.concat([bars, candles], ignore_index=False).drop_duplicates()  # Add today's data
                if not bars.index.is_monotonic_increasing:
                    print(f"{self.trader.profile['name']} {self.stock['symbol']}: Non-monotonic bars, sorting...")
                    bars = bars.sort_index()

                # Add indicators
                bars["slow_k"], bars["slow_d"] = talib.STOCH(bars["high"], bars["low"], bars["close"],
                                                             fastk_period=self.trader.profile["k_period"],
                                                             slowk_period=self.trader.profile["d_period"],
                                                             slowd_period=self.trader.profile["d_period"])
                bars["rsi"] = talib.RSI(bars["close"], timeperiod=self.trader.profile["rsi_period"])
                bars["atr"] = talib.ATR(bars["high"], bars["low"], bars["close"], timeperiod=self.trader.profile["atr_period"])
                bars["ema_k"] = talib.EMA(bars["close"], timeperiod=self.trader.profile["k_period"])
                bars["ema_d"] = talib.EMA(bars["close"], timeperiod=self.trader.profile["d_period"])
                for sma_period in self.trader.profile["sma_periods"]:
                    bars[f"sma_{sma_period}"] = talib.SMA(bars["close"], timeperiod=sma_period)

                current_candle = bars.iloc[-1]
                prev_candle = bars.iloc[-2]
                
                current_spy_candle = spy_candles.iloc[-1]
                prev_spy_candle = spy_candles.iloc[-2]

                current_qqq_candle = qqq_candles.iloc[-1]
                prev_qqq_candle = qqq_candles.iloc[-2]

                inputs = [position["longOpenProfitLoss"] / position["averagePrice"],  # profit/loss percent
                          self.rel_change(prev_candle.open, current_candle.open),
                          self.rel_change(prev_candle.high, current_candle.high),
                          self.rel_change(prev_candle.low, current_candle.low),
                          self.rel_change(prev_candle.close, current_candle.close),
                          self.rel_change(prev_candle.volume, current_candle.volume),
                          self.rel_change(prev_candle.vwap, current_candle.vwap),
                          stock_sentiment,  # -1 = negative, 0 = neutral, 1 = positive
                          self.rel_change(prev_spy_candle.close, current_spy_candle.close),
                          self.rel_change(prev_spy_candle.volume, current_spy_candle.volume),
                          spy_sentiment,
                          self.rel_change(prev_qqq_candle.close, current_qqq_candle.close),
                          self.rel_change(prev_qqq_candle.volume, current_qqq_candle.volume),
                          qqq_sentiment,
                          (current_candle.slow_k - 50) / 50,
                          (current_candle.slow_d - 50) / 50,
                          (current_candle.rsi - 50) / 50,
                          Agent.rel_change(prev_candle.atr, current_candle.atr),
                          Agent.rel_change(prev_candle.ema_k, current_candle.ema_k),
                          Agent.rel_change(prev_candle.ema_d, current_candle.ema_d),
                          ]

                for sma_period in self.trader.profile["sma_periods"]:
                    inputs.append(Agent.rel_change(prev_candle[f"sma_{sma_period}"], current_candle[f"sma_{sma_period}"]))

                outputs = self.net.activate(inputs)

                qty_percent = (outputs[1] + 1) * 0.5

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
                        quantity = min(self.trader.profile["cash_limit"], settled_cash) * qty_percent * self.stock["cash_at_risk"] / current_candle.close
                        quantity = round(quantity)
                        if quantity > 0:
                            self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="BUY")

                            action = {"side": "Buy", "type": "long", "quantity": quantity, "price": current_candle.close,
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": now_date}
                            print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                            self.trader.logs[self.stock["symbol"]].append(action)
                elif outputs[0] < -0.5:  # Sell
                    account = self.trader.schwab_api.get_account()
                    unsettled_cash = account["currentBalances"]["unsettledCash"]
                    settled_cash = account["currentBalances"]["cashAvailableForTrading"] - unsettled_cash
                    quantity = round(qty_percent * position["longQuantity"])
                    price = quantity * current_candle.close
                    if price >= 1:
                        if position["longQuantity"] - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                            self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="SELL")
                        else:
                            self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="SELL")

                        # profit = price - cost
                        action = {"side": "Sell", "type": "long", "quantity": quantity, "price": current_candle.close,
                                  "profit": price - (position["averageLongPrice"] * quantity),
                                  "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                  "datetime": now_date}
                        print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                        self.trader.logs[self.stock["symbol"]].append(action)

                time.sleep(self.trader.profile["interval"] * 60)
            else:
                next_open = self.trader.clock[0].next_open
                wait_time = (next_open - now_date).total_seconds()
                wait_time += self.trader.profile["interval"] * 60 + 10  # Wait for yahoo finance to update
                print(f"{self.trader.profile['name']} {self.stock['symbol']}: Pausing trading. Waiting until market opens in {wait_time / 3600} hours")

                self.save_memory()

                time.sleep(wait_time)
                print(f"{self.trader.profile['name']} {self.stock['symbol']}: Resuming trading")