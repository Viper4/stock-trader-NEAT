import datetime as dt
import pandas as pd
import pytz
import time
from Agents.base_agent import Agent
from neat import nn
from HMM import models


class Trading(Agent):
    def __init__(self, settings, stock, trader):
        super().__init__(settings, None, stock)
        self.trader = trader
        self.net = None
        self.genome = None

    def create_net(self, values, active):
        self.net = nn.RecurrentNetwork.create(self.genome, self.config)
        self.net.values = values
        self.net.active = active

    def update_net(self, bars, days):
        print(f"{self.profile.name} {self.stock['symbol']}: Running network over {days} days")

        for row in bars.itertuples():
            inputs = Agent.generate_inputs(row, 0.0, self.profile.ma_periods)

            self.net.activate(inputs)

        print(f"{self.profile.name} {self.stock['symbol']}: Network memory updated")

    def run(self):
        if self.running:
            return
        print(f"{self.trader.profile['name']} {self.stock['symbol']}: Starting trading")
        self.running = True

        max_ma_period = max(self.trader.profile.ma_periods)
        max_period = max(self.trader.profile["k_period"], self.trader.profile["d_period"],
                         self.trader.profile["rsi_period"], self.trader.profile["atr_period"], max_ma_period)

        long_regime_predictor = models.HMMRegimePrediction()
        short_regime_predictor = models.HMMRegimePrediction()

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status():
                # Get today's candles
                candles, prev_stock_close = self.trader.scraper.get_latest_candles(self.stock["symbol"], interval=str(
                    self.trader.profile["interval"]) + "m")
                spy_candles, prev_spy_close = self.trader.scraper.get_latest_candles("SPY", interval=str(
                    self.trader.profile["interval"]) + "m")
                qqq_candles, prev_qqq_close = self.trader.scraper.get_latest_candles("QQQ", interval=str(
                    self.trader.profile["interval"]) + "m")
                candles["close_spy"] = spy_candles["close"]
                candles["volume_spy"] = spy_candles["volume"]

                candles["close_qqq"] = qqq_candles["close"]
                candles["volume_qqq"] = qqq_candles["volume"]

                # Get current position
                position = self.trader.schwab_api.get_position(self.stock["symbol"])

                # Get historical data on symbol
                start_date = now_date - dt.timedelta(days=max_period)
                end_date = now_date - dt.timedelta(days=1)
                spy_bars = self.trader.generate_data("SPY", "-SA", self.trader.profile, start_date, end_date)
                qqq_bars = self.trader.generate_data("QQQ", "-SA", self.trader.profile, start_date, end_date)
                bars = self.trader.generate_data(self.stock["symbol"], "-SA", self.trader.profile, start_date, end_date, spy_bars=spy_bars, qqq_bars=qqq_bars)

                bars = pd.concat([bars, candles], ignore_index=False).drop_duplicates()

                current_bar = bars.iloc[-1]
                current_bar["sentiment"] = self.trader.finbert.get_api_sentiment(self.stock["symbol"], now_date - dt.timedelta(days=3), now_date)
                current_bar["sentiment_spy"] = self.trader.finbert.get_api_sentiment("SPY", now_date - dt.timedelta(days=3), now_date)
                current_bar["sentiment_qqq"] = self.trader.finbert.get_api_sentiment("QQQ", now_date - dt.timedelta(days=3), now_date)

                long_regime_predictor.fit(bars, self.stock["long_term_features"], self.stock["long_term_seed"])
                current_bar["long_regime"] = long_regime_predictor.predict_probability(bars)[-1]

                short_regime_predictor.fit(bars, self.stock["short_term_features"], self.stock["short_term_seed"])
                current_bar["short_regime"] = short_regime_predictor.predict_probability(bars)[-1]

                inputs = self.generate_inputs(current_bar, position["longOpenProfitLoss"] / position["averagePrice"], self.trader.profile.ma_periods)

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
                        quantity = min(self.trader.profile["cash_limit"], settled_cash) * qty_percent * self.stock[
                            "cash_at_risk"] / current_bar.close
                        quantity = round(quantity)
                        if quantity > 0:
                            self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity,
                                                                side="BUY")

                            action = {"side": "Buy", "type": "long", "quantity": quantity,
                                      "price": current_bar.close,
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": now_date}
                            print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                            self.trader.logs[self.stock["symbol"]].append(action)
                elif outputs[0] < -0.5:  # Sell
                    account = self.trader.schwab_api.get_account()
                    unsettled_cash = account["currentBalances"]["unsettledCash"]
                    settled_cash = account["currentBalances"]["cashAvailableForTrading"] - unsettled_cash
                    quantity = round(qty_percent * position["longQuantity"])
                    price = quantity * current_bar.close
                    if price >= 1:
                        if position["longQuantity"] - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                            self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity,
                                                                side="SELL")
                        else:
                            self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity,
                                                                side="SELL")

                        # profit = price - cost
                        action = {"side": "Sell", "type": "long", "quantity": quantity, "price": current_bar.close,
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
                print(
                    f"{self.trader.profile['name']} {self.stock['symbol']}: Pausing trading. Waiting until market opens in {wait_time / 3600} hours")

                self.trader.save_memory(self.net, f"{self.profile.name.replace(' ', '-')}-{self.stock['symbol']}")

                time.sleep(wait_time)
                print(f"{self.trader.profile['name']} {self.stock['symbol']}: Resuming trading")

    def stop(self):
        print(f"Stopping {self.trader.profile['name']} {self.stock['symbol']} trading agent...")
        self.trader.save_memory(self.net, f"{self.profile.name.replace(' ', '-')}-{self.stock['symbol']}")
        self.running = False
