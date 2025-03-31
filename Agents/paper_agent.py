import datetime as dt
import pytz
import time
from Agents.base_agent import Agent


class PaperTrading(Agent):
    def __init__(self, settings, profile, stock, finbert, trader, scraper):
        super().__init__(settings, profile, stock)
        self.finbert = finbert
        self.trader = trader
        self.scraper = scraper
        self.net = None
        self.genome = None

    def run(self):
        if self.running:
            return
        print(f"{self.profile.interval}m {self.stock['symbol']}: Starting trading")
        self.running = True
        cum_stock_price = 0
        cum_stock_vol = 0
        prev_stock_candle = None

        prev_sp500_candle = None

        prev_nasdaq_candle = None

        max_period = max(self.profile.k_period, self.profile.d_period, self.profile.rsi_period)
        k_period = self.days_to_bars(self.profile.k_period, self.profile.interval)
        d_period = self.days_to_bars(self.profile.d_period, self.profile.interval)
        rsi_period = self.days_to_bars(self.profile.rsi_period, self.profile.interval)

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status():
                # Stock candles for today
                stock_candles, prev_close = self.scraper.get_latest_stock_candles(self.stock["symbol"], interval=str(
                    self.profile.interval) + "m")
                stock_latest = stock_candles[-1]
                cum_stock_price += stock_latest["volume"] * ((stock_latest["high"] + stock_latest["low"] + stock_latest["close"]) / 3)
                cum_stock_vol += stock_latest["volume"]
                stock_latest["vwap"] = cum_stock_price / cum_stock_vol if cum_stock_vol > 0 else 0

                if prev_stock_candle is None:
                    if len(stock_candles) >= 2:
                        prev_stock_candle = stock_candles[-2]
                    else:
                        prev_stock_candle = stock_latest
                        prev_stock_candle["close"] = prev_close
                    prev_stock_candle["vwap"] = (prev_stock_candle["high"] + prev_stock_candle["low"] + prev_stock_candle["close"]) / 3

                # SP500 candles for today
                sp500_candles, prev_sp500_close = self.scraper.get_stock_latests("SPY", interval=str(
                    self.profile.interval) + "m")
                sp500_latest = sp500_candles[-1]

                if prev_sp500_candle is None:
                    if len(sp500_candles) >= 2:
                        prev_sp500_candle = sp500_candles[-2]
                    else:
                        prev_sp500_candle = stock_latest
                        prev_sp500_candle["close"] = prev_sp500_close
                    prev_sp500_candle["vwap"] = (prev_sp500_candle["high"] + prev_sp500_candle["low"] +
                                                 prev_sp500_candle["close"]) / 3

                # NASDAQ candles for today
                nasdaq_candles, prev_nasdaq_close = self.scraper.get_stock_latests("QQQ", interval=str(
                    self.profile.interval) + "m")
                nasdaq_latest = nasdaq_candles[-1]

                if prev_nasdaq_candle is None:
                    if len(nasdaq_candles) >= 2:
                        prev_nasdaq_candle = nasdaq_candles[-2]
                    else:
                        prev_nasdaq_candle = stock_latest
                        prev_nasdaq_candle["close"] = prev_nasdaq_close
                    prev_nasdaq_candle["vwap"] = (prev_nasdaq_candle["high"] + prev_nasdaq_candle["low"] +
                                                  prev_nasdaq_candle["close"]) / 3

                # Get current position
                position = self.trader.get_position(self.stock["symbol"])
                position_qty = float(position.qty)

                stock_sentiment = self.finbert.get_api_sentiment(self.stock["symbol"], now_date - dt.timedelta(days=3), now_date)
                sp500_sentiment = self.finbert.get_api_sentiment("SPY", now_date - dt.timedelta(days=3), now_date)
                nasdaq_sentiment = self.finbert.get_api_sentiment("QQQ", now_date - dt.timedelta(days=3), now_date)

                # Get historical data for momentum indicators
                stock_bars = self.trader.get_bars(self.stock["symbol"], self.trader.alpaca_api,
                                                  self.profile.interval,
                                                  now_date - dt.timedelta(days=max_period + 1),
                                                  now_date - dt.timedelta(days=1),
                                                  500000,
                                                  False)
                stock_bars.append(stock_candles)  # Add today's data

                last_index = len(stock_bars) - 1

                inputs = [
                          float(position.unrealized_plpc),  # profit/loss percent
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

                if self.stock["shorting"] and position_qty < 0:
                    inputs[8] = -1
                outputs = self.net.activate(inputs)

                qty_percent = (outputs[1] + 1) * 0.5

                asset = self.profile.alpaca_api.get_asset(symbol=self.stock["symbol"])
                if not asset.tradable:
                    print(f"{self.stock['symbol']}: Not tradable.")
                else:
                    if outputs[0] > 0.5:  # Buy
                        quantity = self.profile.settled_cash * qty_percent * self.stock["cash_at_risk"] / stock_latest["close"]
                        if not asset.fractionable:
                            quantity = round(quantity)
                        price = quantity * stock_latest["close"] * (1 - self.stock["transaction_fee"])
                        if price >= 1:  # Alpaca doesn't allow trades under $1
                            self.profile.settled_cash -= price
                            self.profile.alpaca_api.submit_order(symbol=self.stock["symbol"], qty=quantity, side="buy", type="market", time_in_force="day")

                            action = {"side": "Buy", "type": "long", "quantity": quantity, "price": stock_latest["close"],
                                      "settled_cash": self.profile.settled_cash, "unsettled_cash": self.profile.unsettled_cash,
                                      "datetime": now_date}
                            print(f"{self.profile.interval}m {self.stock['symbol']}: {action}")
                            self.profile.logs[self.stock["symbol"]].append(action)
                    elif outputs[0] < -0.5:  # Sell
                        quantity = qty_percent * position_qty
                        if not asset.fractionable:
                            quantity = round(quantity)
                        price = quantity * stock_latest["close"] * (1 - self.stock["transaction_fee"])
                        if price >= 1:
                            if position_qty - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                self.profile.alpaca_api.submit_order(symbol=self.stock["symbol"], qty=position_qty, side="sell", type="market", time_in_force="day")
                                price = position_qty * stock_latest["close"]
                            else:
                                self.profile.alpaca_api.submit_order(symbol=self.stock["symbol"], qty=quantity, side="sell", type="market", time_in_force="day")
                            self.profile.unsettled_cash += price
                            self.session["pending_sales"].enqueue((price, self.trader.consecutive_days))

                            action = {"side": "Sell", "type": "long", "quantity": quantity, "price": stock_latest["close"],
                                      "profit": price - (float(position.avg_entry_price) * quantity),
                                      "settled_cash": self.profile.settled_cash, "unsettled_cash": self.profile.unsettled_cash,
                                      "datetime": now_date}
                            print(f"{self.profile.interval}m {self.stock['symbol']}: {action}")
                            self.profile.logs[self.stock["symbol"]].append(action)
                prev_stock_candle = stock_latest
                prev_sp500_candle = sp500_latest
                prev_nasdaq_candle = nasdaq_latest

                time.sleep(self.profile.interval * 60)
            else:
                cum_stock_price = 0.0
                cum_stock_vol = 0.0

                next_open = self.session["clock"][0].next_open
                wait_time = (next_open - now_date).total_seconds()
                wait_time += self.profile.interval * 60 + 10  # Wait for yahoo finance to update
                print(f"{self.profile.interval}m {self.stock['symbol']}: Pausing trading. Waiting until market opens in {wait_time / 3600} hours")
                time.sleep(wait_time)
                print(f"{self.profile.interval}m {self.stock['symbol']}: Resuming trading")

