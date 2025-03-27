import datetime as dt
import pytz
import time
from base_agent import Agent


class PaperTrading(Agent):
    def __init__(self, settings, session, stock, finbert, trader, scraper):
        super().__init__(settings, session, stock)
        self.finbert = finbert
        self.trader = trader
        self.scraper = scraper
        self.net = None
        self.genome = None

    def run(self):
        if self.running:
            return
        print(f"{self.session['interval']}m {self.stock['symbol']}: Starting trading")
        self.running = True
        cum_stock_price = 0
        cum_stock_vol = 0
        prev_stock_candle = None

        prev_sp500_candle = None

        prev_nasdaq_candle = None

        max_period = max(self.session["k_period"], self.session["d_period"], self.session["rsi_period"])
        k_period = self.days_to_bars(self.session["k_period"], self.session["interval"])
        d_period = self.days_to_bars(self.session["d_period"], self.session["interval"])
        rsi_period = self.days_to_bars(self.session["rsi_period"], self.session["interval"])
        d_alpha = 2 / (d_period + 1)
        prev_d_ema = None
        k_alpha = 2 / (k_period + 1)
        prev_k_ema = None
        
        gain = 0
        loss = 0

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status(self.session):
                # Stock candles for today
                stock_candles, prev_close = self.scraper.get_latest_stock_candles(self.stock["symbol"], interval=str(
                    self.session["interval"]) + "m")
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
                    self.session["interval"]) + "m")
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
                    self.session["interval"]) + "m")
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
                position = self.trader.get_position(self.stock["symbol"], self.session)
                position_qty = float(position.qty)

                stock_sentiment = self.finbert.get_api_sentiment(self.stock["symbol"],
                                                                 now_date - dt.timedelta(days=2), now_date)
                sp500_sentiment = self.finbert.get_api_sentiment("SPY",
                                                                 now_date - dt.timedelta(days=2), now_date)
                nasdaq_sentiment = self.finbert.get_api_sentiment("QQQ",
                                                                 now_date - dt.timedelta(days=2), now_date)

                # Get historical data for momentum indicators
                stock_bars = self.trader.get_bars(self.stock["symbol"], self.trader.alpaca_api,
                                                  self.session["interval"],
                                                  now_date - dt.timedelta(days=max_period + 1),
                                                  now_date - dt.timedelta(days=1),
                                                  500000,
                                                  False)
                stock_bars.append(stock_candles)  # Add today's data

                last_index = len(stock_bars) - 1
                k_percent = self.calculate_k_percent(stock_bars[last_index - min(k_period, last_index):last_index])

                # %D = d_ema(%K, N) or SMA(%K, N)
                d_ema = self.calculate_ema(stock_latest["close"], d_alpha, prev_d_ema)
                prev_d_ema = d_ema
                k_ema = self.calculate_ema(stock_latest["close"], k_alpha, prev_k_ema)
                prev_k_ema = k_ema

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

                inputs = [1,  # -1 = shorting, 1 = longing
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
                          k_percent,
                          d_ema,
                          k_ema,
                          rsi]

                if self.stock["shorting"] and position_qty < 0:
                    inputs[8] = -1
                outputs = self.net.activate(inputs)

                qty_percent = (outputs[1] + 1) * 0.5

                asset = self.session["alpaca_api"].get_asset(symbol=self.stock["symbol"])
                if not asset.tradable:
                    print(f"{self.stock['symbol']}: Not tradable.")
                else:
                    if outputs[0] > 0.5:  # Buy
                        if self.stock["shorting"] and asset.shortable and position_qty < 0:
                            quantity = qty_percent * position_qty
                            quantity = round(quantity)  # Shorts don't allow fractional qty
                            price = quantity * stock_latest["close"] * (1 - self.stock["transaction_fee"])
                            if abs(price) >= 1:
                                if abs(position_qty - quantity) < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                    self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=abs(position_qty), side="buy", type="market", time_in_force="day")
                                    price = position_qty * stock_latest["close"] * (1 - self.stock["transaction_fee"])
                                else:
                                    self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=abs(quantity), side="buy", type="market", time_in_force="day")
                                cost = float(position.avg_entry_price) * quantity
                                self.session["settled_cash"] += price - cost

                                action = {"side": "Buy", "type": "short", "quantity": abs(quantity), "price": stock_latest["close"],
                                          "profit": price - cost,
                                          "settled_cash": self.session["settled_cash"],
                                          "unsettled_cash": self.session["unsettled_cash"],
                                          "datetime": now_date}
                                print(f"{self.session['interval']}m {self.stock['symbol']}: {action}")
                                self.session["logs"][self.stock["symbol"]].append(action)
                        else:
                            quantity = self.session["settled_cash"] * qty_percent * self.stock["cash_at_risk"] / stock_latest["close"]
                            if not asset.fractionable:
                                quantity = round(quantity)
                            price = quantity * stock_latest["close"] * (1 - self.stock["transaction_fee"])
                            if price >= 1:  # Alpaca doesn't allow trades under $1
                                self.session["settled_cash"] -= price
                                self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=quantity, side="buy", type="market", time_in_force="day")

                                action = {"side": "Buy", "type": "long", "quantity": quantity, "price": stock_latest["close"],
                                          "settled_cash": self.session["settled_cash"], "unsettled_cash": self.session["unsettled_cash"],
                                          "datetime": now_date}
                                print(f"{self.session['interval']}m {self.stock['symbol']}: {action}")
                                self.session["logs"][self.stock["symbol"]].append(action)
                    elif outputs[0] < -0.5:  # Sell
                        if self.stock["shorting"] and asset.shortable and position_qty <= 0:
                            if abs(float(position.cost_basis)) < self.session["short_limit"]:
                                quantity = -qty_percent * (min(self.session["settled_cash"], self.session["short_limit"]) - abs(float(position.cost_basis))) * self.stock["cash_at_risk"] / stock_latest["close"]
                                quantity = round(quantity)  # Shorts don't allow fractional qty
                                price = quantity * stock_latest["close"] * (1 - self.stock["transaction_fee"])
                                if abs(price) >= 1:  # Alpaca doesn't allow trades under $1
                                    self.session["unsettled_cash"] -= price
                                    self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=abs(quantity), side="sell", type="market", time_in_force="day")

                                    action = {"side": "Sell", "type": "short", "quantity": abs(quantity), "price": stock_latest["close"],
                                              "settled_cash": self.session["settled_cash"],
                                              "unsettled_cash": self.session["unsettled_cash"],
                                              "datetime": now_date}
                                    print(f"{self.session['interval']}m {self.stock['symbol']}: {action}")
                                    self.session["logs"][self.stock["symbol"]].append(action)
                        elif position_qty > 0:
                            quantity = qty_percent * position_qty
                            if not asset.fractionable:
                                quantity = round(quantity)
                            price = quantity * stock_latest["close"] * (1 - self.stock["transaction_fee"])
                            if price >= 1:
                                if position_qty - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                    self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=position_qty, side="sell", type="market", time_in_force="day")
                                    price = position_qty * stock_latest["close"]
                                else:
                                    self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=quantity, side="sell", type="market", time_in_force="day")
                                self.session["unsettled_cash"] += price
                                self.session["pending_sales"].enqueue((price, self.trader.consecutive_days))

                                action = {"side": "Sell", "type": "long", "quantity": quantity, "price": stock_latest["close"],
                                          "profit": price - (float(position.avg_entry_price) * quantity),
                                          "settled_cash": self.session["settled_cash"], "unsettled_cash": self.session["unsettled_cash"],
                                          "datetime": now_date}
                                print(f"{self.session['interval']}m {self.stock['symbol']}: {action}")
                                self.session["logs"][self.stock["symbol"]].append(action)
                prev_stock_candle = stock_latest
                prev_sp500_candle = sp500_latest
                prev_nasdaq_candle = nasdaq_latest

                time.sleep(self.session["interval"] * 60)
            else:
                cum_stock_price = 0.0
                cum_stock_vol = 0.0

                next_open = self.session["clock"][0].next_open
                wait_time = (next_open - now_date).total_seconds()
                wait_time += self.session["interval"] * 60 + 10  # Wait for yahoo finance to update
                print(f"{self.session['interval']}m {self.stock['symbol']}: Pausing trading. Waiting until market opens in {wait_time / 3600} hours")
                time.sleep(wait_time)
                print(f"{self.session['interval']}m {self.stock['symbol']}: Resuming trading")

