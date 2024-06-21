import neat
import datetime as dt
import pytz
import time
import os
from multiprocessing import Pool
import threading
import saving
import visualize
import plot
import numpy as np


class Agent:
    def __init__(self, settings, session, stock):
        self.running = False
        self.settings = settings
        self.session = session
        self.stock = stock

        if self.settings["processes"] >= os.cpu_count():
            print("Using " + str(self.settings["processes"]) + " workers to train but system only has " + str(os.cpu_count()) + " cores.")
            if input("Proceed? (y/n): ") != "y":
                exit(0)

        self.population_path = self.settings["save_path"] + "\\Populations"
        saving.SaveSystem.make_dir(self.population_path)

        self.genome_path = self.settings["save_path"] + "\\Genomes"
        saving.SaveSystem.make_dir(self.genome_path)

        self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])

    @staticmethod
    def rel_change(a, b):
        if a == 0:
            return 0
        return (b - a) / a

    @staticmethod
    def max_drawdown(portfolio_values):
        portfolio_values = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max
        max_drawdown = np.max(drawdowns)
        return max_drawdown


# Separate from classes so instances don't get cached to RAM and slow things down
def eval_genome(bars, sentiments, start_cash, genome, config, cash_at_risk, log_training, profit_window, fitness_multipliers, shorting, shortable, fractionable, short_limit):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    start_date = bars[0]["timestamp"].date()
    settled_cash = start_cash
    unsettled_cash = 0
    pending_sales = []
    start_equity = start_cash
    profit_sum = 0.0
    num_windows = 0
    shares = 0.0
    cost = 0.0
    consecutive_days = 1
    log = []
    portfolio_values = [start_equity]

    # Start at 1 to have previous bar for relative change
    num_bars = len(bars)
    for i in range(1, num_bars):
        bar = bars[i]
        prev_bar = bars[i-1]
        prev_date = prev_bar["timestamp"].date()
        date = bar["timestamp"].date()
        if date != prev_date:  # Check pending sales to settle cash after 1 day of sale
            consecutive_days += 1
            for j in reversed(range(len(pending_sales))):
                sale_price, sale_day, is_short = pending_sales[j]
                if consecutive_days - sale_day > 1:
                    settled_cash += sale_price
                    if not is_short:
                        unsettled_cash -= sale_price
                    pending_sales.pop(j)

        inputs = [Agent.rel_change(cost, bar["close"] * shares),  # plpc
                  Agent.rel_change(prev_bar["open"], bar["open"]),
                  Agent.rel_change(prev_bar["high"], bar["high"]),
                  Agent.rel_change(prev_bar["low"], bar["low"]),
                  Agent.rel_change(prev_bar["close"], bar["close"]),
                  Agent.rel_change(prev_bar["volume"], bar["volume"]),
                  Agent.rel_change(prev_bar["vwap"], bar["vwap"]),
                  sentiments[i],
                  1]
        if shorting and shares < 0:
            inputs[0] *= -1
            inputs[8] = -1

        outputs = net.activate(inputs)

        qty_percent = (outputs[1] + 1) * 0.5
        if outputs[0] > 0.5:  # Buy
            if shorting and shortable and shares < 0:
                quantity = qty_percent * shares
                if not fractionable:
                    quantity = round(quantity)
                price = quantity * bar["close"] * 0.999  # Transaction fee
                if abs(price) >= 1:
                    if abs(shares - quantity) < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                        price = shares * bar["close"] * 0.999  # Transaction fee
                        profit = price - cost
                        if log_training:
                            action = {"side": "Buy", "type": "short", "quantity": -quantity, "price": bar["close"],
                                      "profit": profit, "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash,
                                      "datetime": bar["timestamp"].to_pydatetime()}
                            log.append(action)
                        shares = 0.0
                        cost = 0.0
                    else:
                        avg_cost = cost / abs(shares)
                        shares -= quantity
                        cost = avg_cost * shares
                        profit = price - (avg_cost * quantity)
                        if log_training:
                            action = {"side": "Buy", "type": "short", "quantity": -quantity, "price": bar["close"],
                                      "profit": profit, "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash,
                                      "datetime": bar["timestamp"].to_pydatetime()}
                            log.append(action)
                    unsettled_cash += price
                    pending_sales.append((profit, consecutive_days, True))
            else:
                quantity = qty_percent * settled_cash * cash_at_risk / bar["close"]
                if not fractionable:
                    quantity = round(quantity)
                price = quantity * bar["close"]
                if price >= 1:  # Alpaca doesn't allow trades under $1
                    cost += price
                    shares += quantity
                    settled_cash -= price

                    if log_training:
                        action = {"side": "Buy", "type": "long", "quantity": quantity, "price": bar["close"],
                                  "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                  "datetime": bar["timestamp"].to_pydatetime()}
                        log.append(action)
        elif outputs[0] < -0.5:  # Sell
            if shorting and shortable and shares <= 0:
                if abs(cost) < short_limit:
                    quantity = -qty_percent * (short_limit - abs(cost)) * cash_at_risk / bar["close"]
                    if not fractionable:
                        quantity = round(quantity)
                    price = quantity * bar["close"] * 0.999  # Transaction fee
                    if abs(price) >= 1:  # Alpaca doesn't allow trades under $1
                        cost += price
                        shares += quantity
                        unsettled_cash -= price

                        if log_training:
                            action = {"side": "Sell", "type": "short", "quantity": quantity, "price": bar["close"],
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": bar["timestamp"].to_pydatetime()}
                            log.append(action)
            elif shares > 0:
                quantity = qty_percent * shares
                if not fractionable:
                    quantity = round(quantity)
                price = quantity * bar["close"] * 0.999  # Transaction fee
                if price >= 1:
                    if shares - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                        price = shares * bar["close"] * 0.999  # Transaction fee
                        if log_training:
                            action = {"side": "Sell", "type": "long", "quantity": quantity, "price": bar["close"],
                                      "profit": price - cost, "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash + price,
                                      "datetime": bar["timestamp"].to_pydatetime()}
                            log.append(action)
                        shares = 0.0
                        cost = 0.0
                    else:
                        avg_cost = cost / shares
                        shares -= quantity
                        cost = avg_cost * shares
                        if log_training:
                            action = {"side": "Sell", "type": "long", "quantity": quantity, "price": bar["close"],
                                      "profit": price - (avg_cost * quantity), "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash + price,
                                      "datetime": bar["timestamp"].to_pydatetime()}
                            log.append(action)
                    unsettled_cash += price
                    pending_sales.append((price, consecutive_days, False))
        if i == num_bars-1 or (date - start_date).days >= profit_window:
            if shares < 0:
                equity = unsettled_cash + settled_cash + shares * bar["close"] - cost
            else:
                equity = unsettled_cash + settled_cash + bar["close"] * shares
            profit_sum += equity - start_equity
            num_windows += 1
            start_equity = equity
            start_date = date
            portfolio_values.append(equity)

    avg_factor = (profit_sum / num_windows) * fitness_multipliers["average"]
    total_factor = profit_sum * fitness_multipliers["total"]
    risk_factor = Agent.max_drawdown(portfolio_values) * fitness_multipliers["risk"]
    return avg_factor + total_factor - risk_factor, log


class Training(Agent):
    def __init__(self, settings, session, stock, bars, sentiments):
        super().__init__(settings, session, stock)
        self.started = False
        self.best_genome = None  # Saving population object adds 10s to each gen
        self.consecutive_gens = 0
        self.start_cash = 100000.0
        self.bars = bars
        self.sentiments = sentiments
        self.batch_index = 0
        self.genome_file_path = os.path.join(self.genome_path, self.stock["genome_filename"])
        self.population_file_path = os.path.join(self.population_path, self.stock["population_filename"])
        self.shortable = True
        self.fractionable = True

    def eval_genomes(self, genomes, config):
        while not self.running:
            time.sleep(1)

        # self.pool doesn't work: cant pickle Pool(). Separate class doesn't work: leaks memory
        pool = Pool(processes=self.settings["processes"])
        jobs = []

        if isinstance(self.bars[self.batch_index], int):
            sub_index = self.bars[self.batch_index]
            print(f"Evaluating genomes on substitute batch {sub_index}")
            bars = self.bars[sub_index]
            sentiments = self.sentiments[sub_index]
        else:
            print(f"Evaluating genomes on batch {self.batch_index}")
            bars = self.bars[self.batch_index]
            sentiments = self.sentiments[self.batch_index]

        for genome_id, genome in genomes:
            jobs.append(pool.apply_async(eval_genome, (bars, sentiments, self.start_cash,
                                                       genome, self.config, self.stock["cash_at_risk"],
                                                       self.settings["log_training"], self.session["profit_window"],
                                                       self.session["fitness_multipliers"], self.stock["shorting"],
                                                       self.shortable, self.fractionable, self.session["short_limit"])))

        best_log = None
        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness, log = job.get()
            if self.best_genome is None or self.best_genome.fitness < genome.fitness:
                best_log = log
                self.best_genome = genome

        if best_log is not None and self.settings["log_training"]:
            plot.plot_log(self.session["alpaca_api"], self.stock["symbol"], best_log, 30, True)
        pool.close()
        pool.join()
        pool.terminate()

        self.consecutive_gens += 1
        if 0 < self.settings["gen_stagger"] <= self.consecutive_gens:
            self.consecutive_gens = 0
            self.running = False
        self.batch_index += 1
        if self.batch_index >= len(self.bars):
            self.batch_index = 0

    def run(self):
        if self.running:
            return
        self.running = True
        asset = self.session["alpaca_api"].get_asset(symbol=self.stock["symbol"])
        self.shortable = asset.shortable
        self.fractionable = asset.fractionable
        if not self.started:
            print(f"Starting {self.session['interval']}m {self.stock['symbol']} training agent...")
            save_system = saving.SaveSystem(1, self.genome_file_path, self.settings["gen_stagger"], self.population_file_path)
            if os.path.exists(self.population_file_path):
                p = save_system.load_population(self.population_file_path)
            else:
                p = neat.Population(self.config)
            if self.settings["print_stats"]:
                p.add_reporter(neat.StdOutReporter(True))

            p.add_reporter(save_system)
            threading.Thread(target=p.run, args=(self.eval_genomes, None)).start()
        else:
            print(f"Resuming {self.session['interval']}m {self.stock['symbol']} training agent...")
        self.started = True

    def plot(self):
        node_names = {-9: 'plpc', -8: 'open%', -7: 'high%', -6: 'low%', -5: 'close%', -4: 'volume%', -3: 'vwap%', -2: "positive", -1: "negative", 0: 'buy/sell', 1: 'amount'}
        visualize.draw_net(self.config, self.best_genome, view=True, node_names=node_names, show_disabled=False)


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
        cum_price = 0
        cum_vol = 0

        prev_data = None

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status():
                candles, prev_close = self.trader.scraper.get_latest_candles(self.stock["symbol"], interval=str(self.trader.profile["interval"]) + "m")
                latest = candles[-1]
                cum_price += latest["volume"] * ((latest["high"] + latest["low"] + latest["close"]) / 3)
                cum_vol += latest["volume"]
                latest["vwap"] = cum_price / cum_vol if cum_vol > 0 else 0

                if prev_data is None:
                    if len(candles) >= 2:
                        prev_data = candles[-2]
                    else:
                        prev_data = latest
                        prev_data["close"] = prev_close
                    prev_data["vwap"] = (prev_data["high"] + prev_data["low"] + prev_data["close"]) / 3

                position = self.trader.schwab_api.get_position(self.stock["symbol"])
                if position["shortQuantity"] > 0:
                    position_qty = -position["shortQuantity"]
                    sentiment = self.trader.finbert.get_api_sentiment(self.stock["symbol"],now_date - dt.timedelta(days=2), now_date)
                    inputs = [position["shortOpenProfitLoss"] / position["averagePrice"],
                              self.rel_change(prev_data["open"], latest["open"]),
                              self.rel_change(prev_data["high"], latest["high"]),
                              self.rel_change(prev_data["low"], latest["low"]),
                              self.rel_change(prev_data["close"], latest["close"]),
                              self.rel_change(prev_data["volume"], latest["volume"]),
                              self.rel_change(prev_data["vwap"], latest["vwap"]),
                              sentiment,
                              -1]
                else:
                    position_qty = position["longQuantity"]
                    sentiment = self.trader.finbert.get_api_sentiment(self.stock["symbol"], now_date - dt.timedelta(days=2), now_date)
                    inputs = [position["longOpenProfitLoss"] / position["averagePrice"],
                              self.rel_change(prev_data["open"], latest["open"]),
                              self.rel_change(prev_data["high"], latest["high"]),
                              self.rel_change(prev_data["low"], latest["low"]),
                              self.rel_change(prev_data["close"], latest["close"]),
                              self.rel_change(prev_data["volume"], latest["volume"]),
                              self.rel_change(prev_data["vwap"], latest["vwap"]),
                              sentiment,
                              1]

                outputs = self.net.activate(inputs)

                qty_percent = (outputs[1] + 1) * 0.5

                asset = self.trader.alpaca_api.get_asset(symbol=self.stock["symbol"])
                if outputs[0] > 0.5:  # Buy
                    if self.stock["shorting"] and asset.shortable and position["shortQuantity"] > 0:
                        account = self.trader.schwab_api.get_account()
                        unsettled_cash = account["currentBalances"]["unsettledCash"]
                        settled_cash = account["currentBalances"]["cashAvailableForTrading"] - unsettled_cash
                        quantity = round(qty_percent * position_qty)
                        price = quantity * latest["close"]
                        if abs(price) >= 1:
                            if abs(position_qty - quantity) < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=-quantity, side="BUY")
                            else:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=-quantity, side="BUY")

                            action = {"side": "Buy", "type": "short", "quantity": -quantity, "price": latest["close"],
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
                        used_cash = market_value + unsettled_cash  # Cash in stocks + unsettled cash
                        if self.trader.profile["cash_limit"] < 0 or used_cash < self.trader.profile["cash_limit"]:
                            quantity = self.trader.profile["cash_limit"] * qty_percent * self.stock["cash_at_risk"] / latest["close"]
                            quantity = round(quantity)
                            if quantity > 0:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="BUY")

                                action = {"side": "Buy", "type": "long", "quantity": quantity, "price": latest["close"],
                                          "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                          "datetime": now_date}
                                print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                                self.trader.logs[self.stock["symbol"]].append(action)
                elif outputs[0] < -0.5:  # Sell
                    if self.stock["shorting"] and asset.shortable and position_qty <= 0:
                        account = self.trader.schwab_api.get_account()
                        unsettled_cash = account["currentBalances"]["unsettledCash"]
                        settled_cash = account["currentBalances"]["cashAvailableForTrading"] - unsettled_cash
                        cost = position["averagePrice"] * position_qty

                        if abs(cost) < self.trader.profile["short_limit"]:
                            quantity = round(-qty_percent * (self.trader.profile["short_limit"] - abs(cost)) * self.stock["cash_at_risk"] / latest["close"])
                            if quantity > 0:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=-quantity, side="SELL")

                                action = {"side": "Sell", "type": "short", "quantity": quantity, "price": latest["close"],
                                          "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                          "datetime": now_date}
                                print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                                self.trader.logs[self.stock["symbol"]].append(action)
                    elif position_qty > 0:
                        account = self.trader.schwab_api.get_account()
                        unsettled_cash = account["currentBalances"]["unsettledCash"]
                        settled_cash = account["currentBalances"]["cashAvailableForTrading"] - account["currentBalances"]["unsettledCash"]
                        quantity = round(qty_percent * position_qty)
                        price = quantity * latest["close"]
                        if price >= 1:
                            if position_qty - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="SELL")
                            else:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="SELL")

                            # profit = price - cost
                            action = {"side": "Sell", "type": "long", "quantity": quantity, "price": latest["close"],
                                      "profit": price - (position["averageLongPrice"] * quantity),
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": now_date}
                            print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                            self.trader.logs[self.stock["symbol"]].append(action)
                prev_data = latest

                time.sleep(self.trader.profile["interval"] * 60)
            else:
                cum_price = 0.0
                cum_vol = 0.0

                next_open = self.trader.clock[0].next_open
                wait_time = (next_open - now_date).total_seconds()
                wait_time += self.trader.profile["interval"] * 60 + 10  # Wait for yahoo finance to update
                print(f"{self.trader.profile['name']} {self.stock['symbol']}: Pausing trading. Waiting until market opens in {wait_time / 3600} hours")
                time.sleep(wait_time)
                print(f"{self.trader.profile['name']} {self.stock['symbol']}: Resuming trading")


class PaperTrading(Agent):
    def __init__(self, settings, session, stock, finbert, trader, scraper):
        super().__init__(settings, session, stock)
        self.finbert = finbert
        self.trader = trader
        self.scraper = scraper
        self.net = None

    def run(self):
        if self.running:
            return
        print(f"{self.session['interval']}m {self.stock['symbol']}: Starting trading")
        self.running = True
        cum_price = 0
        cum_vol = 0

        prev_data = None

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status(self.session):
                candles, prev_close = self.scraper.get_latest_candles(self.stock["symbol"], interval=str(self.session["interval"]) + "m")
                latest = candles[-1]
                cum_price += latest["volume"] * ((latest["high"] + latest["low"] + latest["close"]) / 3)
                cum_vol += latest["volume"]
                latest["vwap"] = cum_price / cum_vol if cum_vol > 0 else 0

                if prev_data is None:
                    if len(candles) >= 2:
                        prev_data = candles[-2]
                    else:
                        prev_data = latest
                        prev_data["close"] = prev_close
                    prev_data["vwap"] = (prev_data["high"] + prev_data["low"] + prev_data["close"]) / 3

                position = self.trader.get_position(self.stock["symbol"], self.session)
                position_qty = float(position.qty)

                sentiment = self.finbert.get_api_sentiment(self.stock["symbol"], now_date - dt.timedelta(days=2), now_date)
                inputs = [float(position.unrealized_plpc),
                          self.rel_change(prev_data["open"], latest["open"]),
                          self.rel_change(prev_data["high"], latest["high"]),
                          self.rel_change(prev_data["low"], latest["low"]),
                          self.rel_change(prev_data["close"], latest["close"]),
                          self.rel_change(prev_data["volume"], latest["volume"]),
                          self.rel_change(prev_data["vwap"], latest["vwap"]),
                          sentiment,
                          1]
                if self.stock["shorting"] and position.side == "short":
                    inputs[8] = -1
                outputs = self.net.activate(inputs)

                qty_percent = (outputs[1] + 1) * 0.5

                asset = self.session["alpaca_api"].get_asset(symbol=self.stock["symbol"])
                if not asset.tradable:
                    print(f"{self.stock['symbol']}: Not tradable.")
                else:
                    if outputs[0] > 0.5:  # Buy
                        if self.stock["shorting"] and asset.shortable and position.side == "short":
                            quantity = qty_percent * position_qty
                            if not asset.fractionable:
                                quantity = round(quantity)
                            price = quantity * latest["close"] * 0.999  # Transaction fee
                            if abs(price) >= 1:
                                if abs(position_qty - quantity) < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                    self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=-position_qty, side="buy", type="market", time_in_force="day")
                                    price = position_qty * latest["close"] * 0.999  # Transaction fee
                                else:
                                    self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=-quantity, side="buy", type="market", time_in_force="day")
                                cost = float(position.avg_entry_price) * quantity
                                self.session["unsettled_cash"] += price
                                self.session["pending_sales"].append((price - cost, self.trader.consecutive_days, True))

                                action = {"side": "Buy", "type": "short", "quantity": quantity, "price": latest["close"],
                                          "profit": price - cost,
                                          "settled_cash": self.session["settled_cash"],
                                          "unsettled_cash": self.session["unsettled_cash"],
                                          "datetime": now_date}
                                print(f"{self.session['interval']}m {self.stock['symbol']}: {action}")
                                self.session["logs"][self.stock["symbol"]].append(action)
                        else:
                            quantity = self.session["settled_cash"] * qty_percent * self.stock["cash_at_risk"] / latest["close"]
                            if not asset.fractionable:
                                quantity = round(quantity)
                            price = quantity * latest["close"] * 0.999  # Transaction fee
                            if price >= 1:  # Alpaca doesn't allow trades under $1
                                self.session["settled_cash"] -= price
                                self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=quantity, side="buy", type="market", time_in_force="day")

                                action = {"side": "Buy", "type": "long", "quantity": quantity, "price": latest["close"],
                                          "settled_cash": self.session["settled_cash"], "unsettled_cash": self.session["unsettled_cash"],
                                          "datetime": now_date}
                                print(f"{self.session['interval']}m {self.stock['symbol']}: {action}")
                                self.session["logs"][self.stock["symbol"]].append(action)
                    elif outputs[0] < -0.5:  # Sell
                        if self.stock["shorting"] and asset.shortable and position_qty <= 0:
                            if abs(float(position.cost_basis)) < self.session["short_limit"]:
                                quantity = -qty_percent * (self.session["short_limit"] - abs(float(position.cost_basis))) * self.stock["cash_at_risk"] / latest["close"]
                                if not asset.fractionable:
                                    quantity = round(quantity)
                                price = quantity * latest["close"] * 0.999  # Transaction fee
                                if abs(price) >= 1:  # Alpaca doesn't allow trades under $1
                                    self.session["unsettled_cash"] -= price
                                    self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=-quantity, side="sell", type="market", time_in_force="day")

                                    action = {"side": "Sell", "type": "short", "quantity": -quantity, "price": latest["close"],
                                              "settled_cash": self.session["settled_cash"],
                                              "unsettled_cash": self.session["unsettled_cash"],
                                              "datetime": now_date}
                                    print(f"{self.session['interval']}m {self.stock['symbol']}: {action}")
                                    self.session["logs"][self.stock["symbol"]].append(action)
                        elif position_qty > 0:
                            quantity = qty_percent * position_qty
                            if not asset.fractionable:
                                quantity = round(quantity)
                            price = quantity * latest["close"] * 0.999  # Transaction fee
                            if price >= 1:
                                if position_qty - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                    self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=position_qty, side="sell", type="market", time_in_force="day")
                                    price = position_qty * latest["close"]
                                else:
                                    self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=quantity, side="sell", type="market", time_in_force="day")
                                self.session["unsettled_cash"] += price
                                self.session["pending_sales"].append((price, self.trader.consecutive_days, False))

                                action = {"side": "Sell", "type": "long", "quantity": quantity, "price": latest["close"],
                                          "profit": price - (float(position.avg_entry_price) * quantity),
                                          "settled_cash": self.session["settled_cash"], "unsettled_cash": self.session["unsettled_cash"],
                                          "datetime": now_date}
                                print(f"{self.session['interval']}m {self.stock['symbol']}: {action}")
                                self.session["logs"][self.stock["symbol"]].append(action)
                prev_data = latest

                time.sleep(self.session["interval"] * 60)
            else:
                cum_price = 0.0
                cum_vol = 0.0

                next_open = self.session["clock"][0].next_open
                wait_time = (next_open - now_date).total_seconds()
                wait_time += self.session["interval"] * 60 + 10  # Wait for yahoo finance to update
                print(f"{self.session['interval']}m {self.stock['symbol']}: Pausing trading. Waiting until market opens in {wait_time / 3600} hours")
                time.sleep(wait_time)
                print(f"{self.session['interval']}m {self.stock['symbol']}: Resuming trading")


class Validation(Agent):
    def __init__(self, settings, session, stock, finbert):
        super().__init__(settings, session, stock)
        self.finbert = finbert

    def validate(self, bars, genome, shorting, asset, short_limit):
        start_time = time.time()
        net = neat.nn.RecurrentNetwork.create(genome, self.config)
        start_date = bars[0]["timestamp"].date()
        settled_cash = 100000
        start_equity = 100000
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

        # Start at 1 to have previous bar for relative change
        num_bars = len(bars)
        for i in range(1, num_bars):
            bar = bars[i]
            prev_bar = bars[i - 1]
            prev_date = prev_bar["timestamp"].date()
            date = bar["timestamp"].date()
            if date != prev_date:  # Check pending sales to settle cash after 1 day of sale
                consecutive_days += 1
                for j in reversed(range(len(pending_sales))):
                    sale_price, sale_day, is_short = pending_sales[j]
                    if consecutive_days - sale_day >= 1:
                        settled_cash += sale_price
                        if not is_short:
                            unsettled_cash -= sale_price
                        pending_sales.pop(j)

            backtest_date = bar["timestamp"].to_pydatetime()
            sentiment = self.finbert.get_saved_sentiment(self.stock["symbol"],
                                                         backtest_date - dt.timedelta(days=2),
                                                         backtest_date)
            inputs = [self.rel_change(cost, bar["close"] * shares),  # plpc
                      self.rel_change(prev_bar["open"], bar["open"]),
                      self.rel_change(prev_bar["high"], bar["high"]),
                      self.rel_change(prev_bar["low"], bar["low"]),
                      self.rel_change(prev_bar["close"], bar["close"]),
                      self.rel_change(prev_bar["volume"], bar["volume"]),
                      self.rel_change(prev_bar["vwap"], bar["vwap"]),
                      sentiment,
                      1]
            if shares < 0:
                inputs[0] *= -1
                inputs[8] = -1

            outputs = net.activate(inputs)

            qty_percent = (outputs[1] + 1) * 0.5
            if outputs[0] > 0.5:  # Buy
                if shorting and asset.shortable and shares < 0:
                    quantity = qty_percent * shares
                    if not asset.fractionable:
                        quantity = round(quantity)
                    price = quantity * bar["close"] * 0.999  # Transaction fee
                    if abs(price) >= 1:
                        if abs(shares - quantity) < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                            price = shares * bar["close"] * 0.999  # Transaction fee
                            profit = price - cost
                            action = {"side": "Buy", "type": "short", "quantity": -quantity, "price": bar["close"],
                                      "profit": profit, "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash,
                                      "datetime": bar["timestamp"].to_pydatetime()}
                            log.append(action)
                            shares = 0.0
                            cost = 0.0
                        else:
                            avg_cost = cost / abs(shares)
                            shares -= quantity
                            cost = avg_cost * shares
                            profit = price - (avg_cost * quantity)
                            action = {"side": "Buy", "type": "short", "quantity": -quantity, "price": bar["close"],
                                      "profit": profit, "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash,
                                      "datetime": bar["timestamp"].to_pydatetime()}
                            log.append(action)
                        unsettled_cash += price
                        pending_sales.append((profit, consecutive_days, True))
                        short_buys += 1
                else:
                    quantity = qty_percent * settled_cash * self.stock["cash_at_risk"] / bar["close"]
                    if not asset.fractionable:
                        quantity = round(quantity)
                    price = quantity * bar["close"]
                    if price >= 1:  # Alpaca doesn't allow trades under $1
                        cost += price
                        shares += quantity
                        settled_cash -= price

                        action = {"side": "Buy", "type": "long", "quantity": quantity, "price": bar["close"],
                                  "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                  "datetime": bar["timestamp"].to_pydatetime()}
                        log.append(action)
                        long_buys += 1
            elif outputs[0] < -0.5:  # Sell
                if shorting and asset.shortable and shares <= 0:
                    if abs(cost) < short_limit:
                        quantity = -qty_percent * (short_limit - abs(cost)) * self.stock["cash_at_risk"] / bar["close"]
                        if not asset.fractionable:
                            quantity = round(quantity)
                        price = quantity * bar["close"]
                        if abs(price) >= 1:  # Alpaca doesn't allow trades under $1
                            cost += price
                            shares += quantity
                            unsettled_cash -= price

                            action = {"side": "Sell", "type": "short", "quantity": quantity, "price": bar["close"],
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": bar["timestamp"].to_pydatetime()}
                            log.append(action)
                            short_sells += 1
                elif shares > 0:
                    quantity = qty_percent * shares
                    if not asset.fractionable:
                        quantity = round(quantity)
                    price = quantity * bar["close"] * 0.999  # Transaction fee
                    if price >= 1:
                        if shares - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                            price = shares * bar["close"] * 0.999  # Transaction fee
                            action = {"side": "Sell", "type": "long", "quantity": quantity, "price": bar["close"],
                                      "profit": price - cost, "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash + price,
                                      "datetime": bar["timestamp"].to_pydatetime()}
                            log.append(action)
                            shares = 0.0
                            cost = 0.0
                        else:
                            avg_cost = cost / shares
                            shares -= quantity
                            cost = avg_cost * shares
                            action = {"side": "Sell", "type": "long", "quantity": quantity, "price": bar["close"],
                                      "profit": price - (avg_cost * quantity), "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash + price,
                                      "datetime": bar["timestamp"].to_pydatetime()}
                            log.append(action)
                        unsettled_cash += price
                        pending_sales.append((price, consecutive_days, False))
                        long_sells += 1
            if i == num_bars - 1 or (date - start_date).days >= self.session["profit_window"]:
                if shares < 0:
                    equity = unsettled_cash + settled_cash + shares * bar["close"] - cost
                else:
                    equity = unsettled_cash + settled_cash + bar["close"] * shares
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
        stock_change = bars[-1]['close'] - bars[0]['close']
        print(f"Simulation finished in {str(time.time() - start_time)} seconds over {consecutive_days} trading days and {num_windows} profit windows"
              f"\n Stock change: ${round(stock_change, 2)} {round(100 * (stock_change / bars[0]['close']), 4)}%"
              f"\n Total profit: ${round(profit_sum, 2)} {round(100 * (profit_sum / 100000), 4)}%"
              f"\n Average {self.session['profit_window']} day profit: ${round(avg_profit, 2)} {round(avg_profit / 100000, 4)}%"
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
                            print(" Inputs")
                            print(f"  PLPC: {action[key][0]}")
                            print(f"  Open: {action[key][1]}")
                            print(f"  High: {action[key][2]}")
                            print(f"  Low: {action[key][3]}")
                            print(f"  Close: {action[key][4]}")
                            print(f"  Volume: {action[key][5]}")
                            print(f"  VWAP: {action[key][6]}")
                            print(f"  News sentiment: {action[key][7]}")
                        elif key == "outputs":
                            print(" Outputs")
                            print(f"  Buy/Sell: {action[key][0]}")
                            print(f"  Quantity: {action[key][1]}")
                        else:
                            print(f" {key}: {action[key]}")
                else:
                    print("Index not in range of log")
