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
def eval_genome(bars, sentiments, start_cash, genome, config, cash_at_risk, log_training, profit_window, fitness_multipliers):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    start_date = bars[0]["timestamp"].date()
    solid_cash = start_cash
    liquid_cash = 0
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
        if date != prev_date:  # Check pending sales to settle cash after 2 days of sale
            consecutive_days += 1

        inputs = [Agent.rel_change(cost, bar["close"] * shares),  # plpc
                  Agent.rel_change(prev_bar["open"], bar["open"]),
                  Agent.rel_change(prev_bar["high"], bar["high"]),
                  Agent.rel_change(prev_bar["low"], bar["low"]),
                  Agent.rel_change(prev_bar["close"], bar["close"]),
                  Agent.rel_change(prev_bar["volume"], bar["volume"]),
                  Agent.rel_change(prev_bar["vwap"], bar["vwap"]),
                  sentiments[i]
                  ]
        outputs = net.activate(inputs)

        qty_percent = (outputs[1] + 1) * 0.5
        if outputs[0] > 0.5:  # Buy
            quantity = qty_percent * solid_cash * cash_at_risk / bar["close"]
            price = quantity * bar["close"]
            if price >= 1:  # Alpaca doesn't allow trades under $1
                cost += price
                shares += quantity
                solid_cash -= price

                if log_training:
                    action = {"side": "Buy", "quantity": quantity, "price": bar["close"],
                              "solid_cash": solid_cash, "liquid_cash": liquid_cash,
                              "datetime": bar["timestamp"].to_pydatetime()}
                    log.append(action)
        elif outputs[0] < -0.5 and shares > 0:  # Sell
            quantity = qty_percent * shares
            price = quantity * bar["close"]
            if price >= 1:
                if shares - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                    price = shares * bar["close"]
                    if log_training:
                        action = {"side": "Sell", "quantity": quantity, "price": bar["close"],
                                  "profit": price - cost, "solid_cash": solid_cash,
                                  "liquid_cash": liquid_cash + price, "datetime": bar["timestamp"].to_pydatetime()}
                        log.append(action)
                    shares = 0.0
                    cost = 0.0
                else:
                    avg_cost = cost / shares
                    shares -= quantity
                    cost = avg_cost * shares
                    if log_training:
                        action = {"side": "Sell", "quantity": quantity, "price": bar["close"],
                                  "profit": price - (avg_cost * quantity), "solid_cash": solid_cash,
                                  "liquid_cash": liquid_cash + price, "datetime": bar["timestamp"].to_pydatetime()}
                        log.append(action)
                solid_cash += price
        if i == num_bars-1 or (date - start_date).days >= profit_window:
            equity = liquid_cash + solid_cash + bar["close"] * shares
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
        self.best_genome = None  # Do this instead of self.p.best_genome since pickling population object adds 10s to each gen
        self.consecutive_gens = 0
        self.start_cash = 100000.0
        self.bars = bars
        self.sentiments = sentiments
        self.genome_file_path = os.path.join(self.genome_path, self.stock["genome_filename"])
        self.population_file_path = os.path.join(self.population_path, self.stock["population_filename"])

    def eval_genomes(self, genomes, config):
        while not self.running:
            time.sleep(1)

        # There's probably a better way to do this. self.pool doesn't work: cant pickle Pool(). Separate class doesn't work: leaks memory
        pool = Pool(processes=self.settings["processes"])
        jobs = []
        for genome_id, genome in genomes:
            jobs.append(pool.apply_async(eval_genome, (self.bars, self.sentiments, self.start_cash,
                                                       genome, self.config, self.stock["cash_at_risk"],
                                                       self.settings["log_training"], self.session["profit_window"],
                                                       self.session["fitness_multipliers"])))

        best_log = None
        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness, log = job.get()
            if self.best_genome is None or self.best_genome.fitness < genome.fitness:
                best_log = log
                self.best_genome = genome

        if best_log is not None and self.settings["log_training"]:
            plot.plot_log(self.session["alpaca_api"], self.stock["symbol"], best_log, 15)
        pool.close()
        pool.join()
        pool.terminate()

        self.consecutive_gens += 1
        if 0 < self.settings["gen_stagger"] <= self.consecutive_gens:
            self.consecutive_gens = 0
            self.running = False

    def run(self):
        self.running = True
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
    def __init__(self, settings, stock, finbert, trader, scraper):
        super().__init__(settings, None, stock)
        self.finbert = finbert
        self.trader = trader
        self.scraper = scraper
        self.net = None

    def update_net(self, genome):
        self.net = neat.nn.RecurrentNetwork.create(genome, self.config)

        # Preparing the network with past 30 days data
        now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
        bars = self.trader.get_bars(self.stock["symbol"], self.trader.alpaca_api, self.trader.profile["interval"], now_date - dt.timedelta(days=30), now_date - dt.timedelta(minutes=16))
        for i in range(1, len(bars)):
            bar = bars[i]
            prev_bar = bars[i-1]
            backtest_date = bars[i]["timestamp"].to_pydatetime()
            sentiment = self.finbert.get_saved_sentiment(self.stock["symbol"],
                                                         backtest_date - dt.timedelta(days=2),
                                                         backtest_date)
            inputs = [0,  # plpc
                      self.rel_change(prev_bar["open"], bar["open"]),
                      self.rel_change(prev_bar["high"], bar["high"]),
                      self.rel_change(prev_bar["low"], bar["low"]),
                      self.rel_change(prev_bar["close"], bar["close"]),
                      self.rel_change(prev_bar["volume"], bar["volume"]),
                      self.rel_change(prev_bar["vwap"], bar["vwap"]),
                      sentiment
                      ]
            self.net.activate(inputs)
        print(f"{self.trader.profile['name']} {self.stock['symbol']}: Updated network")

    def run(self):
        print(f"{self.trader.profile['name']} {self.stock['symbol']}: Starting trading")
        self.running = True
        cum_price = 0
        cum_vol = 0

        prev_data = None

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status():
                candles, prev_close = self.scraper.get_latest_candles(self.stock["symbol"], interval=str(self.trader.profile["interval"]) + "m")
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
                position_qty = position["longQuantity"]

                sentiment = self.finbert.get_api_sentiment(self.stock["symbol"], now_date - dt.timedelta(days=2), now_date)
                inputs = [position["longOpenProfitLoss"],
                          self.rel_change(prev_data["open"], latest["open"]),
                          self.rel_change(prev_data["high"], latest["high"]),
                          self.rel_change(prev_data["low"], latest["low"]),
                          self.rel_change(prev_data["close"], latest["close"]),
                          self.rel_change(prev_data["volume"], latest["volume"]),
                          self.rel_change(prev_data["vwap"], latest["vwap"]),
                          sentiment
                          ]
                outputs = self.net.activate(inputs)

                qty_percent = (outputs[1] + 1) * 0.5

                if outputs[0] > 0.5:  # Buy
                    account = self.trader.schwab_api.get_account()
                    total_cash = account["currentBalances"]["cashBalance"]
                    solid_cash = account["currentBalances"]["cashAvailableForTrading"]
                    liquid_cash = total_cash - solid_cash
                    if "longMarketValue" in account["currentBalances"]:
                        market_value = account["currentBalances"]["longMarketValue"]
                    else:
                        market_value = 0
                    used_cash = market_value + liquid_cash  # Cash in stocks + liquid cash
                    if self.trader.profile["cash_limit"] < 0 or used_cash < self.trader.profile["cash_limit"]:
                        quantity = self.trader.profile["cash_limit"] * qty_percent * self.stock["cash_at_risk"] / latest["close"]
                        quantity = round(quantity)
                        if quantity > 0:
                            self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="BUY")

                            action = {"side": "Buy", "quantity": quantity, "price": latest["close"],
                                      "solid_cash": solid_cash, "liquid_cash": liquid_cash,
                                      "datetime": now_date}
                            print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                            self.trader.logs[self.stock["symbol"]].append(action)
                elif outputs[0] < -0.5 and position_qty > 0:  # Sell
                    account = self.trader.schwab_api.get_account()
                    solid_cash = account["currentBalances"]["cashAvailableForTrading"]
                    liquid_cash = account["currentBalances"]["cashBalance"] - solid_cash
                    quantity = qty_percent * position_qty
                    quantity = round(quantity)
                    price = quantity * latest["close"]
                    if price >= 1:
                        if position_qty - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                            self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="SELL")
                        else:
                            self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="SELL")

                        action = {"side": "Sell", "quantity": quantity, "price": latest["close"],
                                  "profit": position["longOpenProfitLoss"],
                                  "solid_cash": solid_cash, "liquid_cash": liquid_cash,
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

    def update_net(self, genome):
        self.net = neat.nn.RecurrentNetwork.create(genome, self.config)

        # Preparing the network with past 30 days data
        now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
        bars = self.trader.get_bars(self.stock["symbol"], self.session["alpaca_api"], self.session["interval"], now_date - dt.timedelta(days=30), now_date - dt.timedelta(minutes=16))
        for i in range(1, len(bars)):
            bar = bars[i]
            prev_bar = bars[i-1]
            backtest_date = bars[i]["timestamp"].to_pydatetime()
            sentiment = self.finbert.get_saved_sentiment(self.stock["symbol"],
                                                         backtest_date - dt.timedelta(days=2),
                                                         backtest_date)
            inputs = [0,  # plpc
                      self.rel_change(prev_bar["open"], bar["open"]),
                      self.rel_change(prev_bar["high"], bar["high"]),
                      self.rel_change(prev_bar["low"], bar["low"]),
                      self.rel_change(prev_bar["close"], bar["close"]),
                      self.rel_change(prev_bar["volume"], bar["volume"]),
                      self.rel_change(prev_bar["vwap"], bar["vwap"]),
                      sentiment
                      ]
            self.net.activate(inputs)
        print(f" {self.stock['symbol']}: Updated network")

    def run(self):
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
                          sentiment
                          ]
                outputs = self.net.activate(inputs)

                qty_percent = (outputs[1] + 1) * 0.5

                asset = self.session["alpaca_api"].get_asset(symbol=self.stock["symbol"])
                if not asset.tradable:
                    print(f"{self.stock['symbol']}: Not tradable.")
                else:
                    if outputs[0] > 0.5:  # Buy
                        quantity = self.session["solid_cash"] * qty_percent * self.stock["cash_at_risk"] / latest["close"]
                        if not asset.fractionable:
                            quantity = round(quantity)
                        price = quantity * latest["close"]
                        if price >= 1:  # Alpaca doesn't allow trades under $1
                            self.session["solid_cash"] -= price
                            self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=quantity, side="buy", type="market", time_in_force="day")

                            action = {"side": "Buy", "quantity": quantity, "price": latest["close"],
                                      "solid_cash": self.session["solid_cash"], "liquid_cash": self.session["liquid_cash"],
                                      "datetime": now_date}
                            print(f"{self.session['interval']}m {self.stock['symbol']}: {action}")
                            self.session["logs"][self.stock["symbol"]].append(action)
                    elif outputs[0] < -0.5 and position_qty > 0:  # Sell
                        quantity = qty_percent * position_qty
                        if not asset.fractionable:
                            quantity = round(quantity)
                        price = quantity * latest["close"]
                        if price >= 1:
                            if position_qty - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=position_qty, side="sell", type="market", time_in_force="day")
                                price = position_qty * latest["close"]
                            else:
                                self.session["alpaca_api"].submit_order(symbol=self.stock["symbol"], qty=quantity, side="sell", type="market", time_in_force="day")
                            self.session["liquid_cash"] += price
                            self.session["pending_sales"].append((price, self.trader.consecutive_days))

                            action = {"side": "Sell", "quantity": quantity, "price": latest["close"],
                                      "profit": price - (float(position.avg_entry_price) * quantity),
                                      "solid_cash": self.session["solid_cash"], "liquid_cash": self.session["liquid_cash"],
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

    def validate(self, bars, genome):
        start_time = time.time()
        net = neat.nn.RecurrentNetwork.create(genome, self.config)
        start_date = bars[0]["timestamp"].date()
        solid_cash = 100000
        start_equity = 100000
        liquid_cash = 0.0
        pending_sales = []
        profit_sum = 0.0
        num_windows = 0
        shares = 0.0
        cost = 0.0
        consecutive_days = 1
        log = []
        num_sell = 0
        num_buy = 0
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
            if date != prev_date:  # Check pending sales to settle cash after 2 days of sale
                consecutive_days += 1
                for j in reversed(range(len(pending_sales))):
                    sale = pending_sales[j]
                    if consecutive_days - sale[1] > 2:
                        solid_cash += sale[0]
                        liquid_cash -= sale[0]
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
                      sentiment
                      ]
            outputs = net.activate(inputs)

            qty_percent = (outputs[1] + 1) * 0.5
            if outputs[0] > 0.5:  # Buy
                quantity = qty_percent * solid_cash * self.stock["cash_at_risk"] / bar["close"]
                price = quantity * bar["close"]
                if price >= 1:  # Alpaca doesn't allow trades under $1
                    cost += price
                    shares += quantity
                    solid_cash -= price

                    action = {"inputs": inputs, "outputs": outputs,
                              "side": "Buy", "quantity": quantity, "price": bar["close"],
                              "solid_cash": solid_cash, "liquid_cash": liquid_cash,
                              "datetime": bar["timestamp"].to_pydatetime().astimezone(tz=pytz.timezone('US/Eastern'))}
                    log.append(action)
                    num_buy += 1
            elif outputs[0] < -0.5 and shares > 0:  # Sell
                quantity = qty_percent * shares
                price = quantity * bar["close"]
                if price >= 1:
                    if shares - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                        price = shares * bar["close"]
                        action = {"inputs": inputs, "outputs": outputs,
                                  "side": "Sell", "quantity": quantity, "price": bar["close"],
                                  "profit": price - cost, "solid_cash": solid_cash,
                                  "liquid_cash": liquid_cash + price,
                                  "datetime": bar["timestamp"].to_pydatetime().astimezone(tz=pytz.timezone('US/Eastern'))}
                        log.append(action)
                        num_sell += 1
                        shares = 0.0
                        cost = 0.0
                    else:
                        avg_cost = cost / shares
                        shares -= quantity
                        cost = avg_cost * shares
                        action = {"inputs": inputs, "outputs": outputs,
                                  "side": "Sell", "quantity": quantity, "price": bar["close"],
                                  "profit": price - (avg_cost * quantity), "solid_cash": solid_cash,
                                  "liquid_cash": liquid_cash + price,
                                  "datetime": bar["timestamp"].to_pydatetime().astimezone(tz=pytz.timezone('US/Eastern'))}
                        log.append(action)
                        num_sell += 1
                    liquid_cash += price
                    pending_sales.append((price, consecutive_days))
            if i == num_bars - 1 or (date - start_date).days >= self.session["profit_window"]:
                equity = liquid_cash + solid_cash + bar["close"] * shares
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
              f"\n Total buys: {num_buy}"
              f"\n Total sells: {num_sell}"
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
