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


class Agent:
    def __init__(self, settings, alpaca_api, option):
        self.running = False
        self.settings = settings
        self.option = option

        if self.settings["processes"] >= os.cpu_count():
            print("Using " + str(self.settings["processes"]) + " workers to train but system only has " + str(os.cpu_count()) + " cores.")
            if input("Proceed? (y/n): ") != "y":
                exit(0)

        self.population_path = self.settings["save_path"] + "/Populations"
        self.make_dir(self.population_path)

        self.genome_path = self.settings["save_path"] + "/Genomes"
        self.make_dir(self.genome_path)

        self.log_path = self.settings["save_path"] + "/Logs"
        self.make_dir(self.log_path)

        self.alpaca_api = alpaca_api

    @staticmethod
    def make_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def rel_change(a, b):
        if a == 0:
            return 0
        return (b - a) / a


class Training(Agent):
    def __init__(self, settings, alpaca_api, option, bars, sentiments):
        super().__init__(settings, alpaca_api, option)
        self.started = False
        self.best_genome = None  # Do this instead of self.p.best_genome since pickling population object adds 10s to each gen
        self.consecutive_gens = 0
        self.start_cash = 100000.0
        self.bars = bars
        self.sentiments = sentiments
        self.genome_file_path = os.path.join(self.genome_path, self.option["genome_filename"])
        self.population_file_path = os.path.join(self.population_path, self.option["population_filename"])

    def eval_genome(self, genome, config):
        net = neat.nn.RecurrentNetwork.create(genome, config)
        start_date = self.bars[0]["timestamp"].date()
        solid_cash = self.start_cash
        start_equity = self.start_cash
        liquid_cash = 0.0
        pending_sales = []
        profit_sum = 0.0
        num_windows = 0
        shares = 0.0
        cost = 0.0
        consecutive_days = 0
        log = []

        # Start at 1 to have previous bar for relative change
        num_bars = len(self.bars)
        for i in range(1, num_bars):
            bar = self.bars[i]
            prev_bar = self.bars[i-1]
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

            inputs = [self.rel_change(cost, bar["close"] * shares),  # plpc
                      self.rel_change(prev_bar["open"], bar["open"]),
                      self.rel_change(prev_bar["high"], bar["high"]),
                      self.rel_change(prev_bar["low"], bar["low"]),
                      self.rel_change(prev_bar["close"], bar["close"]),
                      self.rel_change(prev_bar["volume"], bar["volume"]),
                      self.rel_change(prev_bar["vwap"], bar["vwap"]),
                      self.sentiments[i]
                      ]
            output = net.activate(inputs)

            qty_percent = (output[1] + 1) * 0.5
            if output[0] > 0.5:  # Buy
                quantity = qty_percent * solid_cash * self.option["cash_at_risk"] / bar["close"]
                price = quantity * bar["close"]
                if price >= 1:  # Alpaca doesn't allow trades under $1
                    cost += price
                    shares += quantity
                    solid_cash -= price

                    if self.settings["log_training"]:
                        action = {"side": "Buy", "quantity": quantity, "price": price, "solid_cash": solid_cash, "liquid_cash": liquid_cash, "datetime": bar["timestamp"].to_pydatetime().astimezone(tz=pytz.timezone('US/Central'))}
                        log.append(action)
            elif output[0] < -0.5 and shares > 0:  # Sell
                quantity = qty_percent * shares
                price = quantity * bar["close"]
                if price >= 1:
                    if shares - quantity < 0.0001:  # Alpaca doesn't allow selling < 1e-9 qty
                        price = shares * bar["close"]
                        if self.settings["log_training"]:
                            action = {"side": "Sell", "quantity": quantity, "price": price,
                                      "profit": price - cost, "solid_cash": solid_cash,
                                      "liquid_cash": liquid_cash + price, "datetime": bar["timestamp"].to_pydatetime().astimezone(tz=pytz.timezone('US/Central'))}
                            log.append(action)
                        shares = 0.0
                        cost = 0.0
                    else:
                        avg_cost = cost / shares
                        shares -= quantity
                        cost = avg_cost * shares
                        if self.settings["log_training"]:
                            action = {"side": "Sell", "quantity": quantity, "price": price,
                                      "profit": price - (avg_cost * quantity), "solid_cash": solid_cash,
                                      "liquid_cash": liquid_cash + price, "datetime": bar["timestamp"].to_pydatetime().astimezone(tz=pytz.timezone('US/Central'))}
                            log.append(action)
                    liquid_cash += price
                    pending_sales.append((price, consecutive_days))
            if i == num_bars-1 or (date - start_date).days >= self.settings["profit_window"]:
                equity = liquid_cash + solid_cash + bar["close"] * shares
                profit_sum += equity - start_equity
                num_windows += 1
                start_equity = equity
                start_date = date
        time.sleep(0.15)

        return profit_sum / num_windows, log

    def eval_genomes(self, genomes, config):
        while not self.running:
            time.sleep(1)

        # There's probably a better way to do this. self.pool doesn't work: cant pickle Pool(). Separate class doesn't work: leaks memory
        pool = Pool(processes=self.settings["processes"])
        jobs = []
        for genome_id, genome in genomes:
            jobs.append(pool.apply_async(self.eval_genome, (genome, config)))

        best_log = None
        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness, log = job.get()
            if self.best_genome is None or self.best_genome.fitness < genome.fitness:
                best_log = log
                self.best_genome = genome
        if best_log is not None and self.settings["log_training"]:
            plot.Plot.plot_log(self.alpaca_api, self.option["symbol"], best_log, self.option["interval"])
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
            print("Starting {0} training agent...".format(self.option["symbol"]))
            config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])

            save_system = saving.SaveSystem(1, self.genome_file_path, self.settings["gen_stagger"], self.population_file_path)
            if os.path.exists(self.population_file_path):
                p = save_system.load_population(self.population_file_path)
            else:
                p = neat.Population(config)
            if self.settings["print_stats"]:
                p.add_reporter(neat.StdOutReporter(True))
            p.add_reporter(save_system)
            threading.Thread(target=p.run, args=(self.eval_genomes, None)).start()
        else:
            print("Resuming {0} training agent...".format(self.option["symbol"]))
        self.started = True

    def plot(self):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])
        node_names = {-9: 'plpc', -8: 'open%', -7: 'high%', -6: 'low%', -5: 'close%', -4: 'volume%', -3: 'vwap%', -2: "positive", -1: "negative", 0: 'buy/sell', 1: 'amount'}
        visualize.draw_net(config, self.best_genome, view=True, node_names=node_names, show_disabled=False)


class Trading(Agent):
    def __init__(self, settings, alpaca_api, option, finbert, trader, scraper):
        super().__init__(settings, alpaca_api, option)
        self.finbert = finbert
        self.trader = trader
        self.scraper = scraper
        self.net = None
        self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])

    def update_net(self, genome):
        self.net = neat.nn.RecurrentNetwork.create(genome, self.config)

    def run(self):
        print(f"{self.option['symbol']}: Starting trading")
        self.running = True
        cum_price = 0
        cum_vol = 0

        prev_data = None

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status():
                candles, prev_close = self.scraper.get_latest_candles(self.option["symbol"], interval=str(self.option["interval"]) + "m")
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

                position = self.trader.get_position(self.option["symbol"])
                position_qty = float(position.qty)

                sentiment = self.finbert.get_api_sentiment(self.option["symbol"], now_date - dt.timedelta(days=2), now_date)
                inputs = [float(position.unrealized_plpc),
                          self.rel_change(prev_data["open"], latest["open"]),
                          self.rel_change(prev_data["high"], latest["high"]),
                          self.rel_change(prev_data["low"], latest["low"]),
                          self.rel_change(prev_data["close"], latest["close"]),
                          self.rel_change(prev_data["volume"], latest["volume"]),
                          self.rel_change(prev_data["vwap"], latest["vwap"]),
                          sentiment
                          ]
                output = self.net.activate(inputs)

                qty_percent = (output[1] + 1) * 0.5
                if output[0] > 0.5:  # Buy
                    quantity = self.trader.solid_cash * qty_percent * self.option["cash_at_risk"] / latest["close"]
                    price = quantity * latest["close"]
                    if price >= 1:  # Alpaca doesn't allow trades under $1
                        self.trader.solid_cash -= price
                        self.alpaca_api.submit_order(symbol=self.option["symbol"], qty=quantity, side="buy", type="market", time_in_force="day")

                        action = {"side": "Buy", "quantity": quantity, "price": price,
                                  "solid_cash": self.trader.solid_cash, "liquid_cash": self.trader.liquid_cash,
                                  "datetime": now_date.astimezone(tz=pytz.timezone('US/Central'))}
                        print(f"{self.option['symbol']}: {action}")
                        self.trader.logs[self.option["symbol"]].append(action)
                elif output[0] < -0.5 and position_qty > 0:  # Sell
                    quantity = qty_percent * position_qty
                    price = quantity * latest["close"]
                    if price >= 1:
                        if position_qty - quantity < 0.0001:  # Alpaca doesn't allow selling < 1e-9 qty
                            self.alpaca_api.submit_order(symbol=self.option["symbol"], qty=position_qty, side="sell", type="market", time_in_force="day")
                            price = position_qty * latest["close"]
                        else:
                            self.alpaca_api.submit_order(symbol=self.option["symbol"], qty=quantity, side="sell", type="market", time_in_force="day")
                        self.trader.liquid_cash += price
                        self.trader.pending_sales.append((price, self.trader.consecutive_days))

                        action = {"side": "Sell", "quantity": quantity, "price": price,
                                  "profit": price - (float(position.avg_entry_price) * quantity),
                                  "solid_cash": self.trader.solid_cash, "liquid_cash": self.trader.liquid_cash,
                                  "datetime": now_date.astimezone(tz=pytz.timezone('US/Central'))}
                        print(f"{self.option['symbol']}: {action}")
                        self.trader.logs[self.option["symbol"]].append(action)
                prev_data = latest

                time.sleep(self.option["interval"] * 60)
            else:
                print(f"{self.option['symbol']}: Stopping trading. Waiting until next market open.")
                cum_price = 0.0
                cum_vol = 0.0

                next_open = self.alpaca_api.get_clock().next_open
                wait_time = (next_open - now_date).total_seconds()
                wait_time += self.option["interval"] * 60 + 10  # Wait for yahoo finance to update
                time.sleep(wait_time)
