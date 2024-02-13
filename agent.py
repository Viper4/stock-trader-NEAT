import neat
import datetime as dt
import pytz
import time
import os
from multiprocessing import Pool
import threading
import trainer
import saving
import visualize
import realtime_candles


class Agent(object):
    def __init__(self, settings, alpaca_api, finbert):
        self.running = False
        self.settings = settings

        if self.settings["processes"] >= os.cpu_count():
            print("Using " + str(self.settings["processes"]) + " workers to train but system only has " + str(os.cpu_count()) + " cores.")
            if input("Proceed? (y/n): ") != "y":
                exit(0)

        self.population_path = self.settings["save_path"] + "/Populations"
        if not os.path.exists(self.population_path):
            os.mkdir(self.population_path)

        self.genome_path = self.settings["save_path"] + "/Genomes"
        if not os.path.exists(self.genome_path):
            os.mkdir(self.genome_path)

        self.alpaca_api = alpaca_api
        self.finbert = finbert

    @staticmethod
    def rel_change(a, b):
        if a == 0:
            return 0
        return (b - a) / a


class Training(Agent):
    def __init__(self, settings, alpaca_api, finbert, option, bars, sentiments):
        super().__init__(settings, alpaca_api, finbert)
        self.started = False
        self.best_genome = None  # Do this instead of self.p.best_genome since pickling population object adds 10s to each gen
        self.consecutive_gens = 0
        self.start_cash = 50000.0
        self.ticker_option = option
        self.bars = bars
        self.sentiments = sentiments
        self.genome_file_path = os.path.join(self.genome_path, self.ticker_option["genome_filename"])
        self.population_file_path = os.path.join(self.population_path, self.ticker_option["population_filename"])

    def eval_genome(self, genome, config):
        net = neat.nn.RecurrentNetwork.create(genome, config)
        start_date = self.bars[0]["timestamp"].date()
        current_cash = self.start_cash
        start_equity = self.start_cash
        profit_sum = 0.0
        num_windows = 0
        shares = 0.0
        cost = 0.0

        # Start at 1 to have previous bar for relative change
        num_bars = len(self.bars)
        for i in range(1, num_bars):
            bar = self.bars[i]
            prev_bar = self.bars[i-1]
            date = bar["timestamp"].date()

            sentiment = self.sentiments[i]
            inputs = [self.rel_change(cost, bar["close"] * shares),  # plpc
                      self.rel_change(prev_bar["open"], bar["open"]),
                      self.rel_change(prev_bar["high"], bar["high"]),
                      self.rel_change(prev_bar["low"], bar["low"]),
                      self.rel_change(prev_bar["close"], bar["close"]),
                      self.rel_change(prev_bar["volume"], bar["volume"]),
                      self.rel_change(prev_bar["vwap"], bar["vwap"]),
                      sentiment[0], sentiment[1],  # positive, negative from 0 to 1
                      ]
            output = net.activate(inputs)

            qty_output = (output[1] + 1) * 0.5
            if output[0] > 0.5:  # Buy
                quantity = qty_output * current_cash * self.ticker_option["cash_at_risk"] / bar["close"]
                price = quantity * bar["close"]
                if price >= 1:  # Alpaca doesn't allow trades under $1
                    cost += price
                    shares += quantity
                    current_cash -= price
            elif output[0] < -0.5 and shares > 0:  # Sell
                quantity = qty_output * shares
                price = quantity * bar["close"]
                if price >= 1:
                    if shares - quantity < 0.0001:  # Alpaca doesn't allow selling < 1e-9 qty
                        current_cash += bar["close"] * shares
                        shares = 0.0
                        cost = 0.0
                    else:
                        cost_per_share = cost / shares
                        shares -= quantity
                        current_cash += price
                        cost = cost_per_share * shares
            if i == num_bars-1 or (date - start_date).days >= self.settings["profit_window"]:
                equity = current_cash + bar["close"] * shares
                profit_sum += equity - start_equity
                num_windows += 1
                start_equity = equity
                start_date = date
        time.sleep(0.15)

        return profit_sum / num_windows

    def eval_genomes(self, genomes, config):
        while not self.running:
            time.sleep(1)

        # There's probably a better way to do this. self.pool doesn't work: cant pickle Pool(). Separate class doesn't work: leaks memory
        pool = Pool(processes=self.settings["processes"])
        jobs = []
        for genome_id, genome in genomes:
            jobs.append(pool.apply_async(self.eval_genome, (genome, config)))

        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get()
            if self.best_genome is None or self.best_genome.fitness < genome.fitness:
                self.best_genome = genome
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
            print("Starting {0} training agent...".format(self.ticker_option["symbol"]))
            config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])

            save_system = saving.SaveSystem(1, self.genome_file_path, self.settings["gen_stagger"], self.population_file_path)
            if os.path.exists(self.population_file_path):
                p = save_system.restore_population(self.population_file_path)
            else:
                p = neat.Population(config)
            if self.settings["print_stats"]:
                p.add_reporter(neat.StdOutReporter(True))
            p.add_reporter(save_system)
            threading.Thread(target=p.run, args=(self.eval_genomes, None)).start()
        else:
            print("Resuming {0} training agent...".format(self.ticker_option["symbol"]))
        self.started = True

    def plot(self):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])
        node_names = {-9: 'plpc', -8: 'open%', -7: 'high%', -6: 'low%', -5: 'close%', -4: 'volume%', -3: 'vwap%', -2: "positive", -1: "negative", 0: 'buy/sell', 1: 'amount'}
        visualize.draw_net(config, self.best_genome, view=True, node_names=node_names, show_disabled=False)


class Trader(Agent):
    def __init__(self, settings, alpaca_api, finbert):
        super().__init__(settings, alpaca_api, finbert)
        self.trainer = trainer.Trainer(settings, alpaca_api, finbert)
        self.training_thread = None
        print("TRADER agent created with settings: {0}\n".format(settings))

    def start_training(self):
        if self.training_thread is not None:
            self.training_thread.join()
        self.training_thread = threading.Thread(target=self.trainer.start_training)
        self.training_thread.start()

    def trading_loop(self, config):
        nets = {}
        cum_prices = {}
        cum_vols = {}
        for symbol in self.trainer.agents:
            nets[symbol] = neat.nn.RecurrentNetwork.create(self.trainer.agents[symbol].best_genome, config)
            cum_prices[symbol] = 0
            cum_vols[symbol] = 0

        prev_vwap = {}

        # EST time
        extended_start = dt.time(hour=4)
        extended_end = dt.time(hour=20)
        now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
        previous_date = now_date
        displayed_acc = False
        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            extended_trading = self.settings["extended_hours"] and extended_start < now_date.time() < extended_end
            if self.alpaca_api.get_clock().is_open or extended_trading:
                if self.trainer.running:
                    self.trainer.stop_training()
                    self.trainer.running = False
                    self.training_thread.join()
                    for symbol in self.trainer.agents:
                        if self.trainer.agents[symbol].best_genome is not None:
                            nets[symbol] = neat.nn.RecurrentNetwork.create(self.trainer.agents[symbol].best_genome, config)

                displayed_acc = False
                account = self.alpaca_api.get_account()
                current_cash = float(account.cash)
                positions = {}
                for position in self.alpaca_api.list_positions():
                    positions[position.symbol] = {"quantity": float(position.qty), "plpc": float(position.unrealized_plpc), "pl": float(position.unrealized_pl)}

                if now_date.isoformat() != previous_date.isoformat():
                    cum_prices.clear()
                    cum_vols.clear()
                    prev_vwap.clear()
                    for option in self.settings["ticker_options"]:
                        cum_prices[option["symbol"]] = 0
                        cum_vols[option["symbol"]] = 0

                for option in self.settings["ticker_options"]:
                    ticker = option["symbol"]
                    if ticker not in positions:
                        positions[ticker] = {"quantity": 0, "plpc": 0, "pl": 0}
                    candles = realtime_candles.get_latest_candles(ticker, interval=str(option["data_interval"] + "m"))

                    latest = candles[-1]
                    prev = candles[-2] if len(candles) >= 2 else latest

                    if ticker not in prev_vwap:
                        prev_vwap[ticker] = (latest["high"] + latest["low"] + latest["close"]) / 3

                    cum_prices[ticker] += latest["volume"] * ((latest["high"] + latest["low"] + latest["close"]) / 3)
                    cum_vols[ticker] += latest["volume"]
                    vwap = cum_prices[option["symbol"]] / cum_vols[ticker] if cum_vols[ticker] > 0 else 0

                    sentiment = self.finbert.get_api_sentiment(ticker, now_date - dt.timedelta(days=2), now_date)
                    inputs = [positions[ticker]["plpc"],
                              self.rel_change(prev["open"], latest["open"]),
                              self.rel_change(prev["high"], latest["high"]),
                              self.rel_change(prev["Low"], latest["low"]),
                              self.rel_change(prev["Close"], latest["close"]),
                              self.rel_change(prev["Volume"], latest["volume"]),
                              self.rel_change(prev_vwap[ticker], vwap),
                              sentiment[0], sentiment[1],  # positive, negative from 0 to 1
                              ]

                    output = nets[ticker].activate(inputs)
                    print("{0}\n Inputs: {1}\n Output: {2}".format(ticker, inputs, output))

                    qty_output = (output[1] + 1) * 0.5
                    if output[0] > 0.5:  # Buy
                        quantity = current_cash * qty_output * option["cash_at_risk"] / latest["close"]
                        price = quantity * latest["close"]
                        if price >= 1:  # Alpaca doesn't allow trades under $1
                            print("{0} BUY (qty: {1}, price: ${2}) at {3} EST".format(ticker, quantity, price, now_date.strftime("%Y-%m-%d %H:%M:%S")))
                            if self.alpaca_api.get_clock().is_open:
                                self.alpaca_api.submit_order(symbol=ticker, qty=quantity, side="buy", type="market", time_in_force="day")
                            else:
                                self.alpaca_api.submit_order(symbol=ticker, qty=quantity, side="buy", type="limit", time_in_force="day", extended_hours=True)
                    elif output[0] < -0.5 and positions[ticker]["quantity"] > 0:  # Sell
                        quantity = qty_output * positions[ticker]["quantity"]
                        price = quantity * latest["close"]
                        if price >= 1:
                            print("{0} SELL (qty: {1}, price: ${2}, profit: {3}) at {4} EST".format(ticker, quantity, price, positions[ticker]["pl"], now_date.strftime("%Y-%m-%d %H:%M:%S")))
                            if positions[ticker]["quantity"] - quantity < 0.0001:  # Alpaca doesn't allow selling < 1e-9 qty
                                if self.alpaca_api.get_clock().is_open:
                                    self.alpaca_api.submit_order(symbol=ticker, qty=positions[ticker]["quantity"], side="sell", type="market", time_in_force="day")
                                else:
                                    self.alpaca_api.submit_order(symbol=ticker, qty=positions[ticker]["quantity"], side="sell", type="limit", time_in_force="day", extended_hours=True)
                            else:
                                if self.alpaca_api.get_clock().is_open:
                                    self.alpaca_api.submit_order(symbol=ticker, qty=quantity, side="sell", type="market", time_in_force="day")
                                else:
                                    self.alpaca_api.submit_order(symbol=ticker, qty=quantity, side="sell", type="limit", time_in_force="day", extended_hours=True)

                    prev_vwap[ticker] = vwap
                previous_date = now_date
                time.sleep(self.settings["trade_delay"])
            else:
                if not displayed_acc:
                    account = self.alpaca_api.get_account()
                    open_positions = self.alpaca_api.list_positions()
                    bought_shares = {}
                    for position in open_positions:
                        bought_shares[position.symbol] = float(position.qty)
                    print("\nMarket is closed. Current Account Details:"
                          "\n Balance Change: " + str(float(account.equity) - float(account.last_equity)) +
                          "\n Cash: " + str(account.cash) +
                          "\n Equity: " + str(account.equity) +
                          "\n Bought shares: " + str(bought_shares) + "\n")
                    displayed_acc = True
                next_open = self.alpaca_api.get_clock().next_open
                next_ext_open = next_open - dt.timedelta(hours=2, minutes=30)
                wait_time = (next_ext_open - now_date).total_seconds() if self.settings["extended_hours"] else (next_open - now_date).total_seconds()
                wait_time += 300  # Wait extra 5 minutes so yahoo finance can update
                if not self.trainer.running:
                    print("Starting training while waiting for market to open in {0} hours.\n-----".format(wait_time / 3600))
                    self.start_training()
                time.sleep(wait_time)

    def run(self):
        print("Starting trader agent...")
        self.running = True
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])
        for option in self.settings["ticker_options"]:
            if option["genome_filename"] is None:
                print("No genome filename provided for " + option["symbol"])
                exit(0)
            else:
                self.trainer.agents[option["symbol"]].best_genome = saving.SaveSystem.restore_genome(os.path.join(self.genome_path, option["genome_filename"]))
        self.trading_loop(config)
