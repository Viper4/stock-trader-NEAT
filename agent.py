import neat
import yfinance as yf
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import datetime as dt
import pytz
import time
import os
import multiprocessing as mp
import threading
import trainer
import saving
import visualize


class Agent:

    def __init__(self, settings, finn_client, alpaca_client):
        self.running = False
        self.settings = settings

        if self.settings["processes"] >= os.cpu_count():
            print("Using " + str(self.settings["processes"]) + " processes to train but system only has " + str(os.cpu_count()) + " cores.")
            if input("Proceed? (y/n): ") != "y":
                exit(0)

        self.population_path = self.settings["save_path"] + "/Populations"
        if not os.path.exists(self.population_path):
            os.mkdir(self.population_path)

        self.genome_path = self.settings["save_path"] + "/Genomes"
        if not os.path.exists(self.genome_path):
            os.mkdir(self.genome_path)

        self.finn_client = finn_client
        self.alpaca_client = alpaca_client

    @staticmethod
    def rel_change(a, b):
        if b == 0:
            return 0
        return (a - b) / b


class Training(Agent):

    def __init__(self, settings, finn_client, alpaca_client, option_index):
        super().__init__(settings, finn_client, alpaca_client)
        self.started = False
        self.best_genome = None
        self.start_cash = float(alpaca_client.get_account().cash)

        self.ticker_option = settings["ticker_options"][option_index]
        self.genome_file_path = os.path.join(self.genome_path, self.ticker_option["genome_filename"])
        self.population_file_path = os.path.join(self.population_path, self.ticker_option["population_filename"])

        self.gen_stagger = settings["gen_stagger"]
        self.consecutive_gens = 0

        self.stats = neat.StatisticsReporter()

        now_date = dt.datetime.now(pytz.timezone("US/Central"))

        start_date = now_date - dt.timedelta(days=self.ticker_option["backtest_days"])
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc

        bars_entity = self.alpaca_client.get_bars(
            self.ticker_option["symbol"],
            timeframe=TimeFrame(self.ticker_option["data_interval"], TimeFrameUnit.Minute),
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            limit=10000,
            sort="asc")
        self.bars = bars_entity._raw  # Array of bars
        print(self.ticker_option["symbol"] + ": Saved " + str(len(self.bars)) + " bars from " + start_date.strftime("%Y-%m-%d %H:%M:%S") + " to " + end_date.strftime("%Y-%m-%d %H:%M:%S") + " at " + str(self.ticker_option["data_interval"]) + "m intervals")

        print(self.ticker_option["symbol"] + " TRAINING agent created\n")

    def eval_genome(self, genome_id, genome, config):
        net = neat.nn.RecurrentNetwork.create(genome, config)

        shares = 0.0
        current_cash = self.start_cash
        cost = 0.0

        for i in range(1, len(self.bars)):
            bar = self.bars[i]
            previous_bar = self.bars[i-1]

            inputs = [self.rel_change(bar["c"] * shares, cost),  # plpc
                      self.rel_change(bar["o"], previous_bar["o"]),
                      self.rel_change(bar["h"], previous_bar["h"]),
                      self.rel_change(bar["l"], previous_bar["l"]),
                      self.rel_change(bar["c"], previous_bar["c"]),
                      self.rel_change(bar["v"], previous_bar["v"]),
                      self.rel_change(bar["vw"], previous_bar["vw"])]
            output = net.activate(inputs)

            quantity = ((output[1] * 0.5) + 0.5) * self.ticker_option["max_quantity"]
            price = bar["c"] * quantity
            if price >= 1:  # Alpaca doesn't allow trades under $1
                if output[0] > 0.5 and price <= current_cash:  # Wants to buy
                    cost += price
                    shares += quantity
                    current_cash -= price
                elif output[0] < -0.5 and shares > 0:  # Wants to sell
                    if shares - quantity < 0.00001:  # Alpaca doesn't allow selling < 1e-9 qty
                        current_cash += bar["c"] * shares
                        shares = 0.0
                        cost = 0.0
                    else:
                        cost_per_share = cost / shares
                        shares -= quantity
                        current_cash += price
                        cost = cost_per_share * shares
        time.sleep(0.15)

        current_cash += self.bars[-1]["c"] * shares  # Add remaining liquid assets to cash

        return [genome_id, current_cash - self.start_cash]  # Fitness equals profit

    def eval_genomes(self, genomes, config):
        while not self.running:
            time.sleep(1)

        pool = mp.Pool(processes=self.settings["processes"])  # Cant do self.pool since it tries pickling the pool object
        async_jobs = []
        for genome_id, genome in genomes:
            async_jobs.append(pool.apply_async(self.eval_genome, (genome_id, genome, config)))

        result_dict = {}
        for job in async_jobs:
            pair = job.get()
            result_dict[pair[0]] = pair[1]
        pool.close()
        pool.join()

        for genome_id, genome in genomes:
            genome.fitness = result_dict[genome_id]
            if self.best_genome is None or self.best_genome.fitness < genome.fitness:
                self.best_genome = genome

        self.consecutive_gens += 1
        if 0 < self.gen_stagger <= self.consecutive_gens:
            self.consecutive_gens = 0
            self.running = False

    def run(self):
        self.running = True
        if not self.started:
            print("Running " + self.ticker_option["symbol"] + " training agent...")
            config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])

            save_system = saving.SaveSystem(1, self.genome_file_path, self.gen_stagger, self.population_file_path)
            if os.path.exists(self.population_file_path):
                p = save_system.restore_population(self.population_file_path)
            else:
                p = neat.Population(config)
            p.add_reporter(neat.StdOutReporter(True))
            #stats = neat.StatisticsReporter()
            p.add_reporter(self.stats)
            p.add_reporter(save_system)
            #pe = neat.ParallelEvaluator(self.settings["processes"], self.eval_genome)
            #threading.Thread(target=p.run, args=(pe.evaluate, self.settings["generations"])).start()
            threading.Thread(target=p.run, args=(self.eval_genomes, self.settings["generations"])).start()
        else:
            print("Resuming " + self.ticker_option["symbol"] + " training agent...")
        self.started = True

    def plot(self):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])
        node_names = {0: 'plpc', 1: 'open%', 2: 'high%', 3: 'low%', 4: 'close%', 5: 'volume%', 6: 'vwap%'}
        visualize.draw_net(config, self.best_genome, True, node_names=node_names)
        #visualize.plot_stats(self.stats, ylog=False, view=True)
        #visualize.plot_species(self.stats, view=True)


class Trader(Agent):

    def __init__(self, settings, finn_client, alpaca_client):
        super().__init__(settings, finn_client, alpaca_client)
        self.trainer = trainer.Trainer(settings, finn_client, alpaca_client)
        print("\nTRADER agent created with settings: " + str(settings) + "\n")

    def trading_loop(self, config):
        nets = {}
        cum_prices = {}
        cum_vols = {}
        costs = {}
        for i in range(len(self.trainer.best_genomes)):
            ticker = self.settings["ticker_options"][i]["symbol"]
            nets[ticker] = neat.nn.RecurrentNetwork.create(self.trainer.best_genomes[i], config)
            cum_prices[ticker] = 0
            cum_vols[ticker] = 0
            costs[ticker] = 0

        prev_data = {}
        prev_vwap = {}

        previous_date = dt.datetime.now(pytz.timezone("US/Central"))
        displayed_acc = False
        training_thread = None
        while self.running:
            if self.finn_client.market_status(exchange='US')["isOpen"]:
                if self.trainer.running:
                    self.trainer.stop_training()
                    training_thread.join()
                    for i in range(len(self.trainer.best_genomes)):
                        option = self.settings["ticker_options"][i]
                        nets[option["symbol"]] = neat.nn.RecurrentNetwork.create(self.trainer.best_genomes[i], config)
                displayed_acc = False
                account = self.alpaca_client.get_account()
                current_cash = float(account.cash)
                positions = {}
                for position in self.alpaca_client.list_positions():
                    positions[position.symbol] = {"quantity": float(position.qty), "plpc": float(position.unrealized_plpc)}

                now_date = dt.datetime.now(pytz.timezone("US/Central"))
                if now_date.strftime('%Y-%m-%d') != previous_date.strftime('%Y-%m-%d'):
                    cum_prices.clear()
                    cum_vols.clear()
                    prev_data.clear()
                    prev_vwap.clear()
                    for option in self.settings["ticker_options"]:
                        cum_prices[option["symbol"]] = 0
                        cum_vols[option["symbol"]] = 0

                for option in self.settings["ticker_options"]:
                    ticker = option["symbol"]
                    if ticker not in positions:
                        positions[ticker] = {"quantity": 0, "plpc": 0}
                    ticker_df = yf.download(tickers=ticker, period="1d", interval=str(option["data_interval"]) + "m")
                    current_data = ticker_df.iloc[-1]

                    if ticker not in prev_data:
                        prev_data[ticker] = ticker_df.iloc[-2]
                        prev_vwap[ticker] = (prev_data[ticker]["High"] + prev_data[ticker]["Low"] + prev_data[ticker]["Close"]) / 3

                        #prev_data[ticker] = current_data
                        #prev_vwap[ticker] = 0

                    cum_prices[ticker] += current_data["Volume"] * ((current_data["High"] + current_data["Low"] + current_data["Close"]) / 3)
                    cum_vols[ticker] += current_data["Volume"]
                    vwap = cum_prices[option["symbol"]] / cum_vols[ticker] if cum_vols[ticker] > 0 else 0

                    print(str(positions[ticker]["plpc"]) + " " + str(self.rel_change(current_data["Close"], costs[ticker])))
                    inputs = [positions[ticker]["plpc"],
                              self.rel_change(current_data["Open"], prev_data[ticker]["Open"]),
                              self.rel_change(current_data["High"], prev_data[ticker]["High"]),
                              self.rel_change(current_data["Low"], prev_data[ticker]["Low"]),
                              self.rel_change(current_data["Close"], prev_data[ticker]["Close"]),
                              self.rel_change(current_data["Volume"], prev_data[ticker]["Volume"]),
                              self.rel_change(vwap, prev_vwap[ticker])]
                    output = nets[ticker].activate(inputs)

                    quantity = ((output[1] * 0.5) + 0.5) * option["max_quantity"]
                    price = current_data["Close"] * quantity

                    if price >= 1:  # Alpaca doesn't allow trades under $1
                        if output[0] > 0.5 and price <= current_cash:  # Wants to buy
                            costs[ticker] += price
                            print(ticker + " BUY (" + str(quantity) + ", $" + str(price) + ") at " + now_date.strftime("%Y-%m-%d %H:%M:%S"))
                            self.alpaca_client.submit_order(symbol=ticker, qty=quantity, side="buy", type="market", time_in_force="day")
                        elif output[0] < -0.5 and positions[ticker]["quantity"] > 0:  # Wants to sell
                            print(ticker + " SELL (" + str(quantity) + ", $" + str(price) + ") at " + now_date.strftime("%Y-%m-%d %H:%M:%S"))
                            cost_per_share = costs[ticker] / positions[ticker]["quantity"]
                            if positions[ticker]["quantity"] - quantity < 0.00001:  # Alpaca doesn't allow selling < 1e-9 qty
                                self.alpaca_client.submit_order(symbol=ticker, qty=positions[ticker]["quantity"], side="sell", type="market", time_in_force="day")
                                costs[ticker] = 0
                            else:
                                self.alpaca_client.submit_order(symbol=ticker, qty=quantity, side="sell", type="market", time_in_force="day")
                                costs[ticker] = cost_per_share * (positions[ticker]["quantity"] - quantity)
                    prev_data[ticker] = current_data
                    prev_vwap[ticker] = vwap
                previous_date = now_date
            else:
                if not displayed_acc:
                    account = self.alpaca_client.get_account()
                    open_positions = self.alpaca_client.list_positions()
                    bought_shares = {}
                    for position in open_positions:
                        bought_shares[position.symbol] = float(position.qty)
                    print("\nMarket is closed. Current Account Details:"
                          "\n Balance Change: " + str(float(account.equity) - float(account.last_equity)) +
                          "\n Cash: " + str(account.cash) +
                          "\n Equity: " + str(account.equity) +
                          "\n Bought shares: " + str(bought_shares) + "\n")
                    displayed_acc = True
                if not self.trainer.running:
                    print("Starting training while waiting for market to open.\n-----")
                    if training_thread is not None:
                        training_thread.join()
                    training_thread = threading.Thread(target=self.trainer.start_training)
                    training_thread.start()
            time.sleep(self.settings["run_interval"])

    def run(self):
        print("Running trader agent...")
        self.running = True
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])
        for option in self.settings["ticker_options"]:
            if option["genome_filename"] is None:
                print("No filename provided for " + option["symbol"])
            else:
                self.trainer.best_genomes.append(saving.SaveSystem.restore_genome(os.path.join(self.genome_path, option["genome_filename"])))
        self.trading_loop(config)
