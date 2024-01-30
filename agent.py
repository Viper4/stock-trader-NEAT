import neat
import yfinance as yf
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import datetime as dt
import pytz
import time
import os
from multiprocessing import Pool
import threading
import trainer
import saving
import visualize


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
        if b == 0:
            return 0
        return (a - b) / b


class Training(Agent):
    def __init__(self, settings, alpaca_api, finbert, option_index, start_date, end_date):
        super().__init__(settings, alpaca_api, finbert)
        self.started = False
        self.best_genome = None  # Do this instead of self.p.best_genome since pickling population object adds 10s to each gen
        self.consecutive_gens = 0

        self.start_cash = 50000.0

        self.ticker_option = settings["ticker_options"][option_index]
        self.genome_file_path = os.path.join(self.genome_path, self.ticker_option["genome_filename"])
        self.population_file_path = os.path.join(self.population_path, self.ticker_option["population_filename"])

        bars_entity = self.alpaca_api.get_bars(
            symbol=self.ticker_option["symbol"],
            timeframe=TimeFrame(self.ticker_option["data_interval"], TimeFrameUnit.Minute),
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            limit=30000,
            sort="asc")
        self.bars = bars_entity.__dict__["_raw"]  # Array of bars
        print("{0}: Saved {1} bars from {2} to {3} at {4}m intervals".format(self.ticker_option["symbol"], len(self.bars), start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S"), self.ticker_option["data_interval"]))

        '''
        # This is way too expensive for CPU or Cuda
        self.sentiments = [[0, 0, 0]]  # First bar is skipped
        for i in range(1, len(self.bars)):
            backtest_date = dt.datetime.strptime(self.bars[i]["t"], "%Y-%m-%dT%H:%M:%SZ")
            sentiment = self.finbert.get_saved_sentiment(self.ticker_option["symbol"], backtest_date - dt.timedelta(days=2), backtest_date)
            self.sentiments.append(sentiment)
            print("{0}: {1}".format(i, sentiment))
            time.sleep(0.015)
        print("{0}: Saved {1} sentiments".format(self.ticker_option["symbol"], len(self.sentiments)))'''

        print("{0} TRAINING agent created\n".format(self.ticker_option["symbol"]))

    def eval_genome(self, genome, config):
        net = neat.nn.RecurrentNetwork.create(genome, config)

        shares = 0.0
        current_cash = self.start_cash
        cost = 0.0

        # Start at 1 to have a previous bar for relative change
        for i in range(1, len(self.bars)):
            bar = self.bars[i]
            previous_bar = self.bars[i-1]

            '''sentiment = self.sentiments[i]

            inputs = [sentiment[0], sentiment[1],  # positive, negative
                      self.rel_change(bar["c"] * shares, cost),  # plpc
                      self.rel_change(bar["o"], previous_bar["o"]),
                      self.rel_change(bar["h"], previous_bar["h"]),
                      self.rel_change(bar["l"], previous_bar["l"]),
                      self.rel_change(bar["c"], previous_bar["c"]),
                      self.rel_change(bar["v"], previous_bar["v"]),
                      self.rel_change(bar["vw"], previous_bar["vw"])]'''
            inputs = [self.rel_change(bar["c"] * shares, cost),  # plpc
                      self.rel_change(bar["o"], previous_bar["o"]),
                      self.rel_change(bar["h"], previous_bar["h"]),
                      self.rel_change(bar["l"], previous_bar["l"]),
                      self.rel_change(bar["c"], previous_bar["c"]),
                      self.rel_change(bar["v"], previous_bar["v"]),
                      self.rel_change(bar["vw"], previous_bar["vw"]),
                      ]
            output = net.activate(inputs)

            quantity = current_cash * self.ticker_option["cash_at_risk"] / previous_bar["c"]
            price = quantity * bar["c"]

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

        return current_cash - self.start_cash  # Fitness equals profit

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
            threading.Thread(target=p.run, args=(self.eval_genomes, self.settings["generations"])).start()
        else:
            print("Resuming {0} training agent...".format(self.ticker_option["symbol"]))
        self.started = True

    def plot(self):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])
        node_names = {-7: 'plpc', -6: 'open%', -5: 'high%', -4: 'low%', -3: 'close%', -2: 'volume%', -1: 'vwap%', 0: 'buy/sell', 1: 'amount'}
        visualize.draw_net(config, self.best_genome, True, node_names=node_names)


class Trader(Agent):
    def __init__(self, settings, alpaca_api, finbert):
        super().__init__(settings, alpaca_api, finbert)
        self.trainer = trainer.Trainer(settings, alpaca_api, finbert)
        print("\nTRADER agent created with settings: {0}\n".format(settings))

    def trading_loop(self, config):
        nets = {}
        cum_prices = {}
        cum_vols = {}
        for i in range(len(self.trainer.best_genomes)):
            ticker = self.settings["ticker_options"][i]["symbol"]
            nets[ticker] = neat.nn.RecurrentNetwork.create(self.trainer.best_genomes[i], config)
            cum_prices[ticker] = 0
            cum_vols[ticker] = 0

        prev_data = {}
        prev_vwap = {}

        previous_date = dt.datetime.now(pytz.UTC)
        displayed_acc = False
        training_thread = None
        while self.running:
            if self.alpaca_api.get_clock().is_open:
                if self.trainer.running:
                    self.trainer.stop_training()
                    training_thread.join()
                    for i in range(len(self.trainer.best_genomes)):
                        option = self.settings["ticker_options"][i]
                        nets[option["symbol"]] = neat.nn.RecurrentNetwork.create(self.trainer.best_genomes[i], config)
                displayed_acc = False
                account = self.alpaca_api.get_account()
                current_cash = float(account.cash)
                positions = {}
                for position in self.alpaca_api.list_positions():
                    positions[position.symbol] = {"quantity": float(position.qty), "plpc": float(position.unrealized_plpc)}

                now_date = dt.datetime.now(pytz.UTC)
                if now_date.isoformat() != previous_date.isoformat():
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

                    '''sentiment = self.finbert.get_api_sentiment(ticker, now_date - dt.timedelta(days=2), now_date)

                    inputs = [sentiment[0], sentiment[1],  # positive, negative
                              positions[ticker]["plpc"],
                              self.rel_change(current_data["Open"], prev_data[ticker]["Open"]),
                              self.rel_change(current_data["High"], prev_data[ticker]["High"]),
                              self.rel_change(current_data["Low"], prev_data[ticker]["Low"]),
                              self.rel_change(current_data["Close"], prev_data[ticker]["Close"]),
                              self.rel_change(current_data["Volume"], prev_data[ticker]["Volume"]),
                              self.rel_change(vwap, prev_vwap[ticker])]'''
                    inputs = [positions[ticker]["plpc"],
                              self.rel_change(current_data["Open"], prev_data[ticker]["Open"]),
                              self.rel_change(current_data["High"], prev_data[ticker]["High"]),
                              self.rel_change(current_data["Low"], prev_data[ticker]["Low"]),
                              self.rel_change(current_data["Close"], prev_data[ticker]["Close"]),
                              self.rel_change(current_data["Volume"], prev_data[ticker]["Volume"]),
                              self.rel_change(vwap, prev_vwap[ticker])]
                    output = nets[ticker].activate(inputs)

                    quantity = current_cash * option["cash_at_risk"] / prev_data[ticker]["Close"]
                    price = quantity * current_data["Close"]

                    if price >= 1:  # Alpaca doesn't allow trades under $1
                        if output[0] > 0.5 and price <= current_cash:  # Wants to buy
                            print(ticker + " BUY (" + str(quantity) + ", $" + str(price) + ") at " + now_date.strftime("%Y-%m-%d %H:%M:%S"))
                            self.alpaca_api.submit_order(symbol=ticker, qty=quantity, side="buy", type="market", time_in_force="day")
                        elif output[0] < -0.5 and positions[ticker]["quantity"] > 0:  # Wants to sell
                            print(ticker + " SELL (" + str(quantity) + ", $" + str(price) + ") at " + now_date.strftime("%Y-%m-%d %H:%M:%S"))
                            if positions[ticker]["quantity"] - quantity < 0.00001:  # Alpaca doesn't allow selling < 1e-9 qty
                                self.alpaca_api.submit_order(symbol=ticker, qty=positions[ticker]["quantity"], side="sell", type="market", time_in_force="day")
                            else:
                                self.alpaca_api.submit_order(symbol=ticker, qty=quantity, side="sell", type="market", time_in_force="day")
                    prev_data[ticker] = current_data
                    prev_vwap[ticker] = vwap
                previous_date = now_date
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
                if not self.trainer.running:
                    print("Starting training while waiting for market to open.\n-----")
                    if training_thread is not None:
                        training_thread.join()
                    training_thread = threading.Thread(target=self.trainer.start_training)
                    training_thread.start()
            time.sleep(self.settings["trade_delay"])

    def run(self):
        print("Starting trader agent...")
        self.running = True
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])
        for option in self.settings["ticker_options"]:
            if option["genome_filename"] is None:
                print("No genome filename provided for " + option["symbol"])
                exit(0)
            else:
                self.trainer.best_genomes.append(saving.SaveSystem.restore_genome(os.path.join(self.genome_path, option["genome_filename"])))
        self.trading_loop(config)
