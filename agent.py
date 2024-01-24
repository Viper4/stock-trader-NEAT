import neat
import pickle
import yfinance as yf
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import datetime as dt
import pytz
import time
import os
import multiprocessing as mp
import glob


class Agent:

    def __init__(self, settings, finn_client, alpaca_client):
        self.settings = settings
        if self.settings["processes"] >= os.cpu_count():
            print("Using " + str(self.settings["processes"]) + " processes to train but system only has " + str(os.cpu_count()) + " cores.")
            if input("Proceed? (y/n): ") != "y":
                exit(0)
        self.finn_client = finn_client
        self.alpaca_client = alpaca_client
        #self.earnings = {}
        #self.sentiments = {}
        #now_date = dt.datetime.now(pytz.timezone("US/Central"))
        #start_date = now_date - dt.timedelta(days=settings["backtest_days"])

        #for ticker in self.settings["tickers"]:
        #    self.earnings[ticker] = finn_client.company_earnings(ticker, limit=5)
        #    self.sentiments[ticker] = finn_client.stock_insider_sentiment(ticker, start_date, now_date)["data"]

        #print("Saved company earnings: " + str(self.earnings) + "\n")
        #print("Saved insider sentiments: " + str(self.sentiments) + "\n")

    def rel_change(self, a, b):
        if b == 0:
            return 0
        return (a - b) / b

    def stop(self):
        exit(0)


class Training(Agent):

    def __init__(self, settings, finn_client, alpaca_client):
        super().__init__(settings, finn_client, alpaca_client)
        self.start_cash = 1000.0
        self.best_genome = None
        if self.settings["genome_filename"] is not None:
            try:
                with open(os.path.join(self.settings["save_path"], self.settings["genome_filename"]), "rb") as file:
                    self.best_genome = pickle.load(file)
            except FileNotFoundError:
                print("No genome save file found.\n")

        now_date = dt.datetime.now(pytz.timezone("US/Central"))
        start_date = now_date - dt.timedelta(days=settings["backtest_days"])
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc
        self.bars = {}
        for ticker in self.settings["tickers"]:
            bars_entity = self.alpaca_client.get_bars(
                ticker,
                timeframe=TimeFrame(self.settings["data_interval"], TimeFrameUnit.Minute),
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                limit=10000,
                sort="asc")
            self.bars[ticker] = bars_entity._raw
            print(ticker + ": Saved " + str(len(self.bars[ticker])) + " bars from " + start_date.strftime("%Y-%m-%d-%H:%M:%S") + " to " + end_date.strftime("%Y-%m-%d-%H:%M:%S"))

        print("\nTRAINING agent created with settings: " + str(settings) + "\n")

    def eval_genome(self, genome_id, genome, config):
        net = neat.nn.RecurrentNetwork.create(genome, config)

        bought_stocks = {}
        final_prices = {}
        current_cash = self.start_cash
        current_equity = current_cash
        last_equity = current_equity

        for ticker in self.settings["tickers"]:
            bought_stocks[ticker] = 0
            for i in range(1, len(self.bars[ticker])):
                bar = self.bars[ticker][i]
                previous_bar = self.bars[ticker][i-1]

                if i == len(self.bars[ticker]) - 1:
                    final_prices[ticker] = bar["c"]

                current_equity = current_cash + bought_stocks[ticker] * bar["c"]

                inputs = [self.rel_change(current_equity, last_equity),
                          self.rel_change(bar["o"], previous_bar["o"]),
                          self.rel_change(bar["h"], previous_bar["h"]),
                          self.rel_change(bar["l"], previous_bar["l"]),
                          self.rel_change(bar["c"], previous_bar["c"]),
                          self.rel_change(bar["v"], previous_bar["v"]),
                          self.rel_change(bar["vw"], previous_bar["vw"])]
                output = net.activate(inputs)

                quantity = ((output[1] * 0.5) + 0.5) * self.settings["max_quantity"]
                price = bar["c"] * quantity
                if price >= 1:  # Alpaca doesn't allow trades under $1
                    if output[0] > 0.5 and price <= current_cash:  # Wants to buy
                        bought_stocks[ticker] += quantity
                        current_cash -= price
                    elif output[0] < -0.5 and bought_stocks[ticker] > 0:  # Wants to sell
                        if bought_stocks[ticker] < quantity:
                            current_cash += bar["c"] * bought_stocks[ticker]
                            bought_stocks[ticker] = 0.0
                        else:
                            bought_stocks[ticker] -= quantity
                            current_cash += price
                last_equity = current_equity
            time.sleep(0.15)

        for ticker in bought_stocks:
            current_cash += final_prices[ticker] * bought_stocks[ticker]  # Add remaining liquid assets to cash
        return [genome_id, current_cash - self.start_cash]  # Fitness equals profit

    def eval_genomes(self, genomes, config):
        async_results = []
        pool = mp.Pool(processes=self.settings["processes"])
        for genome_id, genome in genomes:
            async_results.append(pool.apply_async(self.eval_genome, (genome_id, genome, config)))

        result_dict = {}
        for result in async_results:
            pair = result.get(timeout=None)
            result_dict[pair[0]] = pair[1]

        for genome_id, genome in genomes:
            genome.fitness = result_dict[genome_id]
            if self.best_genome is None or self.best_genome.fitness < genome.fitness:
                self.best_genome = genome

        with open(os.path.join(self.settings["save_path"], self.settings["genome_filename"]), "wb") as file:
            pickle.dump(self.best_genome, file)

    def run(self):
        print("Running training agent...")
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])
        checkpointer = neat.Checkpointer(generation_interval=100, filename_prefix=str(os.path.join(self.settings["checkpoint_path"], self.settings["checkpoint_prefix"])))
        p = neat.Population(config) if self.settings["checkpoint_filename"] is None else checkpointer.restore_checkpoint(os.path.join(self.settings["checkpoint_path"], self.settings["checkpoint_filename"]))
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.StatisticsReporter())
        p.add_reporter(checkpointer)
        self.best_genome = p.run(self.eval_genomes, self.settings["generations"])

        print("Best fitness -> {}".format(self.best_genome))


class Trader(Agent):

    def __init__(self, settings, finn_client, alpaca_client):
        super().__init__(settings, finn_client, alpaca_client)
        self.training_agent = None
        print("\nTRADER agent created with settings: " + str(settings) + "\n")

    def trading_loop(self, genome, config):
        net = neat.nn.RecurrentNetwork.create(genome, config)
        cum_prices = {}
        cum_vols = {}
        bought_stocks = {}
        for ticker in self.settings["tickers"]:
            cum_prices[ticker] = 0
            cum_vols[ticker] = 0
            bought_stocks[ticker] = 0

        prev_data = {}
        prev_vwap = {}

        previous_date = dt.datetime.now(pytz.timezone("US/Central"))
        displayed_acc = False
        while True:
            if self.finn_client.market_status(exchange='US')["isOpen"]:
                if self.training_agent is not None:
                    net = neat.nn.RecurrentNetwork.create(self.training_agent.best_genome, config)
                    checkpoint_files = glob.glob(self.settings["checkpoint_path"] + "\\*")
                    latest_file = max(checkpoint_files, key=os.path.getctime)
                    with open(latest_file, "rb") as file:
                        self.settings["checkpoint_filename"] = file.name
                    self.training_agent.stop()
                    self.training_agent = None
                displayed_acc = False
                alpaca_account = self.alpaca_client.get_account()
                current_cash = float(alpaca_account.cash)
                now_date = dt.datetime.now(pytz.timezone("US/Central"))
                if now_date.strftime('%Y-%m-%d') != previous_date.strftime('%Y-%m-%d'):
                    cum_prices.clear()
                    cum_vols.clear()
                    prev_data.clear()
                    prev_vwap.clear()
                    for ticker in self.settings["tickers"]:
                        cum_prices[ticker] = 0
                        cum_vols[ticker] = 0
                
                for ticker in self.settings["tickers"]:
                    ticker_df = yf.download(tickers=ticker, period="1d", interval=str(self.settings["data_interval"]) + "m")
                    current_data = ticker_df.iloc[-1]

                    if ticker not in prev_data:
                        prev_data[ticker] = ticker_df.iloc[-2]
                        prev_vwap[ticker] = (prev_data[ticker]["High"] + prev_data[ticker]["Low"] + prev_data[ticker]["Close"]) / 3

                        #prev_data[ticker] = current_data
                        #prev_vwap[ticker] = 0

                    cum_prices[ticker] += current_data["Volume"] * ((current_data["High"] + current_data["Low"] + current_data["Close"]) / 3)
                    cum_vols[ticker] += current_data["Volume"]
                    vwap = cum_prices[ticker] / cum_vols[ticker] if cum_vols[ticker] > 0 else 0

                    #bought_amount = float(bought_stocks[ticker]) if ticker in bought_stocks else 0.0
                    #inputs = [current_cash, bought_amount, current_data["Open"], current_data["High"], current_data["Low"], current_data["Close"], current_data["Volume"], vwap]
                    inputs = [self.rel_change(alpaca_account.equity, alpaca_account.last_equity),
                              self.rel_change(current_data["Open"], prev_data[ticker]["Open"]),
                              self.rel_change(current_data["High"], prev_data[ticker]["High"]),
                              self.rel_change(current_data["Low"], prev_data[ticker]["Low"]),
                              self.rel_change(current_data["Close"], prev_data[ticker]["Close"]),
                              self.rel_change(current_data["Volume"], prev_data[ticker]["Volume"]),
                              self.rel_change(vwap, prev_vwap[ticker])]
                    output = net.activate(inputs)

                    quantity = ((output[1] * 0.5) + 0.5) * self.settings["max_quantity"]
                    price = current_data["Close"] * quantity

                    if price >= 1:  # Alpaca doesn't allow trades under $1
                        if output[0] > 0.5 and price <= current_cash:  # Wants to buy
                            print(ticker + " BUY (" + str(quantity) + ", $" + str(price) + ") at " + str(now_date))
                            bought_stocks[ticker] += quantity
                            self.alpaca_client.submit_order(symbol=ticker, qty=quantity, side="buy", type="market", time_in_force="day")
                        elif output[0] < -0.5 and bought_stocks[ticker] > 0:  # Wants to sell
                            print(ticker + " SELL (" + str(quantity) + ", $" + str(price) + ") at " + str(now_date))
                            if bought_stocks[ticker] < quantity:
                                bought_stocks[ticker] = 0
                                self.alpaca_client.submit_order(symbol=ticker, qty=bought_stocks[ticker], side="sell", type="market", time_in_force="day")
                            else:
                                bought_stocks[ticker] -= quantity
                                self.alpaca_client.submit_order(symbol=ticker, qty=quantity, side="sell", type="market", time_in_force="day")
                    prev_data[ticker] = current_data
                    prev_vwap[ticker] = vwap
                previous_date = now_date
            else:
                if not displayed_acc:
                    account = self.alpaca_client.get_account()
                    print("\nMarket is closed. Current Account Details:"
                          "\n Balance Change: " + str(float(account.equity) - float(account.last_equity)) +
                          "\n Cash: " + str(account.cash) +
                          "\n Equity: " + str(account.equity) +
                          "\n Bought stocks: " + str(bought_stocks) + "\n")
                    displayed_acc = True
                if self.training_agent is None:
                    print("Starting training while waiting for market to open.\n-----")
                    self.training_agent = Training(self.settings, self.finn_client, self.alpaca_client)
                self.training_agent.run()
            time.sleep(self.settings["run_interval"])

    def run(self):
        print("Running trader agent...")
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])
        if self.settings["genome_filename"] is None:
            print("No file selected to run from.")
        else:
            with open(os.path.join(self.settings["save_path"], self.settings["genome_filename"]), "rb") as file:
                best_genome = pickle.load(file)
            self.trading_loop(best_genome, config)
