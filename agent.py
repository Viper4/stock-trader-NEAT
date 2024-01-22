import neat
import pickle
import yfinance as yf
from alpaca_trade_api.rest import TimeFrame
import datetime as dt
import time
import os
import pytz
#from concurrent.futures import ProcessPoolExecutor


class Agent:
    def __init__(self, settings, finn_client, alpaca_client):
        self.settings = settings
        self.finn_client = finn_client
        self.alpaca_client = alpaca_client
        self.earnings = {}
        self.sentiments = {}
        now_date = dt.datetime.now(pytz.timezone("US/Central"))
        start_date = now_date - dt.timedelta(days=settings["backtest_days"])
        for ticker in self.settings["tickers"]:
            self.earnings[ticker] = finn_client.company_earnings(ticker, limit=5)
            self.sentiments[ticker] = finn_client.stock_insider_sentiment(ticker, start_date, now_date)["data"]

        print("Saved company earnings: " + str(self.earnings))
        print("\nSaved insider sentiments: " + str(self.sentiments))

    def stop(self):
        exit(0)


class Training(Agent):

    def __init__(self, settings, finn_client, alpaca_client):
        super().__init__(settings, finn_client, alpaca_client)
        self.generations = settings["generations"]
        self.checkpoint_prefix = settings["checkpoint_prefix"]
        self.start_cash = 1000.0
        self.best_genome = None
        if self.settings["genome_filename"] is not None:
            try:
                with open(os.path.join(self.settings["save_path"], self.settings["genome_filename"]), "rb") as file:
                    self.best_genome = pickle.load(file)
            except FileNotFoundError:
                print("No genome save file found.")

        now_date = dt.datetime.now(pytz.timezone("US/Central"))
        start_date = now_date - dt.timedelta(days=settings["backtest_days"])
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc

        '''self.bars = {}

        for ticker in self.settings["tickers"]:
            self.bars[ticker] = self.alpaca_client.get_bars(
                self.settings["tickers"],
                timeframe=TimeFrame.Minute,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                limit=10000
            )'''

        self.bars = self.alpaca_client.get_bars(
            self.settings["tickers"],
            timeframe=TimeFrame.Minute,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            limit=20000
        )

        print("\nSaved " + str(len(self.bars)) + " bars from " + start_date.strftime("%Y-%m-%d-%H:%M:%S") + " to " + end_date.strftime("%Y-%m-%d-%H:%M:%S"))
        print("\nTRAINING agent created with settings: " + str(settings) + "\n")

    def pass_bars(self, genomes, config, ticker_bars):
        for genome_id, genome in genomes:
            current_cash = self.start_cash
            net = neat.nn.RecurrentNetwork.create(genome, config)
            bought_stocks = {}
            final_prices = {ticker_bars[len(ticker_bars) - 1].S: ticker_bars[len(ticker_bars) - 1].c}
            for i in range(len(ticker_bars)):
                bar = ticker_bars[i]
                previous_bar = ticker_bars[i - 1]

                if bar.S != previous_bar.S:
                    final_prices[previous_bar.S] = previous_bar.c

                bought_amount = bought_stocks[bar.S] if bar.S in bought_stocks else 0

                recent_report = None
                if len(self.earnings[bar.S]) > 0:
                    bar_date = dt.datetime.strptime(str(bar.t).split(" ")[0], "%Y-%m-%d")
                    previous_date = dt.datetime.strptime(self.earnings[bar.S][-1]["period"], "%Y-%m-%d")
                    for i in range(len(self.earnings[bar.S])):
                        report_date = dt.datetime.strptime(self.earnings[bar.S][i]["period"], "%Y-%m-%d")
                        if report_date <= bar_date:
                            if abs((report_date - bar_date).days) < abs((previous_date - bar_date).days):
                                recent_report = self.earnings[bar.S][i]
                                break
                            else:
                                recent_report = self.earnings[bar.S][i-1]
                                break
                        previous_date = report_date

                if recent_report is None:
                    output = net.activate([current_cash, bought_amount, bar.o, bar.h, bar.l, bar.c, bar.v, bar.vw, 0, 0])
                else:
                    output = net.activate([current_cash, bought_amount, bar.o, bar.h, bar.l, bar.c, bar.v, bar.vw, recent_report["actual"], recent_report["estimate"]])
                quantity = ((output[1] * 0.5) + 0.5) * self.settings["max_quantity"]
                price = bar.c * quantity
                if output[0] > 0.5 and price <= current_cash:  # Wants to buy
                    bought_stocks[bar.S] = bought_stocks[bar.S] + quantity if bar.S in bought_stocks.keys() else quantity
                    current_cash -= price
                elif output[0] < -0.5 and bar.S in bought_stocks.keys() and bought_stocks[bar.S] > 0:  # Wants to sell
                    if bought_stocks[bar.S] < quantity:
                        current_cash += bar.c * bought_stocks[bar.S]
                        bought_stocks[bar.S] = 0
                    else:
                        bought_stocks[bar.S] -= quantity
                        current_cash += price

            for stock in bought_stocks:
                current_cash += final_prices[stock] * bought_stocks[stock]  # Add remaining liquid assets to cash
            genome.fitness = current_cash - self.start_cash  # Fitness equals profit

            if self.best_genome is None or self.best_genome.fitness < genome.fitness:
                self.best_genome = genome

    def eval_genomes(self, genomes, config):
        #annotation_dict = {}
        #best_genome_id = -1
        #best_cash = 0

        for genome_id, genome in genomes:
            current_cash = self.start_cash
            net = neat.nn.RecurrentNetwork.create(genome, config)
            bought_stocks = {}
            final_prices = {self.bars[len(self.bars) - 1].S: self.bars[len(self.bars) - 1].c}
            for i in range(len(self.bars)):
                bar = self.bars[i]
                previous_bar = self.bars[i - 1]

                if bar.S != previous_bar.S:
                    final_prices[previous_bar.S] = previous_bar.c

                bought_amount = bought_stocks[bar.S] if bar.S in bought_stocks else 0

                recent_report = None
                if len(self.earnings[bar.S]) > 0:
                    bar_date = dt.datetime.strptime(str(bar.t).split(" ")[0], "%Y-%m-%d")
                    previous_date = dt.datetime.strptime(self.earnings[bar.S][-1]["period"], "%Y-%m-%d")
                    for i in range(len(self.earnings[bar.S])):
                        report_date = dt.datetime.strptime(self.earnings[bar.S][i]["period"], "%Y-%m-%d")
                        if report_date <= bar_date:
                            if abs((report_date - bar_date).days) < abs((previous_date - bar_date).days):
                                recent_report = self.earnings[bar.S][i]
                                break
                            else:
                                recent_report = self.earnings[bar.S][i-1]
                                break
                        previous_date = report_date

                if recent_report is None:
                    output = net.activate([current_cash, bought_amount, bar.o, bar.h, bar.l, bar.c, bar.v, bar.vw, 0, 0])
                else:
                    output = net.activate([current_cash, bought_amount, bar.o, bar.h, bar.l, bar.c, bar.v, bar.vw, recent_report["actual"], recent_report["estimate"]])
                quantity = ((output[1] * 0.5) + 0.5) * self.settings["max_quantity"]
                price = bar.c * quantity
                if output[0] > 0.5 and price <= current_cash:  # Wants to buy
                    bought_stocks[bar.S] = bought_stocks[bar.S] + quantity if bar.S in bought_stocks.keys() else quantity
                    current_cash -= price
                elif output[0] < -0.5 and bar.S in bought_stocks.keys() and bought_stocks[bar.S] > 0:  # Wants to sell
                    if bought_stocks[bar.S] < quantity:
                        current_cash += bar.c * bought_stocks[bar.S]
                        bought_stocks[bar.S] = 0
                    else:
                        bought_stocks[bar.S] -= quantity
                        current_cash += price

            for stock in bought_stocks:
                current_cash += final_prices[stock] * bought_stocks[stock]  # Add remaining liquid assets to cash
            genome.fitness = current_cash - self.start_cash  # Fitness equals profit

            if self.best_genome is None or self.best_genome.fitness < genome.fitness:
                self.best_genome = genome

        """for ticker in self.settings["tickers"]:
            ticker_bars = self.bars[ticker]
            with ProcessPoolExecutor(4) as exe:
                exe.map(self.pass_bars, [genomes, config, ticker_bars])"""

        with open(os.path.join(self.settings["save_path"], self.settings["genome_filename"]), "wb") as f:
            pickle.dump(self.best_genome, f)

    def run(self):
        print("Running training agent...")
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    self.settings["config_path"])
        checkpointer = neat.Checkpointer(generation_interval=100, filename_prefix=os.path.join(self.settings["save_path"], self.settings["checkpoint_prefix"]))
        p = neat.Population(config) if self.settings["checkpoint_filename"] is None else checkpointer.restore_checkpoint(os.path.join(self.settings["save_path"], self.settings["checkpoint_filename"]))
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.StatisticsReporter())
        p.add_reporter(checkpointer)
        self.best_genome = p.run(self.eval_genomes, self.generations)

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
        previous_date = dt.datetime.today()
        displayed_acc = False
        while True:
            if self.finn_client.market_status(exchange='US')["isOpen"]:
                if self.training_agent is not None:
                    net = neat.nn.RecurrentNetwork.create(self.training_agent.best_genome, config)
                    self.training_agent.stop()
                displayed_acc = False
                current_cash = self.alpaca_client.get_account().cash
                now_date = dt.datetime.today()
                if now_date.strftime('%Y-%m-%d') != previous_date.strftime('%Y-%m-%d'):
                    cum_prices = {}
                    cum_vols = {}

                for ticker in self.settings["tickers"]:
                    stock_df = yf.download(tickers=ticker, start=now_date-dt.timedelta(minutes=1), end=now_date)
                    cum_prices[ticker] += stock_df["Volume"] * ((stock_df["High"] + stock_df["Low"] + stock_df["Close"]) / 3)
                    cum_vols[ticker] += stock_df["Volume"]

                    bought_amount = bought_stocks[ticker] if ticker in bought_stocks else 0

                    recent_report = None
                    if len(self.earnings[ticker]) > 0:
                        recent_report = self.earnings[ticker][0]

                    if recent_report is None:
                        output = net.activate([current_cash, bought_amount, stock_df["Open"], stock_df["High"], stock_df["Low"], stock_df["Close"], stock_df["Volume"], cum_prices[ticker] / cum_vols[ticker], 0, 0])
                    else:
                        output = net.activate([current_cash, bought_amount, stock_df["Open"], stock_df["High"], stock_df["Low"], stock_df["Close"], stock_df["Volume"], cum_prices[ticker] / cum_vols[ticker], recent_report["actual"], recent_report["estimate"]])
                    quantity = ((output[1] * 0.5) + 0.5) * self.settings["max_quantity"]
                    price = stock_df["Close"] * quantity
                    if output[0] > 0.5 and price <= current_cash:  # Wants to buy
                        bought_stocks[ticker] = bought_stocks[ticker] + quantity if ticker in bought_stocks.keys() else quantity
                        self.alpaca_client.submit_order(symbol=ticker, qty=quantity, side="buy", type="market", time_in_force="gtc")
                    elif output[0] < -0.5 and ticker in bought_stocks.keys() and bought_stocks[ticker] > 0:  # Wants to sell
                        bought_stocks[ticker] = 0 if bought_stocks[ticker] < quantity else bought_stocks[ticker] - quantity
                        self.alpaca_client.submit_order(symbol=ticker, qty=quantity, side="sell", type="market", time_in_force="gtc")
                previous_date = now_date
            else:
                if not displayed_acc:
                    account = self.alpaca_client.get_account()
                    print("\nMarket is closed. Current Account Details:\n Balance Change: " + str(float(account.equity) - float(account.last_equity)) + "\n Cash: " + str(account.cash) + "\n Equity: " + str(account.equity) + "\n Bought stocks: " + str(bought_stocks) + "\n")
                    displayed_acc = True
                if self.training_agent is None:
                    print("Starting training while waiting for market to open.\n-----")
                    self.training_agent = Training(self.settings, self.finn_client, self.alpaca_client)
                self.training_agent.run()
            time.sleep(60)

    def run(self):
        print("Running trader agent...")
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    self.settings["config_path"])
        if self.settings["genome_filename"] is None:
            print("No file selected to run from")
        else:
            with open(os.path.join(self.settings["save_path"], self.settings["genome_filename"]), "rb") as file:
                best_genome = pickle.load(file)
            self.trading_loop(best_genome, config)
