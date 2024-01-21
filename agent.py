import neat
import yfinance as yf
from alpaca_trade_api.rest import TimeFrame
import datetime as dt
import time
import os
import pytz


class Agent:
    def __init__(self, settings, finn_client, alpaca_client):
        self.settings = settings
        self.finn_client = finn_client
        self.alpaca_client = alpaca_client
        #self.candlestick_figs = []
        self.earnings = {}
        self.sentiments = {}
        now_date = dt.datetime.now(pytz.timezone("US/Central"))
        start_date = now_date - dt.timedelta(days=settings["backtest_days"])
        for ticker in self.settings["tickers"]:
            self.earnings[ticker] = finn_client.company_earnings(ticker, limit=10)
            self.sentiments[ticker] = finn_client.stock_insider_sentiment(ticker, start_date, now_date)["data"]

            '''bars_df = self.bars.df.loc[self.bars.df["symbol"] == ticker]
            fig = go.Figure(data=[go.Candlestick(x=bars_df.index, open=bars_df["open"], high=bars_df["high"], low=bars_df["low"], close=bars_df["close"])])
            fig.update_layout(title="Candlestick chart for " + symbol,
                              xaxis_title="Date",
                              yaxis_title="Price ($USD)")
            self.candlestick_figs.append(fig)'''
        print("Saved company earnings: " + str(self.earnings))
        print("Saved insider sentiments: " + str(self.sentiments))

    def stop(self):
        exit(0)


class Training(Agent):

    def __init__(self, settings, finn_client, alpaca_client):
        super().__init__(settings, finn_client, alpaca_client)
        self.generations = settings["generations"]
        self.checkpoint_prefix = settings["checkpoint_prefix"]
        self.best_genome = None
        self.start_cash = 1000.0

        now_date = dt.datetime.now(pytz.timezone("US/Central"))
        start_date = now_date - dt.timedelta(days=settings["backtest_days"])
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc
        self.bars = self.alpaca_client.get_bars(
            self.settings["tickers"],
            timeframe=TimeFrame.Minute,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            limit=10000
        )

        print("Saved " + str(len(self.bars)) + " bars from " + start_date.strftime("%Y-%m-%d-%H:%M:%S") + " to " + end_date.strftime("%Y-%m-%d-%H:%M:%S"))
        print("TRAINING agent created with settings: " + str(settings))

    def eval_genomes(self, genomes, config):
        best_fitness = 0
        annotation_dict = {}
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

                bar_date = dt.datetime.strptime(str(bar.t).split(" ")[0], "%Y-%m-%d")
                previous_date = dt.datetime.strptime(self.earnings[bar.S][-1]["period"], "%Y-%m-%d")
                recent_report = None
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
                    output = net.activate([current_cash, bought_amount, bar.o, bar.h, bar.l, bar.c, bar.v, bar.vw, 0])
                else:
                    output = net.activate([current_cash, bought_amount, bar.o, bar.h, bar.l, bar.c, bar.v, bar.vw, recent_report["surprisePercent"]])
                quantity = ((output[1] * 0.5) + 0.5) * self.settings["max_quantity"]
                price = bar.c * quantity
                if output[0] > 0.5 and price <= current_cash:  # Wants to buy
                    bought_stocks[bar.S] = bought_stocks[bar.S] + quantity if bar.S in bought_stocks.keys() else quantity
                    current_cash -= price
                    arrow = dict(x=bar.t, y=bar.c, xref="x", yref="y", text=str(genome_id) + " Buy $" + str(price)[:4],
                                 arrowhead=2,
                                 arrowwidth=1.5,
                                 arrowcolor='rgb(255,0,0)')
                    if genome_id in annotation_dict.keys():
                        annotation_dict[bar.S + str(genome_id)].append(arrow)
                    else:
                        annotation_dict[bar.S + str(genome_id)] = [arrow]
                elif output[0] < -0.5 and bar.S in bought_stocks.keys() and bought_stocks[bar.S] > 0:  # Wants to sell
                    if bought_stocks[bar.S] < quantity:
                        current_cash += bar.c * bought_stocks[bar.S]
                        bought_stocks[bar.S] = 0
                    else:
                        bought_stocks[bar.S] -= quantity
                        current_cash += price
                    arrow = dict(x=bar.t, y=bar.c, xref="x", yref="y", text=str(genome_id) + " Sell $" + str(price)[:4],
                                 arrowhead=2,
                                 arrowwidth=1.5,
                                 arrowcolor='rgb(0,255,0)')
                    if genome_id in annotation_dict.keys():
                        annotation_dict[bar.S + str(genome_id)].append(arrow)
                    else:
                        annotation_dict[bar.S + str(genome_id)] = [arrow]

            for stock in bought_stocks:
                current_cash += final_prices[stock] * bought_stocks[stock]  # Add remaining liquid assets to cash
            genome.fitness = current_cash - self.start_cash  # Fitness equals profit
            if best_fitness < genome.fitness:
                best_fitness = genome.fitness
                self.best_genome = genome

        '''for i in range(len(self.symbols)):
            key = self.symbols[i] + str(best_genome_id)
            if key in annotation_dict.keys():
                self.candlestick_figs[i].update_layout(annotations=annotation_dict[key])
            self.candlestick_figs[i].show()'''

    def run(self, local_dir):
        print("Running training agent...")
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    os.path.join(local_dir, self.settings["config_filename"]))
        checkpointer = neat.Checkpointer(generation_interval=100, filename_prefix=self.checkpoint_prefix)
        p = neat.Population(config) if self.settings["trained_filename"] is None else checkpointer.restore_checkpoint(os.path.join(local_dir, self.settings["trained_filename"]))
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.StatisticsReporter())
        p.add_reporter(checkpointer)
        self.best_genome = p.run(self.eval_genomes, self.generations)

        print("Best fitness -> {}".format(self.best_genome))


class Trader(Agent):

    def __init__(self, settings, finn_client, alpaca_client):
        super().__init__(settings, finn_client, alpaca_client)
        self.training_agent = None
        print("TRADER agent created with settings: " + str(settings))

    def trading_loop(self, genome, config):
        net = neat.nn.RecurrentNetwork.create(genome, config)
        cum_prices = {}
        cum_vols = {}
        bought_stocks = {}
        previous_date = dt.datetime.today()
        displayed_summary = False
        while True:
            if self.finn_client.market_status(exchange='US')["isOpen"]:
                if self.training_agent is not None:
                    self.training_agent.stop()
                displayed_summary = False
                current_cash = self.alpaca_client.get_account()["cash"]
                now_date = dt.datetime.today()
                if now_date.strftime('%Y-%m-%d') != previous_date.strftime('%Y-%m-%d'):
                    cum_prices = {}
                    cum_vols = {}

                for ticker in self.settings["tickers"]:
                    stock_df = yf.download(tickers=ticker, start=now_date-dt.timedelta(minutes=1), end=now_date)
                    cum_prices[ticker] += stock_df["Volume"] * ((stock_df["High"] + stock_df["Low"] + stock_df["Close"]) / 3)
                    cum_vols[ticker] += stock_df["Volume"]

                    bought_amount = bought_stocks[ticker] if ticker in bought_stocks else 0

                    previous_date = dt.datetime.strptime(self.earnings[ticker][-1]["period"], "%Y-%m-%d")
                    recent_report = None
                    for i in range(len(self.earnings[ticker])):
                        report_date = dt.datetime.strptime(self.earnings[ticker][i]["period"], "%Y-%m-%d")
                        if report_date >= now_date:
                            if abs((report_date - now_date).days) < abs((previous_date - now_date).days):
                                recent_report = self.earnings[ticker][i]
                                break
                            else:
                                recent_report = self.earnings[ticker][i-1]
                                break
                        previous_date = report_date

                    if recent_report is None:
                        output = net.activate([current_cash, bought_amount, stock_df["Open"], stock_df["High"], stock_df["Low"], stock_df["Close"], stock_df["Volume"], cum_prices[ticker] / cum_vols[ticker], 0])
                    else:
                        output = net.activate([current_cash, bought_amount, stock_df["Open"], stock_df["High"], stock_df["Low"], stock_df["Close"], stock_df["Volume"], cum_prices[ticker] / cum_vols[ticker], recent_report["surprisePercent"]])
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
                if not displayed_summary:
                    print("Market closed. Account cash: " + str(self.alpaca_client.get_account()["cash"]) + " Bought stocks: " + str(bought_stocks))
                    displayed_summary = True
                local_dir = os.path.dirname(__file__)
                if self.training_agent is None:
                    self.training_agent = Training(self.settings, self.finn_client, self.alpaca_client)
                self.training_agent.run(local_dir)
            time.sleep(60)

    def run(self, local_dir):
        print("Running trader agent...")
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    os.path.join(local_dir, self.settings["config_filename"]))
        checkpointer = neat.Checkpointer()
        p = neat.Population(config) if self.settings["trained_filename"] is None else checkpointer.restore_checkpoint(os.path.join(local_dir, self.settings["trained_filename"]))
        self.trading_loop(p.best_genome, config)
