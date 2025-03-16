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
        self.alphas = {}

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

    @staticmethod
    def time_to_minutes(t):
        return t.hour * 60 + t.minute

    @staticmethod
    def calculate_k_percent(bars):
        """
        Stochastic oscillator indicator
        %K = [(Current Close - Lowest Low) / (Highest High - Lowest Low)] * 100
        %D = EMA(%K, 3) or SMA(%K, 3)
        """
        min_low = bars[0]["low"]
        max_high = bars[0]["high"]
        for i in range(1, len(bars)):
            if bars[i]["low"] < min_low:
                min_low = bars[i]["low"]
            if bars[i]["high"] > max_high:
                max_high = bars[i]["high"]
        denom = max_high - min_low
        if denom == 0:
            return 0
        return 2 * (((bars[-1]["close"] - min_low) / denom) - 0.5)  # Scale to [-1, 1]

    @staticmethod
    def calculate_ema(close_price, alpha, prev_ema):
        """
        Exponential moving average
        EMA = current price * (2 / (period + 1)) + EMA(prev) * (1 - (2 / (period + 1)))
        period is the number of days to calculate the EMA
        alpha = 2 / (period + 1)
        """
        if prev_ema is None:
            prev_ema = close_price
        return close_price * alpha + prev_ema * (1 - alpha)

    @staticmethod
    def calculate_sma(bars):
        total = 0
        if len(bars) == 0:
            return 0
        for i in range(len(bars)):
            total += bars[i]["close"]
        return total / len(bars)

    @staticmethod
    def calculate_rsi(bars):
        """
        Relative Strength Index indicator
        RS = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))
        """
        if len(bars) <= 1:
            return 0
        gain = 0
        loss = 0
        for i in range(1, len(bars)):
            change = bars[i]["close"] - bars[i - 1]["close"]
            if change > 0:
                gain += change
            elif change < 0:
                loss += abs(change)

        avg_gain = gain / (len(bars) - 1)
        avg_loss = loss / (len(bars) - 1)
        if avg_loss == 0:
            rsi = 100
        else:
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        return 2 * ((rsi - 50) / 50)  # Scale between -1 and 1


# Separate from classes so instances don't get cached to RAM and slow things down
def eval_genome(stock_bars, sp500_bars, nasdaq_bars,
                stock_sentiments, sp500_sentiments, nasdaq_sentiments,
                start_cash, genome, config, cash_at_risk, log_training, profit_window,
                fitness_multipliers, shorting, shortable, fractionable, short_limit,
                transaction_fee, k_period, d_period, rsi_period):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    start_date = stock_bars[0]["timestamp"].date()
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
    prev_ema = None
    alpha = 2 / (d_period + 1)

    # Start at 1 to have previous bar for relative change
    num_bars = len(stock_bars)
    for i in range(1, num_bars):
        stock_bar = stock_bars[i]
        sp500_bar = sp500_bars[i]
        nasdaq_bar = nasdaq_bars[i]
        prev_stock_bar = stock_bars[i-1]
        prev_sp500_bar = sp500_bars[i-1]
        prev_nasdaq_bar = nasdaq_bars[i-1]
        prev_date = prev_stock_bar["timestamp"].date()
        date = stock_bar["timestamp"].date()
        if date != prev_date:  # Check pending sales to settle cash after 1 day of sale
            consecutive_days += 1
            for j in reversed(range(len(pending_sales))):
                sale_price, sale_day = pending_sales[j]
                if consecutive_days - sale_day > 1:
                    settled_cash += sale_price
                    unsettled_cash -= sale_price
                    pending_sales.pop(j)

        k_percent = Agent.calculate_k_percent(stock_bars[i - min(k_period, i):i])

        # %D = EMA(%K, N)
        ema = Agent.calculate_ema(stock_bar["close"], alpha, prev_ema)
        norm_ema = 2 * ((ema - stock_bar["close"]) / stock_bar["close"])
        prev_ema = ema
        k_sma = Agent.calculate_sma(stock_bars[i - min(k_period, i):i])
        norm_k_sma = 2 * ((k_sma - stock_bar["close"]) / stock_bar["close"])
        d_sma = Agent.calculate_sma(stock_bars[i - min(d_period, i):i])
        norm_d_sma = 2 * ((d_sma - stock_bar["close"]) / stock_bar["close"])

        rsi = Agent.calculate_rsi(stock_bars[i - min(rsi_period, i):i])

        inputs = [1,  # -1 = short, 1 = long
                  Agent.rel_change(cost, stock_bar["close"] * shares),  # plpc
                  Agent.rel_change(prev_stock_bar["open"], stock_bar["open"]),
                  Agent.rel_change(prev_stock_bar["high"], stock_bar["high"]),
                  Agent.rel_change(prev_stock_bar["low"], stock_bar["low"]),
                  Agent.rel_change(prev_stock_bar["close"], stock_bar["close"]),
                  Agent.rel_change(prev_stock_bar["volume"], stock_bar["volume"]),
                  Agent.rel_change(prev_stock_bar["vwap"], stock_bar["vwap"]),
                  stock_sentiments[i],  # -1 = negative, 0 = neutral, 1 = positive
                  Agent.rel_change(prev_sp500_bar["close"], sp500_bar["close"]),
                  Agent.rel_change(prev_sp500_bar["volume"], sp500_bar["volume"]),
                  sp500_sentiments[i],
                  Agent.rel_change(prev_nasdaq_bar["close"], nasdaq_bar["close"]),
                  Agent.rel_change(prev_nasdaq_bar["volume"], nasdaq_bar["volume"]),
                  nasdaq_sentiments[i],
                  k_percent,
                  norm_ema,
                  norm_k_sma,
                  norm_d_sma,
                  rsi]
        if shorting and shares < 0:
            inputs[0] = -1
            inputs[1] = Agent.rel_change(stock_bar["close"] * abs(shares), cost)

        outputs = net.activate(inputs)

        qty_percent = (outputs[1] + 1) * 0.5
        if outputs[0] > 0.5:  # Buy
            if shorting and shortable and shares < 0:
                quantity = qty_percent * abs(shares)
                quantity = round(quantity)  # Shorts don't allow fractional qty
                price = quantity * stock_bar["close"] * (1 - transaction_fee)
                if price >= 1:
                    if abs(shares) - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                        price = abs(shares) * stock_bar["close"] * (1 - transaction_fee)
                        profit = cost - price
                        shares = 0.0
                        cost = 0.0
                    else:
                        avg_cost = cost / abs(shares)
                        shares += quantity
                        cost = avg_cost * abs(shares)
                        profit = (avg_cost * quantity) - price

                    # TODO: REMOVE THIS LATER. Using this to incentivize shorting
                    if profit > 0:
                        profit *= 1.5
                    settled_cash += profit

                    if log_training:
                        action = {"side": "Buy", "type": "short", "quantity": quantity, "price": stock_bar["close"],
                                  "profit": profit, "settled_cash": settled_cash,
                                  "unsettled_cash": unsettled_cash,
                                  "datetime": stock_bar["timestamp"].to_pydatetime()}
                        log.append(action)
            else:
                quantity = qty_percent * settled_cash * cash_at_risk / stock_bar["close"]
                if not fractionable:
                    quantity = round(quantity)
                price = quantity * stock_bar["close"]
                if price >= 1:  # Alpaca doesn't allow trades under $1
                    cost += price
                    shares += quantity
                    settled_cash -= price

                    if log_training:
                        action = {"side": "Buy", "type": "long", "quantity": quantity, "price": stock_bar["close"],
                                  "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                  "datetime": stock_bar["timestamp"].to_pydatetime()}
                        log.append(action)
        elif outputs[0] < -0.5:  # Sell
            if shorting and shortable and shares <= 0:
                quantity = qty_percent * (short_limit - cost) * cash_at_risk / stock_bar["close"]
                quantity = round(quantity)  # Shorts don't allow fractional qty
                price = quantity * stock_bar["close"] * (1 - transaction_fee)
                if cost + price < short_limit:
                    if price >= 1:  # Alpaca doesn't allow trades under $1
                        cost += price
                        shares -= quantity

                        if log_training:
                            action = {"side": "Sell", "type": "short", "quantity": abs(quantity), "price": stock_bar["close"],
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": stock_bar["timestamp"].to_pydatetime()}
                            log.append(action)
            else:
                quantity = qty_percent * shares
                if not fractionable:
                    quantity = round(quantity)
                price = quantity * stock_bar["close"] * (1 - transaction_fee)
                if price >= 1:
                    if shares - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                        price = shares * stock_bar["close"] * (1 - transaction_fee)
                        profit = price - cost
                        shares = 0.0
                        cost = 0.0
                    else:
                        avg_cost = cost / shares
                        shares -= quantity
                        cost = avg_cost * shares
                        profit = price - (avg_cost * quantity)
                    unsettled_cash += price
                    pending_sales.append((price, consecutive_days))

                    if log_training:
                        action = {"side": "Sell", "type": "long", "quantity": quantity, "price": stock_bar["close"],
                                  "profit": profit, "settled_cash": settled_cash,
                                  "unsettled_cash": unsettled_cash,
                                  "datetime": stock_bar["timestamp"].to_pydatetime()}
                        log.append(action)
        if i == num_bars-1 or (date - start_date).days >= profit_window:
            if shares < 0:
                equity = unsettled_cash + settled_cash + shares * stock_bar["close"] - cost
            else:
                equity = unsettled_cash + settled_cash + stock_bar["close"] * shares
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
    def __init__(self, settings, session, stock,
                 stock_bars, sp500_bars, nasdaq_bars,
                 stock_sentiments, sp500_sentiments, nasdaq_sentiments):
        super().__init__(settings, session, stock)
        self.started = False
        self.best_genome = None  # Saving population object adds 10s to each gen
        self.consecutive_gens = 0
        self.start_cash = 500000.0
        self.stock_bars = stock_bars
        self.sp500_bars = sp500_bars
        self.nasdaq_bars = nasdaq_bars
        self.stock_sentiments = stock_sentiments
        self.sp500_sentiments = sp500_sentiments
        self.nasdaq_sentiments = nasdaq_sentiments
        self.batch_index = 0
        self.genome_file_path = os.path.join(self.genome_path, self.stock["genome_filename"])
        self.population_file_path = os.path.join(self.population_path, self.stock["population_filename"])
        self.shortable = True
        self.fractionable = True
        self.cum_fitness = {}

    def eval_genomes(self, genomes, config):
        while not self.running:
            time.sleep(1)

        # self.pool doesn't work: cant pickle Pool(). Separate class doesn't work: leaks memory
        pool = Pool(processes=self.settings["processes"])
        jobs = []

        if isinstance(self.stock_bars[self.batch_index], int):
            sub_index = self.stock_bars[self.batch_index]
            print(f"Evaluating genomes on substitute batch {sub_index}")
            stock_bars = self.stock_bars[sub_index]
            sp500_bars = self.sp500_bars[sub_index]
            nasdaq_bars = self.nasdaq_bars[sub_index]
            stock_sentiments = self.stock_sentiments[sub_index]
            sp500_sentiments = self.sp500_sentiments[sub_index]
            nasdaq_sentiments = self.nasdaq_sentiments[sub_index]
        else:
            print(f"Evaluating genomes on batch {self.batch_index}")
            stock_bars = self.stock_bars[self.batch_index]
            sp500_bars = self.sp500_bars[self.batch_index]
            nasdaq_bars = self.nasdaq_bars[self.batch_index]
            stock_sentiments = self.stock_sentiments[self.batch_index]
            sp500_sentiments = self.sp500_sentiments[self.batch_index]
            nasdaq_sentiments = self.nasdaq_sentiments[self.batch_index]

        for genome_id, genome in genomes:
            jobs.append(pool.apply_async(eval_genome, (stock_bars, sp500_bars, nasdaq_bars,
                                                       stock_sentiments, sp500_sentiments, nasdaq_sentiments,
                                                       self.start_cash, genome, self.config, self.stock["cash_at_risk"],
                                                       self.settings["log_training"], self.session["profit_window"],
                                                       self.session["fitness_multipliers"], self.stock["shorting"],
                                                       self.shortable, self.fractionable, self.session["short_limit"],
                                                       self.stock["transaction_fee"],
                                                       int((self.session["k_period"] * 6.5 * 60) / self.session["interval"]),
                                                       int((self.session["d_period"] * 6.5 * 60) / self.session["interval"]),
                                                       int((self.session["rsi_period"] * 6.5 * 60) / self.session["interval"]))))

        best_log = None
        best_genome_id = -1
        for i in range(len(jobs)):
            # Genome's fitness based on total fitness over all batches
            job, (genome_id, genome) = jobs[i], genomes[i]
            batch_fitness, log = job.get()
            if genome_id not in self.cum_fitness:
                self.cum_fitness[genome_id] = []
            if len(self.cum_fitness[genome_id]) >= self.session["batches"]:
                self.cum_fitness[genome_id].pop(0)
            self.cum_fitness[genome_id].append(batch_fitness)
            genome.fitness = sum(self.cum_fitness[genome_id])
            if self.best_genome is None or genome.fitness > self.best_genome.fitness:
                best_log = log
                self.best_genome = genome

            if self.best_genome == genome:
                best_genome_id = genome_id

        if self.best_genome is not None and best_genome_id in self.cum_fitness:
            print(f"Fitness across batches: {self.cum_fitness[best_genome_id]} - id {best_genome_id}")
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
        if self.batch_index >= len(self.stock_bars):
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
        node_names = {-9: 'plpc', -8: 'O%', -7: 'H%', -6: 'L%', -5: 'C%', -4: 'V%', -3: 'vwap%', -2: "sentiment", -1: 'buy/sell', 0: 'amount'}
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
        cum_stock_price = 0
        cum_stock_vol = 0
        prev_stock_candle = None

        prev_sp500_candle = None

        prev_nasdaq_candle = None

        max_period = max(self.trader.profile["k_period"], self.trader.profile["d_period"], self.trader.profile["rsi_period"])
        k_period = int((self.trader.profile["k_period"] * 6.5 * 60) / self.trader.profile["interval"])
        d_period = int((self.trader.profile["d_period"] * 6.5 * 60) / self.trader.profile["interval"])
        rsi_period = int((self.trader.profile["rsi_period"] * 6.5 * 60) / self.trader.profile["interval"])
        alpha = 2 / (d_period + 1)
        prev_ema = None

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.trader.get_market_status():
                # Stock candles for today
                stock_candles, prev_stock_close = self.trader.scraper.get_latest_candles(self.stock["symbol"], interval=str(self.trader.profile["interval"]) + "m")
                stock_latest = stock_candles[-1]
                cum_stock_price += stock_latest["volume"] * ((stock_latest["high"] + stock_latest["low"] + stock_latest["close"]) / 3)
                cum_stock_vol += stock_latest["volume"]
                stock_latest["vwap"] = cum_stock_price / cum_stock_vol if cum_stock_vol > 0 else 0

                if prev_stock_candle is None:
                    if len(stock_candles) >= 2:
                        prev_stock_candle = stock_candles[-2]
                    else:
                        prev_stock_candle = stock_latest
                        prev_stock_candle["close"] = prev_stock_close
                    prev_stock_candle["vwap"] = (prev_stock_candle["high"] + prev_stock_candle["low"] +
                                                 prev_stock_candle["close"]) / 3

                # SP500 candles for today
                sp500_candles, prev_sp500_close = self.trader.scraper.get_latest_candles("SPY", interval=str(self.trader.profile["interval"]) + "m")
                sp500_latest = sp500_candles[-1]

                if prev_sp500_candle is None:
                    if len(sp500_candles) >= 2:
                        prev_sp500_candle = sp500_candles[-2]
                    else:
                        prev_sp500_candle = stock_latest
                        prev_sp500_candle["close"] = prev_sp500_close

                # NASDAQ candles for today
                nasdaq_candles, prev_nasdaq_close = self.trader.scraper.get_latest_candles("QQQ", interval=str(self.trader.profile["interval"]) + "m")
                nasdaq_latest = nasdaq_candles[-1]

                if prev_nasdaq_candle is None:
                    if len(nasdaq_candles) >= 2:
                        prev_nasdaq_candle = nasdaq_candles[-2]
                    else:
                        prev_nasdaq_candle = stock_latest
                        prev_nasdaq_candle["close"] = prev_nasdaq_close

                # Get current position
                position = self.trader.schwab_api.get_position(self.stock["symbol"])
                stock_sentiment = self.trader.finbert.get_api_sentiment(self.stock["symbol"],
                                                                        now_date - dt.timedelta(days=2), now_date)
                sp500_sentiment = self.trader.finbert.get_api_sentiment("SPY",
                                                                        now_date - dt.timedelta(days=2), now_date)
                nasdaq_sentiment = self.trader.finbert.get_api_sentiment("QQQ",
                                                                        now_date - dt.timedelta(days=2), now_date)

                # Get historical data for momentum indicators

                stock_bars = self.trader.get_bars(self.stock["symbol"], self.trader.alpaca_api, self.trader.profile["interval"], now_date - dt.timedelta(days=max_period + 1), now_date - dt.timedelta(days=1))
                stock_bars.append(stock_candles)  # Add today's data

                last_index = len(stock_bars) - 1
                k_percent = self.calculate_k_percent(stock_bars[last_index - min(k_period, last_index):last_index])

                # %D = EMA(%K, N) or SMA(%K, N)
                ema = self.calculate_ema(stock_latest["close"], alpha, prev_ema)
                norm_ema = 2 * ((ema - stock_latest["close"]) / stock_latest["close"])
                prev_ema = ema
                k_sma = Agent.calculate_sma(stock_bars[last_index - min(k_period, last_index):last_index])
                norm_k_sma = 2 * ((k_sma - stock_latest["close"]) / stock_latest["close"])
                d_sma = Agent.calculate_sma(stock_candles[last_index - min(d_period, last_index):last_index])
                norm_d_sma = 2 * ((d_sma - stock_latest["close"]) / stock_latest["close"])

                rsi = self.calculate_rsi(stock_candles[last_index - min(rsi_period, last_index):last_index])

                inputs = [0,  # -1 = shorting, 1 = longing
                          0,  # profit/loss percent
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
                          norm_ema,
                          norm_k_sma,
                          norm_d_sma,
                          rsi]

                if "shortQuantity" in position and position["shortQuantity"] > 0:
                    inputs[0] = -1
                    position_qty = -position["shortQuantity"]
                    if position["shortOpenProfitLoss"] > 0:
                        inputs[1] = position["shortOpenProfitLoss"] / position["averagePrice"]
                else:
                    inputs[0] = 1
                    position_qty = position["longQuantity"]
                    if position["longOpenProfitLoss"] > 0:
                        inputs[1] = position["longOpenProfitLoss"] / position["averagePrice"]

                outputs = self.net.activate(inputs)

                qty_percent = (outputs[1] + 1) * 0.5

                asset = self.trader.alpaca_api.get_asset(symbol=self.stock["symbol"])
                if outputs[0] > 0.5:  # Buy
                    if self.stock["shorting"] and asset.shortable and position_qty < 0:
                        account = self.trader.schwab_api.get_account()
                        unsettled_cash = account["currentBalances"]["unsettledCash"]
                        settled_cash = account["currentBalances"]["cashAvailableForTrading"] - unsettled_cash
                        quantity = abs(round(qty_percent * position_qty))
                        price = quantity * stock_latest["close"]
                        if abs(price) >= 1:
                            if abs(position_qty - quantity) < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=abs(quantity), side="BUY")
                            else:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=abs(quantity), side="BUY")

                            action = {"side": "Buy", "type": "short", "quantity": abs(quantity), "price": stock_latest["close"],
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
                        used_cash = market_value + unsettled_cash
                        if used_cash < self.trader.profile["cash_limit"]:
                            quantity = min(self.trader.profile["cash_limit"], settled_cash) * qty_percent * self.stock["cash_at_risk"] / stock_latest["close"]
                            quantity = round(quantity)
                            if quantity > 0:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="BUY")

                                action = {"side": "Buy", "type": "long", "quantity": quantity, "price": stock_latest["close"],
                                          "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                          "datetime": now_date}
                                print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                                self.trader.logs[self.stock["symbol"]].append(action)
                elif outputs[0] < -0.5:  # Sell
                    account = self.trader.schwab_api.get_account()
                    unsettled_cash = account["currentBalances"]["unsettledCash"]
                    settled_cash = account["currentBalances"]["cashAvailableForTrading"] - unsettled_cash
                    if self.stock["shorting"] and asset.shortable and position_qty <= 0:
                        cost = position["averagePrice"] * position_qty

                        if abs(cost) < self.trader.profile["short_limit"]:
                            quantity = round(-qty_percent * (self.trader.profile["short_limit"] - abs(cost)) * self.stock["cash_at_risk"] / stock_latest["close"])
                            if quantity > 0:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=abs(quantity), side="SELL")

                                action = {"side": "Sell", "type": "short", "quantity": quantity, "price": stock_latest["close"],
                                          "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                          "datetime": now_date}
                                print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                                self.trader.logs[self.stock["symbol"]].append(action)
                    else:
                        quantity = round(qty_percent * position_qty)
                        price = quantity * stock_latest["close"]
                        if price >= 1:
                            if position_qty - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty and assume sell all with small qty
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="SELL")
                            else:
                                self.trader.schwab_api.submit_order(symbol=self.stock["symbol"], quantity=quantity, side="SELL")

                            # profit = price - cost
                            action = {"side": "Sell", "type": "long", "quantity": quantity, "price": stock_latest["close"],
                                      "profit": price - (position["averageLongPrice"] * quantity),
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": now_date}
                            print(f"{self.trader.profile['name']} {self.stock['symbol']}: {action}")
                            self.trader.logs[self.stock["symbol"]].append(action)
                prev_stock_candle = stock_latest
                prev_sp500_candle = sp500_latest
                prev_nasdaq_candle = nasdaq_latest

                time.sleep(self.trader.profile["interval"] * 60)
            else:
                cum_stock_price = 0.0
                cum_stock_vol = 0.0

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
        cum_stock_price = 0
        cum_stock_vol = 0
        prev_stock_candle = None

        prev_sp500_candle = None

        prev_nasdaq_candle = None

        max_period = max(self.session["k_period"], self.session["d_period"], self.session["rsi_period"])
        k_period = int((self.session["k_period"] * 6.5 * 60) / self.session["interval"])
        d_period = int((self.session["d_period"] * 6.5 * 60) / self.session["interval"])
        rsi_period = int((self.session["rsi_period"] * 6.5 * 60) / self.session["interval"])
        alpha = 2 / (d_period + 1)
        prev_ema = None

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
                sp500_candles, prev_sp500_close = self.scraper.get_latest_candles("SPY", interval=str(
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
                nasdaq_candles, prev_nasdaq_close = self.scraper.get_latest_candles("QQQ", interval=str(
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
                                                  now_date - dt.timedelta(days=1))
                stock_bars.append(stock_candles)  # Add today's data

                last_index = len(stock_bars) - 1
                k_percent = self.calculate_k_percent(stock_bars[last_index - min(k_period, last_index):last_index])

                # %D = EMA(%K, N) or SMA(%K, N)
                ema = self.calculate_ema(stock_latest["close"], alpha, prev_ema)
                norm_ema = 2 * ((ema - stock_latest["close"]) / stock_latest["close"])
                prev_ema = ema
                k_sma = Agent.calculate_sma(stock_candles[last_index - min(k_period, last_index):last_index])
                norm_k_sma = 2 * ((k_sma - stock_latest["close"]) / stock_latest["close"])
                d_sma = Agent.calculate_sma(stock_candles[last_index - min(d_period, last_index):last_index])
                norm_d_sma = 2 * ((d_sma - stock_latest["close"]) / stock_latest["close"])

                rsi = self.calculate_rsi(stock_candles[last_index - min(rsi_period, last_index):last_index])

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
                          norm_ema,
                          norm_k_sma,
                          norm_d_sma,
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
                                self.session["pending_sales"].append((price, self.trader.consecutive_days))

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


class Validation(Agent):
    def __init__(self, settings, session, stock, finbert):
        super().__init__(settings, session, stock)
        self.finbert = finbert

    def validate(self, stock_bars, sp500_bars, nasdaq_bars,
                 genome, shorting, asset, short_limit,
                 k_period, d_period, rsi_period):
        start_time = time.time()
        net = neat.nn.RecurrentNetwork.create(genome, self.config)
        start_date = stock_bars[0]["timestamp"].date()
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

        prev_ema = None

        k_period_days = k_period
        d_period_days = d_period
        rsi_period_days = rsi_period
        k_period = (k_period * 6.5 * 60) / self.session["interval"]
        d_period = (d_period * 6.5 * 60) / self.session["interval"]
        rsi_period = (rsi_period * 6.5 * 60) / self.session["interval"]
        alpha = 2 / (d_period + 1)

        # Start at 1 to have previous bar for relative change
        num_bars = len(stock_bars)
        for i in range(1, num_bars):
            stock_bar = stock_bars[i]
            sp500_bar = sp500_bars[i]
            nasdaq_bar = nasdaq_bars[i]
            prev_stock_bar = stock_bars[i - 1]
            prev_sp500_bar = sp500_bars[i - 1]
            prev_nasdaq_bar = nasdaq_bars[i - 1]
            prev_date = prev_stock_bar["timestamp"].date()
            date = stock_bar["timestamp"].date()
            if date != prev_date:  # Check pending sales to settle cash after 1 day of sale
                consecutive_days += 1
                for j in reversed(range(len(pending_sales))):
                    sale_price, sale_day = pending_sales[j]
                    if consecutive_days - sale_day >= 1:
                        settled_cash += sale_price
                        unsettled_cash -= sale_price
                        pending_sales.pop(j)

            backtest_date = stock_bar["timestamp"].to_pydatetime()
            stock_sentiment = self.finbert.get_saved_sentiment(self.stock["symbol"],
                                                         backtest_date - dt.timedelta(days=2),
                                                         backtest_date)
            sp500_sentiment = self.finbert.get_saved_sentiment("SPY",
                                                               backtest_date - dt.timedelta(days=2),
                                                               backtest_date)
            nasdaq_sentiment = self.finbert.get_saved_sentiment("QQQ",
                                                               backtest_date - dt.timedelta(days=2),
                                                               backtest_date)

            k_percent = Agent.calculate_k_percent(stock_bars[i - min(k_period, i):i])

            # %D = EMA(%K, N) or SMA(%K, N)
            ema = Agent.calculate_ema(stock_bar["close"], alpha, prev_ema)
            norm_ema = 2 * ((ema - stock_bar["close"]) / stock_bar["close"])
            prev_ema = ema
            k_sma = Agent.calculate_sma(stock_bars[i - min(k_period, i):i])
            norm_k_sma = 2 * ((k_sma - stock_bar["close"]) / stock_bar["close"])
            d_sma = Agent.calculate_sma(stock_bars[i - min(d_period, i):i])
            norm_d_sma = 2 * ((d_sma - stock_bar["close"]) / stock_bar["close"])

            rsi = Agent.calculate_rsi(stock_bars[i - min(rsi_period, i):i])

            inputs = [1,  # -1 = short, 1 = long
                      Agent.rel_change(cost, stock_bar["close"] * shares),  # plpc
                      Agent.rel_change(prev_stock_bar["open"], stock_bar["open"]),
                      Agent.rel_change(prev_stock_bar["high"], stock_bar["high"]),
                      Agent.rel_change(prev_stock_bar["low"], stock_bar["low"]),
                      Agent.rel_change(prev_stock_bar["close"], stock_bar["close"]),
                      Agent.rel_change(prev_stock_bar["volume"], stock_bar["volume"]),
                      Agent.rel_change(prev_stock_bar["vwap"], stock_bar["vwap"]),
                      stock_sentiment,  # -1 = negative, 0 = neutral, 1 = positive
                      Agent.rel_change(prev_sp500_bar["close"], sp500_bar["close"]),
                      Agent.rel_change(prev_sp500_bar["volume"], sp500_bar["volume"]),
                      sp500_sentiment,
                      Agent.rel_change(prev_nasdaq_bar["close"], nasdaq_bar["close"]),
                      Agent.rel_change(prev_nasdaq_bar["volume"], nasdaq_bar["volume"]),
                      nasdaq_sentiment,
                      k_percent,
                      norm_ema,
                      norm_k_sma,
                      norm_d_sma,
                      rsi]
            if shorting and shares < 0:
                inputs[0] = -1
                inputs[1] = Agent.rel_change(stock_bar["close"] * abs(shares), cost)

            outputs = net.activate(inputs)

            qty_percent = (outputs[1] + 1) * 0.5
            if outputs[0] > 0.5:  # Buy
                if shorting and asset.shortable and shares < 0:
                    quantity = qty_percent * abs(shares)
                    quantity = round(quantity)  # Shorts don't allow fractional qty
                    price = quantity * stock_bar["close"] * (1 - self.stock["transaction_fee"])
                    if price >= 1:
                        if abs(shares) - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                            price = abs(shares) * stock_bar["close"] * (1 - self.stock["transaction_fee"])
                            profit = cost - price
                            shares = 0.0
                            cost = 0.0
                        else:
                            avg_cost = cost / abs(shares)
                            shares += quantity
                            cost = avg_cost * abs(shares)
                            profit = (avg_cost * quantity) - price
                        settled_cash += profit

                        short_buys += 1
                        action = {"inputs": inputs, "outputs": outputs,
                                  "side": "Buy", "type": "short", "quantity": abs(quantity), "price": stock_bar["close"],
                                  "profit": profit, "settled_cash": settled_cash,
                                  "unsettled_cash": unsettled_cash,
                                  "datetime": stock_bar["timestamp"].to_pydatetime()}
                        log.append(action)
                else:
                    quantity = qty_percent * settled_cash * self.stock["cash_at_risk"] / stock_bar["close"]
                    if not asset.fractionable:
                        quantity = round(quantity)
                    price = quantity * stock_bar["close"]
                    if price >= 1:  # Alpaca doesn't allow trades under $1
                        cost += price
                        shares += quantity
                        settled_cash -= price

                        action = {"inputs": inputs, "outputs": outputs,
                                  "side": "Buy", "type": "long", "quantity": quantity, "price": stock_bar["close"],
                                  "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                  "datetime": stock_bar["timestamp"].to_pydatetime()}
                        log.append(action)
                        long_buys += 1
            elif outputs[0] < -0.5:  # Sell
                if shorting and asset.shortable and shares <= 0:
                    quantity = qty_percent * (short_limit - cost) * self.stock["cash_at_risk"] / stock_bar["close"]
                    quantity = round(quantity)  # Shorts don't allow fractional qty
                    price = quantity * stock_bar["close"] * (1 - self.stock["transaction_fee"])
                    if cost + price < short_limit:
                        if price >= 1:  # Alpaca doesn't allow trades under $1
                            cost += price
                            shares -= quantity

                            action = {"inputs": inputs, "outputs": outputs,
                                      "side": "Sell", "type": "short", "quantity": abs(quantity), "price": stock_bar["close"],
                                      "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                                      "datetime": stock_bar["timestamp"].to_pydatetime()}
                            log.append(action)
                            short_sells += 1
                elif shares > 0:
                    quantity = qty_percent * shares
                    if not asset.fractionable:
                        quantity = round(quantity)
                    price = quantity * stock_bar["close"] * (1 - self.stock["transaction_fee"])
                    if price >= 1:
                        if shares - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                            price = shares * stock_bar["close"] * (1 - self.stock["transaction_fee"])
                            action = {"inputs": inputs, "outputs": outputs,
                                      "side": "Sell", "type": "long", "quantity": quantity, "price": stock_bar["close"],
                                      "profit": price - cost, "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash + price,
                                      "datetime": stock_bar["timestamp"].to_pydatetime()}
                            log.append(action)
                            shares = 0.0
                            cost = 0.0
                        else:
                            avg_cost = cost / shares
                            shares -= quantity
                            cost = avg_cost * shares
                            action = {"inputs": inputs, "outputs": outputs,
                                      "side": "Sell", "type": "long", "quantity": quantity, "price": stock_bar["close"],
                                      "profit": price - (avg_cost * quantity), "settled_cash": settled_cash,
                                      "unsettled_cash": unsettled_cash + price,
                                      "datetime": stock_bar["timestamp"].to_pydatetime()}
                            log.append(action)
                        unsettled_cash += price
                        pending_sales.append((price, consecutive_days))
                        long_sells += 1
            if i == num_bars - 1 or (date - start_date).days >= self.session["profit_window"]:
                if shares < 0:
                    equity = unsettled_cash + settled_cash + shares * stock_bar["close"] - cost
                else:
                    equity = unsettled_cash + settled_cash + stock_bar["close"] * shares
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
        stock_change = stock_bars[-1]['close'] - stock_bars[0]['close']
        print(f"Simulation finished in {str(time.time() - start_time)} seconds over {consecutive_days} trading days and {num_windows} profit windows"
              f"\n Stock change: ${round(stock_change, 2)} {round(100 * (stock_change / stock_bars[0]['close']), 4)}%"
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
                            print("-Inputs")
                            print(f" |Short/Long: {action[key][0]}")
                            print(f" |PLPC: {action[key][1]}")
                            print(f" |Open: {action[key][2]}")
                            print(f" |High: {action[key][3]}")
                            print(f" |Low: {action[key][4]}")
                            print(f" |Close: {action[key][5]}")
                            print(f" |Volume: {action[key][6]}")
                            print(f" |VWAP: {action[key][7]}")
                            print(f" |{stock_bars[0]['symbol']} Sentiment: {action[key][8]}")
                            print(f" |S&P 500 Close: {action[key][9]}")
                            print(f" |S&P 500 Volume: {action[key][10]}")
                            print(f" |S&P 500 Sentiment: {action[key][11]}")
                            print(f" |NASDAQ Close: {action[key][12]}")
                            print(f" |NASDAQ Volume: {action[key][13]}")
                            print(f" |NASDAQ Sentiment: {action[key][14]}")
                            print(f" |%K: {action[key][15]}")
                            print(f" |{k_period_days}-day EMA: {action[key][16]}")
                            print(f" |{k_period_days}-day SMA: {action[key][17]}")
                            print(f" |{d_period_days}-day SMA: {action[key][18]}")
                            print(f" |{rsi_period_days}-day RSI: {action[key][19]}")
                        elif key == "outputs":
                            print("-Outputs")
                            print(f" |Buy/Sell: {action[key][0]}")
                            print(f" |Quantity: {action[key][1]}")
                        else:
                            print(f"-{key}: {action[key]}")
                else:
                    print("Index not in range of log")
