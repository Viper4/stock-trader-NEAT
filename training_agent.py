import neat
import time
import os
from multiprocessing import Pool
import threading
import saving
import visualize
import plot
from base_agent import Agent
from data_structures import Queue


def eval_genome(args):
    (stock_bars, sp500_bars, nasdaq_bars,
     stock_sentiments, sp500_sentiments, nasdaq_sentiments,
     start_cash, genome, config, cash_at_risk, log_training, profit_window, fitness_multipliers,
     short, fractionable, short_limit, transaction_fee,
     indicator_data) = args
    net = neat.nn.RecurrentNetwork.create(genome, config)
    start_date = stock_bars[0]["timestamp"].date()
    settled_cash = start_cash
    unsettled_cash = 0
    pending_sales = Queue()
    start_equity = start_cash
    running_max_equity = start_equity
    max_drawdown = 0.0
    profit_sum = 0.0
    num_windows = 0
    shares = 0.0
    cost = 0.0
    consecutive_days = 1
    log = []
    sp500_index = 0
    nasdaq_index = 0

    # Start at 1 to have previous bar for relative change
    num_bars = len(stock_bars)
    for i in range(1, num_bars):
        stock_bar = stock_bars[i]
        prev_stock_bar = stock_bars[i-1]
        date = stock_bar["timestamp"].date()
        prev_date = prev_stock_bar["timestamp"].date()
        if date != prev_date:
            consecutive_days += 1
            while not pending_sales.is_empty():
                sale_price, sale_day = pending_sales.head.value
                if consecutive_days - sale_day > 1:
                    settled_cash += sale_price
                    unsettled_cash -= sale_price
                    pending_sales.dequeue()
                else:
                    break

        # Dealing with mismatch in length of bars for sp500 and nasdaq
        if sp500_index + 1 < len(sp500_bars):
            sp500_date = sp500_bars[sp500_index + 1]["timestamp"].date()
            if sp500_date <= date:
                sp500_index += 1

        if nasdaq_index + 1 < len(nasdaq_bars):
            nasdaq_date = nasdaq_bars[nasdaq_index + 1]["timestamp"].date()
            if nasdaq_date <= date:
                nasdaq_index += 1

        sp500_bar = sp500_bars[sp500_index]
        nasdaq_bar = nasdaq_bars[nasdaq_index]
        prev_sp500_bar = sp500_bars[min(0, sp500_index - 1)]
        prev_nasdaq_bar = nasdaq_bars[min(0, nasdaq_index - 1)]

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
                  sp500_sentiments[sp500_index],
                  Agent.rel_change(prev_nasdaq_bar["close"], nasdaq_bar["close"]),
                  Agent.rel_change(prev_nasdaq_bar["volume"], nasdaq_bar["volume"]),
                  nasdaq_sentiments[nasdaq_index],
                  indicator_data["k_percent"][i],
                  indicator_data["d_ema"][i],
                  indicator_data["k_ema"][i],
                  indicator_data["rsi"][i]
                  ]
        if short and shares < 0:
            inputs[0] = -1
            inputs[1] = Agent.rel_change(stock_bar["close"] * abs(shares), cost)

        outputs = net.activate(inputs)

        qty_percent = (outputs[1] + 1) * 0.5
        if outputs[0] > 0.5:  # Buy
            if short and shares < 0:
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
                    profit *= 2.0
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
            if short and shares <= 0:
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
                    pending_sales.enqueue((price, consecutive_days))

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
            if equity > running_max_equity:
                running_max_equity = equity
            max_drawdown = max(running_max_equity - equity, max_drawdown)

    avg_factor = (profit_sum / num_windows) * fitness_multipliers["average"]
    total_factor = profit_sum * fitness_multipliers["total"]
    risk_factor = max_drawdown * fitness_multipliers["risk"]
    return avg_factor + total_factor - risk_factor, log


class Training(Agent):
    def __init__(self, settings, session, stock,
                 stock_bars, sp500_bars, nasdaq_bars,
                 stock_sentiments, sp500_sentiments, nasdaq_sentiments,
                 indicator_data):
        super().__init__(settings, session, stock)
        self.started = False
        self.best_genome = None  # Saving population object adds 10s to each gen
        self.consecutive_gens = 0
        self.stock_bars = stock_bars
        self.sp500_bars = sp500_bars
        self.nasdaq_bars = nasdaq_bars
        self.stock_sentiments = stock_sentiments
        self.sp500_sentiments = sp500_sentiments
        self.nasdaq_sentiments = nasdaq_sentiments
        self.indicator_data = indicator_data
        self.data_batch_index = 0
        self.genome_file_path = os.path.join(self.genome_path, self.stock["genome_filename"])
        self.population_file_path = os.path.join(self.population_path, self.stock["population_filename"])
        self.shortable = True
        self.fractionable = True
        self.cum_fitness = {}

    def eval_genomes(self, genomes, config):
        while not self.running:
            time.sleep(1)

        if isinstance(self.stock_bars[self.data_batch_index], int):
            sub_index = self.stock_bars[self.data_batch_index]
            print(f"Evaluating genomes on substitute data batch {sub_index}")
            b_stock_bars = self.stock_bars[sub_index]
            b_sp500_bars = self.sp500_bars[sub_index]
            b_nasdaq_bars = self.nasdaq_bars[sub_index]
            b_stock_sentiments = self.stock_sentiments[sub_index]
            b_sp500_sentiments = self.sp500_sentiments[sub_index]
            b_nasdaq_sentiments = self.nasdaq_sentiments[sub_index]
            b_indicator_data = self.indicator_data[sub_index]
        else:
            print(f"Evaluating genomes on data batch {self.data_batch_index}")
            b_stock_bars = self.stock_bars[self.data_batch_index]
            b_sp500_bars = self.sp500_bars[self.data_batch_index]
            b_nasdaq_bars = self.nasdaq_bars[self.data_batch_index]
            b_stock_sentiments = self.stock_sentiments[self.data_batch_index]
            b_sp500_sentiments = self.sp500_sentiments[self.data_batch_index]
            b_nasdaq_sentiments = self.nasdaq_sentiments[self.data_batch_index]
            b_indicator_data = self.indicator_data[self.data_batch_index]

        pool = Pool(processes=self.settings["processes"])
        args = []
        for genome_id, genome in genomes:
            args.append(
                (
                    b_stock_bars,
                    b_sp500_bars,
                    b_nasdaq_bars,
                    b_stock_sentiments,
                    b_sp500_sentiments,
                    b_nasdaq_sentiments,
                    self.session["start_cash"],
                    genome,
                    self.config,
                    self.stock["cash_at_risk"],
                    self.settings["log_training"],
                    self.session["profit_window"],
                    self.session["fitness_multipliers"],
                    self.stock["shorting"] and self.shortable,
                    self.fractionable,
                    self.session["short_limit"],
                    self.stock["transaction_fee"],
                    b_indicator_data
                )
            )

        # Use map_async to send each tuple of arguments
        results_async = pool.map_async(eval_genome, args)

        # Wait for the results
        results = results_async.get()

        best_log = None
        best_genome_id = -1
        for i, (fitness, log) in enumerate(results):
            genome_id, genome = genomes[i]
            self.cum_fitness.setdefault(genome_id, []).append(fitness)
            if len(self.cum_fitness[genome_id]) >= self.session["data_batches"]:
                self.cum_fitness[genome_id].pop(0)

            genome.fitness = sum(self.cum_fitness[genome_id])

            if self.best_genome is None or genome.fitness > self.best_genome.fitness:
                best_log = log
                self.best_genome = genome
                best_genome_id = genome_id

        if best_genome_id in self.cum_fitness:
            print(f"Fitness across data batches: {self.cum_fitness[best_genome_id]} - id {best_genome_id}")
        if best_log is not None and self.settings["log_training"]:
            plot.plot_log(self.session["alpaca_api"], self.stock["symbol"], best_log, 30, True)

        pool.close()
        pool.join()
        pool.terminate()
        self.consecutive_gens += 1

        if 0 < self.settings["gen_stagger"] <= self.consecutive_gens:
            self.consecutive_gens = 0
            self.running = False

        self.data_batch_index = (self.data_batch_index + 1) % len(self.stock_bars)

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
        node_names = {-18: 'Position',
                      -17: 'PLPC',
                      -16: 'Open%',
                      -15: 'High%',
                      -14: 'Low%',
                      -13: 'Close%',
                      -12: 'Vol%',
                      -11: "VWAP%",
                      -10: 'Stock Sent',
                      -9: 'SPY Close%',
                      -8: 'SPY Vol%',
                      -7: 'SPY Sent',
                      -6: 'QQQ Close%',
                      -5: 'QQQ Vol%',
                      -4: 'QQQ Sent',
                      -3: 'K%',
                      -2: 'EMA D',
                      -1: 'EMA K',
                      0: 'RSI'}
        visualize.draw_net(self.config, self.best_genome, view=True, node_names=node_names, show_disabled=False)
