import neat
import time
import os
from multiprocessing import Pool
import threading
import saving
import visualize
import plot
from Agents.base_agent import Agent
from data_structures import Queue
from constants import POPULATION_DIR, GENOME_DIR
from tqdm import tqdm


def eval_genome(args):
    (columns, ma_periods,
     start_cash, genome, config, cash_at_risk, log_training, profit_window, fitness_multipliers,
     fractionable, transaction_fee) = args
    net = neat.nn.RecurrentNetwork.create(genome, config)
    start_date = columns["index"][0].to_pydatetime()
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
    log = []
    prev_date = None

    for i in tqdm(range(len(columns["index"]))):
        date = columns["index"][i]

        # Check to settle cash after each day
        if prev_date is not None and (date - prev_date).days >= 1:
            while not pending_sales.is_empty():
                sale_price, sale_date = pending_sales.head.value
                if (date - sale_date).days >= 1:
                    settled_cash += sale_price
                    unsettled_cash -= sale_price
                    pending_sales.dequeue()
                else:
                    break

        inputs = Agent.generate_inputs_fast(columns, i, Agent.rel_change(cost, columns["close"][i] * shares), ma_periods)

        outputs = net.activate(inputs)  # MAJOR SLOWDOWN

        qty_percent = (outputs[1] + 1) * 0.5
        if outputs[0] > 0.5:  # Buy
            quantity = qty_percent * settled_cash * cash_at_risk / columns["close"][i]
            if not fractionable:
                quantity = round(quantity)
            price = quantity * columns["close"][i]
            if price >= 1:  # Alpaca doesn't allow trades under $1
                cost += price
                shares += quantity
                settled_cash -= price

                if log_training:
                    action = {"side": "Buy", "type": "long", "quantity": quantity, "price": columns["close"][i],
                              "settled_cash": settled_cash, "unsettled_cash": unsettled_cash,
                              "datetime": date.to_pydatetime()}
                    log.append(action)
        elif outputs[0] < -0.5:  # Sell
            quantity = qty_percent * shares
            if not fractionable:
                quantity = round(quantity)
            price = quantity * columns["close"][i] * (1 - transaction_fee)
            if price >= 1:
                if shares - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                    price = shares * columns["close"][i] * (1 - transaction_fee)
                    profit = price - cost
                    shares = 0.0
                    cost = 0.0
                else:
                    avg_cost = cost / shares
                    shares -= quantity
                    cost = avg_cost * shares
                    profit = price - (avg_cost * quantity)
                unsettled_cash += price
                pending_sales.enqueue((price, date))

                if log_training:
                    action = {"side": "Sell", "type": "long", "quantity": quantity, "price": columns["close"][i],
                              "profit": profit, "settled_cash": settled_cash,
                              "unsettled_cash": unsettled_cash,
                              "datetime": date.to_pydatetime()}
                    log.append(action)

        if i == len(columns["index"]) - 1 or (date - start_date).days >= profit_window:
            equity = unsettled_cash + settled_cash + columns["close"][i] * shares
            profit_sum += equity - start_equity
            num_windows += 1
            start_equity = equity
            start_date = date
            if equity > running_max_equity:
                running_max_equity = equity
            max_drawdown = max(running_max_equity - equity, max_drawdown)

        prev_date = date

    avg_factor = (profit_sum / num_windows) * fitness_multipliers["average"]
    total_factor = profit_sum * fitness_multipliers["total"]
    risk_factor = max_drawdown * fitness_multipliers["risk"]
    return avg_factor + total_factor - risk_factor, log


class Training(Agent):
    def __init__(self, settings, profile, stock, stock_bars):
        super().__init__(settings, profile, stock)
        self.started = False
        self.best_genome = None  # Saving population object adds 10s to each gen
        self.consecutive_gens = 0
        self.stock_bars = stock_bars
        self.data_batch_index = 0
        self.genome_file_path = os.path.join(GENOME_DIR, self.stock["genome_filename"])
        self.population_file_path = os.path.join(POPULATION_DIR, self.stock["population_filename"])
        self.fractionable = True
        self.cum_fitness = {}

    def eval_genomes(self, genomes, config):
        while not self.running:
            time.sleep(1)

        if isinstance(self.stock_bars[self.data_batch_index], int):
            sub_index = self.stock_bars[self.data_batch_index]
            print(f"Evaluating genomes on substitute data batch {sub_index}")
            b_stock_bars = self.stock_bars[sub_index]
        else:
            print(f"Evaluating genomes on data batch {self.data_batch_index}")
            b_stock_bars = self.stock_bars[self.data_batch_index]

        columns = {}
        for column in b_stock_bars.columns:
            columns[column] = b_stock_bars[column].tolist()

        columns["index"] = b_stock_bars.index.tolist()

        pool = Pool(processes=self.settings["processes"])
        args = []
        for genome_id, genome in genomes:
            args.append(
                (
                    columns,
                    self.profile.ma_periods,
                    self.profile.start_cash,
                    genome,
                    self.config,
                    self.stock["cash_at_risk"],
                    self.settings["log_training"],
                    self.profile.profit_window,
                    self.profile.fitness_multipliers,
                    self.fractionable,
                    self.stock["transaction_fee"],
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
            if len(self.cum_fitness[genome_id]) > self.profile.data_batches:
                self.cum_fitness[genome_id].pop(0)

            genome.fitness = sum(self.cum_fitness[genome_id])

            if best_genome_id == -1 or genome.fitness > self.best_genome.fitness:
                best_log = log
                self.best_genome = genome
                best_genome_id = genome_id

        if best_genome_id in self.cum_fitness:
            print(f"Best fitness across data batches: {self.cum_fitness[best_genome_id]} - id {best_genome_id}")
        if best_log is not None and self.settings["log_training"]:
            plot.plot_log(self.profile.alpaca_api, self.stock["symbol"], best_log, 30, True)

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
        asset = self.profile.alpaca_api.get_asset(symbol=self.stock["symbol"])
        self.fractionable = asset.fractionable
        if not self.started:
            print(f"Starting {self.profile.interval}m {self.stock['symbol']} training agent...")
            save_system = saving.SaveSystem(1, self.genome_file_path, self.settings["gen_stagger"],
                                            self.population_file_path)
            if os.path.exists(self.population_file_path):
                p = save_system.load_population(self.population_file_path)
            else:
                p = neat.Population(self.config)
            if self.settings["print_stats"]:
                p.add_reporter(neat.StdOutReporter(True))

            p.add_reporter(save_system)
            threading.Thread(target=p.run, args=(self.eval_genomes, None)).start()
        else:
            print(f"Resuming {self.profile.interval}m {self.stock['symbol']} training agent...")
        self.started = True

    def stop(self):
        print(f"Stopping {self.profile.interval}m {self.stock['symbol']} training agent...")
        self.running = False

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
