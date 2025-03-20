import neat
import time
import os
import threading
import pandas as pd
import saving
import visualize
from base_agent import Agent
from numba import njit
import numpy as np
from data_structures import NumbaFloatIntQueue


@njit
def rel_change(a, b):
    if a == 0:
        return 0
    return (b - a) / a


@njit
def eval_genome(stock_opens,
                stock_highs,
                stock_lows,
                stock_closes, sp500_closes, nasdaq_closes,
                stock_volumes, sp500_volumes, nasdaq_volumes,
                stock_vwaps,
                stock_sentiments, sp500_sentiments, nasdaq_sentiments,
                stock_days, sp500_days, nasdaq_days,
                start_cash, network, cash_at_risk, profit_window,
                total_multiplier, avg_multiplier, risk_multiplier,
                short, fractionable, short_limit, transaction_fee,
                k_percents, d_emas, k_emas, rsis):
    settled_cash = start_cash
    unsettled_cash = 0
    pending_sales_queue = NumbaFloatIntQueue()
    start_equity = start_cash
    running_max_equity = start_equity
    max_drawdown = 0.0
    profit_sum = 0.0
    num_windows = 0
    shares = 0.0
    cost = 0.0
    sp500_index = 0
    nasdaq_index = 0

    # Start at 1 to have previous bar for relative change
    num_bars = len(stock_closes)
    for i in range(1, num_bars):
        if stock_days[i] != stock_days[i - 1]:
            while not pending_sales_queue.is_empty():
                sale = pending_sales_queue.peek()
                if stock_days[i] - sale.value2 > 1:
                    settled_cash += sale.value1
                    unsettled_cash -= sale.value1
                    pending_sales_queue.dequeue()
                else:
                    break

        # Dealing with mismatch in length of bar_dfs for sp500 and nasdaq
        if sp500_index + 1 < len(sp500_closes):
            sp500_day = sp500_days[sp500_index + 1]
            if sp500_day <= stock_days[i]:
                sp500_index += 1

        if nasdaq_index + 1 < len(nasdaq_closes):
            nasdaq_day = nasdaq_days[nasdaq_index + 1]
            if nasdaq_day <= stock_days[i]:
                nasdaq_index += 1

        inputs = np.array(
            [
                1,  # -1 = short, 1 = long
                rel_change(cost, stock_closes[i] * shares),  # plpc
                rel_change(stock_opens[i - 1], stock_opens[i]),
                rel_change(stock_highs[i - 1], stock_highs[i]),
                rel_change(stock_lows[i - 1], stock_lows[i]),
                rel_change(stock_closes[i - 1], stock_closes[i]),
                rel_change(stock_volumes[i - 1], stock_volumes[i]),
                rel_change(stock_vwaps[i - 1], stock_vwaps[i]),
                stock_sentiments[i],  # -1 = negative, 0 = neutral, 1 = positive
                rel_change(sp500_closes[sp500_index - 1], sp500_closes[sp500_index]),
                rel_change(sp500_volumes[sp500_index - 1], sp500_volumes[sp500_index]),
                sp500_sentiments[sp500_index],
                rel_change(nasdaq_closes[nasdaq_index - 1], nasdaq_closes[nasdaq_index]),
                rel_change(nasdaq_volumes[nasdaq_index - 1], nasdaq_volumes[nasdaq_index]),
                nasdaq_sentiments[nasdaq_index],
                k_percents[i],
                d_emas[i],
                k_emas[i],
                rsis[i]
            ],
            dtype=np.float32
        )
        if short and shares < 0:
            inputs[0] = -1
            inputs[1] = rel_change(stock_closes[i] * abs(shares), cost)

        outputs = net.activate(inputs)

        qty_percent = (outputs[1] + 1) * 0.5
        if outputs[0] > 0.5:  # Buy
            if short and shares < 0:
                quantity = qty_percent * abs(shares)
                quantity = round(quantity)  # Shorts don't allow fractional qty
                price = quantity * stock_closes[i] * (1 - transaction_fee)
                if price >= 1:
                    if abs(shares) - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                        price = abs(shares) * stock_closes[i] * (1 - transaction_fee)
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
                        profit *= 2
                    settled_cash += profit
            else:
                quantity = qty_percent * settled_cash * cash_at_risk / stock_closes[i]
                if not fractionable:
                    quantity = round(quantity)
                price = quantity * stock_closes[i]
                if price >= 1:  # Alpaca doesn't allow trades under $1
                    cost += price
                    shares += quantity
                    settled_cash -= price
        elif outputs[0] < -0.5:  # Sell
            if short and shares <= 0:
                quantity = qty_percent * (short_limit - cost) * cash_at_risk / stock_closes[i]
                quantity = round(quantity)  # Shorts don't allow fractional qty
                price = quantity * stock_closes[i] * (1 - transaction_fee)
                if cost + price < short_limit:
                    if price >= 1:  # Alpaca doesn't allow trades under $1
                        cost += price
                        shares -= quantity
            else:
                quantity = qty_percent * shares
                if not fractionable:
                    quantity = round(quantity)
                price = quantity * stock_closes[i] * (1 - transaction_fee)
                if price >= 1:
                    if shares - quantity < 0.001:  # Alpaca doesn't allow selling < 1e-9 qty
                        price = shares * stock_closes[i] * (1 - transaction_fee)
                        shares = 0.0
                        cost = 0.0
                    else:
                        avg_cost = cost / shares
                        shares -= quantity
                        cost = avg_cost * shares
                    unsettled_cash += price

                    pending_sales_queue.enqueue(price, stock_days[i])

        if i == num_bars - 1 or stock_days[i] >= profit_window:
            if shares < 0:
                equity = unsettled_cash + settled_cash + shares * stock_closes[i] - cost
            else:
                equity = unsettled_cash + settled_cash + stock_closes[i] * shares
            profit_sum += equity - start_equity
            num_windows += 1
            start_equity = equity
            if equity > running_max_equity:
                running_max_equity = equity
            max_drawdown = max(running_max_equity - equity, max_drawdown)

    avg_factor = (profit_sum / num_windows) * avg_multiplier
    total_factor = profit_sum * total_multiplier
    risk_factor = max_drawdown * risk_multiplier
    return avg_factor + total_factor - risk_factor


class TrainingGPU(Agent):
    def __init__(self, settings, session, stock,
                 stock_bar_dfs, sp500_bar_dfs, nasdaq_bar_dfs,
                 stock_sentiments, sp500_sentiments, nasdaq_sentiments,
                 indicator_data):
        super().__init__(settings, session, stock)
        self.started = False
        self.best_genome = None  # Saving population object adds 10s to each gen
        self.consecutive_gens = 0
        self.stock_bar_dfs = stock_bar_dfs
        self.sp500_bar_dfs = sp500_bar_dfs
        self.nasdaq_bar_dfs = nasdaq_bar_dfs
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

        if isinstance(self.stock_bar_dfs[self.data_batch_index], int):
            sub_index = self.stock_bar_dfs[self.data_batch_index]
            print(f"Evaluating genomes on substitute data batch {sub_index}")
            b_stock_bar_df = self.stock_bar_dfs[sub_index]
            b_sp500_bar_df = self.sp500_bar_dfs[sub_index]
            b_nasdaq_bar_df = self.nasdaq_bar_dfs[sub_index]
            b_stock_sentiments = self.stock_sentiments[sub_index]
            b_sp500_sentiments = self.sp500_sentiments[sub_index]
            b_nasdaq_sentiments = self.nasdaq_sentiments[sub_index]
            b_indicator_data = self.indicator_data[sub_index]
        else:
            print(f"Evaluating genomes on data batch {self.data_batch_index}")
            b_stock_bar_df = self.stock_bar_dfs[self.data_batch_index]
            b_sp500_bar_df = self.sp500_bar_dfs[self.data_batch_index]
            b_nasdaq_bar_df = self.nasdaq_bar_dfs[self.data_batch_index]
            b_stock_sentiments = self.stock_sentiments[self.data_batch_index]
            b_sp500_sentiments = self.sp500_sentiments[self.data_batch_index]
            b_nasdaq_sentiments = self.nasdaq_sentiments[self.data_batch_index]
            b_indicator_data = self.indicator_data[self.data_batch_index]

        b_stock_bar_df.index = pd.to_datetime(b_stock_bar_df.index)
        b_sp500_bar_df.index = pd.to_datetime(b_sp500_bar_df.index)
        b_nasdaq_bar_df.index = pd.to_datetime(b_nasdaq_bar_df.index)
        first_timestamp = b_stock_bar_df.index[0]

        best_genome_id = -1
        for genome_id, genome in genomes:
            fitness = eval_genome(b_stock_bar_df["open"].to_numpy(),
                                  b_stock_bar_df["high"].to_numpy(),
                                  b_stock_bar_df["low"].to_numpy(),
                                  b_stock_bar_df["close"].to_numpy(),
                                  b_sp500_bar_df["close"].to_numpy(),
                                  b_nasdaq_bar_df["close"].to_numpy(),
                                  b_stock_bar_df["volume"].to_numpy(),
                                  b_sp500_bar_df["volume"].to_numpy(),
                                  b_nasdaq_bar_df["volume"].to_numpy(),
                                  b_stock_bar_df["vwap"].to_numpy(),
                                  b_stock_sentiments,
                                  b_sp500_sentiments,
                                  b_nasdaq_sentiments,
                                  (b_stock_bar_df.index - first_timestamp).days.to_numpy(),
                                  (b_sp500_bar_df.index - first_timestamp).days.to_numpy(),
                                  (b_nasdaq_bar_df.index - first_timestamp).days.to_numpy(),
                                  self.session["start_cash"],
                                  neat.nn.RecurrentNetwork.create(genome, config),
                                  self.stock["cash_at_risk"],
                                  self.session["profit_window"],
                                  self.session["fitness_multipliers"],
                                  self.stock["shorting"] and self.shortable,
                                  self.fractionable,
                                  self.session["short_limit"],
                                  self.stock["transaction_fee"],
                                  np.array(b_indicator_data["k_percent"], dtype=np.float32),
                                  np.array(b_indicator_data["d_ema"], dtype=np.float32),
                                  np.array(b_indicator_data["k_ema"], dtype=np.float32),
                                  np.array(b_indicator_data["rsi"], dtype=np.float32))

            self.cum_fitness.setdefault(genome_id, []).append(fitness)
            if len(self.cum_fitness[genome_id]) >= self.session["data_batches"]:
                self.cum_fitness[genome_id].pop(0)

            genome.fitness = sum(self.cum_fitness[genome_id])

            if self.best_genome is None or genome.fitness > self.best_genome.fitness:
                self.best_genome = genome
                best_genome_id = genome_id

        if best_genome_id in self.cum_fitness:
            print(f"Fitness across data batches: {self.cum_fitness[best_genome_id]} - id {best_genome_id}")

        self.consecutive_gens += 1

        if 0 < self.settings["gen_stagger"] <= self.consecutive_gens:
            self.consecutive_gens = 0
            self.running = False

        self.data_batch_index = (self.data_batch_index + 1) % len(self.stock_bar_dfs)

    def run(self):
        if self.running:
            return
        self.running = True
        asset = self.session["alpaca_api"].get_asset(symbol=self.stock["symbol"])
        self.shortable = asset.shortable
        self.fractionable = asset.fractionable
        if not self.started:
            print(f"Starting {self.session['interval']}m {self.stock['symbol']} training agent...")
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
