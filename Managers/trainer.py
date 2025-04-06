import random
import time
import datetime as dt
from Managers.base_manager import Manager, Profile
from Agents.training_agent import Training
from alpaca_trade_api.rest import TimeFrameUnit
import os
from constants import TRAINING_DIR


class Trainer(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)
        self.cycles = 0

        self.symbols = []
        self.largest_backtest = 0
        self.largest_data_batch_size = 0
        self.one_agent = False
        self.processes = settings["processes"]

        self.profiles = []
        for i in range(len(settings["profiles"])):
            self.profiles.append(Profile(settings, i))

        self.update_profiles()

        self.create_agents()
        
    def update_profiles(self):
        self.symbols.clear()
        for profile in self.profiles:
            profile.update()

            for symbol in profile.stocks:
                self.symbols.append(symbol)
                if self.largest_backtest < profile.data_batch_size:
                    self.largest_backtest = profile.data_batch_size
                if self.largest_data_batch_size < profile.data_batches:
                    self.largest_data_batch_size = profile.data_batches

            if len(self.settings["profiles"]) == 1 and len(profile.stocks) == 1 and self.settings["gen_stagger"] != 0:
                print(f"{profile.name}: Only training 1 agent. Setting gen_stagger to 0")
                self.settings["gen_stagger"] = 0
                self.one_agent = True

    def create_agents(self, regenerate=False, save_news=False):
        print("Trainer: Creating agents")
        now_date = dt.datetime.now(dt.timezone.utc)
        earliest_date = now_date - dt.timedelta(days=self.largest_backtest * self.largest_data_batch_size)
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc

        if save_news:
            self.finbert.save_news(self.symbols + ["SPY", "QQQ"], earliest_date, end_date)

        for profile in self.profiles:
            print(profile.name)
            start_date = now_date - dt.timedelta(days=profile.data_batch_size)

            profile.agents.clear()

            stock_bars = {"SPY": [], "QQQ": []}

            for i in range(profile.data_batches):
                time_delta = dt.timedelta(days=i * profile.data_batch_size)

                spy_file_path = os.path.join(TRAINING_DIR, str(profile.interval) + "m-data-SPY" + str(i) + ".gz")
                if not regenerate and os.path.exists(spy_file_path):
                    stock_bars["SPY"].append(self.load_data("SPY", i, spy_file_path))
                else:
                    test_bars = self.get_bars("SPY", profile.alpaca_api, 1,
                                              start_date - time_delta, end_date - time_delta,
                                              100, TimeFrameUnit.Hour, "desc")
                    if test_bars.empty:
                        print(f" SPY{i}: No data from {(start_date - time_delta).isoformat()} to {(end_date - time_delta).isoformat()}!")
                        stock_bars["SPY"].append(None)
                        continue
                    if len(self.finbert.saved_news) == 0:
                        self.finbert.save_news(self.symbols + ["SPY", "QQQ"], earliest_date, end_date)
                    stock_bars["SPY"].append(self.generate_data("SPY", i, profile,
                                                                start_date - time_delta,
                                                                end_date - time_delta,
                                                                spy_file_path,
                                                                None, None,
                                                                training=True))

                qqq_file_path = os.path.join(TRAINING_DIR, str(profile.interval) + "m-data-QQQ" + str(i) + ".gz")
                if not regenerate and os.path.exists(qqq_file_path):
                    stock_bars["QQQ"].append(self.load_data("QQQ", i, qqq_file_path))
                else:
                    test_bars = self.get_bars("QQQ", profile.alpaca_api, 1,
                                              start_date - time_delta, end_date - time_delta,
                                              100, TimeFrameUnit.Hour, "desc")
                    if test_bars.empty:
                        print(f" QQQ{i}: No data from {(start_date - time_delta).isoformat()} to {(end_date - time_delta).isoformat()}!")
                        stock_bars["QQQ"].append(None)
                        continue
                    if len(self.finbert.saved_news) == 0:
                        self.finbert.save_news(self.symbols + ["QQQ"], earliest_date, end_date)
                    stock_bars["QQQ"].append(self.generate_data("QQQ", i, profile,
                                                                start_date - time_delta,
                                                                end_date - time_delta,
                                                                qqq_file_path,
                                                                None, None,
                                                                training=True))

            used_substitutions = {}
            for symbol, stock in profile.stocks.items():
                stock_bars[symbol] = []
                used_substitutions[symbol] = set()

                if stock["training_filename"] is None:
                    print(" No training data filename provided for " + symbol)
                    exit(0)
                
                training_file_path = os.path.join(TRAINING_DIR, stock["training_filename"])
                for i in range(profile.data_batches):
                    file_path = training_file_path.replace(".gz", f"{i}.gz")
                    time_delta = dt.timedelta(days=i * profile.data_batch_size)

                    if not regenerate and os.path.exists(file_path):
                        stock_bars[symbol].append(self.load_data(symbol, i, file_path))
                    else:
                        test_bars = self.get_bars(symbol, profile.alpaca_api, 1,
                                                  start_date - time_delta, end_date - time_delta,
                                                  100, TimeFrameUnit.Hour, "desc")
                        if test_bars.empty:
                            stock_bars[symbol].append(None)
                            found_data = False
                            for j in range(i):
                                if not isinstance(stock_bars[symbol][j], int) and len(stock_bars[symbol][j]) != 0 and j not in used_substitutions[symbol]:
                                    stock_bars[symbol][i] = j
                                    print(f" {symbol}{i}: No data from {(start_date - time_delta).isoformat()} to {(end_date - time_delta).isoformat()}, using {symbol}{j} data")
                                    found_data = True
                                    used_substitutions[symbol].add(j)
                                    break
                            if not found_data:
                                if len(used_substitutions) == 0:
                                    print(f" {symbol}{i}: No data from {(start_date - time_delta).isoformat()} to {(end_date - time_delta).isoformat()}, no data to substitute!")
                                else:
                                    sub_index = random.choice(list(used_substitutions[symbol]))
                                    stock_bars[symbol][i] = sub_index
                                    print(f" {symbol}{i}: No data from {(start_date - time_delta).isoformat()} to {(end_date - time_delta).isoformat()}, using {symbol}{sub_index} data")
                            continue
                        if len(self.finbert.saved_news) == 0:
                            self.finbert.save_news(self.symbols, earliest_date, end_date)
                        stock_bars[symbol].append(self.generate_data(symbol, i, profile,
                                                                     start_date - time_delta,
                                                                     end_date - time_delta,
                                                                     file_path,
                                                                     stock_bars["SPY"][i], stock_bars["QQQ"][i],
                                                                     training=True))

            for symbol, stock in profile.stocks.items():
                profile.agents[symbol] = Training(self.settings, profile, stock, stock_bars[symbol])

        print(f"Trainer: Created {self.symbols} training agents\n")

    def start(self):
        print(f"Starting training... ({self.cycles})")
        self.running = True
        if self.cycles >= self.settings["training_reset"]:
            self.cycles = 0
            self.create_agents(True, True)

        if self.one_agent:
            next(iter(self.profiles[0].agents.values())).run()
        else:
            while self.running:
                self.update_profiles()
                for profile in self.profiles:
                    for symbol in profile.agents:
                        profile.agents[symbol].settings = self.settings
                        profile.agents[symbol].run()
                        while profile.agents[symbol].running:
                            time.sleep(1)
                        if self.settings["visualize"]:
                            profile.agents[symbol].plot()
                        if not self.running:
                            return

    def stop(self):
        print("Stopping training...")
        self.running = False
        self.cycles += 1
        for profile in self.profiles:
            for symbol in profile.agents:
                profile.agents[symbol].stop()
