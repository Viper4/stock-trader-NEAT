import random
import time
import datetime as dt
import os
import alpaca_trade_api as alpaca
from base_manager import Manager
from training_agent import Training
from alpaca_trade_api.rest import URL, TimeFrameUnit


class Trainer(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)
        self.cycles = 0

        self.training_path = self.settings["save_path"] + "\\TrainingData"
        if not os.path.exists(self.training_path):
            os.mkdir(self.training_path)

        self.symbols = []
        self.largest_backtest = 0
        self.largest_data_batch_size = 0
        self.one_agent = False
        self.processes = settings["processes"]

        for profile in settings["profiles"]:
            alpaca_api = alpaca.REST(profile["public_key"], profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))

            self.update_profile(profile, alpaca_api, True)

        self.create_agents()

    def update_profile(self, profile, alpaca_api, no_agents=False):
        print(f"Trainer: Updating {profile['name']}")

        if self.largest_backtest < profile["data_batch_size"]:
            self.largest_backtest = profile["data_batch_size"]

        if self.largest_data_batch_size < profile["data_batches"]:
            self.largest_data_batch_size = profile["data_batches"]

        if len(self.settings["profiles"]) == 1 and len(profile["stocks"]) == 1 and self.settings["gen_stagger"] != 0:
            print(f"{profile['name']}: Only training 1 agent. Setting gen_stagger to 0.")
            profile["gen_stagger"] = 0
            self.one_agent = True

        if profile["name"] in self.sessions:
            agents = self.sessions[profile["name"]]["agents"]
            logs = self.sessions[profile["name"]]["logs"]
        else:
            agents = {}
            logs = {}

        self.sessions[profile["name"]] = {
            "alpaca_api": alpaca_api,
            "agents": agents,
            "logs": logs,
            "stocks": profile["stocks"],
            "data_batch_size": profile["data_batch_size"],
            "data_batches": profile["data_batches"],
            "interval": profile["interval"],
            "k_period": profile["k_period"],
            "d_period": profile["d_period"],
            "rsi_period": profile["rsi_period"],
            "atr_period": profile["atr_period"],
            "sma_periods": profile["sma_periods"],
            "profit_window": profile["profit_window"],
            "fitness_multipliers": profile["fitness_multipliers"],
            "start_cash": profile["start_cash"]
        }

        update_agents = False
        for stock in profile["stocks"]:
            if stock["symbol"] not in self.symbols:
                self.symbols.append(stock["symbol"])
                update_agents = True

        if not no_agents and update_agents:
            self.create_agents(False, True)

    def create_agents(self, regenerate=False, save_news=False):
        print("Trainer: Creating agents")
        now_date = dt.datetime.now(dt.timezone.utc)
        earliest_date = now_date - dt.timedelta(days=self.largest_backtest * self.largest_data_batch_size)
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc

        if save_news:
            self.finbert.save_news(self.symbols + ["SPY", "QQQ"], earliest_date, end_date)

        for profile_name in self.sessions:
            print(profile_name)
            session = self.sessions[profile_name]
            start_date = now_date - dt.timedelta(days=session["data_batch_size"])

            session["agents"].clear()

            stock_bars = {"SPY": [], "QQQ": []}

            for i in range(session["data_batches"]):
                time_delta = dt.timedelta(days=i * session["data_batch_size"])

                spy_file_path = os.path.join(self.training_path, str(session["interval"]) + "m-data-SPY" + str(i) + ".gz")
                if not regenerate and os.path.exists(spy_file_path):
                    stock_bars["SPY"].append(self.load_data("SPY", i, spy_file_path))
                else:
                    test_bars = self.get_bars("SPY", session["alpaca_api"], 1,
                                              start_date - time_delta, end_date - time_delta,
                                              100, TimeFrameUnit.Hour, "desc")
                    if test_bars.empty:
                        print(f" SPY{i}: No data from {(start_date - time_delta).isoformat()} to {(end_date - time_delta).isoformat()}!")
                        stock_bars["SPY"].append(None)
                        continue
                    if len(self.finbert.saved_news) == 0:
                        self.finbert.save_news(self.symbols + ["SPY", "QQQ"], earliest_date, end_date)
                    stock_bars["SPY"].append(self.generate_data("SPY", i, session,
                                                                start_date - time_delta,
                                                                end_date - time_delta,
                                                                spy_file_path, False,
                                                                None, None,
                                                                False))

                qqq_file_path = os.path.join(self.training_path, str(session["interval"]) + "m-data-QQQ" + str(i) + ".gz")
                if not regenerate and os.path.exists(qqq_file_path):
                    stock_bars["QQQ"].append(self.load_data("QQQ", i, qqq_file_path))
                else:
                    test_bars = self.get_bars("QQQ", session["alpaca_api"], 1,
                                              start_date - time_delta, end_date - time_delta,
                                              100, TimeFrameUnit.Hour, "desc")
                    if test_bars.empty:
                        print(f" QQQ{i}: No data from {(start_date - time_delta).isoformat()} to {(end_date - time_delta).isoformat()}!")
                        stock_bars["QQQ"].append(None)
                        continue
                    if len(self.finbert.saved_news) == 0:
                        self.finbert.save_news(self.symbols + ["QQQ"], earliest_date, end_date)
                    stock_bars["QQQ"].append(self.generate_data("QQQ", i, session,
                                                                start_date - time_delta,
                                                                end_date - time_delta,
                                                                qqq_file_path, False,
                                                                None, None,
                                                                False))

            used_substitutions = {}
            for stock in session["stocks"]:
                symbol = stock["symbol"]
                stock_bars[symbol] = []
                used_substitutions[symbol] = set()

                if stock["training_filename"] is None:
                    print(" No training data filename provided for " + symbol)
                    exit(0)
                
                training_file_path = os.path.join(self.training_path, stock["training_filename"])
                for i in range(session["data_batches"]):
                    file_path = training_file_path.replace(".gz", f"{i}.gz")
                    time_delta = dt.timedelta(days=i * session["data_batch_size"])

                    if not regenerate and os.path.exists(file_path):
                        stock_bars[symbol].append(self.load_data(symbol, i, file_path))
                    else:
                        test_bars = self.get_bars(symbol, session["alpaca_api"], 1,
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
                        stock_bars[symbol].append(self.generate_data(symbol, i, session,
                                                                     start_date - time_delta,
                                                                     end_date - time_delta,
                                                                     file_path, True,
                                                                     stock_bars["SPY"][i], stock_bars["QQQ"][i],
                                                                     False))

            for stock in session["stocks"]:
                symbol = stock["symbol"]
                session["agents"][symbol] = Training(self.settings, session, stock, stock_bars[symbol])

        print(f"Trainer: Created {self.symbols} training agents\n")

    def start(self):
        print(f"Starting training... ({self.cycles})")
        self.running = True
        if self.cycles >= self.settings["training_reset"]:
            self.cycles = 0
            self.create_agents(True, True)

        if self.one_agent:
            first_session = self.sessions[next(iter(self.sessions))]
            first_session["agents"][next(iter(first_session["agents"]))].run()
        else:
            while self.running:
                for profile_name in self.sessions:
                    session = self.sessions[profile_name]
                    for symbol in session["agents"]:
                        self.settings, session["alpaca_api"] = self.get_settings_and_alpaca(0)
                        for profile in self.settings["profiles"]:
                            if profile["name"] == profile_name:
                                self.update_profile(profile, session["alpaca_api"])
                                break

                        current_agent = session["agents"][symbol]
                        current_agent.settings = self.settings
                        current_agent.run()
                        while current_agent.running:
                            time.sleep(1)
                        if self.settings["visualize"]:
                            current_agent.plot()
                        if not self.running:
                            return

    def stop(self):
        print("Stopping training...")
        self.running = False
        self.cycles += 1
        for profile_name in self.sessions:
            session = self.sessions[profile_name]
            for symbol in session["agents"]:
                session["agents"][symbol].running = False
