import time
import datetime as dt
import os
import torch.cuda
import saving
import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
from base_manager import Manager
from training_agent import Training
from training_agent_gpu import TrainingGPU
from multiprocessing import Pool


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
        self.gpu_training = settings["gpu"] and torch.cuda.is_available()

        for profile in settings["profiles"]:
            alpaca_api = alpaca.REST(profile["public_key"], profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))

            self.update_profile(profile, alpaca_api, True)

        self.create_agents()

    def update_profile(self, profile, alpaca_api, no_agents=False):
        print(f"Trainer: Updating {profile['name']} profile")

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
            "profit_window": profile["profit_window"],
            "fitness_multipliers": profile["fitness_multipliers"],
            "short_limit": profile["short_limit"],
            "start_cash": profile["start_cash"]
        }

        update_agents = False
        for stock in profile["stocks"]:
            if stock["symbol"] not in self.symbols:
                self.symbols.append(stock["symbol"])
                update_agents = True

        if not no_agents and update_agents:
            self.create_agents(False, True)

    def load_data(self, symbol, i, session, file_path):
        data = saving.SaveSystem.load_data(file_path)
        if len(data) == 4:
            backtest_start, backtest_end, b_sentiments, b_indicators = data
            b_bars = self.get_bars(symbol, session["alpaca_api"], session["interval"], backtest_start, backtest_end, 500000, self.gpu_training)
            if self.gpu_training:
                print(f" {symbol}{i}: Loaded {len(b_bars)} bars, sentiments, and indicator data from {b_bars.index[0]} to {b_bars.index[-1]}")
            else:
                print(f" {symbol}{i}: Loaded {len(b_bars)} bars, sentiments, and indicator data from {b_bars[0]['timestamp']} to {b_bars[-1]['timestamp']}")
            return b_bars, b_sentiments, b_indicators
        else:
            backtest_start, backtest_end, b_sentiments = data
            b_bars = self.get_bars(symbol, session["alpaca_api"], session["interval"], backtest_start, backtest_end, 500000, self.gpu_training)
            if self.gpu_training:
                print(f" {symbol}{i}: Loaded {len(b_bars)} bars and sentiments from {b_bars.index[0]} to {b_bars.index[-1]}")
            else:
                print(f" {symbol}{i}: Loaded {len(b_bars)} bars and sentiments from {b_bars[0]['timestamp']} to {b_bars[-1]['timestamp']}")
            return b_bars, b_sentiments, None

    def generate_data(self, symbol, i, session, start_date, end_date, file_path, indicators):
        bars = self.get_bars(symbol, session["alpaca_api"], session["interval"], start_date, end_date, 500000, self.gpu_training)
        if len(bars) == 0:
            return None

        sentiments = [0]  # We skip first sentiment in training (due to relative change we must start at index 1)

        if indicators:
            print(f" {symbol}{i}: Generating sentiments and indicator data for {len(bars)} bars from {start_date} to {end_date}")

            bar_k_period = Training.days_to_bars(session["k_period"], session["interval"])
            bar_d_period = Training.days_to_bars(session["d_period"], session["interval"])
            bar_rsi_period = Training.days_to_bars(session["rsi_period"], session["interval"])
            k_percent_data = [0]
            d_ema_data = [0]
            k_ema_data = [0]
            rsi_data = [0]
            d_alpha = 2 / (bar_d_period + 1)
            k_alpha = 2 / (bar_k_period + 1)
            prev_d_ema = bars[0]["close"]
            prev_k_ema = bars[0]["close"]

            gain = 0
            loss = 0

            start_time = time.time()

            for i in range(1, len(bars)):
                # Sentiments
                backtest_date = bars[i]["timestamp"].to_pydatetime()
                sentiment = self.finbert.get_saved_sentiment(symbol, backtest_date - dt.timedelta(days=2), backtest_date)
                sentiments.append(sentiment)

                # Indicators
                k_percent_data.append(Training.calculate_k_percent(bars[i - min(bar_k_period, i):i]))
                d_ema_data.append(Training.calculate_ema(bars[i]["close"], d_alpha, prev_d_ema))
                prev_d_ema = d_ema_data[-1]
                k_ema_data.append(Training.calculate_ema(bars[i]["close"], k_alpha, prev_k_ema))
                prev_k_ema = k_ema_data[-1]

                # Calculate RSI
                change = bars[i]["close"] - bars[i - 1]["close"]
                if change > 0:
                    gain += change
                else:
                    loss += abs(change)

                # Remove old data
                start_rsi_index = i - min(bar_rsi_period, i)
                if (i - start_rsi_index) + 1 >= bar_rsi_period:
                    start_change = bars[start_rsi_index]["close"] - bars[start_rsi_index - 1]["close"]
                    if start_change > 0:
                        gain -= change
                    else:
                        loss -= abs(change)

                rsi_data.append(Training.calculate_rsi(gain, loss, (i - start_rsi_index) + 1))

            indicator_data = {
                "k_percent": k_percent_data,
                "d_ema": d_ema_data,
                "k_ema": k_ema_data,
                "rsi": rsi_data
            }
            print(f" {symbol}{i}: Finished generating data in {(time.time() - start_time):.2f}s")
            saving.SaveSystem.save_data((start_date, end_date, sentiments, indicator_data), file_path)
            return bars, sentiments, indicator_data
        else:
            print(f" {symbol}: Generating sentiments for {len(bars)} bars from {start_date} to {end_date}")
            for i in range(1, len(bars)):
                backtest_date = bars[i]["timestamp"].to_pydatetime()
                sentiment = self.finbert.get_saved_sentiment(symbol, backtest_date - dt.timedelta(days=2), backtest_date)
                sentiments.append(sentiment)

            saving.SaveSystem.save_data((start_date, end_date, sentiments), file_path)
            return bars, sentiments, None

    def create_agents(self, regenerate=False, save_news=False):
        print("Trainer: Creating agents")
        now_date = dt.datetime.now(dt.timezone.utc)
        earliest_date = now_date - dt.timedelta(days=self.largest_backtest * self.largest_data_batch_size)
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc

        if save_news:
            print("save1")
            self.finbert.save_news(self.symbols, earliest_date, end_date)

        for profile_name in self.sessions:
            print(profile_name)
            session = self.sessions[profile_name]
            start_date = now_date - dt.timedelta(days=session["data_batch_size"])

            session["agents"].clear()

            pool = Pool(processes=self.processes)
            jobs = []
            stock_bars = {"SPY": [], "QQQ": []}
            stock_sentiments = {"SPY": [], "QQQ": []}
            stock_indicators = {"SPY": [], "QQQ": []}

            for i in range(session["data_batches"]):
                time_delta = dt.timedelta(days=i * session["data_batch_size"])
                if i >= len(stock_bars["SPY"]):
                    stock_bars["SPY"].append([])
                    stock_sentiments["SPY"].append([])
                    stock_indicators["SPY"].append([])

                    stock_bars["QQQ"].append([])
                    stock_sentiments["QQQ"].append([])
                    stock_indicators["QQQ"].append([])

                sp500_file_path = os.path.join(self.training_path, str(session["interval"]) + "m-data-SPY" + str(i) + ".gz")
                if not regenerate and os.path.exists(sp500_file_path):
                    jobs.append((i, "SPY",
                                 pool.apply_async(self.load_data,
                                                  ("SPY", i, session, sp500_file_path))))
                else:
                    one_bar = self.get_bars("SPY", session["alpaca_api"], session["interval"],
                                            start_date - time_delta, end_date - time_delta,
                                            1, self.gpu_training)
                    if len(one_bar) == 0:
                        jobs.append((i, "SPY", None))
                        continue
                    if len(self.finbert.saved_news) == 0:
                        self.finbert.save_news(self.symbols, earliest_date, end_date)
                    jobs.append((i, "SPY",
                                 pool.apply_async(self.generate_data, ("SPY", i, session,
                                                                       start_date - time_delta,
                                                                       end_date - time_delta,
                                                                       sp500_file_path, False))))

                nasdaq_file_path = os.path.join(self.training_path, str(session["interval"]) + "m-data-QQQ" + str(i) + ".gz")
                if not regenerate and os.path.exists(nasdaq_file_path):
                    jobs.append((i, "QQQ",
                                 pool.apply_async(self.load_data,
                                                  ("QQQ", i, session, nasdaq_file_path))))
                else:
                    one_bar = self.get_bars("QQQ", session["alpaca_api"], session["interval"],
                                            start_date - time_delta, end_date - time_delta,
                                            1, self.gpu_training)
                    if len(one_bar) == 0:
                        jobs.append((i, "QQQ", None))
                        continue
                    if len(self.finbert.saved_news) == 0:
                        self.finbert.save_news(self.symbols, earliest_date, end_date)
                    jobs.append((i, "QQQ",
                                 pool.apply_async(self.generate_data, ("QQQ", i, session,
                                                                       start_date - time_delta,
                                                                       end_date - time_delta,
                                                                       nasdaq_file_path, False))))

            for stock in session["stocks"]:
                symbol = stock["symbol"]

                if stock["training_filename"] is None:
                    print(" No training data filename provided for " + symbol)
                    exit(0)
                
                training_file_path = os.path.join(self.training_path, stock["training_filename"])
                for i in range(session["data_batches"]):
                    file_path = training_file_path.replace(".gz", f"{i}.gz")
                    time_delta = dt.timedelta(days=i * session["data_batch_size"])

                    if symbol not in stock_bars:
                        stock_bars[symbol] = []
                        stock_sentiments[symbol] = []
                        stock_indicators[symbol] = []

                    stock_bars[symbol].append([])
                    stock_sentiments[symbol].append([])
                    stock_indicators[symbol].append([])

                    if not regenerate and os.path.exists(file_path):
                        jobs.append((i, symbol,
                                     pool.apply_async(self.load_data,
                                                      (symbol, i, session, file_path))))
                    else:
                        one_bar = self.get_bars(symbol, session["alpaca_api"], session["interval"],
                                                start_date - time_delta, end_date - time_delta,
                                                1, self.gpu_training)
                        if len(one_bar) == 0:
                            jobs.append((i, symbol, None))
                            continue
                        if len(self.finbert.saved_news) == 0:
                            self.finbert.save_news(self.symbols, earliest_date, end_date)
                        jobs.append((i, symbol,
                                     pool.apply_async(self.generate_data,
                                                      (symbol, i, session,
                                                       start_date - time_delta,
                                                       end_date - time_delta,
                                                       file_path, True))))

            for job in jobs:
                i, symbol, async_result = job
                if async_result is None:
                    result = None
                else:
                    result = async_result.get()
                time_delta = dt.timedelta(days=i * session["data_batch_size"])

                if result is None:
                    found_data = False
                    for j in reversed(range(i)):
                        if not isinstance(stock_bars[symbol][j], int) and len(stock_bars[symbol][j]) != 0:
                            stock_bars[symbol][i], stock_sentiments[symbol][i], stock_indicators[symbol][i] = j, j, j
                            print(f" {symbol}{i}: No data from {(start_date - time_delta).isoformat()} to {(end_date - time_delta).isoformat()}, using {symbol}{j} data")
                            found_data = True
                            break
                    if not found_data:
                        stock_bars[symbol][i], stock_sentiments[symbol][i], stock_indicators[symbol][i] = 0, 0, 0
                        print(f" {symbol}{i}: No data from {(start_date - time_delta).isoformat()} to {(end_date - time_delta).isoformat()}, using {symbol}0 data")
                else:
                    stock_bars[symbol][i], stock_sentiments[symbol][i], stock_indicators[symbol][i] = result

            pool.close()
            pool.join()
            pool.terminate()

            for stock in session["stocks"]:
                symbol = stock["symbol"]
                if stock_indicators[symbol][0] is not None:
                    if self.gpu_training:
                        session["agents"][symbol] = TrainingGPU(self.settings, session, stock,
                                                                stock_bars[symbol], stock_bars["SPY"], stock_bars["QQQ"],
                                                                stock_sentiments[symbol], stock_sentiments["SPY"], stock_sentiments["QQQ"],
                                                                stock_indicators[symbol])
                    else:
                        session["agents"][symbol] = Training(self.settings, session, stock,
                                                             stock_bars[symbol], stock_bars["SPY"], stock_bars["QQQ"],
                                                             stock_sentiments[symbol], stock_sentiments["SPY"], stock_sentiments["QQQ"],
                                                             stock_indicators[symbol])

        if self.gpu_training:
            print(f"Trainer: Created {self.symbols} GPU training agents\n")
        else:
            print(f"Trainer: Created {self.symbols} CPU training agents\n")

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