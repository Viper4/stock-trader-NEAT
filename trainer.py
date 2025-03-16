import time
import datetime as dt
import os
import saving
import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
from base_manager import Manager
from training_agent import Training
import torch


class Trainer(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)
        self.cycles = 0

        self.training_path = self.settings["save_path"] + "\\TrainingData"
        if not os.path.exists(self.training_path):
            os.mkdir(self.training_path)

        self.symbols = []
        self.largest_backtest = 0
        self.largest_batch_size = 0
        self.one_agent = False

        for profile in settings["profiles"]:
            alpaca_api = alpaca.REST(profile["public_key"], profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))

            self.update_profile(profile, alpaca_api, True)

        self.create_agents()

    def update_profile(self, profile, alpaca_api, no_agents=False):
        print(f"Trainer: Updating {profile['name']} profile")

        if self.largest_backtest < profile["batch_size"]:
            self.largest_backtest = profile["batch_size"]

        if self.largest_batch_size < profile["batches"]:
            self.largest_batch_size = profile["batches"]

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
            "batch_size": profile["batch_size"],
            "batches": profile["batches"],
            "interval": profile["interval"],
            "k_period": profile["k_period"],
            "d_period": profile["d_period"],
            "rsi_period": profile["rsi_period"],
            "profit_window": profile["profit_window"],
            "fitness_multipliers": profile["fitness_multipliers"],
            "short_limit": profile["short_limit"]
        }

        update_agents = False
        for stock in profile["stocks"]:
            if stock["symbol"] not in self.symbols:
                self.symbols.append(stock["symbol"])
                update_agents = True

        if not no_agents and update_agents:
            self.create_agents(False, True)

    def calculate_ema_gpu(self, close_price, prev_ema, alpha):
        close_price = torch.tensor(close_price, dtype=torch.float32)
        prev_ema = torch.tensor(prev_ema, dtype=torch.float32)
        ema = close_price * alpha + prev_ema * (1 - alpha)
        return ema.cpu().item()  # Bring back to CPU

    def generate_data(self, symbol, session, earliest_date, start_date, end_date, file_path, save_news, indicators):
        bars = self.get_bars(symbol, session["alpaca_api"], session["interval"], start_date, end_date)
        if len(bars) == 0:
            if indicators:
                return None, None, None
            else:
                return None, None

        if save_news or len(self.finbert.saved_news) == 0:
            self.finbert.save_news(self.symbols, earliest_date, end_date)

        sentiments = [0]  # We skip first sentiment in training (due to relative change we must start at index 1)

        if indicators:
            print(f" {symbol}: Generating sentiments and indicator data for {len(bars)} bars from {start_date} to {end_date}")

            bar_k_period = Training.days_to_bars(session["k_period"], session["interval"])
            bar_d_period = Training.days_to_bars(session["d_period"], session["interval"])
            bar_rsi_period = Training.days_to_bars(session["rsi_period"], session["interval"])
            k_percent_data = [0]
            ema_data = [0]
            k_sma_data = [0]
            d_sma_data = [0]
            rsi_data = [0]
            alpha = 2 / (bar_d_period + 1)
            prev_ema = bars[0]["close"]

            for i in range(1, len(bars)):
                # Sentiments
                backtest_date = bars[i]["timestamp"].to_pydatetime()
                sentiment = self.finbert.get_saved_sentiment(symbol, backtest_date - dt.timedelta(days=2), backtest_date)
                sentiments.append(sentiment)

                # Indicators
                k_percent_data.append(Training.calculate_k_percent(bars[i - min(bar_k_period, i):i]))

                ema = Training.calculate_ema(bars[i]["close"], alpha, prev_ema)
                ema_data.append(2 * ((ema - bars[i]["close"]) / bars[i]["close"]))  # Normalize

                k_sma = Training.calculate_sma(bars[i - min(bar_k_period, i):i])
                k_sma_data.append(2 * ((k_sma - bars[i]["close"]) / bars[i]["close"]))

                d_sma = Training.calculate_sma(bars[i - min(bar_d_period, i):i])
                d_sma_data.append(2 * ((d_sma - bars[i]["close"]) / bars[i]["close"]))

                rsi_data.append(Training.calculate_rsi(bars[i - min(bar_rsi_period, i):i]))

            indicator_data = {
                "k_percent": k_percent_data,
                "ema": ema_data,
                "k_sma": k_sma_data,
                "d_sma": d_sma_data,
                "rsi": rsi_data
            }
            saving.SaveSystem.save_data((start_date, end_date, sentiments, indicator_data), file_path)
            return bars, sentiments, indicator_data
        else:
            print(f" {symbol}: Generating sentiments for {len(bars)} bars from {start_date} to {end_date}")
            for i in range(1, len(bars)):
                backtest_date = bars[i]["timestamp"].to_pydatetime()
                sentiment = self.finbert.get_saved_sentiment(symbol, backtest_date - dt.timedelta(days=2), backtest_date)
                sentiments.append(sentiment)

            saving.SaveSystem.save_data((start_date, end_date, sentiments), file_path)
            return bars, sentiments

    def create_agents(self, regenerate=False, save_news=False):
        print("Trainer: Creating agents")
        now_date = dt.datetime.now(dt.timezone.utc)
        earliest_date = now_date - dt.timedelta(days=self.largest_backtest * self.largest_batch_size)
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc

        for profile_name in self.sessions:
            print(profile_name)
            session = self.sessions[profile_name]
            start_date = now_date - dt.timedelta(days=session["batch_size"])

            session["agents"].clear()
            sp500_bars = []  # Each item is one batch's bars
            nasdaq_bars = []
            sp500_sentiments = []  # Each item is one batch's sentiments
            nasdaq_sentiments = []

            # TODO: Repeated code with if statements. Figure out how to condense them to make this cleaner
            for i in range(session["batches"]):
                time_delta = dt.timedelta(days=i * session["batch_size"])

                sp500_file_path = os.path.join(self.training_path, str(session["interval"]) + "m-data-SPY" + str(i) + ".gz")
                if not regenerate and os.path.exists(sp500_file_path):
                    backtest_start, backtest_end, b_sp500_sentiments = saving.SaveSystem.load_data(sp500_file_path)
                    b_sp500_bars = self.get_bars("SPY", session["alpaca_api"], session["interval"], backtest_start,
                                                 backtest_end)
                    if len(b_sp500_bars) != len(b_sp500_sentiments):
                        print(f" SPY{i}: Loaded {len(b_sp500_bars)} bars but have {len(b_sp500_sentiments)} sentiments. Regenerating data")
                        b_sp500_bars, b_sp500_sentiments = self.generate_data("SPY", session, earliest_date - time_delta,
                                                                              start_date - time_delta, end_date - time_delta,
                                                                              sp500_file_path, save_news, False)
                    else:
                        print(f" SPY{i}: Loaded {len(b_sp500_bars)} bars and sentiments from {b_sp500_bars[0]['timestamp']} to {b_sp500_bars[-1]['timestamp']}")
                else:
                    b_sp500_bars, b_sp500_sentiments = self.generate_data("SPY", session, earliest_date - time_delta,
                                                                          start_date - time_delta, end_date - time_delta,
                                                                          sp500_file_path, save_news, False)

                nasdaq_file_path = os.path.join(self.training_path, str(session["interval"]) + "m-data-QQQ" + str(i) + ".gz")
                if not regenerate and os.path.exists(nasdaq_file_path):
                    backtest_start, backtest_end, b_nasdaq_sentiments = saving.SaveSystem.load_data(nasdaq_file_path)
                    b_nasdaq_bars = self.get_bars("QQQ", session["alpaca_api"], session["interval"], backtest_start,
                                                  backtest_end)
                    if len(b_nasdaq_bars) != len(b_nasdaq_sentiments):
                        print(f" QQQ{i}: Loaded {len(b_nasdaq_bars)} bars but have {len(b_nasdaq_sentiments)} sentiments. Regenerating data")
                        b_nasdaq_bars, b_nasdaq_sentiments = self.generate_data("QQQ", session, earliest_date - time_delta,
                                                                                start_date - time_delta, end_date - time_delta,
                                                                                nasdaq_file_path, save_news, False)
                    else:
                        print(f" QQQ{i}: Loaded {len(b_nasdaq_bars)} bars and sentiments from {b_nasdaq_bars[0]['timestamp']} to {b_nasdaq_bars[-1]['timestamp']}")
                else:
                    b_nasdaq_bars, b_nasdaq_sentiments = self.generate_data("QQQ", session, earliest_date - time_delta,
                                                                            start_date - time_delta, end_date - time_delta,
                                                                            nasdaq_file_path, save_news, False)

                sp500_bars.append(b_sp500_bars)
                sp500_sentiments.append(b_sp500_sentiments)
                nasdaq_bars.append(b_nasdaq_bars)
                nasdaq_sentiments.append(b_nasdaq_sentiments)

            for stock in session["stocks"]:
                if stock["training_filename"] is None:
                    print(" No training data filename provided for " + stock["symbol"])
                    exit(0)

                training_file_path = os.path.join(self.training_path, stock["training_filename"])
                stock_bars = []  # Each item is one batch's bars
                stock_sentiments = []  # Each item is one batch's sentiments
                indicator_data = []  # Each item is one batch's indicator data
                for i in range(session["batches"]):
                    file_path = training_file_path.replace(".gz", f"{i}.gz")
                    time_delta = dt.timedelta(days=i * session["batch_size"])

                    if not regenerate and os.path.exists(file_path):
                        backtest_start, backtest_end, b_sentiments, b_indicators = saving.SaveSystem.load_data(file_path)
                        b_bars = self.get_bars(stock["symbol"], session["alpaca_api"], session["interval"], backtest_start, backtest_end)
                        if len(b_bars) != len(b_sentiments):
                            print(f" {stock['symbol']}{i}: Mismatch in data length (bars: {len(b_bars)}, sent: {len(b_sentiments)}, ind: {len(b_indicators['k_percent'])}). Regenerating data")
                            b_bars, b_sentiments, b_indicators = self.generate_data(stock["symbol"], session, earliest_date - time_delta,
                                                                                    start_date - time_delta, end_date - time_delta,
                                                                                    file_path, save_news, True)
                        else:
                            print(f" {stock['symbol']}{i}: Loaded {len(b_bars)} bars, sentiments, and indicator data from {b_bars[0]['timestamp']} to {b_bars[-1]['timestamp']}")
                    else:
                        b_bars, b_sentiments, b_indicators = self.generate_data(stock["symbol"], session, earliest_date - time_delta,
                                                                                start_date - time_delta, end_date - time_delta,
                                                                                file_path, save_news, True)
                    if b_bars is None:
                        for j in reversed(range(len(stock_bars))):
                            if not isinstance(stock_bars[j], int):
                                b_bars, b_sentiments = j, j
                                b_indicators = j
                                print(f" {stock['symbol']}{i}: No data from {(start_date - time_delta).isoformat()} to {(end_date - time_delta).isoformat()}, using {stock['symbol']}{j} data")
                                break
                    stock_bars.append(b_bars)
                    stock_sentiments.append(b_sentiments)
                    indicator_data.append(b_indicators)

                session["agents"][stock["symbol"]] = Training(self.settings, session, stock,
                                                              stock_bars, sp500_bars, nasdaq_bars,
                                                              stock_sentiments, sp500_sentiments, nasdaq_sentiments,
                                                              indicator_data)

        print("Trainer: Created {0} training agents\n".format(self.symbols))

    def start(self):
        print(f"Starting training... ({self.cycles})")
        self.running = True
        if self.cycles >= self.settings["training_reset"]:
            self.cycles = 0
            self.create_agents(True)

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