import agent
import time
import datetime as dt
import pytz
import os
import pickle
import gzip
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit


class Trainer(object):
    def __init__(self, settings, alpaca_api, finbert):
        self.running = False
        self.settings = settings
        self.cycles = 0

        self.training_path = self.settings["save_path"] + "/TrainingData"
        if not os.path.exists(self.training_path):
            os.mkdir(self.training_path)

        if len(self.settings["ticker_options"]) == 1 and self.settings["gen_stagger"] != 0:
            print("Only training 1 agent. Setting gen_stagger to 0.")
            self.settings["gen_stagger"] = 0

        self.alpaca_api = alpaca_api
        self.finbert = finbert
        self.agents = {}
        self.symbols = [option["symbol"] for option in self.settings["ticker_options"]]

        self.create_agents()

    def get_bars(self, option, start, end):
        print("{0}: Getting bars from {1} to {2} at {3}m intervals".format(option["symbol"], start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), option["data_interval"]))

        bars_df = self.alpaca_api.get_bars(
            symbol=option["symbol"],
            timeframe=TimeFrame(option["data_interval"], TimeFrameUnit.Minute),
            start=start.isoformat(),
            end=end.isoformat(),
            limit=30000,
            sort="asc").df.tz_convert("US/Eastern")
        if not self.settings["extended_hours"]:
            bars_df = bars_df.between_time("9:30", "16:00")
        return bars_df.reset_index().to_dict("records")

    def create_agents(self):
        now_date = dt.datetime.now(pytz.UTC)
        start_date = now_date - dt.timedelta(days=self.settings["backtest_days"])
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc

        self.agents.clear()
        self.finbert.save_news(self.symbols, start_date, end_date)
        for option in self.settings["ticker_options"]:
            if option["training_filename"] is None:
                print("No training data filename provided for " + option["symbol"])
                exit(0)

            training_file_path = os.path.join(self.training_path, option["training_filename"])

            if os.path.exists(training_file_path):
                with gzip.open(training_file_path) as f:
                    backtest_start, backtest_end, sentiments = pickle.load(f)
                bars = self.get_bars(option, backtest_start, backtest_end)
                print("{0}: Loaded training data with {1} bars and {2} sentiments".format(option["symbol"], len(bars), len(sentiments)))
            else:
                bars = self.get_bars(option, start_date, end_date)

                print("{0}: Generating sentiments".format(option["symbol"]))
                sentiments = [[0, 0, 0]]  # First bar is skipped
                for i in range(1, len(bars)):
                    backtest_date = bars[i]["timestamp"].to_pydatetime()
                    sentiment = self.finbert.get_saved_sentiment(option["symbol"], backtest_date - dt.timedelta(days=2), backtest_date)
                    sentiments.append(sentiment)
                with gzip.open(training_file_path, 'w', compresslevel=5) as f:
                    data = (start_date, end_date, sentiments)
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print("{0}: Saved backtest range and {1} sentiments to {2}".format(option["symbol"], len(sentiments), training_file_path))

            self.agents[option["symbol"]] = agent.Training(self.settings, self.alpaca_api, self.finbert, option, bars, sentiments)

        print("Trainer: Created {0} training agents\n".format(self.symbols))

    def start_training(self):
        self.running = True
        if self.cycles >= self.settings["training_reset"]:
            self.cycles = 0
            self.create_agents()
        if len(self.agents) > 1:
            i = 0
            while self.running:
                self.agents[self.symbols[i]].run()
                while self.agents[self.symbols[i]].running:
                    time.sleep(1)
                if self.settings["visualize"]:
                    self.agents[self.symbols[i]].plot()
                #if self.settings["print_stats"]:
                #    time.sleep(5)  # Wait for console printout

                i += 1
                if i >= len(self.agents):
                    i = 0
        else:
            self.agents[0].run()

    def stop_training(self):
        self.running = False
        self.cycles += 1
        for symbol in self.agents:
            self.agents[symbol].running = False
