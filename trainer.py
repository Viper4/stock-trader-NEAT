import agent
import time
import datetime as dt
import pytz
import os
import saving
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
            limit=100000,
            sort="asc").df.tz_convert("US/Eastern")
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
                backtest_start, backtest_end, sentiments = saving.SaveSystem.load_data(training_file_path)
                bars = self.get_bars(option, backtest_start, backtest_end)
                print("{0}: Loaded training data with {1} bars and {2} sentiments from {3} to {4}".format(option["symbol"], len(bars), len(sentiments), bars[0]["timestamp"], bars[-1]["timestamp"]))
            else:
                bars = self.get_bars(option, start_date, end_date)

                print("{0}: Generating sentiments".format(option["symbol"]))
                sentiments = [0]
                for i in range(1, len(bars)):
                    backtest_date = bars[i]["timestamp"].to_pydatetime()
                    sentiment = self.finbert.get_saved_sentiment(option["symbol"], backtest_date - dt.timedelta(days=2), backtest_date)
                    sentiments.append(sentiment)

                saving.SaveSystem.save_data((start_date, end_date, sentiments), training_file_path)
                print("{0}: Saved backtest range and {1} sentiments to {2}".format(option["symbol"], len(sentiments), training_file_path))

            self.agents[option["symbol"]] = agent.Training(self.settings, self.alpaca_api, option, bars, sentiments)

        print("Trainer: Created {0} training agents\n".format(self.symbols))

    def start_training(self):
        print("Starting training...")
        self.running = True
        print(self.cycles)
        if self.cycles >= self.settings["training_reset"]:
            self.cycles = 0
            self.create_agents()
        agents_len = len(self.agents)
        if agents_len > 1:
            i = 0
            while self.running:
                current_agent = self.agents[self.symbols[i]]
                current_agent.run()
                while current_agent.running:
                    time.sleep(1)
                if self.settings["visualize"]:
                    current_agent.plot()

                i += 1
                if i >= agents_len:
                    i = 0
        else:
            self.agents[0].run()

    def stop_training(self):
        print("Stopping training...")
        self.running = False
        self.cycles += 1
        for symbol in self.agents:
            self.agents[symbol].running = False


class Trader(object):
    def __init__(self, settings, alpaca_api, finbert):
        self.running = False
        self.settings = settings
        self.trainer = Trainer(settings, alpaca_api, finbert)
        self.alpaca_api = alpaca_api
        self.finbert = finbert
        self.training_thread = None
        self.logs = {}

        self.agents = {}
        self.symbols = [option["symbol"] for option in self.settings["ticker_options"]]

        self.create_agents()

    def create_agents(self):
        self.agents.clear()
        for option in self.settings["ticker_options"]:
            self.agents[option["symbol"]] = agent.Trading(self.settings, self.alpaca_api, option)

        print("Trader: Created {0} trading agents\n".format(self.symbols))

    def get_market_status(self):
        retries = 0
        while True:
            try:
                return self.alpaca_api.get_clock().is_open
            except ConnectionError as e:
                print(f"Connection error: '{e}'. Retrying in 5 seconds... ({retries})")
                time.sleep(5)
                retries += 1

    def start_trading(self):
        starting_liquid = (float(input("Enter starting liquid cash: ")), int(input("Enter pending days: ")))
        account = self.alpaca_api.get_account()
        solid_cash = float(account.cash)
        liquid_cash = 0.0
        pending_sales = []

        if starting_liquid[0] != 0:
            liquid_cash = starting_liquid[0]
            solid_cash -= liquid_cash
            pending_sales.append((liquid_cash, starting_liquid[1]))

        consecutive_days = 0
        while self.running:
            if self.get_market_status():
                print("Market is open. Starting trading...")
                if self.trainer.running:
                    consecutive_days += 1
                    self.trainer.stop_training()
                    self.trainer.running = False
                    self.training_thread.join()
                    for symbol in self.trainer.agents:
                        if self.trainer.agents[symbol].best_genome is not None:
                            self.agents[symbol].update_net(self.trainer.agents[symbol].best_genome)

                    for j in reversed(range(len(pending_sales))):
                        sale = pending_sales[j]
                        if consecutive_days - sale[1] > 2:
                            solid_cash += sale[0]
                            liquid_cash -= sale[0]
                            pending_sales.pop(j)

                positions = {}
                for position in self.alpaca_api.list_positions():
                    positions[position.symbol] = {"quantity": float(position.qty),
                                                  "price": float(position.current_price),
                                                  "plpc": float(position.unrealized_plpc),
                                                  "pl": float(position.unrealized_pl),
                                                  "avg_entry_price": float(position.avg_entry_price)}

                for symbol in self.symbols:
                    if symbol not in positions:
                        positions[symbol] = {"quantity": 0, "price": 0, "plpc": 0, "pl": 0, "avg_entry_price": 0}
                    if not self.agents[symbol].running:
                        self.agents[symbol].run()
            else:
                account = self.alpaca_api.get_account()
                open_positions = self.alpaca_api.list_positions()
                bought_shares = {}
                for position in open_positions:
                    bought_shares[position.symbol] = float(position.qty)
                balance_change = float(account.equity) - float(account.last_equity)
                print("\nMarket is closed. Account Details:" +
                      "\n Daily Bal Change: " + str(balance_change) +
                      "\n Solid Cash: " + str(solid_cash) +
                      "\n Liquid Cash: " + str(liquid_cash) +
                      "\n Equity: " + str(account.equity) +
                      "\n Bought shares: " + str(bought_shares) + "\n")
