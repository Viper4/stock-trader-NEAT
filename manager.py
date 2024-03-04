import agent
import time
import datetime as dt
import pytz
import os
import saving
import plot
import candle_scraper as cs
import threading
import alpaca_trade_api as alpaca
import alpaca_trade_api.entity
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit


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
        print(f"{option['symbol']}: Getting bars from {start.strftime('%Y-%m-%d %H:%M:%S')} to {end.strftime('%Y-%m-%d %H:%M:%S')} at {option['interval']}m intervals")

        bars_df = self.alpaca_api.get_bars(
            symbol=option["symbol"],
            timeframe=TimeFrame(option["interval"], TimeFrameUnit.Minute),
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


class TrainerTest(object):
    def __init__(self, settings, finbert):
        self.running = False
        self.settings = settings
        self.cycles = 0

        self.training_path = self.settings["save_path"] + "/TrainingData"
        if not os.path.exists(self.training_path):
            os.mkdir(self.training_path)

        self.finbert = finbert
        self.symbols = []
        self.largest_backtest = 0
        self.one_agent = False

        self.sessions = {}
        for account in settings["accounts"]:
            base_url = URL("https://paper-api.alpaca.markets") if account["paper"] else URL("https://api.alpaca.markets")
            api = alpaca.REST(account["public_key"], account["secret_key"], base_url=base_url)

            if len(settings["accounts"]) == 1 and len(account["stocks"]) == 1 and account["gen_stagger"] != 0:
                print(f"{account['name']}: Only training 1 agent. Setting gen_stagger to 0.")
                account["gen_stagger"] = 0
                self.one_agent = True

            self.sessions[account["name"]] = {
                "api": api,
                "agents": {},
                "logs": {},
                "interval": account["interval"],
                "backtest_days": account["backtest_days"],
                "stocks": account["stocks"]
            }

            for stock in account["stocks"]:
                self.symbols.append(stock["symbol"])

            if self.largest_backtest < account["backtest_days"]:
                self.largest_backtest = account["backtest_days"]

        self.create_agents()

    def get_bars(self, symbol, session, start, end):
        print(f"{symbol}: Getting bars from {start.strftime('%Y-%m-%d %H:%M:%S')} to {end.strftime('%Y-%m-%d %H:%M:%S')} at {session['interval']}m intervals")

        bars_df = session["api"].get_bars(
            symbol=symbol,
            timeframe=TimeFrame(session["interval"], TimeFrameUnit.Minute),
            start=start.isoformat(),
            end=end.isoformat(),
            limit=100000,
            sort="asc").df.tz_convert("US/Eastern")
        bars_df = bars_df.between_time("9:30", "16:00")
        return bars_df.reset_index().to_dict("records")

    def create_agents(self):
        now_date = dt.datetime.now(pytz.UTC)
        earliest_date = now_date - dt.timedelta(days=self.largest_backtest)
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc
        self.finbert.save_news(self.symbols, earliest_date, end_date)

        for session_key in self.sessions:
            session = self.sessions[session_key]
            start_date = now_date - dt.timedelta(days=session["backtest_days"])

            session["agents"].clear()
            for stock in session["stocks"]:
                if stock["training_filename"] is None:
                    print("No training data filename provided for " + stock["symbol"])
                    exit(0)

                training_file_path = os.path.join(self.training_path, stock["training_filename"])

                if os.path.exists(training_file_path):
                    backtest_start, backtest_end, sentiments = saving.SaveSystem.load_data(training_file_path)
                    bars = self.get_bars(stock["symbol"], session, backtest_start, backtest_end)
                    print("{0}: Loaded training data with {1} bars and {2} sentiments from {3} to {4}".format(stock["symbol"], len(bars), len(sentiments), bars[0]["timestamp"], bars[-1]["timestamp"]))
                else:
                    bars = self.get_bars(stock["symbol"], session, start_date, end_date)

                    print("{0}: Generating sentiments".format(stock["symbol"]))
                    sentiments = [0]
                    for i in range(1, len(bars)):
                        backtest_date = bars[i]["timestamp"].to_pydatetime()
                        sentiment = self.finbert.get_saved_sentiment(stock["symbol"], backtest_date - dt.timedelta(days=2), backtest_date)
                        sentiments.append(sentiment)

                    saving.SaveSystem.save_data((start_date, end_date, sentiments), training_file_path)
                    print("{0}: Saved backtest range and {1} sentiments to {2}".format(stock["symbol"], len(sentiments), training_file_path))

                session["agents"][stock["symbol"]] = agent.Training(self.settings, session["api"], stock, bars, sentiments)

        print("Trainer: Created {0} training agents\n".format(self.symbols))

    def start_training(self):
        print("Starting training...")
        self.running = True
        print(self.cycles)
        if self.cycles >= self.settings["training_reset"]:
            self.cycles = 0
            self.create_agents()

        for session_key in self.sessions:
            session = self.sessions[session_key]
            i = 0
            for symbol in session["agents"]:
                current_agent = session["agents"][symbol]
                current_agent.run()
                while current_agent.running:
                    time.sleep(1)
                if self.settings["visualize"]:
                    current_agent.plot()
                

        '''if self.one_agent:
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
            self.agents[0].run()'''

    def stop_training(self):
        print("Stopping training...")
        self.running = False
        self.cycles += 1
        for symbol in self.agents:
            self.agents[symbol].running = False


class Test(object):
    def __init__(self, settings, alpaca_apis, finbert):
        self.running = False
        self.settings = settings
        self.trainer = Trainer(settings, alpaca_apis, finbert)
        self.finbert = finbert
        self.scraper = cs.Scraper()
        self.training_thread = None
        self.consecutive_days = 0

        self.sessions = {}
        for account in settings["accounts"]:
            base_url = URL("https://paper-api.alpaca.markets") if account["paper"] else URL("https://api.alpaca.markets")
            api = alpaca.REST(account["public_key"], account["secret_key"], base_url=base_url)

            self.sessions[account["name"]] = {
                "api": api,
                "solid_cash": 0.0,
                "liquid_cash": 0.0,
                "pending_sales": [],
                "agents": {},
                "logs": {},
                "clock": None,
                "positions": None,
                "last_get_time": 0.0,
                "interval": account["interval"]
            }

        self.create_agents()

    def create_agents(self):
        print("Trader: Creating agents")
        for session_key, session in self.sessions.items():
            session["agents"].clear()
            for option in self.settings["ticker_options"]:
                session["logs"][option["symbol"]] = []
                session["agents"][option["symbol"]] = agent.Trading(self.settings, session["api"], option, self.finbert, self, self.scraper)
                if option["genome_filename"] is None:
                    print(f" {session_key}: No genome filename provided for {option['symbol']}")
                    exit(0)
                else:
                    try:
                        best_genome = saving.SaveSystem.load_data(
                            os.path.join(session["agents"][option["symbol"]].genome_path, option["genome_filename"]))
                        self.trainer.agents[option["symbol"]].best_genome = best_genome
                        session["agents"][option["symbol"]].update_net(best_genome)
                    except FileNotFoundError:
                        print(f" {session_key}: No genome file found for {option['genome_filename']}")
            print(f" {session_key}: Created {session['agents']} trading agents for \n")

    def get_market_status(self, session_key):
        if self.sessions[session_key]["clock"] is None or time.time() - self.sessions[session_key]["last_get_time"] > 1:
            retries = 0
            while True:
                try:
                    self.sessions[session_key]["clock"] = self.sessions[session_key]["api"].get_clock().is_open
                    self.sessions[session_key]["last_get_time"] = time.time()
                    return self.sessions[session_key]["clock"].is_open
                except ConnectionError as e:
                    print(f"Connection error: '{e}'. Retrying in 5 seconds... ({retries})")
                    time.sleep(5)
                    retries += 1
        else:
            return self.sessions[session_key]["clock"].is_open

    def get_position(self, symbol, session_key):
        session = self.sessions[session_key]
        if session["positions"] is None or time.time() - self.sessions[session_key]["last_get_time"] > 1:
            session["positions"] = session["api"].list_positions()
            session["last_get_time"] = time.time()
        for position in session["positions"]:
            if position.symbol == symbol:
                return position
        return alpaca.entity.Position(raw={
                "symbol": "AMD",
                "qty": "0",
                "avg_entry_price": "0",
                "market_value": "0",
                "cost_basis": "0",
                "unrealized_pl": "0",
                "unrealized_plpc": "0",
              })

    def start_trading(self):
        self.running = True
        self.consecutive_days = 0
        for session_key in self.sessions:
            session = self.sessions[session_key]
            print(session_key)
            starting_liquid = (float(input(" Enter starting liquid cash: ")), int(input(" Enter pending days: ")))
            account = session["api"].get_account()
            session["solid_cash"] = float(account.cash)
            session["liquid_cash"] = 0.0
            session["pending_sales"].clear()

            if starting_liquid[0] != 0:
                session["liquid_cash"] = starting_liquid[0]
                session["solid_cash"] -= session["liquid_cash"]
                session["pending_sales"].append((session["liquid_cash"], starting_liquid[1]))

            for symbol in session["agents"]:
                threading.Thread(target=session["agents"][symbol].run).start()

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            first_session_key = next(iter(self.sessions))
            if self.get_market_status(first_session_key):
                if self.trainer.running:
                    self.trainer.stop_training()
                    self.trainer.running = False
                    self.training_thread.join()
                    for session_key in self.sessions:
                        session = self.sessions[session_key]
                        for symbol in session["agents"]:
                            if self.trainer.agents[symbol].best_genome is not None:
                                session["agents"][symbol].update_net(self.trainer.agents[symbol].best_genome)

                for session in self.sessions.values():
                    for j in reversed(range(len(session["pending_sales"]))):
                        sale = session["pending_sales"][j]
                        if self.consecutive_days - sale[1] > 2:
                            session["solid_cash"] += sale[0]
                            session["liquid_cash"] -= sale[0]
                            session["pending_sales"].pop(j)

                next_close = self.sessions[first_session_key]["clock"].next_close
                wait_time = (next_close - now_date).total_seconds()
                print(f"Market closes in {wait_time / 3600} hours")
                time.sleep(wait_time + 5)
                self.consecutive_days += 1
            else:
                for session_key in self.sessions:
                    session = self.sessions[session_key]
                    api_account = session["api"].get_account()
                    open_positions = session["api"].list_positions()
                    bought_shares = {}
                    for position in open_positions:
                        bought_shares[position.symbol] = float(position.qty)
                    balance_change = float(api_account.equity) - float(api_account.last_equity)
                    print(f"\n{session_key} Details:" +
                          f"\n Daily Bal Change: {balance_change}" +
                          f"\n Solid Cash: {session['solid_cash']}" +
                          f"\n Liquid Cash: {session['liquid_cash']}" +
                          f"\n Equity: {api_account.equity}" +
                          f"\n Bought shares: {bought_shares}\n")

                    saved_log = False
                    for symbol in session["logs"]:
                        if len(session["logs"][symbol]) > 0:
                            if not saved_log:
                                saving.SaveSystem.save_data((session["logs"], balance_change, bought_shares), os.path.join(session["agents"][symbol].log_path, f"{session_key}_{now_date.astimezone(tz=pytz.timezone('US/Central')).strftime('%Y-%m-%d')}.gz"))
                                saved_log = True
                            threading.Thread(target=plot.Plot.plot_log, args=(session["api"], symbol, session["logs"][symbol], session["interval"])).start()
                            session["logs"][symbol].clear()

                next_open = self.sessions[first_session_key]["clock"].next_open
                wait_time = (next_open - now_date).total_seconds()
                print(f"Market opens in {wait_time / 3600} hours\n-----")
                if not self.trainer.running:
                    if self.training_thread is not None:
                        self.training_thread.join()
                    self.training_thread = threading.Thread(target=self.trainer.start_training)
                    self.training_thread.start()
                time.sleep(wait_time + 5)


class Trader(object):
    def __init__(self, settings, alpaca_api, finbert):
        self.running = False
        self.settings = settings
        self.trainer = Trainer(settings, alpaca_api, finbert)
        self.alpaca_api = alpaca_api
        self.finbert = finbert
        self.scraper = cs.Scraper()
        self.training_thread = None
        self.logs = {}
        self.clock = None
        self.positions = None
        self.last_get = 0

        self.consecutive_days = 0
        self.solid_cash = 0.0
        self.liquid_cash = 0.0
        self.pending_sales = []
        self.agents = {}

        self.create_agents()

    def create_agents(self):
        self.agents.clear()
        for option in self.settings["ticker_options"]:
            self.logs[option["symbol"]] = []
            self.agents[option["symbol"]] = agent.Trading(self.settings, self.alpaca_api, option, self.finbert, self, self.scraper)
            if option["genome_filename"] is None:
                print("No genome filename provided for " + option["symbol"])
                exit(0)
            else:
                try:
                    best_genome = saving.SaveSystem.load_data(os.path.join(self.agents[option["symbol"]].genome_path, option["genome_filename"]))
                    self.trainer.agents[option["symbol"]].best_genome = best_genome
                    self.agents[option["symbol"]].update_net(best_genome)
                except FileNotFoundError:
                    print("No genome file found for " + option["genome_filename"])

        print(f"Trader: Created {self.agents.keys()} trading agents\n")

    def get_market_status(self):
        if self.clock is None or time.time() - self.last_get > 1:
            retries = 0
            while True:
                try:
                    self.clock = self.alpaca_api.get_clock().is_open
                    return self.clock.is_open
                except ConnectionError as e:
                    print(f"Connection error: '{e}'. Retrying in 5 seconds... ({retries})")
                    time.sleep(5)
                    retries += 1
        else:
            return self.clock.is_open

    def get_position(self, symbol):
        if self.positions is None or time.time() - self.last_get > 1:
            self.positions = self.alpaca_api.list_positions()
            self.last_get = time.time()
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return alpaca.entity.Position(raw={
                "symbol": "AMD",
                "qty": "0",
                "avg_entry_price": "0",
                "market_value": "0",
                "cost_basis": "0",
                "unrealized_pl": "0",
                "unrealized_plpc": "0",
              })

    def start_trading(self):
        self.running = True
        starting_liquid = (float(input("Enter starting liquid cash: ")), int(input("Enter pending days: ")))
        account = self.alpaca_api.get_account()
        self.solid_cash = float(account.cash)
        self.liquid_cash = 0.0
        self.pending_sales = []

        if starting_liquid[0] != 0:
            self.liquid_cash = starting_liquid[0]
            self.solid_cash -= self.liquid_cash
            self.pending_sales.append((self.liquid_cash, starting_liquid[1]))

        self.consecutive_days = 0

        for symbol in self.agents:
            threading.Thread(target=self.agents[symbol].run).start()
        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            if self.get_market_status():
                if self.trainer.running:
                    self.trainer.stop_training()
                    self.trainer.running = False
                    self.training_thread.join()
                    for symbol in self.trainer.agents:
                        if self.trainer.agents[symbol].best_genome is not None:
                            self.agents[symbol].update_net(self.trainer.agents[symbol].best_genome)

                for j in reversed(range(len(self.pending_sales))):
                    sale = self.pending_sales[j]
                    if self.consecutive_days - sale[1] > 2:
                        self.solid_cash += sale[0]
                        self.liquid_cash -= sale[0]
                        self.pending_sales.pop(j)

                next_close = self.alpaca_api.get_clock().next_close
                wait_time = (next_close - now_date).total_seconds()
                time.sleep(wait_time + 5)
                self.consecutive_days += 1
            else:
                account = self.alpaca_api.get_account()
                open_positions = self.alpaca_api.list_positions()
                bought_shares = {}
                for position in open_positions:
                    bought_shares[position.symbol] = float(position.qty)
                balance_change = float(account.equity) - float(account.last_equity)
                print("\nEnd of Day Account Details:" +
                      "\n Daily Bal Change: " + str(balance_change) +
                      "\n Solid Cash: " + str(self.solid_cash) +
                      "\n Liquid Cash: " + str(self.liquid_cash) +
                      "\n Equity: " + str(account.equity) +
                      "\n Bought shares: " + str(bought_shares) + "\n")

                saved_log = False
                for symbol in self.logs:
                    if len(self.logs[symbol]) > 0:
                        if not saved_log:
                            saving.SaveSystem.save_data((self.logs, balance_change, bought_shares), os.path.join(self.agents[symbol].log_path, f"{now_date.astimezone(tz=pytz.timezone('US/Central')).strftime('%Y-%m-%d')}.gz"))
                            saved_log = True
                        threading.Thread(target=plot.Plot.plot_log, args=(self.alpaca_api, symbol, self.logs[symbol], self.settings["trade_delay"] / 60)).start()
                        self.logs[symbol].clear()

                next_open = self.alpaca_api.get_clock().next_open
                wait_time = (next_open - now_date).total_seconds()
                if not self.trainer.running:
                    print("Market will open in {0} hours.\n-----".format(wait_time / 3600))
                    if self.training_thread is not None:
                        self.training_thread.join()
                    self.training_thread = threading.Thread(target=self.trainer.start_training)
                    self.training_thread.start()
                time.sleep(wait_time + 5)
