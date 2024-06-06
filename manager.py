import json
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
import urllib3
import subprocess


class Manager(object):
    def __init__(self, settings, finbert):
        self.running = False
        self.settings = settings
        self.finbert = finbert
        self.sessions = {}
        self.log_path = self.settings["save_path"] + "\\Logs"
        saving.SaveSystem.make_dir(self.log_path)

    @staticmethod
    def check_internet_connection():
        tries = 0
        while True:
            try:
                subprocess.check_output(["ping", "-c", "1", "8.8.8.8"])
                break
            except subprocess.CalledProcessError:
                print(f"No internet connection. ({tries})")
                time.sleep(5)
                tries += 1

    @staticmethod
    def get_bars(symbol, alpaca_api, interval, start, end):
        tries = 1
        while True:
            try:
                bars_df = alpaca_api.get_bars(
                    symbol=symbol,
                    timeframe=TimeFrame(interval, TimeFrameUnit.Minute),
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=500000,
                    sort="asc",
                    adjustment="all").df.tz_convert("US/Eastern")
                bars_df = bars_df.between_time("9:30", "16:00")
                return bars_df.reset_index().to_dict("records")
            except (ConnectionError, urllib3.exceptions.ProtocolError) as e:
                Manager.check_internet_connection()
                print(f"Error getting bars: '{e}'. Retrying in 5 seconds... ({tries})")
                tries += 1
                time.sleep(5)

    @staticmethod
    def get_settings_and_alpaca(profile_index):
        # Settings
        local_dir = os.path.dirname(__file__)
        settings_path = os.path.join(local_dir, "settings.json")
        with open(settings_path) as file:
            settings = json.load(file)

        # Alpaca API
        first_profile = settings["profiles"][profile_index]
        alpaca_api = alpaca.REST(first_profile["public_key"], first_profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))

        return settings, alpaca_api


class Trainer(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)
        self.cycles = 0

        self.training_path = self.settings["save_path"] + "\\TrainingData"
        if not os.path.exists(self.training_path):
            os.mkdir(self.training_path)

        self.symbols = []
        self.largest_backtest = 0
        self.one_agent = False

        for profile in settings["profiles"]:
            alpaca_api = alpaca.REST(profile["public_key"], profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))
            
            self.update_profile(profile, alpaca_api)

        self.create_agents()
    
    def update_profile(self, profile, alpaca_api):
        print("Updated profile")
        if len(self.settings["profiles"]) == 1 and len(profile["stocks"]) == 1 and profile["gen_stagger"] != 0:
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
            "backtest_days": profile["backtest_days"],
            "interval": profile["interval"],
            "profit_window": profile["profit_window"],
            "fitness_multipliers": profile["fitness_multipliers"]
        }

        update_agents = False
        for stock in profile["stocks"]:
            if stock["symbol"] not in self.symbols:
                self.symbols.append(stock["symbol"])
                update_agents = True

        if update_agents:
            self.create_agents(False, True)

        if self.largest_backtest < profile["backtest_days"]:
            self.largest_backtest = profile["backtest_days"]
    
    def generate_data(self, symbol, session, earliest_date, start_date, end_date, file_path, save_news):
        if save_news or len(self.finbert.saved_news) == 0:
            self.finbert.save_news(self.symbols, earliest_date, end_date)

        bars = self.get_bars(symbol, session["alpaca_api"], session["interval"], start_date, end_date)

        print(f" {symbol}: Generating sentiments for {len(bars)} bars")
        sentiments = [0]
        for i in range(1, len(bars)):
            backtest_date = bars[i]["timestamp"].to_pydatetime()
            sentiment = self.finbert.get_saved_sentiment(symbol,
                                                         backtest_date - dt.timedelta(days=2),
                                                         backtest_date)
            sentiments.append(sentiment)

        saving.SaveSystem.save_data((start_date, end_date, sentiments), file_path)
        print(f" {symbol}: Saved backtest range and {len(sentiments)} sentiments to {file_path}")
        return bars, sentiments

    def create_agents(self, regenerate=False, save_news=False):
        print("Trainer: Creating agents")
        now_date = dt.datetime.now(dt.timezone.utc)
        earliest_date = now_date - dt.timedelta(days=self.largest_backtest)
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc

        for profile_name in self.sessions:
            print(profile_name)
            session = self.sessions[profile_name]
            start_date = now_date - dt.timedelta(days=session["backtest_days"])

            session["agents"].clear()
            for stock in session["stocks"]:
                if stock["training_filename"] is None:
                    print(" No training data filename provided for " + stock["symbol"])
                    exit(0)

                training_file_path = os.path.join(self.training_path, stock["training_filename"])

                if not regenerate and os.path.exists(training_file_path):
                    backtest_start, backtest_end, sentiments = saving.SaveSystem.load_data(training_file_path)
                    bars = self.get_bars(stock["symbol"], session["alpaca_api"], session["interval"], backtest_start, backtest_end)
                    if len(bars) != len(sentiments):
                        print(f" {stock['symbol']}: Loaded {len(bars)} bars but have {len(sentiments)} sentiments. Regenerating training data")
                        bars, sentiments = self.generate_data(stock["symbol"], session, earliest_date, start_date, end_date, training_file_path, save_news)
                    else:
                        print(f" {stock['symbol']}: Loaded {len(bars)} bars and sentiments from {bars[0]['timestamp']} to {bars[-1]['timestamp']}")
                else:
                    bars, sentiments = self.generate_data(stock["symbol"], session, earliest_date, start_date, end_date, training_file_path, save_news)

                session["agents"][stock["symbol"]] = agent.Training(self.settings, session, stock, bars, sentiments)

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


class Trader(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)
        self.schwab_api = settings["schwab"]
        self.trainer = Trainer(settings, finbert)
        self.scraper = cs.Scraper()
        self.training_thread = None
        self.consecutive_days = 0
        self.profile = settings["profiles"][0]
        self.alpaca_api = alpaca.REST(self.profile["public_key"], self.profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))
        self.agents = {}
        self.logs = {}
        self.clock = [None, 0]
        self.positions = [None, 0]

        self.create_agents()

    def update_profile(self):
        print("Updated profile")
        self.settings, self.alpaca_api = self.get_settings_and_alpaca(0)
        self.profile = self.settings["profiles"][0]
        for stock in self.profile["stocks"]:
            if stock["trading"]:
                if stock["symbol"] not in self.agents:
                    self.create_agents()
                    break
                else:
                    self.agents[stock["symbol"]].settings = self.settings
                    self.agents[stock["symbol"]].stock = stock

    def create_agents(self):
        print("Trader: Creating agents")

        now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
        symbols = []
        for stock in self.profile["stocks"]:
            if stock["trading"]:
                symbols.append(stock["symbol"])
        self.finbert.save_news(symbols, now_date - dt.timedelta(days=30), now_date - dt.timedelta(minutes=16))

        for stock in self.profile["stocks"]:
            if stock["trading"]:
                if stock["symbol"] not in self.agents:
                    self.logs[stock["symbol"]] = []
                    self.agents[stock["symbol"]] = agent.Trading(self.settings, stock, self)
                    if stock["genome_filename"] is None:
                        print(f" No genome filename provided for {stock['symbol']}")
                        exit(0)
                    else:
                        try:
                            best_genome = saving.SaveSystem.load_data(os.path.join(self.agents[stock["symbol"]].genome_path, stock["genome_filename"]))
                            self.agents[stock["symbol"]].update_net(best_genome)
                        except FileNotFoundError:
                            print(f" No genome file found for {stock['genome_filename']}")
                else:
                    self.agents[stock["symbol"]].settings = self.settings
                    self.agents[stock["symbol"]].stock = stock

        print(f" Created {', '.join(self.agents.keys())} trading agents\n")

    def get_market_status(self):
        if self.clock[0] is None or time.time() - self.clock[1] > 1:
            tries = 1
            while True:
                try:
                    self.clock[0] = self.alpaca_api.get_clock()
                    self.clock[1] = time.time()
                    return self.clock[0].is_open
                except (ConnectionError, urllib3.exceptions.ProtocolError) as e:
                    self.check_internet_connection()
                    print(f"Error getting clock: '{e}'. Retrying in 5 seconds... ({tries})")
                    time.sleep(5)
                    tries += 1
        return self.clock[0].is_open

    def start(self):
        self.running = True
        self.consecutive_days = 0

        for symbol in self.agents:
            threading.Thread(target=self.agents[symbol].run).start()

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            self.update_profile()
            if self.get_market_status():
                if self.trainer.running:
                    self.trainer.stop()
                    self.training_thread.join()

                    self.finbert.save_news(list(self.agents.keys()), now_date - dt.timedelta(days=30), now_date - dt.timedelta(minutes=16))

                    for symbol in self.agents:
                        trainer_agent = self.trainer.sessions[self.settings["profiles"][0]["name"]]["agents"][symbol]
                        if trainer_agent.best_genome is not None:
                            self.agents[symbol].update_net(trainer_agent.best_genome)

                next_close = self.clock[0].next_close
                wait_time = (next_close - now_date).total_seconds()
                print(f"Market closes in {wait_time / 3600} hours")
                time.sleep(wait_time + 5)
                self.consecutive_days += 1
            else:
                schwab_account = self.schwab_api.get_account()
                if "positions" in schwab_account:
                    positions = schwab_account["positions"]
                else:
                    positions = {}
                total_cash = schwab_account["currentBalances"]["cashBalance"]
                solid_cash = schwab_account["currentBalances"]["cashAvailableForTrading"]
                liquid_cash = total_cash - solid_cash
                bought_shares = {}
                for position in positions:
                    bought_shares[position["instrument"]["symbol"]] = position["longQuantity"]

                if "longMarketValue" in schwab_account["currentBalances"]:
                    market_value = schwab_account["currentBalances"]["longMarketValue"]
                    balance_change = market_value + total_cash - schwab_account["initialBalances"]["accountValue"]
                else:
                    market_value = 0
                    balance_change = 0

                print(f"\n{self.profile['name']} Details:" +
                      f"\n Bal Change: {balance_change}" +
                      f"\n Solid Cash: {solid_cash}" +
                      f"\n Liquid Cash: {liquid_cash}" +
                      f"\n Market Value: {market_value}" +
                      f"\n Bought Shares: {bought_shares}")

                logs_path = os.path.join(self.log_path, f"{self.profile['name']}.gz")
                if os.path.exists(logs_path):
                    previous_logs = saving.SaveSystem.load_data(logs_path)
                else:
                    previous_logs = {}
                for symbol in self.logs:
                    if len(self.logs[symbol]) > 0:
                        if symbol in previous_logs:
                            previous_logs[symbol].extend(self.logs[symbol])
                        else:
                            previous_logs[symbol] = self.logs[symbol]
                        threading.Thread(target=plot.plot_log, args=(self.alpaca_api, symbol, self.logs[symbol], self.profile["interval"])).start()
                        self.logs[symbol].clear()
                saving.SaveSystem.save_data(previous_logs, os.path.join(self.log_path, f"{self.profile['name']}.gz"))

                next_open = self.clock[0].next_open
                wait_time = (next_open - now_date).total_seconds()
                print(f"\nMarket opens in {wait_time / 3600} hours\n-----")
                if not self.trainer.running:
                    if self.training_thread is not None:
                        self.training_thread.join()
                    self.training_thread = threading.Thread(target=self.trainer.start)
                    self.training_thread.start()
                time.sleep(wait_time + 5)


class PaperTrader(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)
        self.trainer = Trainer(settings, finbert)
        self.scraper = cs.Scraper()
        self.training_thread = None
        self.consecutive_days = 0

        for profile in settings["profiles"]:
            alpaca_api = alpaca.REST(profile["public_key"], profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))

            self.sessions[profile["name"]] = {
                "alpaca_api": alpaca_api,
                "solid_cash": 0.0,
                "liquid_cash": 0.0,
                "pending_sales": [],
                "cash_limit": profile["cash_limit"],
                "stocks": profile["stocks"],
                "interval": profile["interval"],
                "agents": {},
                "logs": {},
                "clock": [None, 0],
                "positions": [None, 0],
                "alpaca_api_account": [None, 0]
            }

        self.create_agents()

    def create_agents(self):
        print("Trader: Creating agents")
        for profile_name in self.sessions:
            print(profile_name)
            session = self.sessions[profile_name]
            session["agents"].clear()

            if self.get_market_status(session):
                now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
                symbols = []
                for stock in session["stocks"]:
                    symbols.append(stock["symbol"])
                self.finbert.save_news(symbols, now_date - dt.timedelta(days=30), now_date - dt.timedelta(minutes=16))

            for stock in session["stocks"]:
                session["logs"][stock["symbol"]] = []
                session["agents"][stock["symbol"]] = agent.PaperTrading(self.settings, session, stock, self.finbert, self, self.scraper)
                if stock["genome_filename"] is None:
                    print(f" No genome filename provided for {stock['symbol']}")
                    exit(0)
                else:
                    try:
                        best_genome = saving.SaveSystem.load_data(os.path.join(session["agents"][stock["symbol"]].genome_path, stock["genome_filename"]))
                        session["agents"][stock["symbol"]].update_net(best_genome)
                    except FileNotFoundError:
                        print(f" No genome file found for {stock['genome_filename']}")
            print(f" Created {', '.join(session['agents'].keys())} trading agents\n")

    @staticmethod
    def get_market_status(session):
        if session["clock"][0] is None or time.time() - session["clock"][1] > 1:
            tries = 1
            while True:
                try:
                    session["clock"][0] = session["alpaca_api"].get_clock()
                    session["clock"][1] = time.time()
                    return session["clock"][0].is_open
                except (ConnectionError, urllib3.exceptions.ProtocolError) as e:
                    Manager.check_internet_connection()
                    print(f"Error getting clock: '{e}'. Retrying in 5 seconds... ({tries})")
                    time.sleep(5)
                    tries += 1
        return session["clock"][0].is_open

    @staticmethod
    def get_positions(session):
        if session["positions"][0] is None or time.time() - session["positions"][1] > 1:
            tries = 1
            while True:
                try:
                    session["positions"][0] = session["alpaca_api"].list_positions()
                    session["positions"][1] = time.time()
                    return session["positions"][0]
                except (ConnectionError, urllib3.exceptions.ProtocolError) as e:
                    Manager.check_internet_connection()
                    print(f"Error listing positions: '{e}'. Retrying in 5 seconds... ({tries})")
        return session["positions"][0]

    @staticmethod
    def get_position(symbol, session):
        positions = PaperTrader.get_positions(session)
        for position in positions:
            if position.symbol == symbol:
                return position
        return alpaca.entity.Position(raw={
                "symbol": symbol,
                "qty": "0",
                "avg_entry_price": "0",
                "market_value": "0",
                "cost_basis": "0",
                "unrealized_pl": "0",
                "unrealized_plpc": "0",
              })

    @staticmethod
    def get_api_account(session):
        if session["alpaca_api_account"][0] is None or time.time() - session["alpaca_api_account"][1] > 1:
            tries = 1
            while True:
                try:
                    session["alpaca_api_account"][0] = session["alpaca_api"].get_account()
                    session["alpaca_api_account"][1] = time.time()
                    return session["alpaca_api_account"][0]
                except (ConnectionError, urllib3.exceptions.ProtocolError) as e:
                    Manager.check_internet_connection()
                    print(f"Error getting account: '{e}'. Retrying in 5 seconds... ({tries})")
        return session["alpaca_api_account"][0]

    def start(self):
        self.running = True
        self.consecutive_days = 0
        for profile_name in self.sessions:
            print(profile_name)
            session = self.sessions[profile_name]
            starting_liquid = (float(input(" Enter starting liquid cash: ")), int(input(" Enter pending days: ")))
            #starting_liquid = (0, 0)
            account = self.get_api_account(session)
            session["solid_cash"] = float(account.cash)
            session["liquid_cash"] = 0.0
            session["pending_sales"].clear()

            if starting_liquid[0] > 0:
                session["liquid_cash"] = starting_liquid[0]
                session["solid_cash"] -= session["liquid_cash"]
                session["pending_sales"].append((session["liquid_cash"], starting_liquid[1]))

            for symbol in session["agents"]:
                threading.Thread(target=session["agents"][symbol].run).start()

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            first_profile_name = next(iter(self.sessions))
            if self.get_market_status(self.sessions[first_profile_name]):
                if self.trainer.running:
                    self.trainer.stop()
                    self.training_thread.join()

                    for profile_name in self.sessions:
                        session = self.sessions[profile_name]
                        self.finbert.save_news(list(session["agents"].keys()), now_date - dt.timedelta(days=30), now_date - dt.timedelta(minutes=16))

                        for symbol in session["agents"]:
                            trainer_agent = self.trainer.sessions[profile_name]["agents"][symbol]
                            if trainer_agent.best_genome is not None:
                                session["agents"][symbol].update_net(trainer_agent.best_genome)

                for session in self.sessions.values():
                    for j in reversed(range(len(session["pending_sales"]))):
                        sale = session["pending_sales"][j]
                        if self.consecutive_days - sale[1] > 2:
                            session["solid_cash"] += sale[0]
                            session["liquid_cash"] -= sale[0]
                            session["pending_sales"].pop(j)

                next_close = self.sessions[first_profile_name]["clock"][0].next_close
                wait_time = (next_close - now_date).total_seconds()
                print(f"Market closes in {wait_time / 3600} hours")
                time.sleep(wait_time + 5)
                self.consecutive_days += 1
            else:
                for profile_name in self.sessions:
                    session = self.sessions[profile_name]
                    api_account = self.get_api_account(session)
                    open_positions = self.get_positions(session)
                    bought_shares = {}
                    for position in open_positions:
                        bought_shares[position.symbol] = float(position.qty)
                    balance_change = float(api_account.equity) - float(api_account.last_equity)
                    print(f"\n{profile_name} Details:" +
                          f"\n Daily Bal Change: {balance_change}" +
                          f"\n Solid Cash: {session['solid_cash']}" +
                          f"\n Liquid Cash: {session['liquid_cash']}" +
                          f"\n Equity: {api_account.equity}" +
                          f"\n Bought Shares: {bought_shares}")

                    logs_path = os.path.join(self.log_path, f"{profile_name}.gz")
                    if os.path.exists(logs_path):
                        previous_logs = saving.SaveSystem.load_data(logs_path)
                    else:
                        previous_logs = {}
                    for symbol in session["logs"]:
                        if len(session["logs"][symbol]) > 0:
                            if symbol in previous_logs:
                                previous_logs[symbol].extend(session["logs"]["symbol"])
                            else:
                                previous_logs[symbol] = session["logs"]["symbol"]
                            threading.Thread(target=plot.plot_log, args=(session["alpaca_api"], symbol, session["logs"][symbol], session["interval"])).start()
                            session["logs"][symbol].clear()
                    saving.SaveSystem.save_data(previous_logs, os.path.join(self.log_path, f"{profile_name} (Paper).gz"))
                next_open = self.sessions[first_profile_name]["clock"][0].next_open
                wait_time = (next_open - now_date).total_seconds()
                print(f"\nMarket opens in {wait_time / 3600} hours\n-----")
                if not self.trainer.running:
                    if self.training_thread is not None:
                        self.training_thread.join()
                    self.training_thread = threading.Thread(target=self.trainer.start)
                    self.training_thread.start()
                time.sleep(wait_time + 5)


class Validator(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)

        for profile in settings["profiles"]:
            api = alpaca.REST(profile["public_key"], profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))

            self.sessions[profile["name"]] = {
                "alpaca_api": api,
                "agents": {},
                "stocks": profile["stocks"],
                "interval": profile["interval"],
                "profit_window": profile["profit_window"]
            }

            for stock in profile["stocks"]:
                session = self.sessions[profile["name"]]
                session["agents"][stock["symbol"]] = agent.Validation(settings, session, stock, finbert)

    def start(self):
        self.running = True
        while self.running:
            print("Accounts:")
            i = 1
            ordered_sessions = []
            for profile in self.sessions:
                ordered_sessions.append(self.sessions[profile])
                print(f" {i}: {profile}")
                i += 1
            index = int(input("Enter account index: "))-1
            session = ordered_sessions[index]

            start_date = dt.datetime(year=int(input("Enter start year: ")),
                                   month=int(input("Enter start month: ")),
                                   day=int(input("Enter start day: ")),
                                   hour=16, tzinfo=pytz.timezone("US/Eastern"))
            end_date = dt.datetime(year=int(input("Enter end year: ")),
                                   month=int(input("Enter end month: ")),
                                   day=int(input("Enter end day: ")),
                                   hour=16, tzinfo=pytz.timezone("US/Eastern"))

            self.finbert.save_news(list(session["agents"].keys()), start_date, end_date)
            for stock in session["stocks"]:
                if input(f"Run simulation for {stock['symbol']}? (y/n): ") == "y":
                    if stock["genome_filename"] is None:
                        print(f" No genome filename provided for {stock['symbol']}")
                    else:
                        try:
                            best_genome = saving.SaveSystem.load_data(os.path.join(session["agents"][stock["symbol"]].genome_path, stock["genome_filename"]))
                            bars = self.get_bars(stock["symbol"], session["alpaca_api"], session["interval"], start_date, end_date)
                            print(f"Validating over {len(bars)} bars from {bars[0]['timestamp']} to {bars[-1]['timestamp']}...")
                            session["agents"][stock["symbol"]].validate(bars, best_genome)
                        except FileNotFoundError:
                            print(f" No genome file found for {stock['genome_filename']}")
