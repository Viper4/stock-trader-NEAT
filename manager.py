import agent
import time
import datetime as dt
import pytz
import os

import finbert_news
import saving
import plot
import candle_scraper as cs
import threading
import alpaca_trade_api as alpaca
import alpaca_trade_api.entity
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit
import urllib3


class Manager(object):
    def __init__(self, settings, finbert):
        self.running = False
        self.settings = settings
        self.finbert = finbert
        self.sessions = {}

    @staticmethod
    def get_bars(symbol, session, start, end):
        tries = 1
        while True:
            try:
                bars_df = session["api"].get_bars(
                    symbol=symbol,
                    timeframe=TimeFrame(session["interval"], TimeFrameUnit.Minute),
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=500000,
                    sort="asc",
                    adjustment="all").df.tz_convert("US/Eastern")
                bars_df = bars_df.between_time("9:30", "16:00")
                return bars_df.reset_index().to_dict("records")
            except ConnectionError as e:
                print(f"Error getting bars: '{e}'. Retrying in 5 seconds... ({tries})")
                tries += 1
                time.sleep(5)


class Trainer(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)
        self.cycles = 0

        self.training_path = self.settings["save_path"] + "/TrainingData"
        if not os.path.exists(self.training_path):
            os.mkdir(self.training_path)

        self.symbols = []
        self.largest_backtest = 0
        self.one_agent = False

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
                "stocks": account["stocks"],
                "backtest_days": account["backtest_days"],
                "interval": account["interval"],
                "profit_window": account["profit_window"]
            }

            for stock in account["stocks"]:
                if stock["symbol"] not in self.symbols:
                    self.symbols.append(stock["symbol"])

            if self.largest_backtest < account["backtest_days"]:
                self.largest_backtest = account["backtest_days"]

        self.create_agents()

    def generate_data(self, symbol, session, earliest_date, start_date, end_date, file_path):
        if len(self.finbert.saved_news) == 0:
            self.finbert.save_news(self.symbols, earliest_date, end_date)

        bars = self.get_bars(symbol, session, start_date, end_date)

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

    def create_agents(self, regenerate=False):
        print("Trainer: Creating agents")
        now_date = dt.datetime.now(dt.timezone.utc)
        earliest_date = now_date - dt.timedelta(days=self.largest_backtest)
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc

        for session_key in self.sessions:
            print(session_key)
            session = self.sessions[session_key]
            start_date = now_date - dt.timedelta(days=session["backtest_days"])

            session["agents"].clear()
            for stock in session["stocks"]:
                if stock["training_filename"] is None:
                    print(" No training data filename provided for " + stock["symbol"])
                    exit(0)

                training_file_path = os.path.join(self.training_path, stock["training_filename"])

                if not regenerate and os.path.exists(training_file_path):
                    backtest_start, backtest_end, sentiments = saving.SaveSystem.load_data(training_file_path)
                    bars = self.get_bars(stock["symbol"], session, backtest_start, backtest_end)
                    if len(bars) != len(sentiments):
                        print(f" {stock['symbol']}: Loaded {len(bars)} bars but have {len(sentiments)} sentiments. Regenerating training data")
                        bars, sentiments = self.generate_data(stock["symbol"], session, earliest_date, start_date, end_date, training_file_path)
                    else:
                        print(f" {stock['symbol']}: Loaded {len(bars)} bars and sentiments from {bars[0]['timestamp']} to {bars[-1]['timestamp']}")
                else:
                    bars, sentiments = self.generate_data(stock["symbol"], session, earliest_date, start_date, end_date, training_file_path)

                session["agents"][stock["symbol"]] = agent.Training(self.settings, session, stock, bars, sentiments)

        print("Trainer: Created {0} training agents\n".format(self.symbols))

    def start(self):
        print("Starting training...")
        self.running = True
        print(self.cycles)
        if self.cycles >= self.settings["training_reset"]:
            self.cycles = 0
            self.create_agents(True)

        if self.one_agent:
            first_session = self.sessions[next(iter(self.sessions))]
            first_session["agents"][next(iter(first_session["agents"]))].run()
        else:
            while self.running:
                for session_key in self.sessions:
                    session = self.sessions[session_key]
                    for symbol in session["agents"]:
                        current_agent = session["agents"][symbol]
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
        for session_key in self.sessions:
            session = self.sessions[session_key]
            for symbol in session["agents"]:
                session["agents"][symbol].running = False


class Trader(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)
        self.trainer = Trainer(settings, finbert)
        self.scraper = cs.Scraper()
        self.training_thread = None
        self.consecutive_days = 0

        for account in settings["accounts"]:
            base_url = URL("https://paper-api.alpaca.markets") if account["paper"] else URL("https://api.alpaca.markets")
            api = alpaca.REST(account["public_key"], account["secret_key"], base_url=base_url)

            self.sessions[account["name"]] = {
                "api": api,
                "solid_cash": 0.0,
                "liquid_cash": 0.0,
                "pending_sales": [],
                "cash_limit": account["cash_limit"],
                "stocks": account["stocks"],
                "interval": account["interval"],
                "agents": {},
                "logs": {},
                "clock": [None, 0],
                "positions": [None, 0],
                "api_account": [None, 0]
            }

            '''if input(f"Plot {account['name']} logs? (y/n): ") == "y":
                logs = {}
                log_path = settings["save_path"] + "/Logs"
                for filename in os.listdir(log_path):
                    if account["name"] in filename:
                        filepath = os.path.join(log_path, filename)
                        file_logs, balance_change, bought_shares = saving.SaveSystem.load_data(filepath)
                        print(f" {filename}\n Bal Change: {balance_change}\n Bought Shares: {bought_shares}\n")
                        for symbol in file_logs:
                            if symbol in logs:
                                logs[symbol].extend(file_logs[symbol])
                            else:
                                logs[symbol] = file_logs[symbol]
                for symbol in logs:
                    if len(logs[symbol]) > 0:
                        logs[symbol].sort(key=lambda x: x["datetime"])
                        plot.plot_log(api, symbol, logs[symbol], account["interval"])'''
        self.create_agents()

    def create_agents(self):
        print("Trader: Creating agents")
        for session_key in self.sessions:
            print(session_key)
            session = self.sessions[session_key]
            session["agents"].clear()

            if self.get_market_status(session):
                now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
                symbols = []
                for stock in session["stocks"]:
                    symbols.append(stock["symbol"])
                self.finbert.save_news(symbols, now_date - dt.timedelta(days=30), now_date - dt.timedelta(minutes=16))

            for stock in session["stocks"]:
                session["logs"][stock["symbol"]] = []
                session["agents"][stock["symbol"]] = agent.Trading(self.settings, session, stock, self.finbert, self, self.scraper)
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
                    session["clock"][0] = session["api"].get_clock()
                    session["clock"][1] = time.time()
                    return session["clock"][0].is_open
                except (ConnectionError, urllib3.exceptions.ProtocolError) as e:
                    print(f"Error getting clock: '{e}'. Retrying in 5 seconds... ({tries})")
                    time.sleep(5)
                    tries += 1
        return session["clock"][0].is_open

    @staticmethod
    def list_positions(session):
        if session["positions"][0] is None or time.time() - session["positions"][1] > 1:
            tries = 1
            while True:
                try:
                    session["positions"][0] = session["api"].list_positions()
                    session["positions"][1] = time.time()
                    return session["positions"][0]
                except ConnectionError as e:
                    print(f"Error listing positions: '{e}'. Retrying in 5 seconds... ({tries})")
        return session["positions"][0]

    @staticmethod
    def get_position(symbol, session):
        positions = Trader.list_positions(session)
        for position in positions:
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

    @staticmethod
    def get_api_account(session):
        if session["api_account"][0] is None or time.time() - session["api_account"][1] > 1:
            tries = 1
            while True:
                try:
                    session["api_account"][0] = session["api"].get_account()
                    session["api_account"][1] = time.time()
                    return session["api_account"][0]
                except ConnectionError as e:
                    print(f"Error getting account: '{e}'. Retrying in 5 seconds... ({tries})")
        return session["api_account"][0]

    def start(self):
        self.running = True
        self.consecutive_days = 0
        for session_key in self.sessions:
            print(session_key)
            session = self.sessions[session_key]
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
            first_session_key = next(iter(self.sessions))
            if self.get_market_status(self.sessions[first_session_key]):
                if self.trainer.running:
                    self.trainer.stop()
                    self.training_thread.join()

                    for session_key in self.sessions:
                        session = self.sessions[session_key]
                        self.finbert.save_news(list(session["agents"].keys()), now_date - dt.timedelta(days=30), now_date - dt.timedelta(minutes=16))

                        for symbol in session["agents"]:
                            trainer_agent = self.trainer.sessions[session_key]["agents"][symbol]
                            if trainer_agent.best_genome is not None:
                                session["agents"][symbol].update_net(trainer_agent.best_genome)

                for session in self.sessions.values():
                    for j in reversed(range(len(session["pending_sales"]))):
                        sale = session["pending_sales"][j]
                        if self.consecutive_days - sale[1] > 2:
                            session["solid_cash"] += sale[0]
                            session["liquid_cash"] -= sale[0]
                            session["pending_sales"].pop(j)

                next_close = self.sessions[first_session_key]["clock"][0].next_close
                wait_time = (next_close - now_date).total_seconds()
                print(f"Market closes in {wait_time / 3600} hours")
                time.sleep(wait_time + 5)
                self.consecutive_days += 1
            else:
                for session_key in self.sessions:
                    session = self.sessions[session_key]
                    api_account = self.get_api_account(session)
                    open_positions = self.list_positions(session)
                    bought_shares = {}
                    for position in open_positions:
                        bought_shares[position.symbol] = float(position.qty)
                    balance_change = float(api_account.equity) - float(api_account.last_equity)
                    print(f"\n{session_key} Details:" +
                          f"\n Daily Bal Change: {balance_change}" +
                          f"\n Solid Cash: {session['solid_cash']}" +
                          f"\n Liquid Cash: {session['liquid_cash']}" +
                          f"\n Equity: {api_account.equity}" +
                          f"\n Bought Shares: {bought_shares}")

                    saved_log = False
                    for symbol in session["logs"]:
                        if len(session["logs"][symbol]) > 0:
                            if not saved_log:
                                saving.SaveSystem.save_data((session["logs"], balance_change, bought_shares), os.path.join(session["agents"][symbol].log_path, f"{session_key}_{now_date.astimezone(tz=pytz.timezone('US/Central')).strftime('%Y-%m-%d')}.gz"))
                                saved_log = True
                            threading.Thread(target=plot.plot_log, args=(session["api"], symbol, session["logs"][symbol], session["interval"])).start()
                            session["logs"][symbol].clear()

                next_open = self.sessions[first_session_key]["clock"][0].next_open
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

        for account in settings["accounts"]:
            base_url = URL("https://paper-api.alpaca.markets") if account["paper"] else URL("https://api.alpaca.markets")
            api = alpaca.REST(account["public_key"], account["secret_key"], base_url=base_url)

            self.sessions[account["name"]] = {
                "api": api,
                "agents": {},
                "stocks": account["stocks"],
                "interval": account["interval"],
                "profit_window": account["profit_window"]
            }

            for stock in account["stocks"]:
                session = self.sessions[account["name"]]
                session["agents"][stock["symbol"]] = agent.Validation(settings, session, stock, finbert)

    def start(self):
        self.running = True
        while self.running:
            print("Accounts:")
            i = 1
            ordered_sessions = []
            for account in self.sessions:
                ordered_sessions.append(self.sessions[account])
                print(f" {i}: {account}")
                i += 1
            index = int(input("Enter account index: "))-1
            session = ordered_sessions[index]

            sim_years = float(input("Enter simulation years: "))
            end_date = dt.datetime(year=int(input("Enter end year: ")),
                                   month=int(input("Enter end month: ")),
                                   day=int(input("Enter end day: ")),
                                   hour=16, tzinfo=pytz.timezone("US/Eastern"))
            start_date = end_date - dt.timedelta(days=sim_years*356)

            self.finbert.save_news(list(session["agents"].keys()), start_date, end_date)
            for stock in session["stocks"]:
                if input(f"Run simulation for {stock['symbol']}? (y/n): ") == "y":
                    if stock["genome_filename"] is None:
                        print(f" No genome filename provided for {stock['symbol']}")
                    else:
                        try:
                            best_genome = saving.SaveSystem.load_data(os.path.join(session["agents"][stock["symbol"]].genome_path, stock["genome_filename"]))
                            bars = self.get_bars(stock["symbol"], session, start_date, end_date)
                            print(f"Validating over {len(bars)} bars from {bars[0]['timestamp']} to {bars[-1]['timestamp']}...")
                            session["agents"][stock["symbol"]].validate(bars, best_genome)
                        except FileNotFoundError:
                            print(f" No genome file found for {stock['genome_filename']}")
