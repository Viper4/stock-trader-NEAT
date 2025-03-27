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
from alpaca_trade_api.rest import URL
from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3.exceptions import ProtocolError
from base_manager import Manager
from trainer import Trainer
from paper_agent import PaperTrading
from data_structures import Queue


class PaperTrader(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)
        self.scraper = cs.Scraper()
        self.training_thread = None
        self.consecutive_days = 0

        for profile in settings["profiles"]:
            alpaca_api = alpaca.REST(profile["public_key"], profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))

            self.sessions[profile["name"]] = {
                "alpaca_api": alpaca_api,
                "settled_cash": 0.0,
                "unsettled_cash": 0.0,
                "pending_sales": Queue(),
                "cash_limit": profile["cash_limit"],
                "short_limit": profile["short_limit"],
                "stocks": profile["stocks"],
                "interval": profile["interval"],
                "k_period": profile["k_period"],
                "d_period": profile["d_period"],
                "rsi_period": profile["rsi_period"],
                "agents": {},
                "logs": {},
                "clock": [None, 0],
                "positions": [None, 0],
                "alpaca_api_account": [None, 0]
            }

            print(profile["name"])
            starting_unsettled = (float(input(" Enter starting unsettled cash: ")), int(input(" Enter pending days: ")))
            account = self.get_api_account(self.sessions[profile["name"]])
            self.sessions[profile["name"]]["settled_cash"] = float(account.cash)
            self.sessions[profile["name"]]["unsettled_cash"] = 0.0

            if starting_unsettled[0] > 0:
                self.sessions[profile["name"]]["unsettled_cash"] = starting_unsettled[0]
                self.sessions[profile["name"]]["settled_cash"] -= self.sessions[profile["name"]]["unsettled_cash"]
                self.sessions[profile["name"]]["pending_sales"].enqueue((self.sessions[profile["name"]]["unsettled_cash"], starting_unsettled[1]))

        self.trainer = Trainer(settings, finbert)

        self.create_agents()

    def create_agents(self):
        print("Paper Trader: Creating agents")
        for profile_name in self.sessions:
            session = self.sessions[profile_name]
            session["agents"].clear()

            symbols = []
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            for stock in session["stocks"]:
                symbols.append(stock["symbol"])

            sp500_bars, nasdaq_bars, sp500_sentiments, nasdaq_sentiments = self.generate_prep_data(symbols, now_date, session["alpaca_api"], session["interval"])

            for stock in session["stocks"]:
                session["logs"][stock["symbol"]] = []
                session["agents"][stock["symbol"]] = PaperTrading(self.settings, session, stock, self.finbert, self, self.scraper)
                if stock["genome_filename"] is None:
                    print(f" No genome filename provided for {stock['symbol']}")
                    exit(0)
                else:
                    try:
                        best_genome = saving.SaveSystem.load_data(os.path.join(session["agents"][stock["symbol"]].genome_path, stock["genome_filename"]))
                        self.update_net(session["agents"][stock["symbol"]], best_genome, session["alpaca_api"],
                                        session["interval"], profile_name,
                                        sp500_bars, nasdaq_bars,
                                        sp500_sentiments, nasdaq_sentiments,
                                        session["k_period"], session["d_period"], session["rsi_period"],
                                        30)
                    except FileNotFoundError:
                        print(f" No genome file found for {stock['genome_filename']}")
            print(f" Created {', '.join(session['agents'].keys())} paper trading agents\n")

            for symbol in session['agents']:
                threading.Thread(target=session['agents'][symbol].run).start()

    @staticmethod
    def get_market_status(session):
        if session["clock"][0] is None or time.time() - session["clock"][1] > 1:
            tries = 1
            while True:
                try:
                    session["clock"][0] = session["alpaca_api"].get_clock()
                    session["clock"][1] = time.time()
                    return session["clock"][0].is_open
                except (RequestsConnectionError, ProtocolError) as e:
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
                except (RequestsConnectionError, ProtocolError) as e:
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
                except (RequestsConnectionError, ProtocolError) as e:
                    Manager.check_internet_connection()
                    print(f"Error getting account: '{e}'. Retrying in 5 seconds... ({tries})")
        return session["alpaca_api_account"][0]

    def start(self):
        self.running = True
        self.consecutive_days = 0

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            first_profile_name = next(iter(self.sessions))
            if self.get_market_status(self.sessions[first_profile_name]):
                if self.trainer.running:
                    self.trainer.stop()
                    self.training_thread.join()

                    for profile_name in self.sessions:
                        session = self.sessions[profile_name]
                        sp500_bars, nasdaq_bars, sp500_sentiments, nasdaq_sentiments = self.generate_prep_data(list(session["agents"].keys()), now_date, session["alpaca_api"], session["interval"])

                        for symbol in session["agents"]:
                            trainer_agent = self.trainer.sessions[profile_name]["agents"][symbol]
                            if trainer_agent.best_genome is not None and session["agents"][symbol].genome != trainer_agent.best_genome:
                                self.update_net(session["agents"][symbol], trainer_agent.best_genome, session["alpaca_api"],
                                                session["interval"], profile_name,
                                                sp500_bars, nasdaq_bars,
                                                sp500_sentiments, nasdaq_sentiments,
                                                session["k_period"], session["d_period"], session["rsi_period"],
                                                30)

                for session in self.sessions.values():
                    while not session["pending_sales"].is_empty():
                        sale_price, sale_day = session["pending_sales"].head.value
                        if self.consecutive_days - sale_day > 1:
                            session["settled_cash"] += sale_price
                            session["unsettled_cash"] -= sale_price
                            session["pending_sales"].dequeue()
                        else:
                            break
                    for symbol in session["agents"]:
                        threading.Thread(target=session["agents"][symbol].run).start()

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
                    held_shares = {}
                    for position in open_positions:
                        held_shares[position.symbol] = float(position.qty)
                    balance_change = float(api_account.equity) - float(api_account.last_equity)
                    print(f"\n{profile_name} Details:" +
                          f"\n Daily Bal Change: {balance_change}" +
                          f"\n Settled Cash: {session['settled_cash']}" +
                          f"\n Unsettled Cash: {session['unsettled_cash']}" +
                          f"\n Equity: {api_account.equity}" +
                          f"\n Held Shares: {held_shares}")

                    logs_path = os.path.join(self.log_path, f"{profile_name}.gz")
                    if os.path.exists(logs_path):
                        previous_logs = saving.SaveSystem.load_data(logs_path)
                    else:
                        previous_logs = {}
                    for symbol in session["logs"]:
                        if len(session["logs"][symbol]) > 0:
                            if symbol in previous_logs:
                                previous_logs[symbol].extend(session["logs"][symbol])
                            else:
                                previous_logs[symbol] = session["logs"][symbol]
                            threading.Thread(target=plot.plot_log, args=(session["alpaca_api"], symbol, session["logs"][symbol], session["interval"])).start()
                    saving.SaveSystem.save_data(previous_logs, os.path.join(self.log_path, f"{profile_name} (Paper).gz"))
                    for symbol in session["logs"]:
                        session["logs"][symbol].clear()
                next_open = self.sessions[first_profile_name]["clock"][0].next_open
                wait_time = (next_open - now_date).total_seconds()
                print(f"\nMarket opens in {wait_time / 3600} hours\n-----")
                if not self.trainer.running:
                    if self.training_thread is not None:
                        self.training_thread.join()
                    self.training_thread = threading.Thread(target=self.trainer.start)
                    self.training_thread.start()
                time.sleep(wait_time + 5)