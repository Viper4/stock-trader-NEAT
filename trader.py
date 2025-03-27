import time
import datetime as dt
import pytz
import os
import saving
import plot
import candle_scraper as cs
import threading
import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
from base_manager import Manager
from trainer import Trainer
from schwab_agent import Trading


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

    def update_profile(self, market_status):
        self.settings, self.alpaca_api = self.get_settings_and_alpaca(0)

        self.profile = self.settings["profiles"][0]
        print(f"Trader: Updating {self.profile['name']} profile")
        if market_status:
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
        sp500_bars, nasdaq_bars, sp500_sentiments, nasdaq_sentiments = self.generate_prep_data(symbols, now_date, self.alpaca_api, self.profile["interval"])

        for stock in self.profile["stocks"]:
            if stock["trading"]:
                if stock["symbol"] not in self.agents:
                    self.logs[stock["symbol"]] = []
                    self.agents[stock["symbol"]] = Trading(self.settings, stock, self)
                    if stock["genome_filename"] is None:
                        print(f" No genome filename provided for {stock['symbol']}")
                        exit(0)
                    else:
                        try:
                            best_genome = saving.SaveSystem.load_data(os.path.join(self.agents[stock["symbol"]].genome_path, stock["genome_filename"]))

                            self.update_net(self.agents[stock["symbol"]], best_genome, self.alpaca_api,
                                            self.profile["interval"], self.profile["name"],
                                            sp500_bars, nasdaq_bars,
                                            sp500_sentiments, nasdaq_sentiments,
                                            self.profile["k_period"], self.profile["d_period"], self.profile["rsi_period"],
                                            30)
                        except FileNotFoundError:
                            print(f" No genome file found for {stock['genome_filename']}")
                else:
                    self.agents[stock["symbol"]].settings = self.settings
                    self.agents[stock["symbol"]].stock = stock

        print(f"Created {', '.join(self.agents.keys())} trading agents\n")
        for symbol in self.agents:
            threading.Thread(target=self.agents[symbol].run).start()

    def get_market_status(self):
        if self.clock[0] is None or time.time() - self.clock[1] > 1:
            tries = 1
            while True:
                try:
                    self.clock[0] = self.alpaca_api.get_clock()
                    self.clock[1] = time.time()
                    return self.clock[0].is_open
                except Exception as e:
                    self.check_internet_connection()
                    print(f"Error getting clock: '{e}'. Retrying in 5 seconds... ({tries})")
                    time.sleep(5)
                    tries += 1
        return self.clock[0].is_open

    def start(self):
        self.running = True
        self.consecutive_days = 0

        while self.running:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            market_status = self.get_market_status()
            self.update_profile(market_status)
            if market_status:
                if self.trainer.running:
                    self.trainer.stop()
                    self.training_thread.join()

                    sp500_bars, nasdaq_bars, sp500_sentiments, nasdaq_sentiments = self.generate_prep_data(list(self.agents.keys()), now_date, self.alpaca_api, self.profile["interval"])

                    for symbol in self.agents:
                        trainer_agent = self.trainer.sessions[self.settings["profiles"][0]["name"]]["agents"][symbol]
                        if trainer_agent.best_genome is not None and self.agents[symbol].genome != trainer_agent.best_genome:
                            self.update_net(self.agents[symbol], trainer_agent.best_genome, self.alpaca_api,
                                            self.profile["interval"], self.profile["name"],
                                            sp500_bars, nasdaq_bars,
                                            sp500_sentiments, nasdaq_sentiments,
                                            self.profile["k_period"], self.profile["d_period"], self.profile["rsi_period"],
                                            30)

                for symbol in self.agents:
                    threading.Thread(target=self.agents[symbol].run).start()
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
                total_cash = schwab_account["currentBalances"]["cashAvailableForTrading"]
                unsettled_cash = schwab_account["currentBalances"]["unsettledCash"]
                settled_cash = total_cash - unsettled_cash
                held_shares = {}
                for position in positions:
                    held_shares[position["instrument"]["symbol"]] = position["longQuantity"]

                if "longMarketValue" in schwab_account["currentBalances"]:
                    market_value = schwab_account["currentBalances"]["longMarketValue"]
                    balance_change = market_value + total_cash - schwab_account["initialBalances"]["accountValue"]
                else:
                    market_value = 0
                    balance_change = 0

                print(f"\n{self.profile['name']} Details:" +
                      f"\n Bal Change: {balance_change}" +
                      f"\n Settled Cash: {settled_cash}" +
                      f"\n Unsettled Cash: {unsettled_cash}" +
                      f"\n Market Value: {market_value}" +
                      f"\n Held Shares: {held_shares}")

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
                saving.SaveSystem.save_data(previous_logs, os.path.join(self.log_path, f"{self.profile['name']}.gz"))
                for symbol in self.logs:
                    self.logs[symbol].clear()
                next_open = self.clock[0].next_open
                wait_time = (next_open - now_date).total_seconds()
                print(f"\nMarket opens in {wait_time / 3600} hours\n-----")
                if not self.trainer.running:
                    if self.training_thread is not None:
                        self.training_thread.join()
                    self.training_thread = threading.Thread(target=self.trainer.start)
                    self.training_thread.start()
                time.sleep(wait_time + 5)