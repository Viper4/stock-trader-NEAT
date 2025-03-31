import time
import datetime as dt
import pytz
import os
import saving
import plot
import candle_scraper as cs
import threading
from Managers.base_manager import Manager, Profile
from Managers.trainer import Trainer
from Agents.schwab_agent import Trading
from constants import LOG_DIR, GENOME_DIR


class Trader(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)
        self.schwab_api = settings["schwab"]
        self.trainer = Trainer(settings, finbert)
        self.scraper = cs.Scraper()
        self.training_thread = None
        self.consecutive_days = 0
        self.profile = Profile(settings, 0)
        self.clock = [None, 0]

    def create_agents(self):
        print("Trader: Creating agents")

        symbols = []
        for stock in self.profile.stocks:
            if stock["trading"]:
                symbols.append(stock["symbol"])

        for stock in self.profile.stocks:
            if stock["trading"]:
                if stock["symbol"] not in self.profile.agents:
                    self.profile.logs[stock["symbol"]] = []
                    self.profile.agents[stock["symbol"]] = Trading(self.settings, stock, self)

                    if stock["genome_filename"] is None:
                        print(f"No genome filename provided for {stock['symbol']}")
                        exit(0)
                    else:
                        try:
                            best_genome = saving.SaveSystem.load_data(os.path.join(GENOME_DIR, stock["genome_filename"]))
                            self.profile.agents[stock["symbol"]].genome = best_genome
                        except FileNotFoundError:
                            print(f" No genome file found for {stock['genome_filename']}")
                else:
                    self.profile.agents[stock["symbol"]].profile = self.profile
                    self.profile.agents[stock["symbol"]].settings = self.settings
                    self.profile.agents[stock["symbol"]].stock = stock

        print(f"Created {', '.join(self.profile.agents.keys())} trading agents\n")
        for symbol in self.profile.agents:
            threading.Thread(target=self.profile.agents[symbol].run).start()

    def get_market_status(self):
        if self.clock[0] is None or time.time() - self.clock[1] > 1:
            tries = 1
            while True:
                try:
                    self.clock[0] = self.profile.alpaca_api.get_clock()
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
            self.profile.update()
            if market_status:
                # Update agents with profile
                for stock in self.profile.stocks:
                    if stock["trading"]:
                        if stock["symbol"] not in self.profile.agents:
                            self.create_agents()
                            break
                        else:
                            self.profile.agents[stock["symbol"]].profile = self.profile
                            self.profile.agents[stock["symbol"]].settings = self.settings
                            self.profile.agents[stock["symbol"]].stock = stock

                # Stop training
                if self.trainer.running:
                    self.trainer.stop()
                    self.training_thread.join()

                    for symbol in self.profile.agents:
                        trainer_agent = self.trainer.profiles[self.profile.index].agents[symbol]
                        if trainer_agent.best_genome is not None and self.profile.agents[symbol].genome != trainer_agent.best_genome:
                            self.profile.agents[symbol].genome = trainer_agent.best_genome

                # Update agents with previous 30 days
                start_date = now_date - dt.timedelta(days=30)
                end_date = now_date - dt.timedelta(minutes=16)
                spy_bars = None
                qqq_bars = None
                for symbol in self.profile.agents:
                    memory = self.load_memory(f"{self.profile.name.replace(' ', '-')}-{symbol}")
                    if memory is None:
                        if spy_bars is None:
                            spy_bars = self.generate_data("SPY", "-T", self.profile, start_date, end_date, None,
                                                          False, None, None, False)
                        if qqq_bars is None:
                            qqq_bars = self.generate_data("QQQ", "-T", self.profile, start_date, end_date, None,
                                                          False, None, None, False)

                        bars = self.generate_data(symbol, "-T", self.profile, start_date, end_date, None, True, spy_bars, qqq_bars, False)

                        self.profile.agents[symbol].update_net(bars, 30)
                        self.save_memory(self.profile.agents[symbol].net, f"{self.profile.name.replace(' ', '-')}-{symbol}")
                    else:
                        self.profile.agents[symbol].create_net(memory[0], memory[1])

                for symbol in self.profile.agents:
                    threading.Thread(target=self.profile.agents[symbol].run).start()
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

                log_path = os.path.join(LOG_DIR, f"{self.profile.name}.gz")
                if os.path.exists(log_path):
                    previous_logs = saving.SaveSystem.load_data(log_path)
                else:
                    previous_logs = {}
                for symbol in self.profile.logs:
                    if len(self.profile.logs[symbol]) > 0:
                        if symbol in previous_logs:
                            previous_logs[symbol].extend(self.profile.logs[symbol])
                        else:
                            previous_logs[symbol] = self.profile.logs[symbol]
                        threading.Thread(target=plot.plot_log, args=(self.profile.alpaca_api, symbol, self.profile.logs[symbol], self.profile.interval)).start()
                saving.SaveSystem.save_data(previous_logs, os.path.join(LOG_DIR, f"{self.profile.name}.gz"))
                for symbol in self.profile.logs:
                    self.profile.logs[symbol].clear()
                next_open = self.clock[0].next_open
                wait_time = (next_open - now_date).total_seconds()
                print(f"\nMarket opens in {wait_time / 3600} hours\n-----")
                if not self.trainer.running:
                    if self.training_thread is not None:
                        self.training_thread.join()
                    self.training_thread = threading.Thread(target=self.trainer.start)
                    self.training_thread.start()
                time.sleep(wait_time + 5)
