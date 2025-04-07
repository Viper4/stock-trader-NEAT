import datetime as dt
import pytz
import os
import saving
from Managers.base_manager import Manager, Profile
from Agents.validation_agent import Validation
from multiprocessing import Pool
from constants import GENOME_DIR, VALIDATION_DIR


class Validator(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)

        self.profiles = []
        for i in range(len(settings["profiles"])):
            self.profiles.append(Profile(settings, i))

            for symbol, stock in self.profiles[i].stocks.items():
                self.profiles[i].agents[symbol] = Validation(settings, self.profiles[i], stock, finbert)

    def start(self):
        self.running = True
        while self.running:
            print("Accounts:")
            i = 1
            for profile in self.profiles:
                print(f" {i}: {profile.name}")
                i += 1
            index = int(input("Enter account index: ")) - 1
            profile = self.profiles[index]

            start = input("Enter start date (YYYY-MM-DD): ")
            end = input("Enter end date (YYYY-MM-DD): ")
            start_date = dt.datetime.strptime(start, "%Y-%m-%d").replace(hour=9, minute=30, tzinfo=pytz.timezone("US/Eastern"))
            end_date = dt.datetime.strptime(end, "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=pytz.timezone("US/Eastern"))

            stock_bars = {}
            genomes = {}
            start_cashes = {}
            for symbol, stock in profile.stocks.items():
                if input(f"Run simulation for {symbol}? (y/n): ") == "y":
                    if stock["genome_filename"] is None:
                        print(f" No genome filename provided for {symbol}")
                    else:
                        try:
                            best_genome = saving.SaveSystem.load_data(os.path.join(GENOME_DIR, stock["genome_filename"]))
                            start_cash = input(" Enter starting cash: ")
                            validation_filename = f"{symbol}-{profile.interval}m-{start_date.isoformat().replace(':', ';')}-{end_date.isoformat().replace(':', ';')}.gz"
                            file_path = VALIDATION_DIR + validation_filename
                            if os.path.exists(file_path):
                                stock_bars[symbol] = self.load_data(symbol, "-V", file_path)
                            else:
                                if "SPY" not in stock_bars:
                                    spy_filename = f"SPY-{profile.interval}m-{start_date.isoformat().replace(':', ';')}-{end_date.isoformat().replace(':', ';')}.gz"
                                    spy_path = VALIDATION_DIR + spy_filename
                                    if os.path.exists(spy_path):
                                        stock_bars["SPY"] = self.load_data("SPY", "-V", spy_path)
                                    else:
                                        stock_bars["SPY"] = self.generate_data("SPY", "-V", profile, start_date, end_date, spy_path)
                                if "QQQ" not in stock_bars:
                                    qqq_filename = f"QQQ-{profile.interval}m-{start_date.isoformat().replace(':', ';')}-{end_date.isoformat().replace(':', ';')}.gz"
                                    qqq_path = VALIDATION_DIR + qqq_filename
                                    if os.path.exists(qqq_path):
                                        stock_bars["QQQ"] = self.load_data("QQQ", "-V", qqq_path)
                                    else:
                                        stock_bars["QQQ"] = self.generate_data("QQQ", "-V", profile, start_date, end_date, qqq_path)

                                stock_bars[symbol] = self.generate_data(symbol, "-V", profile, start_date, end_date, file_path, bars_spy=stock_bars["SPY"], bars_qqq=stock_bars["QQQ"], training=False)
                            genomes[symbol] = best_genome
                            start_cashes[symbol] = start_cash
                        except FileNotFoundError:
                            print(f" No genome file found for {stock['genome_filename']}")

            simulations = len(stock_bars)
            if simulations <= 0:
                print("No simulations selected")
                continue
            print(f"Validating {simulations} simulations...")
            pool = Pool(processes=min(simulations, self.settings["processes"]))
            jobs = []

            for symbol in stock_bars:
                if len(stock_bars[symbol]) == 0:
                    print(f"{symbol}: No bars, skipping...")
                    continue
                if symbol not in profile.agents:
                    print(f"{symbol}: No agent, skipping...")
                    continue
                print(f"{symbol}: Validating over {len(stock_bars[symbol])} bars from {stock_bars[symbol].index[0]} to {stock_bars[symbol].index[-1]}...")
                asset = profile.alpaca_api.get_asset(symbol=symbol)

                columns = {}
                for column in stock_bars[symbol].columns:
                    columns[column] = stock_bars[symbol][column].tolist()
                columns["index"] = stock_bars[symbol].index.tolist()

                jobs.append((symbol, pool.apply_async(profile.agents[symbol].validate,
                                                      (columns,
                                                       profile.ma_periods,
                                                       genomes[symbol],
                                                       asset.fractionable,
                                                       start_cashes[symbol]))))

            for job in jobs:
                symbol, async_result = job
                log = async_result.get()
                while True:
                    user_input = input(f"{symbol}: Enter action index or exit: ")
                    if user_input == "exit":
                        break
                    else:
                        i = int(user_input)
                        if len(log) > i >= 0:
                            print("Action at " + str(i))
                            action = log[i]
                            for key in action:
                                if key == "inputs":
                                    print("-Inputs")
                                    for j in range(len(action[key])):
                                        print(f" |{j}: {action[key][j]}")

                                elif key == "outputs":
                                    print("-Outputs")
                                    print(f" |Buy/Sell: {action[key][0]}")
                                    print(f" |Quantity: {action[key][1]}")
                                else:
                                    print(f"-{key}: {action[key]}")
                        else:
                            print("Index not in range of log")

            pool.close()
            pool.join()
