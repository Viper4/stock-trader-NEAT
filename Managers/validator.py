import datetime as dt
import pytz
import os
import saving
from Managers.base_manager import Manager, Profile
from Agents.validation_agent import Validation
from multiprocessing import Pool


class Validator(Manager):
    def __init__(self, settings, finbert):
        super().__init__(settings, finbert)

        self.profiles = []
        for i in range(len(settings["profiles"])):
            profile = settings["profiles"][i]

            self.profiles.append(Profile(settings, i))

            for stock in profile["stocks"]:
                profile.agents[stock["symbol"]] = Validation(settings, profile, stock, finbert)

    def start(self):
        self.running = True
        while self.running:
            print("Accounts:")
            i = 1
            for profile in self.profiles:
                print(f" {i}: {profile}")
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
            for stock in profile.stocks:
                if input(f"Run simulation for {stock['symbol']}? (y/n): ") == "y":
                    if stock["genome_filename"] is None:
                        print(f" No genome filename provided for {stock['symbol']}")
                    else:
                        try:
                            best_genome = saving.SaveSystem.load_data(os.path.join(profile.agents[stock["symbol"]].genome_path, stock["genome_filename"]))
                            start_cash = input(" Enter starting cash: ")
                            validation_filename = f"{stock['symbol']}-{profile.interval}m-{start_date.isoformat().replace(':', ';')}-{end_date.isoformat().replace(':', ';')}.gz"
                            file_path = f"{self.settings['save_path']}\\ValidationData\\{validation_filename}"
                            if os.path.exists(file_path):
                                stock_bars[stock["symbol"]] = self.load_data(stock["symbol"], "-V", file_path)
                            else:
                                if "SPY" not in stock_bars:
                                    spy_filename = f"SPY-{profile.interval}m-{start_date.isoformat().replace(':', ';')}-{end_date.isoformat().replace(':', ';')}.gz"
                                    spy_path = os.path.join(self.settings["save_path"], "ValidationData", spy_filename)
                                    if os.path.exists(spy_path):
                                        stock_bars["SPY"] = self.load_data("SPY", "-V", spy_path)
                                    else:
                                        stock_bars["SPY"] = self.generate_data("SPY", "-V", profile, start_date, end_date, spy_path)
                                if "QQQ" not in stock_bars:
                                    qqq_filename = f"QQQ-{profile.interval}m-{start_date.isoformat().replace(':', ';')}-{end_date.isoformat().replace(':', ';')}.gz"
                                    qqq_path = os.path.join(self.settings["save_path"], "ValidationData", qqq_filename)
                                    if os.path.exists(qqq_path):
                                        stock_bars["QQQ"] = self.load_data("QQQ", "-V", qqq_path)
                                    else:
                                        stock_bars["QQQ"] = self.generate_data("QQQ", "-V", profile, start_date, end_date, qqq_path)

                                stock_bars[stock["symbol"]] = self.generate_data(stock["symbol"], "-V", profile, start_date, end_date, file_path, stock_bars["SPY"], stock_bars["QQQ"], False)
                            genomes[stock["symbol"]] = best_genome
                            start_cashes[stock["symbol"]] = start_cash
                        except FileNotFoundError:
                            print(f" No genome file found for {stock['genome_filename']}")

            simulations = len(stock_bars) - 2  # SPY and QQQ
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
                print(f"{symbol}: Validating over {len(stock_bars[symbol])} bars from {stock_bars[symbol][0]['timestamp']} to {stock_bars[symbol][-1]['timestamp']}...")
                asset = profile.alpaca_api.get_asset(symbol=symbol)
                jobs.append((symbol, pool.apply_async(profile.agents[symbol].validate,
                                                      (stock_bars[symbol],
                                                       profile.sma_periods,
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
                                    print(f" |Short/Long: {action[key][0]}")
                                    print(f" |PLPC: {action[key][1]}")
                                    print(f" |Open: {action[key][2]}")
                                    print(f" |High: {action[key][3]}")
                                    print(f" |Low: {action[key][4]}")
                                    print(f" |Close: {action[key][5]}")
                                    print(f" |Volume: {action[key][6]}")
                                    print(f" |VWAP: {action[key][7]}")
                                    print(f" |{symbol} Sentiment: {action[key][8]}")
                                    print(f" |S&P 500 Close: {action[key][9]}")
                                    print(f" |S&P 500 Volume: {action[key][10]}")
                                    print(f" |S&P 500 Sentiment: {action[key][11]}")
                                    print(f" |NASDAQ Close: {action[key][12]}")
                                    print(f" |NASDAQ Volume: {action[key][13]}")
                                    print(f" |NASDAQ Sentiment: {action[key][14]}")
                                    print(f" |Slow K: {action[key][15]}")
                                    print(f" |Slow D: {action[key][16]}")
                                    print(f" |{profile.k_period}-day EMA: {action[key][17]}")
                                    print(f" |{profile.d_period}-day EMA: {action[key][18]}")
                                    print(f" |{profile.rsi_period}-day RSI: {action[key][19]}")
                                    print(f" |{profile.atr_period}-day ATR: {action[key][20]}")
                                    for i in range(len(profile.sma_periods)):
                                        print(f" |{profile.sma_periods[i]}-day SMA: {action[key][21 + i]}")

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
