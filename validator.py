import datetime as dt
import pytz
import os
import saving
import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
from base_manager import Manager
from validation_agent import Validation
from multiprocessing import Pool


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
                "profit_window": profile["profit_window"],
                "short_limit": profile["short_limit"],
                "k_period": profile["k_period"],
                "d_period": profile["d_period"],
                "rsi_period": profile["rsi_period"]
            }

            for stock in profile["stocks"]:
                session = self.sessions[profile["name"]]
                session["agents"][stock["symbol"]] = Validation(settings, session, stock, finbert)

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

            stock_bars = {}
            genomes = {}
            start_cashes = {}
            shorting = {}
            for stock in session["stocks"]:
                if input(f"Run simulation for {stock['symbol']}? (y/n): ") == "y":
                    if stock["genome_filename"] is None:
                        print(f" No genome filename provided for {stock['symbol']}")
                    else:
                        try:
                            best_genome = saving.SaveSystem.load_data(os.path.join(session["agents"][stock["symbol"]].genome_path, stock["genome_filename"]))
                            start_cash = input(" Enter starting cash: ")
                            stock_bars[stock["symbol"]] = self.get_bars(stock["symbol"],
                                                                        session["alpaca_api"],
                                                                        session["interval"],
                                                                        start_date, end_date,
                                                                        500000, False)
                            genomes[stock["symbol"]] = best_genome
                            start_cashes[stock["symbol"]] = start_cash
                            shorting[stock["symbol"]] = stock["shorting"]
                        except FileNotFoundError:
                            print(f" No genome file found for {stock['genome_filename']}")

            simulations = len(stock_bars)
            if simulations == 0:
                print("No simulations selected")
                continue
            print(f"Validating {simulations} simulations...")
            pool = Pool(processes=min(simulations, self.settings["processes"]))
            jobs = []

            self.finbert.save_news(list(session["agents"].keys()), start_date, end_date)
            sp500_bars = self.get_bars("SPY", session["alpaca_api"], session["interval"], start_date, end_date, 500000, False)
            nasdaq_bars = self.get_bars("QQQ", session["alpaca_api"], session["interval"], start_date, end_date, 500000, False)

            for symbol in stock_bars:
                print(f"{symbol}: Validating over {len(stock_bars[symbol])} bars from {stock_bars[symbol][0]['timestamp']} to {stock_bars[symbol][-1]['timestamp']}...")
                asset = session["alpaca_api"].get_asset(symbol=symbol)
                jobs.append((symbol, pool.apply_async(session["agents"][symbol].validate,
                                 (stock_bars[symbol],
                                  sp500_bars, nasdaq_bars,
                                  genomes[symbol],
                                  shorting[symbol] and asset.shortable,
                                  asset.fractionable,
                                  session["short_limit"],
                                  session["k_period"], session["d_period"], session["rsi_period"],
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
                                    print(f" |%K: {action[key][15]}")
                                    print(f" |{session['d_period']}-day EMA: {action[key][16]}")
                                    print(f" |{session['k_period']}-day EMA: {action[key][17]}")
                                    print(f" |{session['rsi_period']}-day RSI: {action[key][18]}")
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
