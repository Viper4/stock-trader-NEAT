from curl_cffi import requests
import random
import datetime as dt
import os
import time
import pandas as pd

SAVE_PATH = r"C:\Users\vpr16\PythonProjects\StockTraderNEAT\Saves\WebScraping"


class Scraper(object):
    def __init__(self):
        self.user_agents = self.load_user_agents()
        print(f"Scraper initialized with {len(self.user_agents)} user agents.")

    def get_latest_candles(self, symbol, interval="5m"):
        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br"
        }
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range=1d"

        tries = 1
        while True:
            try:
                response = requests.get(url, headers=headers, impersonate="chrome")

                if response.status_code == 200 or response.status_code == 201:
                    data = response.json()

                    df = pd.DataFrame(data=data["chart"]["result"][0]["indicators"]["quote"][0],
                                      index=data["chart"]["result"][0]["timestamp"])

                    prev_close = data["chart"]["result"][0]["meta"]["chartPreviousClose"]  # Yesterday's close

                    if df.empty:
                        print(f"Received no candles for {symbol} retrying in 5 seconds... ({tries})")
                        time.sleep(5)
                        tries += 1
                    else:
                        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
                        df["cum_volume"] = df["volume"].cumsum()
                        df["cum_typical_price"] = (df["typical_price"] * df["volume"]).cumsum()
                        df["vwap"] = df["cum_typical_price"] / df["cum_volume"]

                        return df, prev_close
                else:
                    print(f"{response.status_code} error code fetching candles for {symbol}: {response.content}\nRetrying in 5 seconds... ({tries})")
                    time.sleep(5)
                    tries += 1
            except Exception as e:
                print(f"Error fetching candles for {symbol}. Retrying in 5 seconds... ({tries}): {e}")
                time.sleep(5)
                tries += 1

    @staticmethod
    def save_user_agents():
        user_agents = []
        response = requests.get("https://www.useragents.me/api").json()  # Rate-limit of 15 requests per IP address per hour
        today = dt.datetime.today().strftime("%Y-%m-%d")
        with open(SAVE_PATH + r"\user_agents.txt", "w") as f:
            f.write(today + "\n")
            for entry in response["data"]:
                user_agents.append(entry["ua"])
                f.write(f"{entry['ua']}\n")
        return user_agents

    @staticmethod
    def load_user_agents():
        path = SAVE_PATH + r"\user_agents.txt"
        if os.path.exists(path):
            user_agents = []
            with open(SAVE_PATH + r"\user_agents.txt", "r") as f:
                for i, line in enumerate(f):
                    user_agents.append(line.strip())

                    if i == 0:
                        today = dt.datetime.today().strftime("%Y-%m-%d")
                        '''if today != line.strip():
                            return Scraper.save_user_agents()'''
                    else:
                        user_agents.append(line.strip())
            return user_agents
        else:
            return Scraper.save_user_agents()
