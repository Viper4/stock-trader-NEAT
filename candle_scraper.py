import requests
import random
import datetime as dt
import os
import time

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
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}"

        tries = 1
        while True:
            try:
                response = requests.get(url, headers=headers)

                # Check if the request was successful (status code 200)
                if response.status_code == 200:
                    data = response.json()

                    # Extract OHLCV data from the response
                    current_price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
                    prev_close = data["chart"]["result"][0]["meta"]["previousClose"]  # Yesterday's close
                    quote_data = data["chart"]["result"][0]["indicators"]["quote"][0]
                    formatted_data = []
                    for i in range(len(quote_data["open"])):
                        # Last element is not updated until interval min mark is hit and just shows current price
                        if quote_data["open"][i] != quote_data["high"][i] and quote_data["volume"][i] != 0:
                            formatted_data.append({"open": quote_data["open"][i],
                                                   "high": quote_data["high"][i],
                                                   "low": quote_data["low"][i],
                                                   "close": current_price,
                                                   # "close": quote_data["close"][i],
                                                   "volume": quote_data["volume"][i]})
                    if len(formatted_data) == 0:
                        print(f"Received no candles for {symbol} retrying in 10 seconds... ({tries})")
                        time.sleep(10)
                        tries += 1
                    else:
                        return formatted_data, prev_close
                else:
                    print(
                        f"{response.status_code} error code fetching candles for {symbol}. Retrying in 5 seconds... ({tries})")
                    time.sleep(5)
                    tries += 1
            except ConnectionError as e:
                print(f"Connection error fetching candles for {symbol}. Retrying in 5 seconds... ({tries}): {e}")
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

                    '''if i == 0:
                        today = dt.datetime.today().strftime("%Y-%m-%d")
                        if today != line.strip():
                            return save_user_agents()
                    else:
                        user_agents.append(line.strip())'''
            return user_agents
        else:
            return Scraper.save_user_agents()
