import requests
import random
import datetime as dt
import os

SAVE_PATH = r"C:\Users\vpr16\PythonProjects\StockTraderNEAT\Saves\WebScraping"


r'''def save_proxies(limit):
    proxies = []
    response = requests.get(f"https://proxylist.geonode.com/api/proxy-list?country=US&limit={limit}&page=1&sort_by=lastChecked&sort_type=desc").json()
    data = response["data"]
    today = dt.datetime.today().strftime("%Y-%m-%d")
    with open(SAVE_PATH + r"\proxies.txt", "w") as f:
        f.write(today + "\n")
        for entry in data:
            proxy = f"http://{entry['ip']}:{entry['port']}"
            proxies.append(proxy)
            f.write(f"{proxy}\n")
    return proxies


def load_proxies(limit=50):
    proxies = []
    path = SAVE_PATH + r"\proxies.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    today = dt.datetime.today().strftime("%Y-%m-%d")
                    if today != line.strip():
                        return save_proxies(limit)
                else:
                    if i >= limit:  # First line is for date
                        break
                    proxies.append(line.strip())
        return proxies
    else:
        return save_proxies(limit)'''


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


def load_user_agents():
    path = SAVE_PATH + r"\user_agents.txt"
    if os.path.exists(path):
        user_agents = []
        with open(SAVE_PATH + r"\user_agents.txt", "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    today = dt.datetime.today().strftime("%Y-%m-%d")
                    if today != line.strip():
                        return save_user_agents()
                else:
                    user_agents.append(line.strip())
        return user_agents
    else:
        return save_user_agents()


class Scraper(object):
    def __init__(self):
        #self.proxies = load_proxies(100)
        self.user_agents = load_user_agents()
        print(f"Scraper initialized with {len(self.user_agents)} user_agents.")

    def get_latest_candles(self, symbol, interval="5m"):
        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br"
        }
        #proxy = random.choice(self.proxies)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}"

        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()

            # Extract OHLCV data from the response
            try:
                quote_data = data["chart"]["result"][0]["indicators"]["quote"][0]
                formatted_data = []
                for i in range(len(quote_data["open"])):
                    if quote_data["open"][i] != quote_data["high"][i] and quote_data["volume"][i] != 0:  # Sometimes when market is closed last element is weird
                        formatted_data.append({"open": quote_data["open"][i],
                                               "high": quote_data["high"][i],
                                               "low": quote_data["low"][i],
                                               "close": quote_data["close"][i],
                                               "volume": quote_data["volume"][i]})
                return formatted_data
            except (KeyError, IndexError):
                print("Error: Unable to extract data from the response.")
                return None
        else:
            print(f"Error: Unable to fetch data. Status Code: {response.status_code}")
            return None
