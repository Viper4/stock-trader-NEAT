from neat import nn
import json
import time
import datetime as dt
import pytz
import os
import saving
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit, REST
import requests


class Manager(object):
    def __init__(self, settings, finbert):
        self.running = False
        self.settings = settings
        self.finbert = finbert
        self.sessions = {}
        self.log_path = self.settings["save_path"] + "\\Logs\\"
        saving.SaveSystem.make_dir(self.log_path)

    @staticmethod
    def check_internet_connection():
        tries = 0
        while True:
            try:
                # Try to send a request to a reliable host (e.g., Google)
                response = requests.get("https://www.google.com", timeout=5)
                if response.status_code == 200:
                    break
                else:
                    print("Unable to reach the internet (status code: " + str(response.status_code) + f") ({tries})")
            except (requests.ConnectionError, requests.Timeout) as e:
                print(f"No internet connection. ({tries})")
                time.sleep(5)
                tries += 1

    @staticmethod
    def get_bars(symbol, alpaca_api, interval, start, end, limit, unit=TimeFrameUnit.Minute, sort="asc"):
        tries = 1
        while True:
            try:
                bars_df = alpaca_api.get_bars(
                    symbol=symbol,
                    timeframe=TimeFrame(interval, unit),
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=limit,
                    sort=sort,
                    adjustment="all").df.tz_convert("US/Eastern").between_time("9:30", "16:00")
                return bars_df
            except Exception as e:
                Manager.check_internet_connection()
                print(f"Error getting bars: '{e}'. Retrying in 5 seconds... ({tries})")
                tries += 1
                time.sleep(5)

    @staticmethod
    def get_settings_and_alpaca(profile_index):
        # Settings
        local_dir = os.path.dirname(__file__)
        settings_path = os.path.join(local_dir, "settings.json")
        with open(settings_path) as file:
            settings = json.load(file)

        # Alpaca API
        first_profile = settings["profiles"][profile_index]
        alpaca_api = REST(first_profile["public_key"], first_profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))

        return settings, alpaca_api

    def update_net(self, for_agent, genome, alpaca_api, interval, profile_name,
                   sp500_bars, nasdaq_bars, sp500_sentiments, nasdaq_sentiments,
                   k_period, d_period, rsi_period, sma_periods, days):
        for_agent.net = nn.RecurrentNetwork.create(genome, for_agent.config)
        for_agent.genome = genome

        # Loading network memory
        if for_agent.load_memory():
            print(f"{profile_name} {for_agent.stock['symbol']}: Network memory loaded successfully")
            return

        # TODO: Run the network over the past given days to generate memory

        #for_agent.save_memory()
        print(f"{profile_name} {for_agent.stock['symbol']}: Network memory generated")
