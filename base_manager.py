from neat import nn
import json
import time
import datetime as dt
import pytz
import os
import saving
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit, REST
import requests
import talib
import pandas as pd


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
                    adjustment="all").df.tz_convert("US/Eastern")
                bars_df.drop_duplicates(inplace=True)
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

    def load_data(self, symbol, i, file_path):
        b_bars = saving.SaveSystem.load_data(file_path)
        print(f" {symbol}{i}: Loaded {b_bars.shape[0]} bars from {b_bars.index[0]} to {b_bars.index[-1]}")
        return b_bars

    def generate_data(self, symbol, i, session, start_date, end_date, file_path, gen_indicators, spy_bars, qqq_bars, training):
        if training:
            # Leave most recent 30 days for validation
            now_date = dt.datetime.now(dt.timezone.utc)
            if end_date > now_date - dt.timedelta(days=30):
                end_date = now_date - dt.timedelta(days=30)

        bars = self.get_bars(symbol, session["alpaca_api"], session["interval"], start_date, end_date, 500000)
        if bars.empty:
            print(f" {symbol}{i}: No bars found for {start_date} to {end_date}")
            return None

        start_time = time.time()

        if gen_indicators:
            print(f" {symbol}{i}: Generating {bars.shape[0]} indicator data from {start_date} to {end_date}")

            # Ensure indicator data for training isn't NaN with pre-batch data
            max_sma_period = max(session["sma_periods"])
            max_period = max(session["k_period"], session["d_period"], session["rsi_period"], session["atr_period"], max_sma_period)
            pre_start_date = start_date - dt.timedelta(days=max_period)
            pre_bars = self.get_bars(symbol, session["alpaca_api"], session["interval"], pre_start_date, start_date, 500000)
            init_bars_length = bars.shape[0]
            bars = pd.concat([pre_bars, bars], ignore_index=False).drop_duplicates()
            if not bars.index.is_monotonic_increasing:
                print(f" {symbol}{i}: Non-monotonic bars, sorting...")
                bars = bars.sort_index()

            bars["slow_k"], bars["slow_d"] = talib.STOCH(bars["high"], bars["low"], bars["close"],
                                                                 fastk_period=session["k_period"],
                                                                 slowk_period=session["d_period"],
                                                                 slowd_period=session["d_period"])
            bars["rsi"] = talib.RSI(bars["close"], timeperiod=session["rsi_period"])
            bars["atr"] = talib.ATR(bars["high"], bars["low"], bars["close"], timeperiod=session["atr_period"])
            bars["ema_k"] = talib.EMA(bars["close"], timeperiod=session["k_period"])
            bars["ema_d"] = talib.EMA(bars["close"], timeperiod=session["d_period"])
            for sma_period in session["sma_periods"]:
                bars[f"sma_{sma_period}"] = talib.SMA(bars["close"], timeperiod=sma_period)

            bars = bars[max(0, bars.shape[0] - init_bars_length):]

            print(f" {symbol}{i}: Finished generating {bars.shape[0]} indicator data in {(time.time() - start_time):.2f}s")

        bars = bars.between_time("9:30", "16:00")
        print(f" {symbol}{i}: Generating {bars.shape[0]} sentiments from {start_date} to {end_date}")

        # Cant vectorize since GPU memory is too small
        bars["sentiment"] = 0.0
        for row in bars.itertuples():
            backtest_date = row.Index.to_pydatetime()
            sentiment = self.finbert.get_saved_sentiment(symbol, backtest_date - dt.timedelta(days=3), backtest_date)
            bars.at[row.Index, "sentiment"] = sentiment

        # Combine SPY df with stock df
        if spy_bars is not None:
            spy_bars = spy_bars.reindex(bars.index, method="ffill")
            bars = bars.join(spy_bars, rsuffix="_spy", how="inner")
        # Combine QQQ df with stock df
        if qqq_bars is not None:
            qqq_bars = qqq_bars.reindex(bars.index, method="ffill")
            bars = bars.join(qqq_bars, rsuffix="_qqq", how="inner")

        print(f" {symbol}{i}: Finished generating {bars.shape[0]} data in {(time.time() - start_time):.2f}s")
        saving.SaveSystem.save_data(bars, file_path)

        return bars

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
