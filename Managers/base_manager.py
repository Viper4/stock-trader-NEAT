import json
import time
import datetime as dt
import saving
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit, REST
import requests
import talib
import pandas as pd
from constants import *


class Profile(object):
    def __init__(self, settings, index):
        self.index = -1
        self.alpaca_api = None
        self.name = ""
        self.agents = {}
        self.stocks = []
        self.interval = -1
        self.profit_window = -1
        self.k_period = -1
        self.d_period = -1
        self.rsi_period = -1
        self.atr_period = -1
        self.sma_periods = []
        self.fitness_multipliers = {}
        self.start_cash = -1
        self.cash_limit = -1
        self.data_batch_size = -1
        self.data_batches = -1
        self.schwab = None

        self.update_properties(settings, index)

    def update_properties(self, settings, index):
        profile = settings["profiles"][index]

        self.alpaca_api = REST(profile["public_key"], profile["secret_key"],
                               base_url=URL("https://paper-api.alpaca.markets"))
        self.name = profile["name"]
        self.stocks = profile["stocks"]
        self.interval = profile["interval"]
        self.profit_window = profile["profit_window"]
        self.k_period = profile["k_period"]
        self.d_period = profile["d_period"]
        self.rsi_period = profile["rsi_period"]
        self.atr_period = profile["atr_period"]
        self.sma_periods = profile["sma_periods"]
        self.fitness_multipliers = profile["fitness_multipliers"]
        self.start_cash = profile["start_cash"]
        self.cash_limit = profile["cash_limit"]
        self.data_batch_size = profile["data_batch_size"]
        self.data_batches = profile["data_batches"]

        if "schwab" in settings:
            self.schwab = settings["schwab"]
        else:
            self.schwab = None

    def update(self):
        with open(SETTINGS_PATH) as file:
            settings = json.load(file)

        self.update_properties(settings, self.index)


class Manager(object):
    def __init__(self, settings, finbert):
        self.running = False
        self.settings = settings
        self.finbert = finbert

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

    def save_memory(self, network, filename):
        file_path = os.path.join(VALUES_DIR, filename + ".gz")
        saving.SaveSystem.save_data((network.values, network.active), file_path)

    def load_memory(self, filename):
        file_path = os.path.join(VALUES_DIR, filename + ".gz")
        if os.path.exists(file_path):
            return saving.SaveSystem.load_data(file_path)
        return None

    def load_data(self, symbol, i, file_path):
        b_bars = saving.SaveSystem.load_data(file_path)
        print(f" {symbol}{i}: Loaded {b_bars.shape[0]} bars from {b_bars.index[0]} to {b_bars.index[-1]}")
        return b_bars

    def generate_data(self, symbol, i, profile, start_date, end_date, file_path, gen_indicators, spy_bars, qqq_bars, training):
        if training:
            # Leave most recent 30 days for validation
            now_date = dt.datetime.now(dt.timezone.utc)
            if end_date > now_date - dt.timedelta(days=30):
                end_date = now_date - dt.timedelta(days=30)

        bars = self.get_bars(symbol, profile.alpaca_api, profile.interval, start_date, end_date, 500000)
        if bars.empty:
            print(f" {symbol}{i}: No bars found for {start_date} to {end_date}")
            return None

        start_time = time.time()

        if gen_indicators:
            print(f" {symbol}{i}: Generating {bars.shape[0]} indicator data from {start_date} to {end_date}")

            # Ensure indicator data for training isn't NaN with pre-batch data
            max_sma_period = max(profile.sma_periods)
            max_period = max(profile.k_period, profile.d_period, profile.rsi_period, profile.atr_period, max_sma_period)
            pre_start_date = start_date - dt.timedelta(days=max_period)
            pre_bars = self.get_bars(symbol, profile.alpaca_api, profile.interval, pre_start_date, start_date, 500000)
            init_bars_length = bars.shape[0]
            bars = pd.concat([pre_bars, bars], ignore_index=False).drop_duplicates()
            if not bars.index.is_monotonic_increasing:
                print(f" {symbol}{i}: Non-monotonic bars, sorting...")
                bars = bars.sort_index()

            bars["slow_k"], bars["slow_d"] = talib.STOCH(bars["high"], bars["low"], bars["close"],
                                                         fastk_period=profile.k_period,
                                                         slowk_period=profile.d_period,
                                                         slowd_period=profile.d_period)
            bars["rsi"] = talib.RSI(bars["close"], timeperiod=profile.rsi_period)
            bars["atr"] = talib.ATR(bars["high"], bars["low"], bars["close"], timeperiod=profile.atr_period)
            bars["ema_k"] = talib.EMA(bars["close"], timeperiod=profile.k_period)
            bars["ema_d"] = talib.EMA(bars["close"], timeperiod=profile.d_period)
            for sma_period in profile.sma_periods:
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
        if file_path is not None:
            saving.SaveSystem.save_data(bars, file_path)

        return bars
