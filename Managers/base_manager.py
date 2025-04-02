import json
import time
import datetime as dt
import saving
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit, REST
import requests
import talib
import pandas as pd
from constants import *
import hmm


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
        self.logs = {}

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

    def generate_percent_change(self, bars, sma_periods):
        bars["open_pc"] = bars["open"].pct_change()
        bars["high_pc"] = bars["high"].pct_change()
        bars["low_pc"] = bars["low"].pct_change()
        bars["close_pc"] = bars["close"].pct_change()
        bars["volume_pc"] = bars["volume"].pct_change()
        bars["vwap_pc"] = bars["vwap"].pct_change()
        bars["ema_k_pc"] = bars["ema_k"].pct_change()
        bars["ema_d_pc"] = bars["ema_d"].pct_change()

        bars["close_spy_pc"] = bars["close_spy"].pct_change()
        bars["volume_spy_pc"] = bars["volume_spy"].pct_change()
        bars["atr_spy_pc"] = bars["atr_spy"].pct_change()
        bars["ema_k_spy_pc"] = bars["ema_k_spy"].pct_change()
        bars["ema_d_spy_pc"] = bars["ema_d_spy"].pct_change()

        bars["close_qqq_pc"] = bars["close_qqq"].pct_change()
        bars["volume_qqq_pc"] = bars["volume_qqq"].pct_change()
        bars["atr_qqq_pc"] = bars["atr_qqq"].pct_change()
        bars["ema_k_qqq_pc"] = bars["ema_k_qqq"].pct_change()
        bars["ema_d_qqq_pc"] = bars["ema_d_qqq"].pct_change()

        for sma_period in sma_periods:
            bars[f"sma_{sma_period}_pc"] = bars[f"sma_{sma_period}"].pct_change()
            bars[f"sma_spy_{sma_period}_pc"] = bars[f"sma_spy_{sma_period}"].pct_change()
            bars[f"sma_qqq_{sma_period}_pc"] = bars[f"sma_qqq_{sma_period}"].pct_change()

    def load_data(self, symbol, i, file_path):
        b_bars = saving.SaveSystem.load_data(file_path)
        print(f" {symbol}{i}: Loaded {b_bars.shape[0]} bars from {b_bars.index[0]} to {b_bars.index[-1]}")
        return b_bars

    def generate_data(self, symbol, i, profile, start_date, end_date,
                      file_path=None, bars_spy=None, bars_qqq=None, training=False, gen_hmm=False):
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

        print(f" {symbol}{i}: Finished generating {bars.shape[0]} indicator data in {(time.time() - start_time):.2f}s")

        backtest_bars = bars[max(0, bars.shape[0] - init_bars_length):].copy()
        backtest_bars = backtest_bars.between_time("9:30", "16:00")

        # Cant vectorize since GPU memory is too small
        backtest_bars["sentiment"] = 0.0
        if gen_hmm:
            print(f" {symbol}{i}: Generating {backtest_bars.shape[0]} sentiments and HMM predictions from {start_date} to {end_date}")
            hmm_predictor = hmm.HMMPricePrediction(10, 100)

            augmented_bars = hmm_predictor.augment_bars(bars)
            backtest_bars["hmm_prediction"] = 0.0
            j = 0
            for row in backtest_bars.itertuples():
                backtest_date = row.Index.to_pydatetime()
                sentiment = self.finbert.get_saved_sentiment(symbol, backtest_date - dt.timedelta(days=3), backtest_date)
                backtest_bars.at[row.Index, "sentiment"] = sentiment

                previous_data = augmented_bars[:row.Index]
                if j % 100 == 0:
                    print(f" {symbol}{i}: Fitting HMM {j}/{backtest_bars.shape[0]}")
                    hmm_predictor.fit_augmented(previous_data)
                backtest_bars.at[row.Index, "hmm_prediction"] = hmm_predictor.predict_augmented(row.open, previous_data)
                j += 1
            print(f" {symbol}{i}: Finished generating {bars.shape[0]} sentiments and HMM predictions in {(time.time() - start_time):.2f}s")
        else:
            print(f" {symbol}{i}: Generating {backtest_bars.shape[0]} sentiments from {start_date} to {end_date}")
            for row in backtest_bars.itertuples():
                backtest_date = row.Index.to_pydatetime()
                sentiment = self.finbert.get_saved_sentiment(symbol, backtest_date - dt.timedelta(days=3), backtest_date)
                backtest_bars.at[row.Index, "sentiment"] = sentiment
            print(f" {symbol}{i}: Finished generating {backtest_bars.shape[0]} sentiments in {(time.time() - start_time):.2f}s")

        # Combine SPY df with stock df
        if bars_spy is not None:
            bars_spy = bars_spy.reindex(backtest_bars.index, method="ffill")
            backtest_bars = backtest_bars.join(bars_spy, rsuffix="_spy", how="inner")
        # Combine QQQ df with stock df
        if bars_qqq is not None:
            bars_qqq = bars_qqq.reindex(backtest_bars.index, method="ffill")
            backtest_bars = backtest_bars.join(bars_qqq, rsuffix="_qqq", how="inner")

        self.generate_percent_change(backtest_bars, profile.sma_periods)

        print(f" {symbol}{i}: Finished generating {backtest_bars.shape[0]} data in {(time.time() - start_time):.2f}s")
        if file_path is not None:
            saving.SaveSystem.save_data(backtest_bars, file_path)

        return backtest_bars
