import json
import time
import datetime as dt
import saving
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit, REST
import requests
import talib
import pandas as pd
from constants import *
from HMM.models import HMMRegimePrediction
from tqdm import tqdm


class Profile(object):
    def __init__(self, settings, index):
        self.index = -1
        self.alpaca_api = None
        self.name = ""
        self.agents = {}
        self.stocks = {}
        self.interval = -1
        self.profit_window = -1
        self.k_period = -1
        self.d_period = -1
        self.rsi_period = -1
        self.atr_period = -1
        self.ma_periods = []
        self.fitness_multipliers = {}
        self.regime_settings = {}
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
        for stock in profile["stocks"]:
            self.stocks[stock["symbol"]] = stock
        self.interval = profile["interval"]
        self.profit_window = profile["profit_window"]
        self.k_period = profile["k_period"]
        self.d_period = profile["d_period"]
        self.rsi_period = profile["rsi_period"]
        self.atr_period = profile["atr_period"]
        self.ma_periods = profile["ma_periods"]
        self.fitness_multipliers = profile["fitness_multipliers"]
        self.regime_settings = profile["regime_settings"]
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

    def generate_percent_change(self, bars, ma_periods, gen_spy, gen_qqq):
        bars["open_pc"] = bars["open"].pct_change()
        bars["high_pc"] = bars["high"].pct_change()
        bars["low_pc"] = bars["low"].pct_change()
        bars["close_pc"] = bars["close"].pct_change()
        bars["volume_pc"] = bars["volume"].pct_change()
        bars["vwap_pc"] = bars["vwap"].pct_change()
        bars["atr_pc"] = bars["atr"].pct_change()

        if gen_spy:
            bars["open_spy_pc"] = bars["open_spy"].pct_change()
            bars["high_spy_pc"] = bars["high_spy"].pct_change()
            bars["low_spy_pc"] = bars["low_spy"].pct_change()
            bars["close_spy_pc"] = bars["close_spy"].pct_change()
            bars["volume_spy_pc"] = bars["volume_spy"].pct_change()
            bars["vwap_spy_pc"] = bars["vwap_spy"].pct_change()
            bars["atr_spy_pc"] = bars["atr_spy"].pct_change()

        if gen_qqq:
            bars["open_qqq_pc"] = bars["open_qqq"].pct_change()
            bars["high_qqq_pc"] = bars["high_qqq"].pct_change()
            bars["low_qqq_pc"] = bars["low_qqq"].pct_change()
            bars["close_qqq_pc"] = bars["close_qqq"].pct_change()
            bars["volume_qqq_pc"] = bars["volume_qqq"].pct_change()
            bars["vwap_qqq_pc"] = bars["vwap_qqq"].pct_change()
            bars["atr_qqq_pc"] = bars["atr_qqq"].pct_change()

        for ma_period in ma_periods:
            bars[f"ema_{ma_period}_pc"] = bars[f"ema_{ma_period}"].pct_change()
            bars[f"sma_{ma_period}_pc"] = bars[f"sma_{ma_period}"].pct_change()
            if gen_spy:
                bars[f"ema_{ma_period}_spy_pc"] = bars[f"ema_{ma_period}_spy"].pct_change()
                bars[f"sma_{ma_period}_spy_pc"] = bars[f"sma_{ma_period}_spy"].pct_change()
            if gen_qqq:
                bars[f"ema_{ma_period}_qqq_pc"] = bars[f"ema_{ma_period}_qqq"].pct_change()
                bars[f"sma_{ma_period}_qqq_pc"] = bars[f"sma_{ma_period}_qqq"].pct_change()

    def load_data(self, symbol, i, file_path):
        b_bars = saving.SaveSystem.load_data(file_path)
        print(f" {symbol}{i}: Loaded {b_bars.shape[0]} bars from {b_bars.index[0]} to {b_bars.index[-1]}")
        return b_bars

    def generate_data(self, symbol, i, profile, start_date, end_date,
                      file_path=None, bars_spy=None, bars_qqq=None, training=False):
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

        # Ensure indicator data for training isn't NaN with pre-batch data
        max_ma_period = max(profile.ma_periods)
        max_period = max(profile.k_period, profile.d_period, profile.rsi_period, profile.atr_period, max_ma_period)
        pre_start_date = start_date - dt.timedelta(days=max_period)
        pre_bars = self.get_bars(symbol, profile.alpaca_api, profile.interval, pre_start_date, start_date, 500000)
        init_bars_length = bars.shape[0]
        bars = pd.concat([pre_bars, bars], ignore_index=False).drop_duplicates()
        print(f"\r {symbol}{i}: Generating {bars.shape[0]} indicator data from {pre_start_date} to {end_date}")
        if not bars.index.is_monotonic_increasing:
            print(f" {symbol}{i}: Non-monotonic bars, sorting...")
            bars = bars.sort_index()

        bars["slow_k"], bars["slow_d"] = talib.STOCH(bars["high"], bars["low"], bars["close"],
                                                     fastk_period=profile.k_period,
                                                     slowk_period=profile.d_period,
                                                     slowd_period=profile.d_period)
        bars["slow_k"] = (bars["slow_k"] - 50) / 50
        bars["slow_d"] = (bars["slow_d"] - 50) / 50
        bars["rsi"] = (talib.RSI(bars["close"], timeperiod=profile.rsi_period) - 50) / 50
        bars["atr"] = talib.ATR(bars["high"], bars["low"], bars["close"], timeperiod=profile.atr_period)
        for ma_period in profile.ma_periods:
            bars[f"ema_{ma_period}"] = talib.EMA(bars["close"], timeperiod=ma_period)
            bars[f"sma_{ma_period}"] = talib.SMA(bars["close"], timeperiod=ma_period)

        print(f" {symbol}{i}: Finished generating {bars.shape[0]} indicator data in {(time.time() - start_time):.2f}s")

        backtest_bars = bars[max(0, bars.shape[0] - init_bars_length):].copy()
        backtest_bars = backtest_bars.between_time("9:30", "16:00")

        # Cant vectorize since GPU memory is too small
        print(f"\r {symbol}{i}: Generating {backtest_bars.shape[0]} sentiments and regime predictions from {start_date} to {end_date}")
        long_regime_predictor = HMMRegimePrediction(processes=1)
        short_regime_predictor = HMMRegimePrediction(processes=1)
        unit_map = {"minute": TimeFrameUnit.Minute, "hour": TimeFrameUnit.Hour, "day": TimeFrameUnit.Day,
                    "week": TimeFrameUnit.Week, "month": TimeFrameUnit.Month}
        regime_bars = self.get_bars(symbol, profile.alpaca_api, profile.regime_settings["interval"],
                                    start_date - dt.timedelta(days=profile.regime_settings["fit_days"]), end_date,
                                    500000, unit=unit_map[profile.regime_settings["unit"]])
        HMMRegimePrediction.augment_bars(regime_bars)

        sentiments = []
        short_term_regimes = []
        long_term_regimes = []
        bars_index_list = list(backtest_bars.index)
        prev_regime_slice = None
        prev_long_term_regime = None
        prev_short_term_regime = None
        if symbol == "SPY":
            long_term_features = profile.regime_settings["spy_long_term_features"]
            long_term_seed = profile.regime_settings["spy_long_term_seed"]
            short_term_features = profile.regime_settings["spy_short_term_features"]
            short_term_seed = profile.regime_settings["spy_short_term_seed"]
        elif symbol == "QQQ":
            long_term_features = profile.regime_settings["qqq_long_term_features"]
            long_term_seed = profile.regime_settings["qqq_long_term_seed"]
            short_term_features = profile.regime_settings["qqq_short_term_features"]
            short_term_seed = profile.regime_settings["qqq_short_term_seed"]
        else:
            long_term_features = profile.stocks[symbol]["long_term_features"]
            long_term_seed = profile.stocks[symbol]["long_term_seed"]
            short_term_features = profile.stocks[symbol]["short_term_features"]
            short_term_seed = profile.stocks[symbol]["short_term_seed"]

        for j in tqdm(range(backtest_bars.shape[0])):
            backtest_date = bars_index_list[j].to_pydatetime()

            # Sentiment
            sentiment = self.finbert.get_saved_sentiment(symbol, backtest_date - dt.timedelta(days=3), backtest_date)
            sentiments.append(sentiment)

            # Regime
            regime_slice = regime_bars[:bars_index_list[j]]
            if regime_slice.shape[0] == 0:
                long_term_regimes.append(0.0)
            else:
                if prev_regime_slice is None or prev_regime_slice.shape[0] != regime_slice.shape[0]:
                    sliced_regime_bars = regime_slice.copy()

                    try:
                        long_regime_predictor.fit(sliced_regime_bars, long_term_features, long_term_seed)
                        long_term_regime = long_regime_predictor.predict_probability(sliced_regime_bars)[-1]
                    except IndexError as e:
                        print(f"\rToo little clusters to fit. Skipping validation...")
                        long_term_regime = {"Bull": 0.0, "Bear": 0.0}
                    except ValueError as e:
                        print("\rProblem with data. Skipping...")
                        long_term_regime = {"Bull": 0.0, "Bear": 0.0}

                    try:
                        short_regime_predictor.fit(sliced_regime_bars, short_term_features, short_term_seed)
                        short_term_regime = short_regime_predictor.predict_probability(sliced_regime_bars)[-1]
                    except IndexError as e:
                        print(f"\rToo little clusters to fit. Skipping validation...")
                        short_term_regime = {"Bull": 0.0, "Bear": 0.0}
                    except ValueError as e:
                        print("\rProblem with data. Skipping...")
                        short_term_regime = {"Bull": 0.0, "Bear": 0.0}

                    long_term_regimes.append(long_term_regime["Bull"] - long_term_regime["Bear"])
                    short_term_regimes.append(short_term_regime["Bull"] - short_term_regime["Bear"])

                    prev_regime_slice = sliced_regime_bars
                    prev_long_term_regime = long_term_regime
                    prev_short_term_regime = short_term_regime
                else:
                    long_term_regimes.append(prev_long_term_regime["Bull"] - prev_long_term_regime["Bear"])
                    short_term_regimes.append(prev_short_term_regime["Bull"] - prev_short_term_regime["Bear"])

        backtest_bars["sentiment"] = sentiments
        backtest_bars["long_term_regime"] = long_term_regimes
        backtest_bars["short_term_regime"] = short_term_regimes

        # Combine SPY df with stock df
        if bars_spy is not None:
            bars_spy = bars_spy.reindex(backtest_bars.index, method="ffill")
            backtest_bars = backtest_bars.join(bars_spy, rsuffix="_spy", how="inner")
        # Combine QQQ df with stock df
        if bars_qqq is not None:
            bars_qqq = bars_qqq.reindex(backtest_bars.index, method="ffill")
            backtest_bars = backtest_bars.join(bars_qqq, rsuffix="_qqq", how="inner")

        self.generate_percent_change(backtest_bars, profile.ma_periods, bars_spy is not None, bars_qqq is not None)

        print(f" {symbol}{i}: Finished generating {backtest_bars.shape[0]} data in {(time.time() - start_time):.2f}s")
        if file_path is not None:
            saving.SaveSystem.save_data(backtest_bars, file_path)

        return backtest_bars
