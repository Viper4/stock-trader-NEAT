import json
import time
import datetime as dt
import saving
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit, REST
import requests
from constants import *
from HMM.hmm_models import HMMRegimePrediction
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
        self.fitness_multipliers = {}
        self.general_regime_settings = {}
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
        self.fitness_multipliers = profile["fitness_multipliers"]
        self.general_regime_settings = profile["general_regime_settings"]
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

    def generate_percent_change(self, bars):
        bars["open_pc"] = bars["open"].pct_change()
        bars["high_pc"] = bars["high"].pct_change()
        bars["low_pc"] = bars["low"].pct_change()
        bars["close_pc"] = bars["close"].pct_change()
        bars["volume_pc"] = bars["volume"].pct_change()
        bars["vwap_pc"] = bars["vwap"].pct_change()
        bars["trade_count_pc"] = bars["trade_count"].pct_change()

    def load_data(self, symbol, i, file_path):
        b_bars = saving.SaveSystem.load_data(file_path)
        print(f" {symbol}{i}: Loaded {b_bars.shape[0]} bars from {b_bars.index[0]} to {b_bars.index[-1]}")
        return b_bars

    def generate_data(self, symbol, i, profile, start_date, end_date,
                      file_path=None, training=False):
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
        bars = bars.between_time("9:30", "16:00")

        # Cant vectorize since GPU memory is too small
        #print(f"\r {symbol}{i}: Generating {bars.shape[0]} sentiments and regime predictions from {start_date} to {end_date}")
        #sentiments = []
        print(f"\r {symbol}{i}: Generating {bars.shape[0]} regime predictions from {start_date} to {end_date}")

        unit_map = {"minute": TimeFrameUnit.Minute, "hour": TimeFrameUnit.Hour, "day": TimeFrameUnit.Day,
                    "week": TimeFrameUnit.Week, "month": TimeFrameUnit.Month}
        regime_bars = self.get_bars(symbol, profile.alpaca_api, profile.general_regime_settings["interval"],
                                    start_date - dt.timedelta(days=profile.general_regime_settings["fit_days"]), end_date,
                                    500000, unit=unit_map[profile.general_regime_settings["unit"]])
        HMMRegimePrediction.augment_bars(regime_bars)

        # Load regime predictor settings
        regime_predictors = []
        regime_predictions = []
        for regime_setting in profile.stocks[symbol]["regime_settings"]:
            regime_predictors.append({"model": HMMRegimePrediction(),
                                      "features": regime_setting["features"],
                                      "seed": regime_setting["seed"],
                                      "label_order": regime_setting["label_order"]})
            regime_predictions.append([])

        # Regime label order changes every time we fit.
        prev_regime_slice_size = 0
        for j in tqdm(range(bars.shape[0])):
            #backtest_date = bars.index[j].to_pydatetime()

            # Sentiment
            #sentiment = self.finbert.get_saved_sentiment(symbol, backtest_date - dt.timedelta(days=3), backtest_date)
            #sentiments.append(sentiment)

            # Regime prediction
            regime_slice = regime_bars[:bars.index[j]]

            for k in range(len(regime_predictors)):
                if regime_slice.shape[0] == 0:
                    # No data to predict with
                    regime_predictions[k].append(0.0)
                elif prev_regime_slice_size != regime_slice.shape[0]:
                    # Need to fit and predict with new data
                    sliced_regime_bars = regime_slice.copy()
                    predictions = None
                    try:
                        regime_predictors[k]["model"].fit(sliced_regime_bars, regime_predictors[k]["features"], regime_predictors[k]["seed"])
                        predictions = regime_predictors[k]["model"].predict_probability(sliced_regime_bars)
                        prediction = predictions[-1]
                    except IndexError as e:
                        print("Too little clusters to fit. Skipping validation...")
                        prediction = [0.0, 0.0, 0.0]
                    except ValueError as e:
                        print("Problem with data. Skipping...")
                        prediction = [0.0, 0.0, 0.0]

                    # Didn't get back enough regime labels, so pad with zeros
                    if len(prediction) < 3:
                        print("Padding regimes")
                        print("RAW RESULT:", predictions)
                        for _ in range(len(prediction) - 1, 3):
                            prediction.append(0.0)
                    bull_index = regime_predictors[k]["label_order"]["Bull"]
                    bear_index = regime_predictors[k]["label_order"]["Bear"]
                    regime_predictions[k].append(prediction[bull_index] - prediction[bear_index])
                else:
                    # Same data, so use previous prediction to save time and processing effort
                    regime_predictions[k].append(regime_predictions[k][-1])
            prev_regime_slice_size = regime_slice.shape[0]

        #bars["sentiment"] = sentiments

        for j in range(len(regime_predictors)):
            bars[f"regime_{j}"] = regime_predictions[j]

        self.generate_percent_change(bars)

        print(f" {symbol}{i}: Finished generating {bars.shape[0]} data in {(time.time() - start_time):.2f}s")
        if file_path is not None:
            saving.SaveSystem.save_data(bars, file_path)

        return bars
