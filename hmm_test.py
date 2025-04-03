import numpy as np
from alpaca_trade_api.rest import REST, URL, TimeFrameUnit
import datetime as dt
import pytz
from constants import *
import json
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import Managers.base_manager
import time
import saving
import talib
import ast
from sklearn.feature_selection import f_classif, mutual_info_classif
from scipy.stats import chi2_contingency
import seaborn as sns
from datetime import timedelta
import HMM.models as models
import HMM.feature_selection as feature_selection

DATA_PATH = PROJECT_DIR + "\\HMM\\bars-data-QQQ-1h_2019-1-1_2025-4-1.gz"
TESTED_PATH = PROJECT_DIR + "\\HMM\\tested-QQQ-1h_2019-1-1_2025-4-1.csv"


class CorrelationAnalysis(object):
    def __init__(self, features):
        self.features = features

    def cramers_v(self, x, y):
        """
        Computes Cramér's V statistic for categorical correlation.
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))

    def compute_feature_correlation(self, bars):
        """
        Computes correlation scores between features and the categorical 'regime'.
        Uses ANOVA F-test for numerical features, Mutual Information, and Cramér's V.
        """
        results = {}

        # Encode categorical regime as integers (Bear = 0, Choppy = 1, Bull = 2)
        bars["regime_encoded"] = bars["regime"].astype("category").cat.codes

        for feature in self.features:
            if bars[feature].dtype in [np.float64, np.int64]:  # Numerical Features
                # ANOVA F-test
                f_stat, _ = f_classif(bars[[feature]], bars["regime_encoded"])
                mi = mutual_info_classif(bars[[feature]], bars["regime_encoded"])[0]
                results[feature] = {"ANOVA F-Score": f_stat[0], "Mutual Info": mi}
            else:  # Categorical Features
                cramers_v_score = self.cramers_v(bars[feature], bars["regime"])
                results[feature] = {"Cramér’s V": cramers_v_score}

        return pd.DataFrame(results).T

    def plot_correlation(self, correlation_matrix):
        plt.figure(figsize=(12, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", robust=True, yticklabels=True, xticklabels=True)
        plt.title("Feature Correlation with Market Regime")
        plt.show()


def run_price_test(num_components, num_latent_bars, train_bars_df, test_bars_df):
    print(f"{num_components} components and {num_latent_bars} latent: ")

    hmm_predictor = models.HMMPricePrediction(num_components, num_latent_bars)
    hmm_predictor.fit(train_bars_df)
    hmm_predictor.validate(test_bars_df)
    start_time = time.time()
    predicted_close = hmm_predictor.predict_latest(bars_df)
    print("Predicted:", predicted_close)
    print("Actual:", bars_df.iloc[-1].close)
    print("Percent Error:", 100 * abs(bars_df.iloc[-1].close - predicted_close) / bars_df.iloc[-1].close)
    print(f"Finished in {time.time() - start_time} seconds")


def run_regime_test(train_bars, test_bars, features, plot):
    regime_predictor = models.HMMRegimePrediction(features)
    regime_predictor.validate(train_bars, test_bars, 2, plot)
    start_time = time.time()
    predicted_regimes = regime_predictor.predict_probability(test_bars_df)
    print(f"Regime probabilities:\n", predicted_regimes)
    print("Finished in ", time.time() - start_time)
    print()


def run_regime_search(train_bars, test_bars, features, val_processes=1, n_iterations=-1):
    feature_combinations = []
    print("Generating combinations")
    for i in range(1, 5):
        feature_combinations.extend(itertools.combinations(features, i))
    print(f"Generated {len(feature_combinations)} combinations")

    best_features = []
    best_accuracy = 0

    tested = {}

    if not os.path.exists(TESTED_PATH):
        saving.SaveSystem.make_csv(["Features", "Accuracy"], TESTED_PATH)
    rows = saving.SaveSystem.read_from_csv(TESTED_PATH)
    for row in rows:
        if row[0] == "Features":
            continue
        key = row[0]
        tested[key] = float(row[1])
        if tested[key] > best_accuracy:
            best_features = ast.literal_eval(row[0])
            best_accuracy = tested[key]

    print("Starting search")
    i = 0
    for features in feature_combinations:
        start_time = time.time()
        print(f"\nTest {i}/{len(feature_combinations)} {100 * i / len(feature_combinations):.4f}%:")
        features_list = list(features)

        if n_iterations != -1 and i >= n_iterations:
            break

        key = str(features_list)
        if key not in tested:
            regime_predictor = models.HMMRegimePrediction(features_list)
            accuracy = regime_predictor.validate(train_bars, test_bars, val_processes, False)
            tested[key] = accuracy
            saving.SaveSystem.save_to_csv([features_list, accuracy], TESTED_PATH, "a")
        else:
            accuracy = tested[key]
            print(f"Already tested: {key} - {tested[key]}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = features_list
            print(f"New best accuracy: {best_accuracy}%")
        i += 1
        iteration_time = time.time() - start_time
        eta = (len(feature_combinations) - i) * iteration_time
        print(f"Finished in {iteration_time:.2f} seconds. ETA: {str(timedelta(seconds=eta))}")

    print(f"Search finished")
    print(f"Best features: {best_features}")
    print(f"Best accuracy: {best_accuracy}%")


def get_stats(bars, column, plot):
    minimum = min(bars[column])
    maximum = max(bars[column])
    mean = bars[column].mean()
    median = bars[column].median()
    std_deviation = bars[column].std()

    if plot:
        print("Min: ", minimum)
        print("Max: ", maximum)
        print("Mean: ", mean)
        print("Median: ", median)
        print("Standard Deviation: ", std_deviation)

        plt.figure(figsize=(12, 6))
        plt.hist(bars[column], bins=50)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {column}")
        plt.show()

        plt.figure(figsize=(15, 6))
        plt.plot(bars.index, bars[column], color="black", label=column)

        plt.legend()
        plt.title(f"{column} over time")
        plt.show()

    return minimum, maximum, mean, median, std_deviation


def run_random_forest(bars, features):
    run_regime_test(bars, bars, features, False)
    random_forest = feature_selection.RandomForestFeatureSelection()
    print(random_forest.evaluate(bars, features))


if __name__ == "__main__":
    user_input = input("Enter command (search, price test, reg test, reg best, correlation, stats, random forest): ")
    if not os.path.exists(DATA_PATH):
        symbol = input("Enter symbol: ")
        start = input("Enter start date (YYYY-MM-DD): ")
        end = input("Enter end date (YYYY-MM-DD): ")
        start_date = dt.datetime.strptime(start, "%Y-%m-%d").replace(hour=9, minute=30,
                                                                     tzinfo=pytz.timezone("US/Eastern"))
        end_date = dt.datetime.strptime(end, "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=pytz.timezone("US/Eastern"))
        with open(SETTINGS_PATH) as file:
            settings = json.load(file)
        alpaca_api = REST(settings["profiles"][0]["public_key"], settings["profiles"][0]["secret_key"],
                          base_url=URL("https://paper-api.alpaca.markets"))
        bars_df = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, 1, start_date, end_date, 500000,
                                                         TimeFrameUnit.Hour)

        bars_df["open_pc"] = bars_df["open"].pct_change(fill_method=None)
        bars_df["high_pc"] = bars_df["high"].pct_change(fill_method=None)
        bars_df["low_pc"] = bars_df["low"].pct_change(fill_method=None)
        bars_df["close_pc"] = bars_df["close"].pct_change(fill_method=None)
        bars_df["volume_pc"] = bars_df["volume"].pct_change(fill_method=None)
        bars_df["vwap_pc"] = bars_df["vwap"].pct_change(fill_method=None)
        bars_df["fracocp"] = (bars_df["close"] - bars_df["open"]) / bars_df["open"]
        bars_df["frachp"] = (bars_df["high"] - bars_df["open"]) / bars_df["open"]
        bars_df["fraclp"] = (bars_df["open"] - bars_df["low"]) / bars_df["open"]

        bars_df["sma_1"] = talib.SMA(bars_df["close"], timeperiod=10)
        bars_df["sma_2"] = talib.SMA(bars_df["close"], timeperiod=30)
        bars_df["sma_3"] = talib.SMA(bars_df["close"], timeperiod=50)
        bars_df["sma_4"] = talib.SMA(bars_df["close"], timeperiod=200)
        bars_df["sma_1"] = bars_df["sma_1"].pct_change(fill_method=None)
        bars_df["sma_2"] = bars_df["sma_2"].pct_change(fill_method=None)
        bars_df["sma_3"] = bars_df["sma_3"].pct_change(fill_method=None)
        bars_df["sma_4"] = bars_df["sma_4"].pct_change(fill_method=None)

        bars_df["ema_1"] = talib.EMA(bars_df["close"], timeperiod=10)
        bars_df["ema_2"] = talib.EMA(bars_df["close"], timeperiod=30)
        bars_df["ema_3"] = talib.EMA(bars_df["close"], timeperiod=50)
        bars_df["ema_4"] = talib.EMA(bars_df["close"], timeperiod=200)
        bars_df["ema_1"] = bars_df["ema_1"].pct_change(fill_method=None)
        bars_df["ema_2"] = bars_df["ema_2"].pct_change(fill_method=None)
        bars_df["ema_3"] = bars_df["ema_3"].pct_change(fill_method=None)
        bars_df["ema_4"] = bars_df["ema_4"].pct_change(fill_method=None)

        bars_df["atr"] = talib.ATR(bars_df["high"], bars_df["low"], bars_df["close"], timeperiod=14)
        bars_df["atr"] = bars_df["atr"].pct_change(fill_method=None)
        bars_df["natr"] = talib.NATR(bars_df["high"], bars_df["low"], bars_df["close"], timeperiod=14)
        bars_df["natr"] = bars_df["natr"].pct_change(fill_method=None)
        bars_df["tr"] = talib.TRANGE(bars_df["high"], bars_df["low"], bars_df["close"])
        bars_df["tr"] = bars_df["tr"].pct_change(fill_method=None).replace(np.inf, 1).replace(-np.inf, -1)
        bars_df["rsi"] = (talib.RSI(bars_df["close"], timeperiod=14) - 50) / 50

        slow_k, slow_d = talib.STOCH(bars_df["high"], bars_df["low"], bars_df["close"], fastk_period=5,
                                     slowk_period=3, slowd_period=3)
        bars_df["slow_k"] = (slow_k - 50) / 50
        bars_df["slow_d"] = (slow_d - 50) / 50

        bars_df["three_black_crows"] = talib.CDL3BLACKCROWS(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["three_inside"] = talib.CDL3INSIDE(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["three_lines"] = talib.CDL3LINESTRIKE(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["three_outside"] = talib.CDL3OUTSIDE(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["three_stars"] = talib.CDL3STARSINSOUTH(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["three_whitesoldiers"] = talib.CDL3WHITESOLDIERS(bars_df["open"], bars_df["high"], bars_df["low"],
                                                              bars_df["close"]) / 100
        bars_df["abandoned_baby"] = talib.CDLABANDONEDBABY(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"],
                                                        penetration=0.3) / 100
        bars_df["advance_block"] = talib.CDLADVANCEBLOCK(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["belthold"] = talib.CDLBELTHOLD(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["breakaway"] = talib.CDLBREAKAWAY(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["closing_marubozu"] = talib.CDLCLOSINGMARUBOZU(bars_df["open"], bars_df["high"], bars_df["low"],
                                                            bars_df["close"]) / 100
        bars_df["conceal_baby"] = talib.CDLCONCEALBABYSWALL(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["counterattack"] = talib.CDLCOUNTERATTACK(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["dark_cloud_cover"] = talib.CDLDARKCLOUDCOVER(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"],
                                                           penetration=0.5) / 100
        bars_df["doji"] = talib.CDLDOJI(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["doji_star"] = talib.CDLDOJISTAR(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["dragonfly_doji"] = talib.CDLDRAGONFLYDOJI(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["engulfing"] = talib.CDLENGULFING(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["evening_doji_star"] = talib.CDLEVENINGDOJISTAR(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["evening_star"] = talib.CDLEVENINGSTAR(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["gap_side_by_side"] = talib.CDLGAPSIDESIDEWHITE(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["gravestone_doji"] = talib.CDLGRAVESTONEDOJI(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["hammer"] = talib.CDLHAMMER(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["hanging_man"] = talib.CDLHANGINGMAN(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["harami"] = talib.CDLHARAMI(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["harami_cross"] = talib.CDLHARAMICROSS(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["high_wave"] = talib.CDLHIGHWAVE(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["hikkake"] = talib.CDLHIKKAKE(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["homing_pigeon"] = talib.CDLHOMINGPIGEON(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["identical_three_crows"] = talib.CDLIDENTICAL3CROWS(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["in_neck"] = talib.CDLINNECK(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["inverted_hammer"] = talib.CDLINVERTEDHAMMER(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["kicking"] = talib.CDLKICKING(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["kicking_by_length"] = talib.CDLKICKINGBYLENGTH(bars_df["open"], bars_df["high"], bars_df["low"],
                                                             bars_df["close"]) / 100
        bars_df["ladder_bottom"] = talib.CDLLADDERBOTTOM(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["long_leader"] = talib.CDLLONGLEGGEDDOJI(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["long_line"] = talib.CDLLONGLINE(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["marubozu"] = talib.CDLMARUBOZU(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["matching_low"] = talib.CDLMATCHINGLOW(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["mat_hold"] = talib.CDLMATHOLD(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["morning_doji_star"] = talib.CDLMORNINGDOJISTAR(bars_df["open"], bars_df["high"], bars_df["low"],
                                                             bars_df["close"]) / 100
        bars_df["morning_star"] = talib.CDLMORNINGSTAR(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["on_neck"] = talib.CDLONNECK(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["piercing"] = talib.CDLPIERCING(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["rickshaw_man"] = talib.CDLRICKSHAWMAN(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["rise_fall_three_methods"] = talib.CDLRISEFALL3METHODS(bars_df["open"], bars_df["high"], bars_df["low"],
                                                                    bars_df["close"]) / 100
        bars_df["separating_lines"] = talib.CDLSEPARATINGLINES(bars_df["open"], bars_df["high"], bars_df["low"],
                                                            bars_df["close"]) / 100
        bars_df["shooting_star"] = talib.CDLSHOOTINGSTAR(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["short_line"] = talib.CDLSHORTLINE(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["spinning_top"] = talib.CDLSPINNINGTOP(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["stalled_pattern"] = talib.CDLSTALLEDPATTERN(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["stick_sandwich"] = talib.CDLSTICKSANDWICH(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["takuri"] = talib.CDLTAKURI(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["tasuki_gap"] = talib.CDLTASUKIGAP(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["thrusting"] = talib.CDLTHRUSTING(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["tristar"] = talib.CDLTRISTAR(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["unique_3_river"] = talib.CDLUNIQUE3RIVER(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["upside_gap_2_crows"] = talib.CDLUPSIDEGAP2CROWS(bars_df["open"], bars_df["high"], bars_df["low"],
                                                              bars_df["close"]) / 100
        bars_df["side_gap_3_methods"] = talib.CDLXSIDEGAP3METHODS(bars_df["open"], bars_df["high"], bars_df["low"],
                                                               bars_df["close"]) / 100

        bars_df["ad"] = talib.AD(bars_df["high"], bars_df["low"], bars_df["close"], bars_df["volume"])
        bars_df["adosc"] = talib.ADOSC(bars_df["high"], bars_df["low"], bars_df["close"], bars_df["volume"],
                                       fastperiod=3, slowperiod=10)
        bars_df["obv"] = talib.OBV(bars_df["close"], bars_df["volume"])
        bars_df["ad"] = bars_df["ad"].pct_change(fill_method=None)
        bars_df["adosc"] = bars_df["adosc"].pct_change(fill_method=None)
        bars_df["obv"] = bars_df["obv"].pct_change(fill_method=None)

        bars_df["adx"] = talib.ADX(bars_df["high"], bars_df["low"], bars_df["close"], timeperiod=14)
        bars_df["adx"] = bars_df["adx"].pct_change(fill_method=None)

        bars_df["ht_trendline"] = talib.HT_TRENDLINE(bars_df["close"])
        bars_df["ht_trendline"] = bars_df["ht_trendline"].pct_change()

        bars_df["kama"] = talib.KAMA(bars_df["close"], timeperiod=30)
        bars_df["kama"] = bars_df["kama"].pct_change(fill_method=None)

        bars_df["mama"], bars_df["fama"] = talib.MAMA(bars_df["close"], fastlimit=0.5, slowlimit=0.05)
        bars_df["mama"] = bars_df["mama"].pct_change(fill_method=None)
        bars_df["fama"] = bars_df["fama"].pct_change(fill_method=None)

        bars_df["sar"] = talib.SAR(bars_df["high"], bars_df["low"], acceleration=0.02, maximum=0.2)
        bars_df["sar"] = bars_df["sar"].pct_change(fill_method=None)

        bars_df["volatility"] = bars_df["close_pc"].rolling(window=15).std()
        bars_df.dropna(inplace=True)

        saving.SaveSystem.save_data(bars_df, DATA_PATH)
    else:
        bars_df = saving.SaveSystem.load_data(DATA_PATH)
        bars_df.dropna(inplace=True)

    train_size = int(bars_df.shape[0] * 0.8)
    train_bars_df = bars_df[:train_size].copy()
    test_bars_df = bars_df[train_size + 1:].copy()
    print("Train bars: ", train_bars_df.shape[0])
    print("Test bars: ", test_bars_df.shape[0])
    print("Total bars: ", bars_df.shape[0])

    all_features = [
        "open_pc", "high_pc", "low_pc", "close_pc", "volume_pc", "vwap_pc",
        "fracocp", "frachp", "fraclp",
        "sma_1", "sma_2", "sma_3", "sma_4",
        "ema_1", "ema_2", "ema_3", "ema_4",
        "atr", "natr", "tr", "rsi", "slow_k", "slow_d",
        "three_black_crows", "three_inside", "three_lines", "three_outside",
        "three_stars", "three_whitesoldiers", "abandoned_baby", "advance_block",
        "belthold", "breakaway", "closing_marubozu", "conceal_baby",
        "counterattack", "dark_cloud_cover", "doji", "doji_star",
        "dragonfly_doji", "engulfing", "evening_doji_star", "evening_star",
        "gap_side_by_side", "gravestone_doji", "hammer", "hanging_man",
        "harami", "harami_cross", "high_wave", "hikkake", "homing_pigeon",
        "identical_three_crows", "in_neck", "inverted_hammer", "kicking",
        "kicking_by_length", "ladder_bottom", "long_leader", "long_line",
        "marubozu", "matching_low", "mat_hold", "morning_doji_star",
        "morning_star", "on_neck", "piercing", "rickshaw_man",
        "rise_fall_three_methods", "separating_lines", "shooting_star",
        "short_line", "spinning_top", "stalled_pattern", "stick_sandwich",
        "takuri", "tasuki_gap", "thrusting", "tristar", "unique_3_river",
        "upside_gap_2_crows", "side_gap_3_methods",
        "ad", "adosc", "obv", "adx", "ht_trendline", "kama", "mama", "fama", "sar",
        "volatility"
    ]

    if user_input == "search":
        run_regime_search(train_bars_df, test_bars_df, all_features, 2)
    elif user_input == "price test":
        run_price_test(input("Number of components: "), int(input("Number of latent bars: ")), train_bars_df, test_bars_df)
    elif user_input == "reg test":
        # ["close_pc", "ema_1"] accuracy isn't good but looking at the graph it actually seems the best
        test_features = ast.literal_eval(input("Type features, Ex: ['close_pc', 'ema_1']: "))
        run_regime_test(train_bars_df, test_bars_df, test_features, True)
    elif user_input == "correlation":
        feature_input = input("Type features or 'all', Ex: ['close_pc', 'ema_1']: ")
        if feature_input == "all":
            corr_features = all_features
        else:
            corr_features = ast.literal_eval(feature_input)
        run_regime_test(train_bars_df, test_bars_df, corr_features, True)
        correlation = CorrelationAnalysis(corr_features)
        correlation_matrix = correlation.compute_feature_correlation(test_bars_df)
        print("Correlation matrix:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(correlation_matrix)
        correlation.plot_correlation(correlation_matrix)
    elif user_input == "reg best":
        results = []
        rows = saving.SaveSystem.read_from_csv(TESTED_PATH)
        for row in rows:
            if row[0] == "Features":
                continue
            key = row[0]
            accuracy = float(row[1])
            results.append((key, accuracy))
        results.sort(key=lambda x: x[1], reverse=True)
        best_features, best_accuracy = results[0]
        print("Best features: ", best_features)
        print("Best accuracy: ", best_accuracy)

        save_path = TESTED_PATH.replace(".csv", "_sorted.csv")
        saving.SaveSystem.delete_file(save_path)
        saving.SaveSystem.make_csv(["Features", "Accuracy"], save_path)

        sorted_features = []
        sorted_accuracies = []
        for feature_settings, accuracy in results:
            sorted_features.append(feature_settings)
            sorted_accuracies.append(accuracy)
            saving.SaveSystem.save_to_csv([feature_settings, accuracy], save_path, "a")

        plt.figure(figsize=(12, 6))
        plt.barh(sorted_features[:200], sorted_accuracies[:200], color="skyblue")
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Feature Combinations")
        plt.title("Feature Combination Accuracy Rankings")
        plt.gca().invert_yaxis()  # Best at top
        plt.show()

        run_regime_test(train_bars_df, test_bars_df, ast.literal_eval(best_features), True)
    elif user_input == "stats":
        get_stats(pd.concat([train_bars_df, test_bars_df]), input("Enter column: "), True)
    elif user_input == "random forest":
        run_random_forest(bars_df, all_features)

    '''run_test_price_predict(16, 50, train_bars, bars_df[train_size + 1:])
    run_test_price_predict(8, 50, train_bars, bars_df[train_size + 1:])
    run_test_price_predict(4, 50, train_bars, bars_df[train_size + 1:])
    run_test_price_predict(2, 50, train_bars, bars_df[train_size + 1:])'''
