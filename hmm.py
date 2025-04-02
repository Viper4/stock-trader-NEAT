from hmmlearn.hmm import GaussianHMM
import numpy as np
from sklearn.preprocessing import StandardScaler
from alpaca_trade_api.rest import REST, URL, TimeFrameUnit
import datetime as dt
import pytz
from constants import *
import json
import pandas as pd
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import Managers.base_manager
from multiprocessing import Pool
import time
import saving
import talib
import ast
from sklearn.feature_selection import f_classif, mutual_info_classif
from scipy.stats import chi2_contingency
import seaborn as sns
from datetime import timedelta

DATA_PATH = SAVE_DIR + "HMM\\bars-data-TSLA-1h_2019-1-1_2025-4-1.gz"
TESTED_PATH = SAVE_DIR + "HMM\\tested-TSLA-1h_2019-1-1_2025-4-1.csv"


class HMMRegimePrediction(object):
    def __init__(self, feature_settings):
        self.model = GaussianHMM(n_components=3, n_iter=10000)
        self.scaler = StandardScaler()  # Store scaler for consistent transformation
        self.regime_mapping = None  # Store regime mapping
        self.feature_settings = feature_settings

    def get_features(self, bars):
        """Gets features from bars and scales features."""
        bars.dropna(inplace=True)
        features = bars[self.feature_settings].values
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled, bars

    def fit(self, bars):
        bars.dropna(inplace=True)
        """Fits the HMM model and maps regimes."""
        features_scaled, bars = self.get_features(bars)
        self.model.fit(features_scaled)
        self.map_regimes(bars, features_scaled)

    def predict_probability(self, bars):
        features_scaled, bars = self.get_features(bars)
        prediction_probs = self.model.predict_proba(features_scaled)
        mapped_prediction = {}
        for i in range(prediction_probs.shape[1]):
            mapped_prediction[self.regime_mapping[np.int64(i)]] = float(prediction_probs[-1][i])
        return mapped_prediction

    def predict(self, bars):
        """Predicts market regimes and returns them as mapped labels."""
        features_scaled, bars = self.get_features(bars)
        predicted_regimes = self.model.predict(features_scaled)
        return np.array([self.regime_mapping[r] for r in predicted_regimes])

    def map_regimes(self, bars, features):
        """Assigns correct labels (Bull, Bear, Choppy) based on mean log return."""
        predicted_regimes = self.model.predict(features)
        bars["regime"] = predicted_regimes

        regime_stats = bars.groupby("regime")["close_pc"].mean().sort_values()

        self.regime_mapping = {
            regime_stats.index[0]: "Bear",
            regime_stats.index[1]: "Choppy",
            regime_stats.index[2]: "Bull"
        }

    @staticmethod
    def get_score(bars, std_deviation, threshold):
        # NOTE: Sometimes terrible model just guessing one regime for entire period can get 50% accuracy
        score = 0
        for i in tqdm(range(bars.shape[0] - 1)):
            predicted = bars.iloc[i].regime
            actual_change = bars.iloc[i + 1].returns

            if ((predicted == "Bull" and actual_change > threshold * std_deviation)
                    or (predicted == "Bear" and actual_change < -threshold * std_deviation)
                    or (predicted == "Choppy" and abs(actual_change) <= threshold * std_deviation)):
                score += 1
        return score

    def validate(self, train_bars, test_bars, processes, plot):
        """Trains HMM, evaluates accuracy, and visualizes results."""
        print(f"Training HMM on {train_bars.shape[0]} bars with\nFeatures: {self.feature_settings}")
        try:
            self.fit(train_bars)
        except IndexError as e:
            print("Too little clusters to fit. Skipping validation...")
            return 0.0

        minimum, maximum, mean, median, std_deviation = get_stats(pd.concat([train_bars_df, test_bars_df]), "close_pc", False)
        print(f"Predicting regimes on {test_bars.shape[0]} test bars with stdv {std_deviation}...")
        predicted_labels = self.predict(test_bars)
        test_bars["regime"] = predicted_labels

        # Calculate Accuracy
        correct_predictions = 0
        total_predictions = len(test_bars) - 1  # Ignore last row due to comparing predicted with future price

        if processes > 1:
            pool = Pool(processes=processes)
            args = []
            bars_per_process = test_bars.shape[0] // processes

            for i in range(processes):
                args.append((test_bars[i*bars_per_process:(i+1)*bars_per_process], std_deviation, 0.75))

            results_async = pool.starmap_async(self.get_score, args)
            results = results_async.get()
            for result in results:
                correct_predictions += result

            pool.close()
            pool.join()
        else:
            total_predictions = self.get_score(test_bars, std_deviation)

        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy: {accuracy:.2f}%")

        # Plot stock prices with color-coded regimes
        if plot:
            plt.figure(figsize=(15, 6))
            plt.plot(test_bars.index, test_bars["close"], color="black", label="Stock Price")

            colors = {"Bull": "green", "Bear": "red", "Choppy": "yellow"}
            for regime, color in colors.items():
                plt.fill_between(
                    test_bars.index,
                    test_bars["close"].min(),
                    test_bars["close"].max(),
                    where=test_bars["regime"] == regime,
                    color=color,
                    alpha=0.3,
                    label=regime
                )

            plt.legend()
            plt.title(f"Stock Price with Predicted Market Regimes (Accuracy: {accuracy:.2f}%)")
            plt.show()

        return accuracy


class HMMPricePrediction(object):
    def __init__(self, num_components, num_latent_bars):
        self.model = GaussianHMM(n_components=num_components, init_params="")
        self.model.startprob_ = np.full(num_components, 1 / num_components)  # Uniform probabilities
        self.model.transmat_ = np.full((num_components, num_components), 1 / num_components)  # Equal transition probabilities
        self.model.means_ = np.random.rand(num_components, 3)  # Random means for each state
        self.model.covars_ = np.full((num_components, 3), 0.1)  # Small diagonal covariance values
        self.num_latent_bars = num_latent_bars
        self.pool = Pool(processes=4)

    def augment_bars(self, bars):
        fracocp = (bars["close"] - bars["open"]) / bars["open"]
        frachp = (bars["high"] - bars["open"]) / bars["open"]
        fraclp = (bars["open"] - bars["low"]) / bars["open"]
        new_dataframe = pd.DataFrame(data={"delOpenClose": fracocp,
                                           "delHighOpen": frachp,
                                           "delLowOpen": fraclp},
                                     index=bars.index)

        return new_dataframe

    def get_features(self, dataframe):
        return np.column_stack((dataframe["delOpenClose"], dataframe["delHighOpen"], dataframe["delLowOpen"]))

    def fit_augmented(self, augmented_bars):
        features = self.get_features(augmented_bars)
        self.model.fit(features)

    def fit(self, bars):
        augmented_bars = self.augment_bars(bars)
        features = self.get_features(augmented_bars)
        self.model.fit(features)

    def get_possible_outcomes(self, augmented_bars):
        fracocp = augmented_bars["delOpenClose"]
        frachp = augmented_bars["delHighOpen"]
        fraclp = augmented_bars["delLowOpen"]

        sample_space_fracocp = np.linspace(fracocp.min(), fracocp.max(), 50)
        sample_space_fraclp = np.linspace(fraclp.min(), frachp.max(), 10)
        sample_space_frachp = np.linspace(frachp.min(), frachp.max(), 10)

        return pd.DataFrame(data={"outcome": list(itertools.product(sample_space_fracocp, sample_space_fraclp, sample_space_frachp))})

    def predict(self, open_price, possible_outcomes, features):
        outcome_scores = possible_outcomes["outcome"].apply(lambda outcome: self.model.score(np.vstack((features, outcome))))

        # Take the most probable outcome as the one with the highest score
        most_probable_outcome = possible_outcomes["outcome"].iloc[np.argmax(outcome_scores)]
        return open_price * (1 + most_probable_outcome[0])

    def predict_augmented(self, open_price, augmented_bars):
        augmented_bars = augmented_bars[max(0, augmented_bars.shape[0] - self.num_latent_bars):]
        possible_outcomes = self.get_possible_outcomes(augmented_bars)
        features = self.get_features(augmented_bars)
        return self.predict(open_price, possible_outcomes, features)

    def predict_latest(self, bars):
        augmented_bars = self.augment_bars(bars[max(0, bars.shape[0] - self.num_latent_bars):])
        possible_outcomes = self.get_possible_outcomes(augmented_bars)
        features = self.get_features(augmented_bars)
        return self.predict(bars.iloc[-1].open, possible_outcomes, features)

    def validate(self, bars):
        print("Validating HMM Price Prediction...")

        possible_outcomes = self.get_possible_outcomes(self.augment_bars(bars))
        predicted_close_prices = []

        for i in tqdm(range(self.num_latent_bars, bars.shape[0])):
            # Calculate start and end indices
            previous_data_start_index = max(0, i - self.num_latent_bars)
            # Acquire test data features for these days
            previous_data = self.get_features(self.augment_bars(bars[previous_data_start_index:i]))

            predicted_close_prices.append(self.predict(bars.iloc[i].open, possible_outcomes, previous_data))

        plt.figure(figsize=(30, 10), dpi=80)
        plt.rcParams.update({'font.size': 18})

        x_axis = np.array(bars.index[self.num_latent_bars:], dtype='datetime64[ms]')
        plt.plot(x_axis, bars[self.num_latent_bars:]["close"], 'b+-', label="Actual close prices")
        plt.plot(x_axis, predicted_close_prices, 'ro-', label="Predicted close prices")
        plt.legend(prop={'size': 20})
        plt.show()

        ae = abs(bars[self.num_latent_bars:]["close"] - predicted_close_prices)
        min_ae = min(ae)
        max_ae = max(ae)
        avg_ae = sum(ae) / ae.shape[0]

        plt.figure(figsize=(30, 10), dpi=80)

        print("Min Error: ", min_ae)
        print("Max Error: ", max_ae)
        print("Avg Error: ", avg_ae)
        plt.plot(x_axis, ae, 'go-', label="Error")
        plt.legend(prop={'size': 20})
        plt.show()


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

    hmm_predictor = HMMPricePrediction(num_components, num_latent_bars)
    hmm_predictor.fit(train_bars_df)
    hmm_predictor.validate(test_bars_df)
    start_time = time.time()
    predicted_close = hmm_predictor.predict_latest(bars_df)
    print("Predicted:", predicted_close)
    print("Actual:", bars_df.iloc[-1].close)
    print("Percent Error:", 100 * abs(bars_df.iloc[-1].close - predicted_close) / bars_df.iloc[-1].close)
    print(f"Finished in {time.time() - start_time} seconds")


def run_regime_test(train_bars_df, test_bars_df, features):
    regime_predictor = HMMRegimePrediction(features)
    regime_predictor.validate(train_bars_df, test_bars_df, 8, True)
    start_time = time.time()
    predicted_regime = regime_predictor.predict_probability(test_bars_df)
    print(f"Predicted:", predicted_regime)
    print("Finished in ", time.time() - start_time)


def run_regime_search(train_bars, test_bars, features, val_processes=1, n_iterations=-1):
    feature_combinations = []
    print("Generating combinations")
    for i in tqdm(range(1, 5)):
        feature_combinations.extend(itertools.combinations(all_features, i))
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
            regime_predictor = HMMRegimePrediction(features_list)
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



if __name__ == "__main__":
    user_input = input("Enter command (search, test, correlation): ")
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
        "atr", "natr", "rsi", "slow_k", "slow_d",
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
        run_regime_search(train_bars_df, test_bars_df, all_features, 8)
    elif user_input == "test":
        # ["close_pc", "ema_1"] accuracy isn't good but looking at the graph it actually seems the most accurate

        run_regime_test(train_bars_df, test_bars_df, ["close_pc"])
    elif user_input == "correlation":
        features = all_features
        run_regime_test(train_bars_df, test_bars_df, features)
        correlation = CorrelationAnalysis(features)
        correlation_matrix = correlation.compute_feature_correlation(test_bars_df)
        print("Correlation matrix:\n", correlation_matrix)
        correlation.plot_correlation(correlation_matrix)
    elif user_input == "get_best":
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
        for features, accuracy in results:
            sorted_features.append(features)
            sorted_accuracies.append(accuracy)
            saving.SaveSystem.save_to_csv([features, accuracy], save_path, "a")

        plt.figure(figsize=(12, 6))
        plt.barh(sorted_features[:200], sorted_accuracies[:200], color="skyblue")
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Feature Combinations")
        plt.title("Feature Combination Accuracy Rankings")
        plt.gca().invert_yaxis()  # Best at top
        plt.show()
    elif user_input == "get_stats":
        get_stats(pd.concat([train_bars_df, test_bars_df]), "close", True)

    '''run_test_price_predict(16, 50, train_bars, bars_df[train_size + 1:])
    run_test_price_predict(8, 50, train_bars, bars_df[train_size + 1:])
    run_test_price_predict(4, 50, train_bars, bars_df[train_size + 1:])
    run_test_price_predict(2, 50, train_bars, bars_df[train_size + 1:])'''


