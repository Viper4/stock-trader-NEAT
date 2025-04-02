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
import random
import ast

DATA_PATH = SAVE_DIR + "HMM\\bars-data-TSLA_2020-1-1_2025-4-1.gz"
TESTED_PATH = SAVE_DIR + "HMM\\tested-TSLA_2020-1-1_2025-4-1.csv"


class HMMRegimePrediction(object):
    def __init__(self, feature_settings, extra_settings):
        self.model = GaussianHMM(n_components=3, n_iter=10000)
        self.scaler = StandardScaler()  # Store scaler for consistent transformation
        self.regime_mapping = None  # Store regime mapping
        self.feature_settings = feature_settings
        self.extra_settings = extra_settings

    def get_features(self, bars):
        """Calculates log returns and volatility, then scales features."""
        bars.dropna(inplace=True)
        for k in range(100):
            bars["open_pc"] = bars["open"].pct_change(fill_method=None)
            bars["high_pc"] = bars["high"].pct_change(fill_method=None)
            bars["low_pc"] = bars["low"].pct_change(fill_method=None)
            bars["close_pc"] = bars["close"].pct_change(fill_method=None)
            bars["volume_pc"] = bars["volume"].pct_change(fill_method=None)
            bars["vwap_pc"] = bars["vwap"].pct_change(fill_method=None)
            bars["fracocp"] = (bars["close"] - bars["open"]) / bars["open"]
            bars["frachp"] = (bars["high"] - bars["open"]) / bars["open"]
            bars["fraclp"] = (bars["open"] - bars["low"]) / bars["open"]

            bars["sma_1"] = talib.SMA(bars["close"], timeperiod=self.extra_settings[0])
            bars["sma_2"] = talib.SMA(bars["close"], timeperiod=self.extra_settings[1])
            bars["sma_3"] = talib.SMA(bars["close"], timeperiod=self.extra_settings[2])
            bars["sma_4"] = talib.SMA(bars["close"], timeperiod=self.extra_settings[3])
            bars["sma_1"] = bars["sma_1"].pct_change(fill_method=None)
            bars["sma_2"] = bars["sma_2"].pct_change(fill_method=None)
            bars["sma_3"] = bars["sma_3"].pct_change(fill_method=None)
            bars["sma_4"] = bars["sma_4"].pct_change(fill_method=None)

            bars["ema_1"] = talib.EMA(bars["close"], timeperiod=self.extra_settings[4])
            bars["ema_2"] = talib.EMA(bars["close"], timeperiod=self.extra_settings[5])
            bars["ema_3"] = talib.EMA(bars["close"], timeperiod=self.extra_settings[6])
            bars["ema_4"] = talib.EMA(bars["close"], timeperiod=self.extra_settings[7])
            bars["ema_1"] = bars["ema_1"].pct_change(fill_method=None)
            bars["ema_2"] = bars["ema_2"].pct_change(fill_method=None)
            bars["ema_3"] = bars["ema_3"].pct_change(fill_method=None)
            bars["ema_4"] = bars["ema_4"].pct_change(fill_method=None)

            bars["atr"] = talib.ATR(bars["high"], bars["low"], bars["close"], timeperiod=self.extra_settings[8])
            bars["atr"] = bars["atr"].pct_change(fill_method=None)
            bars["natr"] = talib.NATR(bars["high"], bars["low"], bars["close"], timeperiod=self.extra_settings[9])
            bars["natr"] = bars["natr"].pct_change(fill_method=None)
            bars["rsi"] = (talib.RSI(bars["close"], timeperiod=self.extra_settings[10]) - 50) / 50

            slow_k, slow_d = talib.STOCH(bars["high"], bars["low"], bars["close"], fastk_period=self.extra_settings[11], slowk_period=self.extra_settings[12], slowd_period=self.extra_settings[13])
            bars["slow_k"] = (slow_k - 50) / 50
            bars["slow_d"] = (slow_d - 50) / 50

            bars["three_black_crows"] = talib.CDL3BLACKCROWS(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["three_inside"] = talib.CDL3INSIDE(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["three_lines"] = talib.CDL3LINESTRIKE(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["three_outside"] = talib.CDL3OUTSIDE(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["three_stars"] = talib.CDL3STARSINSOUTH(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["three_whitesoldiers"] = talib.CDL3WHITESOLDIERS(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["abandoned_baby"] = talib.CDLABANDONEDBABY(bars["open"], bars["high"], bars["low"], bars["close"], penetration=self.extra_settings[14]) / 100
            bars["advance_block"] = talib.CDLADVANCEBLOCK(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["belthold"] = talib.CDLBELTHOLD(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["breakaway"] = talib.CDLBREAKAWAY(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["closing_marubozu"] = talib.CDLCLOSINGMARUBOZU(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["conceal_baby"] = talib.CDLCONCEALBABYSWALL(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["counterattack"] = talib.CDLCOUNTERATTACK(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["dark_cloud_cover"] = talib.CDLDARKCLOUDCOVER(bars["open"], bars["high"], bars["low"], bars["close"], penetration=self.extra_settings[15]) / 100
            bars["doji"] = talib.CDLDOJI(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["doji_star"] = talib.CDLDOJISTAR(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["dragonfly_doji"] = talib.CDLDRAGONFLYDOJI(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["engulfing"] = talib.CDLENGULFING(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["evening_doji_star"] = talib.CDLEVENINGDOJISTAR(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["evening_star"] = talib.CDLEVENINGSTAR(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["gap_side_by_side"] = talib.CDLGAPSIDESIDEWHITE(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["gravestone_doji"] = talib.CDLGRAVESTONEDOJI(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["hammer"] = talib.CDLHAMMER(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["hanging_man"] = talib.CDLHANGINGMAN(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["harami"] = talib.CDLHARAMI(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["harami_cross"] = talib.CDLHARAMICROSS(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["high_wave"] = talib.CDLHIGHWAVE(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["hikkake"] = talib.CDLHIKKAKE(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["homing_pigeon"] = talib.CDLHOMINGPIGEON(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["identical_three_crows"] = talib.CDLIDENTICAL3CROWS(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["in_neck"] = talib.CDLINNECK(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["inverted_hammer"] = talib.CDLINVERTEDHAMMER(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["kicking"] = talib.CDLKICKING(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["kicking_by_length"] = talib.CDLKICKINGBYLENGTH(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["ladder_bottom"] = talib.CDLLADDERBOTTOM(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["long_leader"] = talib.CDLLONGLEGGEDDOJI(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["long_line"] = talib.CDLLONGLINE(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["marubozu"] = talib.CDLMARUBOZU(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["matching_low"] = talib.CDLMATCHINGLOW(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["mat_hold"] = talib.CDLMATHOLD(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["morning_doji_star"] = talib.CDLMORNINGDOJISTAR(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["morning_star"] = talib.CDLMORNINGSTAR(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["on_neck"] = talib.CDLONNECK(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["piercing"] = talib.CDLPIERCING(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["rickshaw_man"] = talib.CDLRICKSHAWMAN(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["rise_fall_three_methods"] = talib.CDLRISEFALL3METHODS(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["separating_lines"] = talib.CDLSEPARATINGLINES(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["shooting_star"] = talib.CDLSHOOTINGSTAR(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["short_line"] = talib.CDLSHORTLINE(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["spinning_top"] = talib.CDLSPINNINGTOP(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["stalled_pattern"] = talib.CDLSTALLEDPATTERN(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["stick_sandwich"] = talib.CDLSTICKSANDWICH(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["takuri"] = talib.CDLTAKURI(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["tasuki_gap"] = talib.CDLTASUKIGAP(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["thrusting"] = talib.CDLTHRUSTING(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["tristar"] = talib.CDLTRISTAR(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["unique_3_river"] = talib.CDLUNIQUE3RIVER(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["upside_gap_2_crows"] = talib.CDLUPSIDEGAP2CROWS(bars["open"], bars["high"], bars["low"], bars["close"]) / 100
            bars["side_gap_3_methods"] = talib.CDLXSIDEGAP3METHODS(bars["open"], bars["high"], bars["low"], bars["close"]) / 100

            bars["ad"] = talib.AD(bars["high"], bars["low"], bars["close"], bars["volume"])
            bars["adosc"] = talib.ADOSC(bars["high"], bars["low"], bars["close"], bars["volume"], fastperiod=self.extra_settings[16], slowperiod=self.extra_settings[17])
            bars["obv"] = talib.OBV(bars["close"], bars["volume"])
            bars["ad"] = bars["ad"].pct_change(fill_method=None)
            bars["adosc"] = bars["adosc"].pct_change(fill_method=None)
            bars["obv"] = bars["obv"].pct_change(fill_method=None)

            bars["adx"] = talib.ADX(bars["high"], bars["low"], bars["close"], timeperiod=self.extra_settings[18])
            bars["adx"] = bars["adx"].pct_change(fill_method=None)

            bars["ht_trendline"] = talib.HT_TRENDLINE(bars["close"])
            bars["ht_trendline"] = bars["ht_trendline"].pct_change()

            bars["kama"] = talib.KAMA(bars["close"], timeperiod=self.extra_settings[19])
            bars["kama"] = bars["kama"].pct_change(fill_method=None)

            fastlimit = min(0.05, max(self.extra_settings[20], 0.5))
            slowlimit = min(0.05, max(self.extra_settings[21], 0.5))
            bars["mama"], bars["fama"] = talib.MAMA(bars["close"], fastlimit=fastlimit, slowlimit=slowlimit)
            bars["mama"] = bars["mama"].pct_change(fill_method=None)
            bars["fama"] = bars["fama"].pct_change(fill_method=None)

            bars["sar"] = talib.SAR(bars["high"], bars["low"], acceleration=self.extra_settings[22], maximum=self.extra_settings[23])
            bars["sar"] = bars["sar"].pct_change(fill_method=None)

            bars["returns"] = bars["close"].pct_change(fill_method=None)
            bars["volatility"] = bars["returns"].rolling(window=self.extra_settings[24]).std()
            bars.dropna(inplace=True)

            if bars.empty:
                print("Dataframe is empty. Retrying...")
            else:
                features = bars[self.feature_settings].values
                features_scaled = self.scaler.fit_transform(features)
                return features_scaled, bars
        print("Failed")
        return None

    def fit(self, bars):
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

        regime_stats = bars.groupby("regime")["returns"].mean().sort_values()

        self.regime_mapping = {
            regime_stats.index[0]: "Bear",
            regime_stats.index[1]: "Choppy",
            regime_stats.index[2]: "Bull"
        }

    @staticmethod
    def get_score(bars):
        score = 0
        for i in tqdm(range(bars.shape[0] - 1)):
            predicted = bars.iloc[i].regime
            actual_change = bars.iloc[i + 1].returns

            if ((predicted == "Bull" and actual_change > 0.001)
                    or (predicted == "Bear" and actual_change < -0.001)
                    or (predicted == "Choppy" and abs(actual_change) <= 0.001)):
                score += 1
        return score

    def validate(self, train_bars, test_bars, processes, plot):
        """Trains HMM, evaluates accuracy, and visualizes results."""
        print(f"Training HMM on {train_bars.shape[0]} bars with\nFeatures: {self.feature_settings}\nExtras: {self.extra_settings}")
        try:
            self.fit(train_bars)
        except IndexError as e:
            print("Too little clusters to fit. Skipping validation...")
            return 0.0

        print(f"Predicting regimes on {test_bars.shape[0]} test bars...")
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
                args.append(test_bars[i*bars_per_process:(i+1)*bars_per_process])

            results_async = pool.map_async(self.get_score, args)
            results = results_async.get()
            for result in results:
                correct_predictions += result

            pool.close()
            pool.join()
        else:
            total_predictions = self.get_score(test_bars)

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


def run_regime_test(train_bars_df, test_bars_df, features, extra_settings):
    regime_predictor = HMMRegimePrediction(features, extra_settings)
    regime_predictor.validate(train_bars_df, test_bars_df, 8, True)
    start_time = time.time()
    predicted_regime = regime_predictor.predict_probability(test_bars_df)
    print(f"Predicted:", predicted_regime)
    print("Finished in ", time.time() - start_time)


def run_regime_search(train_bars, test_bars, val_processes=1, n_iterations=-1):
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
        "returns", "volatility"
    ]

    base_extra_settings = [
        10, 30, 50, 200,
        10, 30, 50, 200,
        14, 14,
        14,
        5, 3, 3,
        0.3, 0.5,
        3, 10,
        14,
        30,
        0.5, 0.05,
        0.02, 0.2,
        30
    ]

    feature_combinations = []
    print("Generating combinations")
    for i in tqdm(range(1, 5)):
        feature_combinations.extend(itertools.combinations(all_features, i))
    print(f"Generated {len(feature_combinations)} combinations")

    best_features = []
    best_extra = []
    best_accuracy = 0

    tested = {}

    rows = saving.SaveSystem.read_from_csv(TESTED_PATH)
    for row in rows:
        if row[0] == "Features":
            continue
        key = row[0] + " " + row[1]
        tested[key] = float(row[2])
        if tested[key] > best_accuracy:
            best_features = ast.literal_eval(row[0])
            best_extra = ast.literal_eval(row[1])
            best_accuracy = tested[key]

    print("Starting search")
    i = 0
    start_time = time.time()
    for features in feature_combinations:
        print(f"\nTest {i}/{len(feature_combinations)} {100 * i / len(feature_combinations):.4f}%:")
        features_list = list(features)

        if n_iterations != -1 and i >= n_iterations:
            break

        extra_template = base_extra_settings

        extra_settings = []
        for extra_setting in extra_template:
            if isinstance(extra_setting, int):
                half = extra_setting // 2
                extra_setting += random.randint(-half, half)
            elif isinstance(extra_setting, float):
                extra_setting = random.uniform(0.01, extra_setting * 2)
            extra_settings.append(extra_setting)

        extra_settings = extra_template  # REMOVE LATER

        key = str(features_list) + " " + str(extra_settings)
        if key not in tested:
            regime_predictor = HMMRegimePrediction(features_list, extra_settings)
            accuracy = regime_predictor.validate(train_bars, test_bars, val_processes, False)
            tested[key] = accuracy
            saving.SaveSystem.save_to_csv([features_list, extra_settings, accuracy], TESTED_PATH, "a")
        else:
            accuracy = tested[key]
            print(f"Already tested: {key} - {tested[key]}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = features_list
            best_extra = extra_settings
            print(f"New best accuracy: {best_accuracy}%")
        i += 1

    print(f"Search finished in {time.time() - start_time:.2f} seconds")
    print(f"Best features: {best_features}")
    print(f"Best extras: {best_extra}")
    print(f"Best accuracy: {best_accuracy}%")


if __name__ == "__main__":
    user_input = input("Enter command (search, test): ")
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
        bars_df = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, 5, start_date, end_date, 500000,
                                                         TimeFrameUnit.Minute)
        saving.SaveSystem.save_data(bars_df, DATA_PATH)
    else:
        bars_df = saving.SaveSystem.load_data(DATA_PATH)

    train_size = int(bars_df.shape[0] * 0.8)
    train_bars_df = bars_df[:train_size].copy()
    test_bars_df = bars_df[train_size + 1:].copy()
    print("Train bars: ", train_bars_df.shape[0])
    print("Test bars: ", test_bars_df.shape[0])
    print("Total bars: ", bars_df.shape[0])

    if user_input == "search":
        run_regime_search(train_bars_df, test_bars_df, 8)
    elif user_input == "test":
        extra = [
            10, 30, 50, 200,
            10, 30, 50, 200,
            14, 14,
            14,
            5, 3, 3,
            0.3, 0.5,
            3, 10,
            14,
            30,
            0.5, 0.05,
            0.02, 0.2,
            30
        ]
        run_regime_test(train_bars_df, test_bars_df, ["volatility", "returns"], extra)

    '''run_test_price_predict(16, 50, train_bars, bars_df[train_size + 1:])
    run_test_price_predict(8, 50, train_bars, bars_df[train_size + 1:])
    run_test_price_predict(4, 50, train_bars, bars_df[train_size + 1:])
    run_test_price_predict(2, 50, train_bars, bars_df[train_size + 1:])'''


