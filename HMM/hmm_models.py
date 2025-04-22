from hmmlearn.hmm import GaussianHMM
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
import talib


class HMMRegimePrediction(object):
    def __init__(self):
        self.model = GaussianHMM(n_components=3, n_iter=10000, covariance_type="diag", init_params="")
        self.regime_mapping = None  # Store regime mapping
        self.fitted_feature_settings = None
        self.feature_index_map = {}  # For array indexing to accomodate fast lookups
        self.scaler = StandardScaler()

    @staticmethod
    def augment_bars(bars_df):
        """Generate indicator data and percent change data from bars dataframe and save it to the dataframe."""
        bars_df["open_pc"] = bars_df["open"].pct_change(fill_method=None)
        bars_df["high_pc"] = bars_df["high"].pct_change(fill_method=None)
        bars_df["low_pc"] = bars_df["low"].pct_change(fill_method=None)
        bars_df["close_pc"] = bars_df["close"].pct_change(fill_method=None)
        bars_df["volume_pc"] = bars_df["volume"].pct_change(fill_method=None)
        bars_df["vwap_pc"] = bars_df["vwap"].pct_change(fill_method=None)
        bars_df["trade_count_pc"] = bars_df["trade_count"].pct_change(fill_method=None)
        bars_df["fracocp"] = (bars_df["close"] - bars_df["open"]) / bars_df["open"]
        bars_df["frachp"] = (bars_df["high"] - bars_df["open"]) / bars_df["open"]
        bars_df["fraclp"] = (bars_df["open"] - bars_df["low"]) / bars_df["open"]

        bars_df["sma_a"] = talib.SMA(bars_df["close"], timeperiod=10)
        bars_df["sma_b"] = talib.SMA(bars_df["close"], timeperiod=30)
        bars_df["sma_c"] = talib.SMA(bars_df["close"], timeperiod=50)
        bars_df["sma_d"] = talib.SMA(bars_df["close"], timeperiod=200)
        bars_df["sma_a"] = bars_df["sma_a"].pct_change(fill_method=None)
        bars_df["sma_b"] = bars_df["sma_b"].pct_change(fill_method=None)
        bars_df["sma_c"] = bars_df["sma_c"].pct_change(fill_method=None)
        bars_df["sma_d"] = bars_df["sma_d"].pct_change(fill_method=None)

        bars_df["ema_a"] = talib.EMA(bars_df["close"], timeperiod=10)
        bars_df["ema_b"] = talib.EMA(bars_df["close"], timeperiod=30)
        bars_df["ema_c"] = talib.EMA(bars_df["close"], timeperiod=50)
        bars_df["ema_d"] = talib.EMA(bars_df["close"], timeperiod=200)
        bars_df["ema_a"] = bars_df["ema_a"].pct_change(fill_method=None)
        bars_df["ema_b"] = bars_df["ema_b"].pct_change(fill_method=None)
        bars_df["ema_c"] = bars_df["ema_c"].pct_change(fill_method=None)
        bars_df["ema_d"] = bars_df["ema_d"].pct_change(fill_method=None)

        bars_df["atr"] = talib.ATR(bars_df["high"], bars_df["low"], bars_df["close"], timeperiod=14)
        bars_df["atr"] = bars_df["atr"].pct_change(fill_method=None)
        bars_df["natr"] = talib.NATR(bars_df["high"], bars_df["low"], bars_df["close"], timeperiod=14)
        bars_df["natr"] = bars_df["natr"].pct_change(fill_method=None)
        bars_df["tr"] = talib.TRANGE(bars_df["high"], bars_df["low"], bars_df["close"])
        bars_df["tr"] = bars_df["tr"].pct_change(fill_method=None)
        bars_df["rsi"] = (talib.RSI(bars_df["close"], timeperiod=14) - 50) / 50

        slow_k, slow_d = talib.STOCH(bars_df["high"], bars_df["low"], bars_df["close"], fastk_period=5,
                                     slowk_period=3, slowd_period=3)
        bars_df["slow_k"] = (slow_k - 50) / 50
        bars_df["slow_d"] = (slow_d - 50) / 50

        bars_df["three_black_crows"] = talib.CDL3BLACKCROWS(bars_df["open"], bars_df["high"], bars_df["low"],
                                                            bars_df["close"]) / 100
        bars_df["three_inside"] = talib.CDL3INSIDE(bars_df["open"], bars_df["high"], bars_df["low"],
                                                   bars_df["close"]) / 100
        bars_df["three_lines"] = talib.CDL3LINESTRIKE(bars_df["open"], bars_df["high"], bars_df["low"],
                                                      bars_df["close"]) / 100
        bars_df["three_outside"] = talib.CDL3OUTSIDE(bars_df["open"], bars_df["high"], bars_df["low"],
                                                     bars_df["close"]) / 100
        bars_df["three_stars"] = talib.CDL3STARSINSOUTH(bars_df["open"], bars_df["high"], bars_df["low"],
                                                        bars_df["close"]) / 100
        bars_df["three_whitesoldiers"] = talib.CDL3WHITESOLDIERS(bars_df["open"], bars_df["high"], bars_df["low"],
                                                                 bars_df["close"]) / 100
        bars_df["abandoned_baby"] = talib.CDLABANDONEDBABY(bars_df["open"], bars_df["high"], bars_df["low"],
                                                           bars_df["close"],
                                                           penetration=0.3) / 100
        bars_df["advance_block"] = talib.CDLADVANCEBLOCK(bars_df["open"], bars_df["high"], bars_df["low"],
                                                         bars_df["close"]) / 100
        bars_df["belthold"] = talib.CDLBELTHOLD(bars_df["open"], bars_df["high"], bars_df["low"],
                                                bars_df["close"]) / 100
        bars_df["breakaway"] = talib.CDLBREAKAWAY(bars_df["open"], bars_df["high"], bars_df["low"],
                                                  bars_df["close"]) / 100
        bars_df["closing_marubozu"] = talib.CDLCLOSINGMARUBOZU(bars_df["open"], bars_df["high"], bars_df["low"],
                                                               bars_df["close"]) / 100
        bars_df["conceal_baby"] = talib.CDLCONCEALBABYSWALL(bars_df["open"], bars_df["high"], bars_df["low"],
                                                            bars_df["close"]) / 100
        bars_df["counterattack"] = talib.CDLCOUNTERATTACK(bars_df["open"], bars_df["high"], bars_df["low"],
                                                          bars_df["close"]) / 100
        bars_df["dark_cloud_cover"] = talib.CDLDARKCLOUDCOVER(bars_df["open"], bars_df["high"], bars_df["low"],
                                                              bars_df["close"],
                                                              penetration=0.5) / 100
        bars_df["doji"] = talib.CDLDOJI(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["doji_star"] = talib.CDLDOJISTAR(bars_df["open"], bars_df["high"], bars_df["low"],
                                                 bars_df["close"]) / 100
        bars_df["dragonfly_doji"] = talib.CDLDRAGONFLYDOJI(bars_df["open"], bars_df["high"], bars_df["low"],
                                                           bars_df["close"]) / 100
        bars_df["engulfing"] = talib.CDLENGULFING(bars_df["open"], bars_df["high"], bars_df["low"],
                                                  bars_df["close"]) / 100
        bars_df["evening_doji_star"] = talib.CDLEVENINGDOJISTAR(bars_df["open"], bars_df["high"], bars_df["low"],
                                                                bars_df["close"]) / 100
        bars_df["evening_star"] = talib.CDLEVENINGSTAR(bars_df["open"], bars_df["high"], bars_df["low"],
                                                       bars_df["close"]) / 100
        bars_df["gap_side_by_side"] = talib.CDLGAPSIDESIDEWHITE(bars_df["open"], bars_df["high"], bars_df["low"],
                                                                bars_df["close"]) / 100
        bars_df["gravestone_doji"] = talib.CDLGRAVESTONEDOJI(bars_df["open"], bars_df["high"], bars_df["low"],
                                                             bars_df["close"]) / 100
        bars_df["hammer"] = talib.CDLHAMMER(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["hanging_man"] = talib.CDLHANGINGMAN(bars_df["open"], bars_df["high"], bars_df["low"],
                                                     bars_df["close"]) / 100
        bars_df["harami"] = talib.CDLHARAMI(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["harami_cross"] = talib.CDLHARAMICROSS(bars_df["open"], bars_df["high"], bars_df["low"],
                                                       bars_df["close"]) / 100
        bars_df["high_wave"] = talib.CDLHIGHWAVE(bars_df["open"], bars_df["high"], bars_df["low"],
                                                 bars_df["close"]) / 100
        bars_df["hikkake"] = talib.CDLHIKKAKE(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["homing_pigeon"] = talib.CDLHOMINGPIGEON(bars_df["open"], bars_df["high"], bars_df["low"],
                                                         bars_df["close"]) / 100
        bars_df["identical_three_crows"] = talib.CDLIDENTICAL3CROWS(bars_df["open"], bars_df["high"], bars_df["low"],
                                                                    bars_df["close"]) / 100
        bars_df["in_neck"] = talib.CDLINNECK(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["inverted_hammer"] = talib.CDLINVERTEDHAMMER(bars_df["open"], bars_df["high"], bars_df["low"],
                                                             bars_df["close"]) / 100
        bars_df["kicking"] = talib.CDLKICKING(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["kicking_by_length"] = talib.CDLKICKINGBYLENGTH(bars_df["open"], bars_df["high"], bars_df["low"],
                                                                bars_df["close"]) / 100
        bars_df["ladder_bottom"] = talib.CDLLADDERBOTTOM(bars_df["open"], bars_df["high"], bars_df["low"],
                                                         bars_df["close"]) / 100
        bars_df["long_leader"] = talib.CDLLONGLEGGEDDOJI(bars_df["open"], bars_df["high"], bars_df["low"],
                                                         bars_df["close"]) / 100
        bars_df["long_line"] = talib.CDLLONGLINE(bars_df["open"], bars_df["high"], bars_df["low"],
                                                 bars_df["close"]) / 100
        bars_df["marubozu"] = talib.CDLMARUBOZU(bars_df["open"], bars_df["high"], bars_df["low"],
                                                bars_df["close"]) / 100
        bars_df["matching_low"] = talib.CDLMATCHINGLOW(bars_df["open"], bars_df["high"], bars_df["low"],
                                                       bars_df["close"]) / 100
        bars_df["mat_hold"] = talib.CDLMATHOLD(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["morning_doji_star"] = talib.CDLMORNINGDOJISTAR(bars_df["open"], bars_df["high"], bars_df["low"],
                                                                bars_df["close"]) / 100
        bars_df["morning_star"] = talib.CDLMORNINGSTAR(bars_df["open"], bars_df["high"], bars_df["low"],
                                                       bars_df["close"]) / 100
        bars_df["on_neck"] = talib.CDLONNECK(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["piercing"] = talib.CDLPIERCING(bars_df["open"], bars_df["high"], bars_df["low"],
                                                bars_df["close"]) / 100
        bars_df["rickshaw_man"] = talib.CDLRICKSHAWMAN(bars_df["open"], bars_df["high"], bars_df["low"],
                                                       bars_df["close"]) / 100
        bars_df["rise_fall_three_methods"] = talib.CDLRISEFALL3METHODS(bars_df["open"], bars_df["high"], bars_df["low"],
                                                                       bars_df["close"]) / 100
        bars_df["separating_lines"] = talib.CDLSEPARATINGLINES(bars_df["open"], bars_df["high"], bars_df["low"],
                                                               bars_df["close"]) / 100
        bars_df["shooting_star"] = talib.CDLSHOOTINGSTAR(bars_df["open"], bars_df["high"], bars_df["low"],
                                                         bars_df["close"]) / 100
        bars_df["short_line"] = talib.CDLSHORTLINE(bars_df["open"], bars_df["high"], bars_df["low"],
                                                   bars_df["close"]) / 100
        bars_df["spinning_top"] = talib.CDLSPINNINGTOP(bars_df["open"], bars_df["high"], bars_df["low"],
                                                       bars_df["close"]) / 100
        bars_df["stalled_pattern"] = talib.CDLSTALLEDPATTERN(bars_df["open"], bars_df["high"], bars_df["low"],
                                                             bars_df["close"]) / 100
        bars_df["stick_sandwich"] = talib.CDLSTICKSANDWICH(bars_df["open"], bars_df["high"], bars_df["low"],
                                                           bars_df["close"]) / 100
        bars_df["takuri"] = talib.CDLTAKURI(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["tasuki_gap"] = talib.CDLTASUKIGAP(bars_df["open"], bars_df["high"], bars_df["low"],
                                                   bars_df["close"]) / 100
        bars_df["thrusting"] = talib.CDLTHRUSTING(bars_df["open"], bars_df["high"], bars_df["low"],
                                                  bars_df["close"]) / 100
        bars_df["tristar"] = talib.CDLTRISTAR(bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"]) / 100
        bars_df["unique_3_river"] = talib.CDLUNIQUE3RIVER(bars_df["open"], bars_df["high"], bars_df["low"],
                                                          bars_df["close"]) / 100
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

        bars_df["volatility"] = bars_df["close_pc"].rolling(window=30).std()

        bars_df["macd"], bars_df["macdsignal"], bars_df["macdhist"] = talib.MACD(bars_df["close"], fastperiod=12,
                                                                                 slowperiod=26, signalperiod=9)
        bars_df["macd"] = bars_df["macd"].pct_change(fill_method=None)
        bars_df["macdsignal"] = bars_df["macdsignal"].pct_change(fill_method=None)
        bars_df["macdhist"] = bars_df["macdhist"].pct_change(fill_method=None)

        bars_df.replace(np.nan, 0.0, inplace=True)
        bars_df.replace(np.inf, 1.0, inplace=True)
        bars_df.replace(-np.inf, -1.0, inplace=True)

    def get_features_fit(self, bars, feature_settings):
        """Gets features from bars dataframe, fits scaler, and scales features."""
        return self.scaler.fit_transform(bars[feature_settings].values)

    def get_features_df(self, bars, feature_settings):
        """Gets features from bars dataframe and scales features using scaler calculated with the training data."""
        return (bars[feature_settings].values - self.scaler.mean_) / self.scaler.scale_

    def get_features_array(self, bars, feature_settings):
        """Gets features from an array of bars and scales features using scaler calculated with the training data."""
        feature_indices = [self.feature_index_map[feature] for feature in feature_settings]

        # Extract the required columns from the NumPy array
        features = bars[:, feature_indices]

        # Use mean and stdv from scaler calculated with the training data
        features_scaled = (features - self.scaler.mean_) / self.scaler.scale_

        return features_scaled

    def fit(self, bars, feature_settings, seed=42):
        """Fits the HMM model and maps regimes."""
        np.random.seed(seed)  # For reproducibility

        # Set initial parameters dynamically based on number of features
        # When switching from n features to n+1 features, error occurs due to mismatch in dimensions
        self.model.startprob_ = np.full(3, 1.0 / 3)  # Uniform probabilities
        self.model.transmat_ = np.full((3, 3), 1.0 / 3)  # Equal transition probabilities
        self.model.means_ = np.random.rand(3, len(feature_settings))  # Random means with correct shape
        self.model.covars_ = np.full((3, len(feature_settings)), 0.1)  # Small diagonal covariance values

        features_scaled = self.get_features_fit(bars, feature_settings)
        for i in range(len(bars.columns)):
            self.feature_index_map[bars.columns[i]] = i
        self.model.fit(features_scaled)

        self.fitted_feature_settings = feature_settings

    def predict_probability(self, bars):
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        features_scaled = self.get_features_df(bars, self.fitted_feature_settings)
        prediction_probs = self.model.predict_proba(features_scaled)
        return prediction_probs

    def predict(self, bars):
        """Predicts market regimes and returns them as mapped labels."""
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        features_scaled = self.get_features_df(bars, self.fitted_feature_settings)
        predicted_regimes = self.model.predict(features_scaled)
        return predicted_regimes

    def predict_array(self, features_scaled):
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        predicted_regimes = self.model.predict(features_scaled)
        return predicted_regimes

    def predict_array_prob(self, features_scaled):
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        predicted_regimes = self.model.predict_proba(features_scaled)
        return predicted_regimes

    def predict_and_label_prob(self, bars):
        label_orders = [{"Bear": 0, "Bull": 1, "Choppy": 2},
                        {"Bear": 0, "Choppy": 1, "Bull": 2},
                        {"Bull": 0, "Bear": 1, "Choppy": 2},
                        {"Bull": 0, "Choppy": 1, "Bear": 2},
                        {"Choppy": 0, "Bear": 1, "Bull": 2},
                        {"Choppy": 0, "Bull": 1, "Bear": 2}]

        start_cash = bars.iloc[0].close * 50
        cash = [start_cash] * len(label_orders)
        shares = [0.0] * len(label_orders)

        # Convert to np array for faster processing
        bars_array = bars.to_numpy()
        features_scaled = self.get_features_array(bars_array, self.fitted_feature_settings)
        predicted_regimes = self.predict_array_prob(features_scaled)

        for i in range(len(bars_array)):
            row = bars_array[i]

            # Extract values
            row_close = row[self.feature_index_map["close"]]

            if predicted_regimes[i][0] > 0.5:
                # -Bear, Bull, Choppy
                if shares[0] > 0:
                    cash[0] = (shares[0] * row_close) * 0.995  # 0.5% fee
                    shares[0] = 0.0

                # -Bear, Choppy, Bull
                if shares[1] > 0:
                    cash[1] = (shares[1] * row_close) * 0.995  # 0.5% fee
                    shares[1] = 0.0

                # -Bull, Bear, Choppy
                if cash[2] > 0:
                    shares[2] = cash[2] / row_close
                    cash[2] = 0.0

                # -Bull, Choppy, Bear
                if cash[3] > 0:
                    shares[3] = cash[3] / row_close
                    cash[3] = 0.0

                # -Choppy, Bear, Bull

                # -Choppy, Bull, Bear

            elif predicted_regimes[i][1] > 0.5:
                # Bear, -Bull, Choppy
                if cash[0] > 0:
                    shares[0] = cash[0] / row_close
                    cash[0] = 0.0

                # Bear, -Choppy, Bull

                # Bull, -Bear, Choppy
                if shares[2] > 0:
                    cash[2] = (shares[2] * row_close) * 0.995  # 0.5% fee
                    shares[2] = 0.0

                # Bull, -Choppy, Bear

                # Choppy, -Bear, Bull
                if shares[4] > 0:
                    cash[4] = (shares[4] * row_close) * 0.995  # 0.5% fee
                    shares[4] = 0.0

                # Choppy, -Bull, Bear
                if cash[5] > 0:
                    shares[5] = cash[5] / row_close
                    cash[5] = 0.0
            elif predicted_regimes[i][2] > 0.5:
                # Bear, Bull, -Choppy

                # Bear, Choppy, -Bull
                if cash[1] > 0:
                    shares[1] = cash[1] / row_close
                    cash[1] = 0.0

                # Bull, Bear, -Choppy

                # Bull, Choppy, -Bear
                if shares[3] > 0:
                    cash[3] = (shares[3] * row_close) * 0.995  # 0.5% fee
                    shares[3] = 0.0

                # Choppy, Bear, -Bull
                if cash[4] > 0:
                    shares[4] = cash[4] / row_close
                    cash[4] = 0.0

                # Choppy, Bull, -Bear
                if shares[5] > 0:
                    cash[5] = (shares[5] * row_close) * 0.995  # 0.5% fee
                    shares[5] = 0.0

        best_index = 0
        last_close = bars.iloc[-1].close
        best_profit = (cash[0] + shares[0] * last_close) - start_cash
        for i in range(1, len(label_orders)):
            profit = (cash[i] + shares[i] * last_close) - start_cash
            if profit > best_profit:
                best_index = i
                best_profit = profit

        return predicted_regimes, label_orders[best_index]

    def get_score(self, bars, std_deviation, threshold):
        # NOTE: Sometimes terrible model just guessing one regime for entire period can get 50% accuracy

        label_orders = [["Bear", "Bull", "Choppy"],
                        ["Bear", "Choppy", "Bull"],
                        ["Bull", "Bear", "Choppy"],
                        ["Bull", "Choppy", "Bear"],
                        ["Choppy", "Bear", "Bull"],
                        ["Choppy", "Bull", "Bear"]]

        correct_predictions = [0] * len(label_orders)

        start_cash = bars.iloc[0].close * 50
        cash = [start_cash] * len(label_orders)
        shares = [0.0] * len(label_orders)

        # Convert to np array for faster processing
        bars_array = bars.to_numpy()
        predicted_regimes = []
        features_scaled = self.get_features_array(bars_array, self.fitted_feature_settings)

        for i in tqdm(range(len(bars_array) - 1)):
            row = bars_array[i]
            next_row = bars_array[i + 1]

            # Extract values
            row_close = row[self.feature_index_map["close"]]
            actual_change = next_row[self.feature_index_map["close_pc"]]

            # Make predictions
            try:
                predicted = self.predict_array(features_scaled[:i + 1])[-1].item()
            except ValueError as e:
                print("Problem with prediction:", e)
                predicted = 1
            predicted_regimes.append(predicted)

            bear_correct = actual_change < -threshold * std_deviation
            bull_correct = actual_change > threshold * std_deviation
            choppy_correct = abs(actual_change) <= threshold * std_deviation

            if predicted == 0:
                # -Bear, Bull, Choppy
                correct_predictions[0] += bear_correct
                if shares[0] > 0:
                    cash[0] = (shares[0] * row_close) * 0.995  # 0.5% fee
                    shares[0] = 0.0

                # -Bear, Choppy, Bull
                correct_predictions[1] += bear_correct
                if shares[1] > 0:
                    cash[1] = (shares[1] * row_close) * 0.995  # 0.5% fee
                    shares[1] = 0.0

                # -Bull, Bear, Choppy
                correct_predictions[2] += bull_correct
                if cash[2] > 0:
                    shares[2] = cash[2] / row_close
                    cash[2] = 0.0

                # -Bull, Choppy, Bear
                correct_predictions[3] += bull_correct
                if cash[3] > 0:
                    shares[3] = cash[3] / row_close
                    cash[3] = 0.0

                # -Choppy, Bear, Bull
                correct_predictions[4] += choppy_correct

                # -Choppy, Bull, Bear
                correct_predictions[5] += choppy_correct
            elif predicted == 1:
                # Bear, -Bull, Choppy
                correct_predictions[0] += bull_correct
                if cash[0] > 0:
                    shares[0] = cash[0] / row_close
                    cash[0] = 0.0

                # Bear, -Choppy, Bull
                correct_predictions[1] += choppy_correct

                # Bull, -Bear, Choppy
                correct_predictions[2] += bear_correct
                if shares[2] > 0:
                    cash[2] = (shares[2] * row_close) * 0.995  # 0.5% fee
                    shares[2] = 0.0

                # Bull, -Choppy, Bear
                correct_predictions[3] += choppy_correct

                # Choppy, -Bear, Bull
                correct_predictions[4] += bear_correct
                if shares[4] > 0:
                    cash[4] = (shares[4] * row_close) * 0.995  # 0.5% fee
                    shares[4] = 0.0

                # Choppy, -Bull, Bear
                correct_predictions[5] += bull_correct
                if cash[5] > 0:
                    shares[5] = cash[5] / row_close
                    cash[5] = 0.0
            else:
                # Bear, Bull, -Choppy
                correct_predictions[0] += choppy_correct

                # Bear, Choppy, -Bull
                correct_predictions[1] += bull_correct
                if cash[1] > 0:
                    shares[1] = cash[1] / row_close
                    cash[1] = 0.0

                # Bull, Bear, -Choppy
                correct_predictions[2] += choppy_correct

                # Bull, Choppy, -Bear
                correct_predictions[3] += bear_correct
                if shares[3] > 0:
                    cash[3] = (shares[3] * row_close) * 0.995  # 0.5% fee
                    shares[3] = 0.0

                # Choppy, Bear, -Bull
                correct_predictions[4] += bull_correct
                if cash[4] > 0:
                    shares[4] = cash[4] / row_close
                    cash[4] = 0.0

                # Choppy, Bull, -Bear
                correct_predictions[5] += bear_correct
                if shares[5] > 0:
                    cash[5] = (shares[5] * row_close) * 0.995  # 0.5% fee
                    shares[5] = 0.0

        best_index = 0
        last_close = bars.iloc[-1].close
        best_profit = (cash[0] + shares[0] * last_close) - start_cash
        for i in range(1, len(label_orders)):
            profit = (cash[i] + shares[i] * last_close) - start_cash
            if profit > best_profit:
                best_index = i
                best_profit = profit

        # Update dataframe with predicted regimes
        try:
            predicted_regimes.append(self.predict_array(features_scaled)[-1].item())  # Add the last predicted regime
        except ValueError as e:
            print("Problem with prediction:", e)
            predicted_regimes.append(label_orders[best_index].index("Choppy"))
        bars["regime"] = predicted_regimes

        return correct_predictions[best_index], (best_profit / start_cash) * 100, label_orders[best_index]

    def validate(self, train_bars, test_bars, feature_settings, plot, seed=0, plot_label=""):
        """Trains HMM, evaluates accuracy, and visualizes results."""
        print(f"Training HMM on {train_bars.shape[0]} bars at {seed} seed with\nFeatures: {feature_settings}")

        train_bars.dropna(inplace=True)
        test_bars.dropna(inplace=True)

        try:
            self.fit(train_bars, feature_settings, seed=seed)
        except IndexError as e:
            print(f"Too little clusters to fit. Skipping validation...")
            return 0.0, 0.0
        except ValueError as e:
            print(f"Problem with data. Skipping validation...")
            return 0.0, 0.0

        std_deviation = pd.concat([train_bars, test_bars])["close_pc"].std()
        # std_deviation = hmm_test.get_stats(pd.concat([train_bars, test_bars]), ["close_pc"], False)[4][0]
        print(f"Predicting regimes on {test_bars.shape[0]} test bars with stdv {std_deviation}...")

        # Calculate Accuracy
        total_predictions = len(test_bars) - 1  # Ignore last row due to comparing predicted with future price

        correct_predictions, profit_percent, label_order = self.get_score(test_bars, std_deviation, 0.1)

        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy: {accuracy:.2f}%, profit: {profit_percent:.2f}%, label order: {label_order}")

        # Plot stock prices with color-coded regimes
        if plot:
            stock_change = ((test_bars.iloc[-1].close - test_bars.iloc[0].close) / test_bars.iloc[0].close) * 100
            print(f"Profit percentage: {profit_percent:.2f}%")
            print(f"Stock change: {stock_change:.2f}%")
            print(f"Beat market by: {profit_percent - stock_change:.2f}%")

            plt.figure(figsize=(15, 6))
            plt.plot(test_bars.index, test_bars["close"], color="black", label="Stock Price")

            colors = {"Bull": "green", "Bear": "red", "Choppy": "yellow"}
            for i in range(len(label_order)):
                plt.fill_between(
                    test_bars.index,
                    test_bars["close"].min(),
                    test_bars["close"].max(),
                    where=test_bars["regime"] == i,
                    color=colors[label_order[i]],
                    alpha=0.3,
                    label=label_order[i]
                )
            '''for regime, color in colors.items():
                plt.fill_between(
                    test_bars.index,
                    test_bars["close"].min(),
                    test_bars["close"].max(),
                    where=test_bars["regime"] == regime,
                    color=color,
                    alpha=0.3,
                    label=regime
                )'''

            plt.legend()
            if plot_label != "":
                plt.title(f"{plot_label}\n{feature_settings}-{seed} Stock Price with Regimes\n(Accuracy: {accuracy:.2f}%, profit: {profit_percent:.2f}%)")
            else:
                plt.title(f"{feature_settings}-{seed} Stock Price with Regimes\n(Accuracy: {accuracy:.2f}%, profit: {profit_percent:.2f}%)")
            plt.show()

        return accuracy, profit_percent, label_order


class HMMPricePrediction(object):
    def __init__(self, num_components, num_latent_bars):
        self.model = GaussianHMM(n_components=num_components, init_params="")
        self.num_components = num_components
        self.num_latent_bars = num_latent_bars

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

    def fit_augmented(self, augmented_bars, seed):
        np.random.seed(seed)  # For reproducibility

        # Set initial parameters dynamically based on number of features
        # When switching from n features to n+1 features, error occurs due to mismatch in dimensions
        self.model.startprob_ = np.full(3, 1.0 / 3)  # Uniform probabilities
        self.model.transmat_ = np.full((3, 3), 1.0 / 3)  # Equal transition probabilities
        self.model.means_ = np.random.rand(3, self.num_components)  # Random means with correct shape
        self.model.covars_ = np.full((3, self.num_components), 0.1)  # Small diagonal covariance values

        features = self.get_features(augmented_bars)
        self.model.fit(features)

    def fit(self, bars, seed):
        np.random.seed(seed)  # For reproducibility

        # Set initial parameters dynamically based on number of features
        # When switching from n features to n+1 features, error occurs due to mismatch in dimensions
        self.model.startprob_ = np.full(self.num_components, 1 / self.num_components)  # Uniform probabilities
        self.model.transmat_ = np.full((self.num_components, self.num_components), 1 / self.num_components)  # Equal transition probabilities
        self.model.means_ = np.random.rand(self.num_components, 3)  # Random means for each state
        self.model.covars_ = np.full((self.num_components, 3), 0.1)  # Small diagonal covariance values

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

    def predict_full(self, open_price, possible_outcomes, features):
        outcomes_copy = possible_outcomes.copy()
        scores = []
        predictions = []
        for row in outcomes_copy.itertuples():
            scores.append(self.model.score(np.vstack((features, row.outcome))))
            predictions.append(open_price * (1 + row.outcome[0]))
        outcomes_copy["score"] = scores
        outcomes_copy["predicted_price"] = predictions

        # Sort the outcomes based on probability scores
        outcomes_copy = outcomes_copy.sort_values(by="score", ascending=False)
        return outcomes_copy["predicted_price"]

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

    def validate(self, bars, plot):
        print("Validating HMM Price Prediction...")

        possible_outcomes = self.get_possible_outcomes(self.augment_bars(bars))
        predicted_close_prices = []

        for i in tqdm(range(self.num_latent_bars, bars.shape[0])):
            # Calculate start and end indices
            previous_data_start_index = max(0, i - self.num_latent_bars)
            # Acquire test data features for these days
            previous_data = self.get_features(self.augment_bars(bars[previous_data_start_index:i]))

            predicted_close_prices.append(self.predict(bars.iloc[i].open, possible_outcomes, previous_data))

        x_axis = np.array(bars.index[self.num_latent_bars:], dtype='datetime64[ms]')
        if plot:
            plt.figure(figsize=(30, 10), dpi=80)
            plt.rcParams.update({'font.size': 18})

            plt.plot(x_axis, bars[self.num_latent_bars:]["close"], 'b+-', label="Actual close prices")
            plt.plot(x_axis, predicted_close_prices, 'ro-', label="Predicted close prices")
            plt.legend(prop={'size': 20})
            plt.show()

        ae = abs(bars[self.num_latent_bars:]["close"] - predicted_close_prices)
        min_ae = min(ae)
        max_ae = max(ae)
        avg_ae = sum(ae) / ae.shape[0]

        print("Min Error: ", min_ae)
        print("Max Error: ", max_ae)
        print("Avg Error: ", avg_ae)

        if plot:
            plt.figure(figsize=(30, 10), dpi=80)
            plt.plot(x_axis, ae, 'go-', label="Error")
            plt.legend(prop={'size': 20})
            plt.show()

        return avg_ae
