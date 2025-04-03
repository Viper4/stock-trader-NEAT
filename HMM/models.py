from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import hmm_test


class HMMRegimePrediction(object):
    def __init__(self, processes):
        self.model = GaussianHMM(n_components=3, n_iter=10000, covariance_type="diag", init_params="")
        self.scaler = StandardScaler()  # Store scaler for consistent transformation
        self.regime_mapping = None  # Store regime mapping
        self.processes = processes
        self.fitted_feature_settings = None

    def get_features(self, bars, feature_settings):
        """Gets features from bars and scales features."""
        bars.dropna(inplace=True)
        features = bars[feature_settings].values
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled, bars

    def fit(self, bars, feature_settings, seed=42):
        """Fits the HMM model and maps regimes."""
        bars.dropna(inplace=True)
        self.fitted_feature_settings = feature_settings

        np.random.seed(seed)  # For reproducibility

        # Set initial parameters dynamically based on n_dim
        self.model.startprob_ = np.full(3, 1.0 / 3)  # Uniform probabilities
        self.model.transmat_ = np.full((3, 3), 1.0 / 3)  # Equal transition probabilities
        self.model.means_ = np.random.rand(3, len(feature_settings))  # Random means with correct shape
        self.model.covars_ = np.full((3, len(feature_settings)), 0.1)  # Small diagonal covariance values

        features_scaled, bars = self.get_features(bars, feature_settings)
        self.model.fit(features_scaled)
        self.map_regimes(bars, features_scaled)

    def predict_latest_probability(self, bars):
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        features_scaled, bars = self.get_features(bars[-1:], self.fitted_feature_settings)
        prediction_probs = self.model.predict_proba(features_scaled)[-1]
        mapped_prediction = {}
        for i in range(prediction_probs.shape[0]):
            mapped_prediction[self.regime_mapping[np.int64(i)]] = float(prediction_probs[i])
        return mapped_prediction

    def predict_probability(self, bars):
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        features_scaled, bars = self.get_features(bars, self.fitted_feature_settings)
        prediction_probs = self.model.predict_proba(features_scaled)
        mapped_predictions = [
            {self.regime_mapping[np.int64(i)]: float(prob[i]) for i in range(len(prob))}
            for prob in prediction_probs
        ]
        return mapped_predictions

    def predict(self, bars):
        """Predicts market regimes and returns them as mapped labels."""
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        features_scaled, bars = self.get_features(bars, self.fitted_feature_settings)
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
            actual_change = bars.iloc[i + 1].close_pc

            if ((predicted == "Bull" and actual_change > threshold * std_deviation)
                    or (predicted == "Bear" and actual_change < -threshold * std_deviation)
                    or (predicted == "Choppy" and abs(actual_change) <= threshold * std_deviation)):
                score += 1
        return score

    def validate(self, train_bars, test_bars, feature_settings, plot, seed=0):
        """Trains HMM, evaluates accuracy, and visualizes results."""
        print(f"Training HMM on {train_bars.shape[0]} bars at {seed} seed with\nFeatures: {feature_settings}")

        try:
            self.fit(train_bars, feature_settings, seed=seed)
        except IndexError as e:
            print(f"Too little clusters to fit. Skipping validation...")
            return 0.0

        minimum, maximum, mean, median, std_deviation = hmm_test.get_stats(pd.concat([train_bars, test_bars]), "close_pc", False)
        print(f"Predicting regimes on {test_bars.shape[0]} test bars with stdv {std_deviation}...")
        predicted_labels = self.predict(test_bars)
        test_bars["regime"] = predicted_labels

        # Calculate Accuracy
        correct_predictions = 0
        total_predictions = len(test_bars) - 1  # Ignore last row due to comparing predicted with future price

        pool = Pool(processes=self.processes)
        if self.processes > 1:
            args = []
            bars_per_process = test_bars.shape[0] // self.processes

            for i in range(self.processes):
                # Threshold for Choppy of within 0.25*stdv or ~19.75% of data assuming normal (which it is from close_pc histogram)
                # Hopefully this means a bad combination that guesses Choppy for everything can only get ~20% accuracy
                args.append((test_bars[i*bars_per_process:(i+1)*bars_per_process], std_deviation, 0.25))

            results_async = pool.starmap_async(self.get_score, args)
            results = results_async.get()
            for result in results:
                correct_predictions += result

            pool.close()
            pool.join()
        else:
            correct_predictions = self.get_score(test_bars, std_deviation, 0.25)

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

