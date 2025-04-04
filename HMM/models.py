from hmmlearn.hmm import GaussianHMM
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
        self.regime_mapping = None  # Store regime mapping
        self.processes = processes
        self.fitted_feature_settings = None
        self.feature_index_map = {}

    def get_features(self, bars, feature_settings):
        """Gets features from bars dataframe and scales features."""
        return bars[feature_settings].values

    def get_features_array(self, bars, feature_settings):
        """Gets features from an array of bars and scales features."""
        feature_indices = [self.feature_index_map[feature] for feature in feature_settings]

        # Extract the required columns from the NumPy array
        features = bars[:, feature_indices]

        return features

    def fit(self, bars, feature_settings, seed=42):
        """Fits the HMM model and maps regimes."""
        self.fitted_feature_settings = feature_settings

        np.random.seed(seed)  # For reproducibility

        # Set initial parameters dynamically based on number of features
        # When switching from n features to n+1 features, error occurs sometimes
        self.model.startprob_ = np.full(3, 1.0 / 3)  # Uniform probabilities
        self.model.transmat_ = np.full((3, 3), 1.0 / 3)  # Equal transition probabilities
        self.model.means_ = np.random.rand(3, len(feature_settings))  # Random means with correct shape
        self.model.covars_ = np.full((3, len(feature_settings)), 0.1)  # Small diagonal covariance values

        features = self.get_features(bars, feature_settings)
        for i in range(len(bars.columns)):
            self.feature_index_map[bars.columns[i]] = i
        self.model.fit(features)
        self.map_regimes(bars, features)

    def predict_latest_probability(self, bars):
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        features = self.get_features(bars[-1:], self.fitted_feature_settings)
        prediction_probs = self.model.predict_proba(features)[-1]
        mapped_prediction = {}
        for i in range(prediction_probs.shape[0]):
            mapped_prediction[self.regime_mapping[np.int64(i)]] = float(prediction_probs[i])
        return mapped_prediction

    def predict_probability(self, bars):
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        features = self.get_features(bars, self.fitted_feature_settings)
        prediction_probs = self.model.predict_proba(features)
        mapped_predictions = [
            {self.regime_mapping[np.int64(i)]: float(prob[i]) for i in range(len(prob))}
            for prob in prediction_probs
        ]
        return mapped_predictions

    def predict(self, bars):
        """Predicts market regimes and returns them as mapped labels."""
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        features = self.get_features(bars, self.fitted_feature_settings)
        predicted_regimes = self.model.predict(features)
        return np.array([self.regime_mapping[r] for r in predicted_regimes])

    def predict_array(self, features):
        if self.fitted_feature_settings is None:
            raise ValueError("Model has not been fitted yet.")
        predicted_regimes = self.model.predict(features)
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

    def get_score(self, bars, std_deviation, threshold):
        # NOTE: Sometimes terrible model just guessing one regime for entire period can get 50% accuracy
        correct_predictions = 0
        start_cash = bars.iloc[0].close * 50
        cash = start_cash
        shares = 0.0
        sell_time = None

        # Convert to np array for faster processing
        bars_array = bars.to_numpy()
        bars_index_list = list(bars.index)
        predicted_regimes = []
        # NOTE: sklearn Standard scaler causes look ahead bias.
        # When comparing features calculation once vs. iteratively, the results were different.
        # This is because the scaler got all future data to scale the features while iterative did not.
        # Since our feature data should be normalized between -1 and 1 already, we can just get rid of the scaler.
        features = self.get_features_array(bars_array, self.fitted_feature_settings)

        for i in tqdm(range(len(bars_array) - 1)):
            row = bars_array[i]
            next_row = bars_array[i + 1]

            # Extract values (assuming column order: Index, Close, Close_pc, etc.)
            row_index = bars_index_list[i]  # Retrieve the original index
            row_close = row[self.feature_index_map["close"]]
            actual_change = next_row[self.feature_index_map["close_pc"]]

            predicted = self.predict_array(features[:i+1])[-1]
            predicted_regimes.append(predicted)

            if predicted == "Bull":
                correct_predictions += actual_change > threshold * std_deviation
                if cash > 0 and (sell_time is None or row_index.date() != sell_time):
                    shares = cash / row_close
                    cash = 0.0
            elif predicted == "Bear":
                correct_predictions += actual_change < -threshold * std_deviation
                if shares > 0:
                    cash = (shares * row_close) * 0.995  # 0.5% fee
                    shares = 0.0
                    sell_time = row_index.date()
            else:
                correct_predictions += abs(actual_change) <= threshold * std_deviation

        # Update dataframe with predicted regimes
        predicted_regimes.append(self.predict_array(features)[-1])  # Add the last predicted regime
        bars["regime"] = predicted_regimes

        return correct_predictions, (cash + shares * bars.iloc[-1].close) - start_cash

    def validate(self, train_bars, test_bars, feature_settings, plot, seed=0):
        """Trains HMM, evaluates accuracy, and visualizes results."""
        print(f"Training HMM on {train_bars.shape[0]} bars at {seed} seed with\nFeatures: {feature_settings}")

        train_bars.dropna(inplace=True)
        test_bars.dropna(inplace=True)

        try:
            self.fit(train_bars, feature_settings, seed=seed)
        except IndexError as e:
            print(f"Too little clusters to fit. Skipping validation...")
            return 0.0, 0.0

        minimum, maximum, mean, median, std_deviation = hmm_test.get_stats(pd.concat([train_bars, test_bars]), "close_pc", False)
        print(f"Predicting regimes on {test_bars.shape[0]} test bars with stdv {std_deviation}...")

        # Calculate Accuracy
        correct_predictions = 0
        total_profit = 0.0
        total_predictions = len(test_bars) - 1  # Ignore last row due to comparing predicted with future price

        pool = Pool(processes=self.processes)
        if self.processes > 1:
            args = []
            bars_per_process = test_bars.shape[0] // self.processes

            for i in range(self.processes):
                # Threshold for Choppy needs to be small assuming normal (which it is from close_pc histogram)
                # Hopefully this means a bad combination that guesses Choppy for everything cant get high accuracy
                args.append((test_bars[i*bars_per_process:(i+1)*bars_per_process], std_deviation, 0.1))

            results_async = pool.starmap_async(self.get_score, args)
            results = results_async.get()
            for correct, profit in results:
                correct_predictions += correct
                total_profit += profit

            pool.close()
            pool.join()
        else:
            correct_predictions, total_profit = self.get_score(test_bars, std_deviation, 0.1)

        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy: {accuracy:.2f}%, profit: {total_profit:.2f}")

        # Plot stock prices with color-coded regimes
        if plot:
            profit_percent = (total_profit / (test_bars.iloc[0].close * 50)) * 100
            stock_change = ((test_bars.iloc[-1].close - test_bars.iloc[0].close) / test_bars.iloc[0].close) * 100
            print(f"Profit percentage: {profit_percent:.2f}%")
            print(f"Stock change: {stock_change:.2f}%")
            print(f"Beat market by: {profit_percent - stock_change:.2f}%")

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
            plt.title(f"Stock Price with Predicted Market Regimes (Accuracy: {accuracy:.2f}%, profit: {total_profit:.2f})")
            plt.show()

        return accuracy, total_profit


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

