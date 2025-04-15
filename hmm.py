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
import ast
from datetime import timedelta
import HMM.models as models
import HMM.feature_selection as feature_selection
from HMM.correlation import CorrelationAnalysis
import HMM.trading

DATA_PATH = PROJECT_DIR + "\\HMM\\bars-data-DXYZ-1d_2019-1-1_2025-4-14.gz"
TESTED_PATH = PROJECT_DIR + "\\HMM\\tested2-DXYZ-1d_2019-1-1_2025-4-14.csv"


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


def run_regime_test(train_bars, test_bars, features, seed, plot, plot_label=""):
    regime_predictor = models.HMMRegimePrediction()
    regime_predictor.validate(train_bars, test_bars, features, plot, seed, plot_label)
    start_time = time.time()
    predicted_regimes = regime_predictor.predict_probability(test_bars_df)
    print(f"Last 10 predictions:")
    for i in range(10):
        print(str(i+1) + ": " + str(predicted_regimes[-10 + i]))
    print("Finished in", time.time() - start_time, "seconds")
    print()


def run_regime_search(train_bars, test_bars, features, n_iterations=-1, n_seeds=20):
    feature_combinations = []
    print("Generating combinations")
    for i in range(3, 5):
        feature_combinations.extend(itertools.combinations(features, i))
    print(f"Generated {len(feature_combinations)} combinations")

    tested = {}
    best_accuracy = 0.0
    best_profit = -999999.0

    if not os.path.exists(TESTED_PATH):
        saving.SaveSystem.make_csv(["Features", "Seed", "Accuracy", "Profit%", "Label Order"], TESTED_PATH)
    rows = saving.SaveSystem.read_from_csv(TESTED_PATH)
    for row in rows:
        if row[0] == "Features":
            continue
        key = row[0] + row[1]
        tested[key] = (float(row[2]), float(row[3]))

    print("Starting search")
    regime_predictor = models.HMMRegimePrediction()
    for i in range(len(feature_combinations)):
        if n_iterations != -1 and i >= n_iterations:
            break
        start_time = time.time()
        print(f"\nTest {i}/{len(feature_combinations)} {100 * i / len(feature_combinations):.4f}%:")
        features_list = list(feature_combinations[i])

        for j in range(n_seeds):
            key = str(features_list) + str(j)
            if key not in tested:
                accuracy, profit_percent, label_order = regime_predictor.validate(train_bars, test_bars, features_list, False, seed=j)
                tested[key] = accuracy
                saving.SaveSystem.save_to_csv([features_list, j, accuracy, profit_percent, label_order], TESTED_PATH, "a")
            else:
                accuracy, profit_percent = tested[key]
                print(f"Already tested: {key} - {tested[key]}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"New best accuracy: {best_accuracy}%")

            if profit_percent > best_profit:
                best_profit = profit_percent
                print(f"New best profit: {best_profit}")
        iteration_time = time.time() - start_time
        eta = (len(feature_combinations) - i) * iteration_time
        print(f"Finished in {iteration_time:.2f} seconds. ETA: {str(timedelta(seconds=eta))}")

    print(f"Search finished")


def get_stats(bars, columns, plot):
    minimums = []
    maximums = []
    means = []
    medians = []
    standard_deviations = []
    for i in range(len(columns)):
        minimums.append(min(bars[columns[i]]))
        maximums.append(max(bars[columns[i]]))
        means.append(bars[columns[i]].mean())
        medians.append(bars[columns[i]].median())
        standard_deviations.append(bars[columns[i]].std())

        if plot:
            print(f"Statistics for {columns[i]}:")
            print(" Min: ", minimums[i])
            print(" Max: ", maximums[i])
            print(" Mean: ", means[i])
            print(" Median: ", medians[i])
            print(" Standard Deviation: ", standard_deviations[i])

            plt.figure(figsize=(10, 6))
            plt.hist(bars[columns[i]], bins=50)
            plt.xlabel(columns[i])
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {columns[i]}")
            plt.show()

    if plot:
        plt.figure(figsize=(12, 6))

        for column in columns:
            # Plot column
            plt.plot(bars.index, bars[column], color="blue", label=column, alpha=0.7)

        plt.legend()
        plt.title(f"{columns} over time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()

    return minimums, maximums, means, medians, standard_deviations


def run_random_forest(bars, features, seed):
    run_regime_test(bars, bars, features, seed, False)
    random_forest = feature_selection.RandomForestFeatureSelection()
    random_forest.evaluate(bars, features)


def get_best(sort_by):
    results = []
    rows = saving.SaveSystem.read_from_csv(TESTED_PATH)
    for row in rows:
        if row[0] == "Features":
            continue
        # Features, Seed, Accuracy, Profit%, Label Order
        results.append((row[0], int(row[1]), float(row[2]), float(row[3]), row[4]))

    if sort_by == 1:
        results.sort(key=lambda x: x[2], reverse=True)
    else:
        results.sort(key=lambda x: x[3], reverse=True)
    res_features, res_seed, res_accuracy, res_profit, res_label_order = results[0]
    print("Best features:", res_features)
    print("Best seed:", res_seed)
    print("Best accuracy:", res_accuracy)
    print("Best profit:", res_profit)
    print("Best label order:", res_label_order)

    save_path = TESTED_PATH.replace(".csv", "_sorted.csv")
    saving.SaveSystem.delete_file(save_path)
    saving.SaveSystem.make_csv(["Features", "Seed", "Accuracy", "Profit%", "Label Order"], save_path)

    sorted_keys = []
    sorted_accuracies = []
    sorted_profits = []
    for feature_settings, seed, accuracy, profit_percent, label_order in results:
        sorted_keys.append(feature_settings + " " + str(seed))
        sorted_accuracies.append(accuracy)
        sorted_profits.append(profit_percent)
        saving.SaveSystem.save_to_csv([feature_settings, seed, accuracy, profit_percent, label_order], save_path, "a")

    if sort_by == 1:
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_keys[:100], sorted_accuracies[:100], color="skyblue")
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Feature Combinations")
        plt.title("Feature Combination Accuracy Rankings")
        plt.gca().invert_yaxis()  # Best at top
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_keys[:100], sorted_profits[:100], color="skyblue")
        plt.xlabel("Profit (%)")
        plt.ylabel("Feature Combinations")
        plt.title("Feature Combination Profit Rankings")
        plt.gca().invert_yaxis()  # Best at top
        plt.show()

    return results


if __name__ == "__main__":
    user_input = input("Enter command (brute force, price test, reg test, reg best, correlation, stats, random forest): ")
    with open(SETTINGS_PATH) as file:
        settings = json.load(file)
    alpaca_api = REST(settings["profiles"][0]["public_key"], settings["profiles"][0]["secret_key"],
                      base_url=URL("https://paper-api.alpaca.markets"))
    now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
    unit_map = {"minute": TimeFrameUnit.Minute, "day": TimeFrameUnit.Day, "week": TimeFrameUnit.Week,
                "month": TimeFrameUnit.Month, "hour": TimeFrameUnit.Hour}

    if not os.path.exists(DATA_PATH):
        symbol = input("Enter symbol: ")
        start = input("Enter start date (YYYY-MM-DD): ")
        end = input("Enter end date (YYYY-MM-DD): ")
        interval = int(input("Enter interval (1, 5, 15, 30): "))
        unit_input = input("Enter interval unit (minute, day, week, month, hour): ")
        start_date = dt.datetime.strptime(start, "%Y-%m-%d").replace(hour=9, minute=30,
                                                                     tzinfo=pytz.timezone("US/Eastern"))
        end_date = dt.datetime.strptime(end, "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=pytz.timezone("US/Eastern"))

        if end_date >= now_date - dt.timedelta(minutes=16):
            end_date = now_date - dt.timedelta(minutes=16)
        bars_df = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, interval, start_date, end_date, 500000, unit_map[unit_input])

        models.HMMRegimePrediction.augment_bars(bars_df)

        saving.SaveSystem.save_data(bars_df, DATA_PATH)
    else:
        bars_df = saving.SaveSystem.load_data(DATA_PATH)

    train_size = int(bars_df.shape[0] * 0.7)
    train_bars_df = bars_df[:train_size].copy()
    test_bars_df = bars_df[train_size + 1:].copy()
    print("Train bars:", train_bars_df.shape[0])
    print("Test bars:", test_bars_df.shape[0])
    print("Total bars:", bars_df.shape[0])

    all_features = [
        "open_pc", "high_pc", "low_pc", "close_pc", "volume_pc", "vwap_pc", "trade_count_pc",
        "fracocp", "frachp", "fraclp",
        "sma_a", "sma_b", "sma_c", "sma_d",
        "ema_a", "ema_b", "ema_c", "ema_d",
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
        "volatility", "macd", "macdsignal", "macdhist"
    ]

    if user_input == "brute force":
        num_seeds = int(input("Number of seeds to check (10 is good): "))
        run_regime_search(train_bars_df, test_bars_df, all_features, n_seeds=num_seeds)
    elif user_input == "price test":
        run_price_test(int(input("Number of components: ")), int(input("Number of latent bars: ")), train_bars_df, test_bars_df)
    elif user_input == "reg test":
        test_features = ast.literal_eval(input("Type features, Ex: ['close_pc', 'ema_1']: "))
        test_seed = int(input("Seed: "))
        plot_label = input("Plot label: ")
        run_regime_test(train_bars_df, test_bars_df, test_features, test_seed, True, plot_label)
    elif user_input == "correlation":
        feature_input = input("Type features or 'all', Ex: ['close_pc', 'ema_1']: ")
        seed = int(input("Seed: "))
        if feature_input == "all":
            corr_features = all_features
        else:
            corr_features = ast.literal_eval(feature_input)
        run_regime_test(train_bars_df, test_bars_df, corr_features, seed, True)
        correlation = CorrelationAnalysis(corr_features)
        correlation_matrix = correlation.compute_feature_correlation(test_bars_df)
        print("Correlation matrix:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(correlation_matrix)
        correlation.plot_correlation(correlation_matrix)
    elif user_input == "reg best":
        results = get_best(int(input("Sort by 1 for accuracy or 2 for profit: ")))

        i = 1
        last_feature_settings = None
        for feature_settings, seed, accuracy, profit_percent, label_order in results:
            if feature_settings != last_feature_settings:
                print(f"#{i}:")
                run_regime_test(train_bars_df, test_bars_df, ast.literal_eval(feature_settings), seed, True)
                last_feature_settings = feature_settings
            i += 1
    elif user_input == "stats":
        columns = ast.literal_eval(input("Enter columns, Ex: ['close_pc', 'ema_1']: "))
        get_stats(pd.concat([train_bars_df, test_bars_df]), columns, True)
    elif user_input == "random forest":
        for i in range(10):
            print(f"Seed {i}")
            run_random_forest(bars_df, all_features, i)
    elif user_input == "reg now":
        symbol = input("Enter symbol: ")
        interval = int(input("Enter interval (1, 5, 15, 30): "))
        unit_input = input("Enter interval unit (minute, day, week, month, hour): ")

        profile = settings["profiles"][0]

        regime_settings = []
        for stock in profile["stocks"]:
            if stock["symbol"] == symbol:
                regime_settings = stock["regime_settings"]
                break

        now_date = dt.datetime.now(dt.timezone.utc)
        bars_df = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, interval,
                                                         now_date - dt.timedelta(days=profile["general_regime_settings"]["fit_days"]),
                                                         now_date - dt.timedelta(minutes=16),
                                                         500000, unit_map[unit_input])
        models.HMMRegimePrediction.augment_bars(bars_df)

        averaged_predictions = {"Bull": 0, "Bear": 0, "Choppy": 0}

        for i in range(len(regime_settings)):
            regime_predictor = models.HMMRegimePrediction()
            regime_predictor.fit(bars_df, regime_settings[i]["features"], regime_settings[i]["seed"])
            prediction = regime_predictor.predict_probability(bars_df)[-1].tolist()

            label_order = regime_settings[i]["label_order"]
            ordered_prediction = {"Bull": prediction[label_order["Bull"]] * 100,
                                  "Bear": prediction[label_order["Bear"]] * 100,
                                  "Choppy": prediction[label_order["Choppy"]] * 100}
            print(f"\n{regime_settings[i]['features']} prediction:\n{ordered_prediction}")
            averaged_predictions["Bull"] += ordered_prediction["Bull"]
            averaged_predictions["Bear"] += ordered_prediction["Bear"]
            averaged_predictions["Choppy"] += ordered_prediction["Choppy"]

        averaged_predictions["Bull"] /= len(regime_settings)
        averaged_predictions["Bear"] /= len(regime_settings)
        averaged_predictions["Choppy"] /= len(regime_settings)
        print(f"\nAverage predictions for {symbol}:\n{averaged_predictions}")
    elif user_input == "price now":
        symbol = input("Enter symbol: ")
        interval = int(input("Enter interval (1, 5, 15, 30): "))
        unit_input = input("Enter interval unit (minute, day, week, month, hour): ")
        num_components = int(input("Enter number of components: "))
        num_latent_bars = int(input("Enter number of latent bars: "))

        now_date = dt.datetime.now(dt.timezone.utc)
        latest_bars = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, interval,
                                                         now_date - dt.timedelta(days=num_latent_bars * 24),
                                                         now_date - dt.timedelta(minutes=16),
                                                         500000, unit_map[unit_input])
        price_predictor = models.HMMPricePrediction(num_components, num_latent_bars)
        price_predictor.fit(latest_bars)

        augmented_bars = price_predictor.augment_bars(latest_bars)
        possible_outcomes = price_predictor.get_possible_outcomes(augmented_bars)
        features = price_predictor.get_features(augmented_bars)

        close_price = latest_bars["close"].iloc[-1]
        price_predictions = price_predictor.predict_full(close_price, possible_outcomes, features).tolist()
        print(f"Top 5 Price Predictions for {symbol}:")
        for i in range(0, 5):
            percent_change = (price_predictions[i] - close_price) / close_price
            print(f"{i+1}: {price_predictions[i]} {percent_change * 100}%")
    elif user_input == "reg one":
        symbol = input("Enter symbol: ")
        interval = int(input("Enter interval (1, 5, 15, 30): "))
        unit_input = input("Enter interval unit (minute, day, week, month, hour): ")
        date = dt.datetime.strptime(input("Enter date (YYYY-MM-DD): "), "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=pytz.timezone("US/Eastern"))
        now_date = dt.datetime.now(dt.timezone.utc)
        if date > now_date - dt.timedelta(minutes=16):
            date = now_date - dt.timedelta(minutes=16)

        profile = settings["profiles"][0]

        regime_settings = []
        for stock in profile["stocks"]:
            if stock["symbol"] == symbol:
                regime_settings = stock["regime_settings"]
                break

        now_date = dt.datetime.now(dt.timezone.utc)
        bars_df = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, interval,
                                                         date - dt.timedelta(days=profile["general_regime_settings"]["fit_days"]),
                                                         date,
                                                         500000, unit_map[unit_input])
        models.HMMRegimePrediction.augment_bars(bars_df)

        averaged_predictions = {"Bull": 0, "Bear": 0, "Choppy": 0}

        for i in range(len(regime_settings)):
            regime_predictor = models.HMMRegimePrediction()
            regime_predictor.fit(bars_df, regime_settings[i]["features"], regime_settings[i]["seed"])
            prediction = regime_predictor.predict_probability(bars_df)[-1].tolist()

            label_order = regime_settings[i]["label_order"]
            ordered_prediction = {"Bull": prediction[label_order["Bull"]] * 100,
                                  "Bear": prediction[label_order["Bear"]] * 100,
                                  "Choppy": prediction[label_order["Choppy"]] * 100}
            print(f"\n{regime_settings[i]['features']} prediction:\n{ordered_prediction}")
            averaged_predictions["Bull"] += ordered_prediction["Bull"]
            averaged_predictions["Bear"] += ordered_prediction["Bear"]
            averaged_predictions["Choppy"] += ordered_prediction["Choppy"]

        averaged_predictions["Bull"] /= len(regime_settings)
        averaged_predictions["Bear"] /= len(regime_settings)
        averaged_predictions["Choppy"] /= len(regime_settings)
        print(f"\nAverage predictions on {date.isoformat()} for {symbol}:\n{averaged_predictions}")
    elif user_input == "price one":
        symbol = input("Enter symbol: ")
        interval = int(input("Enter interval (1, 5, 15, 30): "))
        unit_input = input("Enter interval unit (minute, day, week, month, hour): ")
        num_components = int(input("Enter number of components: "))
        num_latent_bars = int(input("Enter number of latent bars: "))
        date = dt.datetime.strptime(input("Enter date (YYYY-MM-DD): "), "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=pytz.timezone("US/Eastern"))
        now_date = dt.datetime.now(dt.timezone.utc)

        if date > now_date - dt.timedelta(minutes=16):
            date = now_date - dt.timedelta(minutes=16)
        latest_bars = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, interval,
                                                             date - dt.timedelta(days=num_latent_bars * 32),
                                                             date,
                                                             500000, unit_map[unit_input])
        price_predictor = models.HMMPricePrediction(num_components, num_latent_bars)
        price_predictor.fit(latest_bars[:-1])

        augmented_bars = price_predictor.augment_bars(latest_bars[:-1])
        possible_outcomes = price_predictor.get_possible_outcomes(augmented_bars)
        features = price_predictor.get_features(augmented_bars)

        close_price = latest_bars["close"].iloc[-2]
        actual_close = latest_bars["close"].iloc[-1]
        price_predictions = price_predictor.predict_full(close_price, possible_outcomes, features).tolist()
        print(f"Top 5 Price Predictions on {date.isoformat()} for {symbol}:")
        for i in range(0, 5):
            percent_change = (price_predictions[i] - close_price) / close_price
            percent_error = abs(price_predictions[i] - actual_close) / price_predictions[i]
            print(f"{i + 1}: {price_predictions[i]} {percent_change * 100}% | Err: {percent_error * 100}%")
    elif user_input == "trade":
        HMM.trading.trade(settings, alpaca_api)
