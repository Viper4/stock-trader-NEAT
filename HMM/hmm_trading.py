import time
import pytz
import datetime as dt
import pandas as pd
import schwab
import candle_scraper as cs
import HMM.hmm_models as models
import Managers.base_manager
from alpaca_trade_api.rest import TimeFrameUnit


def try_get_clock(alpaca_api):
    tries = 1
    while True:
        try:
            return alpaca_api.get_clock()
        except Exception as e:
            print(f"Error getting clock: '{e}'. Retrying in 5 seconds... ({tries})")
            time.sleep(5)
            tries += 1


def trade(settings, alpaca_api):
    print("Select profile:")
    for profile in settings["profiles"]:
        if input(f"{profile['name']} (y/n): ") == "y":
            selected_profile = profile
            break
    else:
        print("No profile selected")
        return

    predictors = {}
    stock_regime_settings = {}
    for stock in selected_profile["stocks"]:
        if stock["trading"]:
            print(f"Select regime settings to use for {stock['symbol']}: ")
            for regime_setting in stock["regime_settings"]:
                if input(f"{regime_setting['features']} (y/n): ") == "y":
                    predictors[stock["symbol"]] = models.HMMRegimePrediction()
                    stock_regime_settings[stock["symbol"]] = regime_setting
                    break

    print(f"Trading every {selected_profile['interval']} minutes with regime settings {stock_regime_settings}")
    if input("Proceed? (y/n): ") != "y":
        return

    regime_interval = selected_profile["general_regime_settings"]["interval"]
    unit_map_tf = {"minute": TimeFrameUnit.Minute, "day": TimeFrameUnit.Day, "week": TimeFrameUnit.Week,
                "month": TimeFrameUnit.Month, "hour": TimeFrameUnit.Hour}
    unit_map_yf = {"minute": "m", "day": "d", "week": "wk", "month": "mo", "hour": "h"}
    unit_tf = unit_map_tf[selected_profile["general_regime_settings"]["unit"]]
    unit_yf = unit_map_yf[selected_profile["general_regime_settings"]["unit"]]
    schwab_api = schwab.Schwab()
    scraper = cs.Scraper()
    while True:
        clock = try_get_clock(alpaca_api)
        if not clock.is_open:
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            wait_time = (clock.next_open - now_date).total_seconds()
            print("Market is closed")
            print(f"Market opens in {wait_time / 3600} hours\n-----")
            time.sleep(wait_time)

            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            print("Market is open")
            print(f"Market closes in {(clock.next_close - now_date).total_seconds() / 3600} hours\n-----")
        now_date = dt.datetime.now(pytz.timezone("US/Eastern"))

        account = schwab_api.get_account()
        for symbol in stock_regime_settings:
            position = schwab_api.get_position(symbol)
            latest_df = scraper.get_latest_candles(symbol, interval=str(regime_interval) + unit_yf)[0]
            start_date = now_date - dt.timedelta(days=selected_profile["general_regime_settings"]["fit_days"])
            end_date = now_date - dt.timedelta(days=1)
            previous_df = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, regime_interval, start_date, end_date, 500000, unit_tf)
            combined_df = pd.concat([previous_df, latest_df], ignore_index=False)
            combined_df.drop_duplicates(inplace=True)
            predictors[symbol].augment_bars(combined_df, False)

            predictors[symbol].fit(combined_df, stock_regime_settings[symbol]["features"],
                                   stock_regime_settings[symbol]["seed"])
            prediction = predictors[symbol].predict_probability(combined_df)[-1]

            label_order = stock_regime_settings[symbol]["label_order"]
            close_price = combined_df.iloc[-1].close

            print(f"{symbol} {now_date.isoformat()}")
            print(f" Shares: {position['longQuantity']}, Avg Price: ${position['averageLongPrice']}, Profit/Loss: {(position['longOpenProfitLoss'] / position['marketValue'])*100:.2f}%")
            print(f" Bull: {prediction[label_order['Bull']]*100:.2f}%, Bear: {prediction[label_order['Bear']]*100:.2f}%, Choppy: {prediction[label_order['Choppy']]*100:.2f}%")

            if prediction[label_order["Bull"]] > 0.5:
                unsettled_cash = account["currentBalances"]["unsettledCash"]
                settled_cash = account["currentBalances"]["cashAvailableForTrading"] - unsettled_cash

                if "longMarketValue" in account["currentBalances"]:
                    market_value = account["currentBalances"]["longMarketValue"]
                else:
                    market_value = 0
                used_cash = market_value + unsettled_cash
                equity = settled_cash + used_cash
                max_cash_to_spend = equity * selected_profile["max_cash_percent"]
                cash_to_spend = max_cash_to_spend - used_cash
                if cash_to_spend > 0:
                    quantity = cash_to_spend / close_price

                    print(f" Buy {quantity} shares at ${close_price}")
                    schwab_api.submit_order(symbol, int(quantity), "BUY")
                else:
                    print(f" Not enough cash to buy")
            elif prediction[label_order["Bear"]] > 0.5:
                quantity = position["longQuantity"]
                profit = quantity * close_price - (position["averageLongPrice"] * quantity)
                print(f" Sell {quantity} shares at {close_price} for ${profit} profit")
                schwab_api.submit_order(symbol, quantity, "SELL")
            else:
                print(f" Holding at ${close_price}")
        time.sleep(selected_profile["interval"] * 60)
