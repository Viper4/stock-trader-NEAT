import time
import pytz
import datetime as dt
import pandas as pd
import schwab
import candle_scraper as cs
import HMM.hmm_models as models
import Managers.base_manager
from alpaca_trade_api.rest import TimeFrameUnit
import logging
from constants import LOG_DIR


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
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{LOG_DIR}\\HMM_trading.log', encoding='utf-8', level=logging.INFO)
    logged_start = False

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
            now_date = dt.datetime.now(pytz.timezone("US/Central"))
            wait_time = (clock.next_open - now_date).total_seconds()
            print("Market is closed")
            print(f"Market opens in {wait_time / 3600} hours\n-----")
            if logged_start:
                logger.info(f"Stopped trading at {now_date}")
            time.sleep(wait_time)

            now_date = dt.datetime.now(pytz.timezone("US/Central"))
            print("Market is open")
            print(f"Market closes in {(clock.next_close - now_date).total_seconds() / 3600} hours\n-----")
            logger.info(f"Started trading at {now_date}")
            logged_start = True
        elif not logged_start:
            now_date = dt.datetime.now(pytz.timezone("US/Central"))
            print("Market is open")
            print(f"Market closes in {(clock.next_close - now_date).total_seconds() / 3600} hours\n-----")
            logger.info(f"Started trading at {now_date}")
            logged_start = True
        now_date = dt.datetime.now(pytz.timezone("US/Central"))

        account = schwab_api.get_account()
        for symbol in stock_regime_settings:
            position = schwab_api.get_position(symbol)
            latest_df = scraper.get_latest_candles(symbol, interval=str(regime_interval) + unit_yf)[0]
            start_date = now_date - dt.timedelta(days=selected_profile["general_regime_settings"]["fit_days"])
            end_date = now_date - dt.timedelta(days=1)
            previous_df = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, regime_interval, start_date, end_date, 500000, unit_tf)
            combined_df = pd.concat([previous_df, latest_df], ignore_index=False)
            combined_df.drop_duplicates(inplace=True)
            predictors[symbol].augment_bars(combined_df, stock_regime_settings[symbol]["features"])

            predictors[symbol].fit(combined_df, stock_regime_settings[symbol]["features"], stock_regime_settings[symbol]["seed"])
            prediction = predictors[symbol].predict_probability(combined_df)[-1]

            label_order = stock_regime_settings[symbol]["label_order"]
            close_price = combined_df.iloc[-1].close
            change = close_price - combined_df.iloc[-2].close

            shares = 0
            avg_price = 0
            profit_loss = 0
            if position["longQuantity"] > 0:
                avg_price = position["averageLongPrice"]
                profit_loss = (position['longOpenProfitLoss'] / position['marketValue'])*100
                shares = position["longQuantity"]

            print(f"{symbol} {now_date.isoformat()}")
            print(f" Shares: {shares}, Avg Price: ${avg_price:.2f}, Profit/Loss: {profit_loss:.2f}%")
            print(f" Current price: ${close_price:.2f}, Day Change: {change:.2f} ({(change / close_price)*100:.2f}%)")
            print(f" Bull: {prediction[label_order['Bull']]*100:.2f}%, Bear: {prediction[label_order['Bear']]*100:.2f}%, Choppy: {prediction[label_order['Choppy']]*100:.2f}%")

            if prediction[label_order["Bull"]] > 0.5:
                unsettled_cash = account["currentBalances"]["unsettledCash"]

                if "marketValue" not in position:
                    market_value = 0
                else:
                    market_value = position["marketValue"]
                #equity = settled_cash + used_cash
                #print(account["currentBalances"])
                equity = account["currentBalances"]["liquidationValue"] - unsettled_cash
                #exit(0)
                max_cash_to_spend = (equity * selected_profile["max_cash_percent"]) / len(stock_regime_settings)
                cash_to_spend = max_cash_to_spend - market_value
                quantity = int(cash_to_spend / close_price)

                if cash_to_spend > 0 and quantity > 0:
                    print(f" Buy {quantity} shares at ${close_price}")
                    schwab_api.submit_order(symbol, quantity, "BUY")
                    logger.info(f"{symbol} {now_date.isoformat()}: Buy {quantity} shares at ${close_price} (Bull: {prediction[label_order['Bull']]*100:.2f}%, Bear: {prediction[label_order['Bear']]*100:.2f}%, Choppy: {prediction[label_order['Choppy']]*100:.2f}%)")
                else:
                    print(f" Reached cash limit of ${max_cash_to_spend}")
                    logger.info(f"{symbol} {now_date.isoformat()}: Reached cash limit of ${max_cash_to_spend} (Bull: {prediction[label_order['Bull']]*100:.2f}%, Bear: {prediction[label_order['Bear']]*100:.2f}%, Choppy: {prediction[label_order['Choppy']]*100:.2f}%)")
            elif prediction[label_order["Bear"]] > 0.5:
                if shares == 0:
                    print(f" No shares to sell")
                    logger.info(f"{symbol} {now_date.isoformat()}: No shares to sell (Bull: {prediction[label_order['Bull']]*100:.2f}%, Bear: {prediction[label_order['Bear']]*100:.2f}%, Choppy: {prediction[label_order['Choppy']]*100:.2f}%)")
                else:
                    profit = shares * close_price - (avg_price * shares)
                    print(f" Sell {shares} shares at {close_price} for ${profit} profit")
                    schwab_api.submit_order(symbol, shares, "SELL")
                    logger.info(f"{symbol} {now_date.isoformat()}: Sell {shares} shares at {close_price} for ${profit} profit (Bull: {prediction[label_order['Bull']]*100:.2f}%, Bear: {prediction[label_order['Bear']]*100:.2f}%, Choppy: {prediction[label_order['Choppy']]*100:.2f}%)")
            else:
                print(f" Holding")
        time.sleep(selected_profile["interval"] * 60)
