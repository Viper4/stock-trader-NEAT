from neat import nn
import json
import time
import datetime as dt
import pytz
import os
import saving
import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit
import subprocess


class Manager(object):
    def __init__(self, settings, finbert):
        self.running = False
        self.settings = settings
        self.finbert = finbert
        self.sessions = {}
        self.log_path = self.settings["save_path"] + "\\Logs\\"
        saving.SaveSystem.make_dir(self.log_path)

    @staticmethod
    def check_internet_connection():
        tries = 0
        while True:
            try:
                subprocess.check_output(["ping", "-c", "1", "8.8.8.8"])
                break
            except subprocess.CalledProcessError:
                print(f"No internet connection. ({tries})")
                time.sleep(5)
                tries += 1

    @staticmethod
    def get_bars(symbol, alpaca_api, interval, start, end):
        tries = 1
        while True:
            try:
                bars_df = alpaca_api.get_bars(
                    symbol=symbol,
                    timeframe=TimeFrame(interval, TimeFrameUnit.Minute),
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=500000,
                    sort="asc",
                    adjustment="all").df
                bars_df = bars_df.tz_convert("US/Eastern")
                bars_df = bars_df.between_time("9:30", "16:00")
                return bars_df.reset_index().to_dict("records")
            except Exception as e:
                Manager.check_internet_connection()
                print(f"Error getting bars: '{e}'. Retrying in 5 seconds... ({tries})")
                tries += 1
                time.sleep(5)

    @staticmethod
    def get_settings_and_alpaca(profile_index):
        # Settings
        local_dir = os.path.dirname(__file__)
        settings_path = os.path.join(local_dir, "settings.json")
        with open(settings_path) as file:
            settings = json.load(file)

        # Alpaca API
        first_profile = settings["profiles"][profile_index]
        alpaca_api = alpaca.REST(first_profile["public_key"], first_profile["secret_key"], base_url=URL("https://paper-api.alpaca.markets"))

        return settings, alpaca_api

    @staticmethod
    def update_net(for_agent, genome, alpaca_api, interval, profile_name, k_period, d_period, rsi_period):
        k_period = for_agent.days_to_bars(k_period, interval)
        d_period = for_agent.days_to_bars(d_period, interval)
        rsi_period = for_agent.days_to_bars(rsi_period, interval)
        d_alpha = 2 / (d_period + 1)
        k_alpha = 2 / (k_period + 1)

        for_agent.net = nn.RecurrentNetwork.create(genome, for_agent.config)

        # Preparing the network with past 20 days data
        now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
        stock_bars = for_agent.trader.get_bars(for_agent.stock["symbol"], alpaca_api, interval, now_date - dt.timedelta(days=20), now_date - dt.timedelta(minutes=16))
        sp500_bars = for_agent.trader.get_bars("SPY", alpaca_api, interval, now_date - dt.timedelta(days=20), now_date - dt.timedelta(minutes=16))
        nasdaq_bars = for_agent.trader.get_bars("QQQ", alpaca_api, interval, now_date - dt.timedelta(days=20), now_date - dt.timedelta(minutes=16))

        prev_d_ema = stock_bars[0]["close"]
        prev_k_ema = stock_bars[0]["close"]

        gain = 0
        loss = 0

        for i in range(1, len(stock_bars)):
            stock_bar = stock_bars[i]
            sp500_bar = sp500_bars[i]
            nasdaq_bar = nasdaq_bars[i]
            prev_stock_bar = stock_bars[i - 1]
            prev_sp500_bar = sp500_bars[i - 1]
            prev_nasdaq_bar = nasdaq_bars[i - 1]
            backtest_date = stock_bars[i]["timestamp"].to_pydatetime()
            stock_sentiment = for_agent.trader.finbert.get_saved_sentiment(for_agent.stock["symbol"],
                                                                           backtest_date - dt.timedelta(days=2),
                                                                           backtest_date)
            sp500_sentiment = for_agent.trader.finbert.get_saved_sentiment("SPY",
                                                                           backtest_date - dt.timedelta(days=2),
                                                                           backtest_date)
            nasdaq_sentiment = for_agent.trader.finbert.get_saved_sentiment("QQQ",
                                                                            backtest_date - dt.timedelta(days=2),
                                                                            backtest_date)

            k_percent = for_agent.calculate_k_percent(stock_bars[i - min(k_period, i):i])

            # %D = EMA(%K, N) or SMA(%K, N)
            d_ema = for_agent.calculate_ema(stock_bar["close"], d_alpha, prev_d_ema)
            prev_d_ema = d_ema
            k_ema = for_agent.calculate_ema(stock_bar["close"], k_alpha, prev_k_ema)
            prev_k_ema = k_ema

            change = stock_bar["close"] - prev_stock_bar["close"]
            if change > 0:
                gain += change
            else:
                loss += abs(change)

            # Remove old data
            start_rsi_index = i - min(rsi_period, i)
            if (i - start_rsi_index) + 1 >= rsi_period:
                start_change = stock_bars[start_rsi_index]["close"] - stock_bars[start_rsi_index - 1]["close"]
                if start_change > 0:
                    gain -= change
                else:
                    loss -= abs(change)

            rsi = for_agent.calculate_rsi(gain, loss, (i - start_rsi_index) + 1)

            inputs = [1,  # -1 = short, 1 = long
                      0,  # plpc
                      for_agent.rel_change(prev_stock_bar["open"], stock_bar["open"]),
                      for_agent.rel_change(prev_stock_bar["high"], stock_bar["high"]),
                      for_agent.rel_change(prev_stock_bar["low"], stock_bar["low"]),
                      for_agent.rel_change(prev_stock_bar["close"], stock_bar["close"]),
                      for_agent.rel_change(prev_stock_bar["volume"], stock_bar["volume"]),
                      for_agent.rel_change(prev_stock_bar["vwap"], stock_bar["vwap"]),
                      stock_sentiment,  # -1 = negative, 0 = neutral, 1 = positive
                      for_agent.rel_change(prev_sp500_bar["close"], sp500_bar["close"]),
                      for_agent.rel_change(prev_sp500_bar["volume"], sp500_bar["volume"]),
                      sp500_sentiment,
                      for_agent.rel_change(prev_nasdaq_bar["close"], nasdaq_bar["close"]),
                      for_agent.rel_change(prev_nasdaq_bar["volume"], nasdaq_bar["volume"]),
                      nasdaq_sentiment,
                      k_percent,
                      d_ema,
                      k_ema,
                      rsi]
            for_agent.net.activate(inputs)
        print(f"{profile_name} {for_agent.stock['symbol']}: Updated network")