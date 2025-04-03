import neat
import os
import numpy as np
from constants import CONFIG_PATH


class Agent:
    def __init__(self, settings, profile, stock):
        self.running = False
        self.settings = settings
        self.profile = profile
        self.stock = stock

        if self.settings["processes"] >= os.cpu_count():
            print("Using " + str(self.settings["processes"]) + " workers to train but system only has " + str(os.cpu_count()) + " cores.")
            self.settings["processes"] = os.cpu_count()

        self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)

    @staticmethod
    def generate_inputs(row, plpc, sma_periods):
        stock_sma_pcs = []
        spy_sma_pcs = []
        qqq_sma_pcs = []
        for sma_period in sma_periods:
            stock_sma_pcs.append(getattr(row, f"sma_{sma_period}_pc"))
            spy_sma_pcs.append(getattr(row, f"sma_{sma_period}_spy_pc"))
            qqq_sma_pcs.append(getattr(row, f"sma_{sma_period}_qqq_pc"))

        return [
            # Stock data
            plpc,  # plpc
            row.open_pc,
            row.high_pc,
            row.low_pc,
            row.close_pc,
            row.volume_pc,
            row.vwap_pc,
            row.sentiment,  # -1 = negative, 0 = neutral, 1 = positive
            row.hmm_regime,  # -1 = Bear, 0 = Choppy, 1 = Bull
            (row.slow_k - 50) / 50,
            (row.slow_d - 50) / 50,
            (row.rsi - 50) / 50,
            row.atr_pc,
            row.ema_k_pc,
            row.ema_d_pc,
            *stock_sma_pcs,

            # SPY data
            row.close_spy_pc,
            row.volume_spy_pc,
            row.sentiment_spy,
            (row.slow_k_spy - 50) / 50,
            (row.slow_d_spy - 50) / 50,
            (row.rsi_spy - 50) / 50,
            row.atr_spy_pc,
            row.ema_k_spy_pc,
            row.ema_d_spy_pc,
            *spy_sma_pcs,

            # QQQ data
            row.close_qqq_pc,
            row.volume_qqq_pc,
            row.sentiment_qqq,
            (row.slow_k_spy - 50) / 50,
            (row.slow_d_spy - 50) / 50,
            (row.rsi_spy - 50) / 50,
            row.atr_spy_pc,
            row.ema_k_spy_pc,
            row.ema_d_spy_pc,
            *qqq_sma_pcs
        ]

    @staticmethod
    def rel_change(a, b, epsilon=1e-6):
        if abs(a) < epsilon:
            return (b - a) / epsilon
        return (b - a) / a

    @staticmethod
    def max_drawdown(portfolio_values):
        portfolio_values = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max
        max_drawdown = np.max(drawdowns)
        return max_drawdown

    @staticmethod
    def time_to_minutes(t):
        return t.hour * 60 + t.minute

    @staticmethod
    def days_to_bars(days, interval):
        """
        6.5 hours per trading day
        """
        return int((days * 6.5 * 60) / interval)

    @staticmethod
    def calculate_k_percent(bars):
        """
        Stochastic oscillator indicator
        %K = [(Current Close - Lowest Low) / (Highest High - Lowest Low)] * 100
        %D = EMA(%K, 3) or SMA(%K, 3)
        Lowest low and highest high over the given period
        """
        min_low = bars[0]["low"]
        max_high = bars[0]["high"]
        for i in range(1, len(bars)):
            if bars[i]["low"] < min_low:
                min_low = bars[i]["low"]
            if bars[i]["high"] > max_high:
                max_high = bars[i]["high"]
        denom = max_high - min_low
        if denom == 0:
            return 0
        return 2 * (((bars[-1]["close"] - min_low) / denom) - 0.5)  # Scale to [-1, 1]

    @staticmethod
    def calculate_ema(close_price, alpha, prev_ema):
        """
        Exponential moving average
        EMA = current price * (2 / (period + 1)) + EMA(prev) * (1 - (2 / (period + 1)))
        period is the number of days to calculate the EMA
        alpha = 2 / (period + 1)
        """
        if prev_ema is None:
            prev_ema = close_price
        ema = close_price * alpha + prev_ema * (1 - alpha)
        return 2 * ((ema - close_price) / close_price)  # Normalize to [-1, 1]

    @staticmethod
    def calculate_rsi(gain, loss, num_bars):
        """
        Relative Strength Index indicator
        RS = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))
        """

        avg_gain = gain / num_bars
        avg_loss = loss / num_bars
        if avg_loss == 0:
            rsi = 100
        else:
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        return 2 * ((rsi - 50) / 50)  # Normalize to [-1, 1]


