import neat
import os
import saving
import numpy as np
import torch
#import cupy as cp


class Agent:
    def __init__(self, settings, session, stock):
        self.running = False
        self.settings = settings
        self.session = session
        self.stock = stock

        if self.settings["processes"] >= os.cpu_count():
            print("Using " + str(self.settings["processes"]) + " workers to train but system only has " + str(os.cpu_count()) + " cores.")
            if input("Proceed? (y/n): ") != "y":
                exit(0)

        self.population_path = self.settings["save_path"] + "\\Populations"
        saving.SaveSystem.make_dir(self.population_path)

        self.genome_path = self.settings["save_path"] + "\\Genomes"
        saving.SaveSystem.make_dir(self.genome_path)

        self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.settings["config_path"])
        self.alphas = {}

    @staticmethod
    def rel_change(a, b):
        if a == 0:
            return 0
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
    def calculate_k_percent_gpu(bars):
        pass

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
    def calculate_ema_gpu(close_price, alpha, prev_ema):
        close_price = torch.tensor(close_price, dtype=torch.float32)
        prev_ema = torch.tensor(prev_ema, dtype=torch.float32)
        ema = close_price * alpha + prev_ema * (1 - alpha)
        return (2 * ((ema - close_price) / close_price)).cpu().item()  # Normalize to [-1, 1]

    @staticmethod
    def calculate_sma(bars):
        total = 0
        if len(bars) == 0:
            return 0
        for i in range(len(bars)):
            total += bars[i]["close"]
        sma = total / len(bars)
        return 2 * ((sma - bars[-1]["close"]) / bars[-1]["close"])  # Normalize to [-1, 1]

    @staticmethod
    def calculate_sma_gpu(bars):
        #close_prices = cp.array([bar["close"] for bar in bars])  # Convert to cupy array
        #return cp.mean(close_prices).get()  # Bring back to CPU
        pass

    @staticmethod
    def calculate_rsi(bars):
        """
        Relative Strength Index indicator
        RS = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))
        """
        if len(bars) <= 1:
            return 0
        gain = 0
        loss = 0
        for i in range(1, len(bars)):
            change = bars[i]["close"] - bars[i - 1]["close"]
            if change > 0:
                gain += change
            elif change < 0:
                loss += abs(change)

        avg_gain = gain / (len(bars) - 1)
        avg_loss = loss / (len(bars) - 1)
        if avg_loss == 0:
            rsi = 100
        else:
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        return 2 * ((rsi - 50) / 50)  # Normalize to [-1, 1]

    @staticmethod
    def calculate_rsi_gpu(bars):
        #changes = cp.array([bars[i]["close"]-bars[i-1]["close"] for i in range(1, len(bars))])
        #gain = cp.sum(changes)
        pass

