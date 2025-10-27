import os.path
import saving
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import datetime as dt
import pytz
import ast
import Managers.base_manager
import talib
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from HMM.hmm_models import HMMRegimePrediction
from tqdm import tqdm
from alpaca_trade_api.rest import REST, URL, TimeFrameUnit
from constants import *


class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, device="cpu"):
        super(TradingLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, device=device)
        self.device = device

        # Classifier head: Buy / Hold / Sell
        self.classifier = nn.Linear(hidden_size, 3, device=device)  # 3 classes: Buy / Hold / Sell

        # Quantity head: How much to buy/sell
        self.quantity = nn.Linear(hidden_size, 1, device=device)  # Predict a quantity between 0 and 1

        self.fitnesses = []
        self.fitness = 0.0

        self.age = 0

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step output

        action_logits = self.classifier(out)
        probs = F.softmax(action_logits, dim=1)
        #quantity_raw = self.quantity(out)
        #quantity = torch.sigmoid(quantity_raw)  # Force between 0 and 1

        #return probs, quantity
        return probs, 1.0

    def predict(self, data):
        x = data.unsqueeze(0).unsqueeze(0).to(self.device)  # [batch, seq_len, input_size]
        return self.forward(x)

    def rebuild_lstm_with_preserved_params(self, new_hidden_size, new_num_layers):
        # Save old parameters
        old_lstm = self.lstm
        old_classifier = self.classifier
        old_quantity = self.quantity

        # Build new LSTM
        self.lstm = nn.LSTM(old_lstm.input_size, new_hidden_size, new_num_layers, batch_first=True).to(self.device)
        self.classifier = nn.Linear(new_hidden_size, 3).to(self.device)
        #self.quantity = nn.Linear(new_hidden_size, 1).to(self.device)

        # Copy matching weights
        with torch.no_grad():
            # LSTM
            for name, param in self.lstm.named_parameters():
                if name in old_lstm.state_dict():
                    old_param = old_lstm.state_dict()[name]

                    # Figure out matching shape
                    min_shape = tuple(min(a, b) for a, b in zip(param.shape, old_param.shape))

                    # Copy matching submatrix
                    if len(param.shape) == 2:  # Weight matrices
                        param[:min_shape[0], :min_shape[1]] = old_param[:min_shape[0], :min_shape[1]]
                    else:  # Bias vectors
                        param[:min_shape[0]] = old_param[:min_shape[0]]

            # Classifier
            for name, param in self.classifier.named_parameters():
                if name in old_classifier.state_dict():
                    old_param = old_classifier.state_dict()[name]
                    min_shape = tuple(min(a, b) for a, b in zip(param.shape, old_param.shape))
                    if len(param.shape) == 2:
                        param[:min_shape[0], :min_shape[1]] = old_param[:min_shape[0], :min_shape[1]]
                    else:
                        param[:min_shape[0]] = old_param[:min_shape[0]]

            '''# Quantity head
            for name, param in self.quantity.named_parameters():
                if name in old_quantity.state_dict():
                    old_param = old_quantity.state_dict()[name]
                    min_shape = tuple(min(a, b) for a, b in zip(param.shape, old_param.shape))
                    if len(param.shape) == 2:
                        param[:min_shape[0], :min_shape[1]] = old_param[:min_shape[0], :min_shape[1]]
                    else:
                        param[:min_shape[0]] = old_param[:min_shape[0]]'''

    def mutate(self, mutation_rate=0.01, network_mutate_rate=0.01, hidden_mutate_strength=1, layers_mutate_strength=1):
        for name, param in self.lstm.named_parameters():
            if name == "bias":
                noise = torch.randn_like(param) * (mutation_rate * 0.5)
            else:
                noise = torch.randn_like(param) * mutation_rate
            param.data += noise

        # Mutate classifier (Buy/Hold/Sell head)
        for name, param in self.classifier.named_parameters():
            if name == "bias":
                noise = torch.randn_like(param) * (mutation_rate * 0.5)
            else:
                noise = torch.randn_like(param) * mutation_rate
            param.data += noise

        '''# Mutate quantity (0-1 sigmoid head)
        for name, param in self.quantity.named_parameters():
            if name == "bias":
                noise = torch.randn_like(param) * (mutation_rate * 0.5)
            else:
                noise = torch.randn_like(param) * mutation_rate
            param.data += noise'''

        hidden_size = self.lstm.hidden_size
        new_hidden_size = hidden_size
        if random.uniform(0, 1) < network_mutate_rate:
            new_hidden_size = max(1, random.randint(hidden_size - hidden_mutate_strength, hidden_size + hidden_mutate_strength))

        num_layers = self.lstm.num_layers
        new_num_layers = num_layers
        if random.uniform(0, 1) < network_mutate_rate:
            new_num_layers = max(1, random.randint(num_layers - layers_mutate_strength, num_layers + layers_mutate_strength))

        if new_hidden_size != hidden_size or new_num_layers != num_layers:
            self.rebuild_lstm_with_preserved_params(new_hidden_size, new_num_layers)

    def clone_model(self):
        clone = TradingLSTM(self.lstm.input_size, self.lstm.hidden_size, self.lstm.num_layers, self.device)
        clone.load_state_dict(self.state_dict())
        return clone

    def training_simulation(self, data, close_prices, train_splits, index):
        self.lstm.eval()

        cash = 10000
        shares = 0
        portfolio_value = 10000
        data = data + torch.randn_like(data) * 0.005  # Try to prevent overfitting

        with torch.no_grad():
            for t in range(data.size(0)):
                x = data[t].unsqueeze(0).unsqueeze(0).to(self.device)  # [batch, seq_len, input_size]
                probs, quantity = self.forward(x)

                action = torch.argmax(probs, dim=1).item()  # 0 = buy, 1 = hold, 2 = sell
                #quantity = quantity.item()  # between 0 and 1

                price = close_prices[t]

                if action == 0:  # Buy
                    buy_amount = cash * quantity
                    shares_to_buy = buy_amount / price
                    cash -= shares_to_buy * price
                    shares += shares_to_buy
                elif action == 2:  # Sell
                    sell_amount = shares * quantity
                    cash += sell_amount * price
                    shares -= sell_amount

                # Portfolio value = cash + current value of shares
                portfolio_value = cash + shares * price

        profit = (portfolio_value - 10000) / 10000

        if len(self.fitnesses) != train_splits:
            self.fitnesses = [0.0] * train_splits

        self.fitnesses[index] = profit
        #self.fitness = profit

        return profit

    def validation_simulation(self, data, close_prices):
        self.lstm.eval()

        cash = 10000
        shares = 0
        portfolio_value = 10000
        actions = []

        with torch.no_grad():
            for t in range(data.size(0)):
                x = data[t].unsqueeze(0).unsqueeze(0).to(self.device)  # [batch, seq_len, input_size]
                probs, quantity = self.forward(x)

                action = torch.argmax(probs, dim=1).item()  # 0 = buy, 1 = hold, 2 = sell
                #quantity = quantity.item()  # between 0 and 1

                actions.append(action)

                price = close_prices[t]

                if action == 0:  # Buy
                    buy_amount = cash * quantity
                    shares_to_buy = buy_amount / price
                    cash -= shares_to_buy * price
                    shares += shares_to_buy
                elif action == 2:  # Sell
                    sell_amount = shares * quantity
                    cash += sell_amount * price
                    shares -= sell_amount

                # Portfolio value = cash + current value of shares
                portfolio_value = cash + shares * price
        return (portfolio_value - 10000) / 10000, actions


class Trainer(object):
    def __init__(self, input_features, hidden_size_range, num_layers_range, num_survivors, population_size, mutation_rate, network_mutate_rate, bars_df, max_generations=-1):
        self.input_features = input_features
        self.hidden_size_range = hidden_size_range
        self.num_layers_range = num_layers_range
        self.num_survivors = num_survivors
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.network_mutate_rate = network_mutate_rate
        self.population = []

        train_size = int(bars_df.shape[0] * 0.85)
        self.train_bars = bars_df[:train_size].copy()
        self.val_bars = bars_df[train_size+1:].copy()
        self.train_splits = 3

        self.max_generations = max_generations

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
        bars_df["sma_a_dst"] = (bars_df["sma_a"] - bars_df["close"]) / bars_df["close"]
        bars_df["sma_b_dst"] = (bars_df["sma_b"] - bars_df["close"]) / bars_df["close"]
        bars_df["sma_c_dst"] = (bars_df["sma_c"] - bars_df["close"]) / bars_df["close"]
        bars_df["sma_d_dst"] = (bars_df["sma_d"] - bars_df["close"]) / bars_df["close"]

        bars_df["ema_a"] = talib.EMA(bars_df["close"], timeperiod=10)
        bars_df["ema_b"] = talib.EMA(bars_df["close"], timeperiod=30)
        bars_df["ema_c"] = talib.EMA(bars_df["close"], timeperiod=50)
        bars_df["ema_d"] = talib.EMA(bars_df["close"], timeperiod=200)
        bars_df["ema_a"] = bars_df["ema_a"].pct_change(fill_method=None)
        bars_df["ema_b"] = bars_df["ema_b"].pct_change(fill_method=None)
        bars_df["ema_c"] = bars_df["ema_c"].pct_change(fill_method=None)
        bars_df["ema_d"] = bars_df["ema_d"].pct_change(fill_method=None)
        bars_df["ema_a_dst"] = (bars_df["ema_a"] - bars_df["close"]) / bars_df["close"]
        bars_df["ema_b_dst"] = (bars_df["ema_b"] - bars_df["close"]) / bars_df["close"]
        bars_df["ema_c_dst"] = (bars_df["ema_c"] - bars_df["close"]) / bars_df["close"]
        bars_df["ema_d_dst"] = (bars_df["ema_d"] - bars_df["close"]) / bars_df["close"]

        bars_df["bb_upper"], bars_df["bb_middle"], bars_df["bb_lower"] = talib.BBANDS(bars_df["close"], timeperiod=5,
                                                                                      nbdevup=2, nbdevdn=2, matype=0)
        bars_df["bb_width"] = (bars_df["bb_upper"] - bars_df["bb_lower"]) / bars_df["bb_lower"]
        bars_df["bb_upper"] = bars_df["bb_upper"].pct_change(fill_method=None)
        bars_df["bb_middle"] = bars_df["bb_middle"].pct_change(fill_method=None)
        bars_df["bb_lower"] = bars_df["bb_lower"].pct_change(fill_method=None)
        bars_df["bb_upper_dst"] = (bars_df["bb_upper"] - bars_df["close"]) / bars_df["close"]
        bars_df["bb_middle_dst"] = (bars_df["bb_middle"] - bars_df["close"]) / bars_df["close"]
        bars_df["bb_lower_dst"] = (bars_df["bb_lower"] - bars_df["close"]) / bars_df["close"]

        bars_df["linearreg"] = talib.LINEARREG(bars_df["close"], timeperiod=14)
        bars_df["linearreg_angle"] = talib.LINEARREG_ANGLE(bars_df["close"], timeperiod=14) / 90
        bars_df["linearreg"] = bars_df["linearreg"].pct_change(fill_method=None)
        bars_df["linearreg_dst"] = (bars_df["linearreg"] - bars_df["close"]) / bars_df["close"]

        bars_df["atr"] = talib.ATR(bars_df["high"], bars_df["low"], bars_df["close"], timeperiod=14)
        bars_df["natr"] = talib.NATR(bars_df["high"], bars_df["low"], bars_df["close"], timeperiod=14)
        bars_df["tr"] = talib.TRANGE(bars_df["high"], bars_df["low"], bars_df["close"])
        bars_df["atr"] = bars_df["atr"].pct_change(fill_method=None)
        bars_df["natr"] = bars_df["natr"].pct_change(fill_method=None)
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
        bars_df["ht_trendline"] = bars_df["ht_trendline"].pct_change(fill_method=None)

        bars_df["kama"] = talib.KAMA(bars_df["close"], timeperiod=30)
        bars_df["kama"] = bars_df["kama"].pct_change(fill_method=None)
        bars_df["kama_dst"] = (bars_df["kama"] - bars_df["close"]) / bars_df["close"]

        bars_df["mama"], bars_df["fama"] = talib.MAMA(bars_df["close"], fastlimit=0.5, slowlimit=0.05)
        bars_df["mama"] = bars_df["mama"].pct_change(fill_method=None)
        bars_df["fama"] = bars_df["fama"].pct_change(fill_method=None)
        bars_df["mama_dst"] = (bars_df["mama"] - bars_df["close"]) / bars_df["close"]
        bars_df["fama_dst"] = (bars_df["fama"] - bars_df["close"]) / bars_df["close"]

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

    @staticmethod
    def add_regime_features(bars_df, regime_settings, full):
        if full:
            regime_predictions = []
            for j in range(len(regime_settings)):
                regime_predictions.append([])

            for i in tqdm(range(bars_df.shape[0]), desc="Generating Regimes"):
                regime_slice = bars_df[bars_df.index[max(0, i - 1000)]:bars_df.index[i]].copy()

                for j in range(len(regime_settings)):
                    '''if i < 500:
                        regime_predictions[j].append(np.nan)
                        continue'''

                    regime_setting = regime_settings[j]
                    try:
                        regime_setting["model"].fit(regime_slice, regime_setting["features"], regime_setting["seed"])
                        predictions = regime_setting["model"].predict_probability(regime_slice)
                        prediction = predictions[-1]
                    except IndexError as e:
                        print("Too little clusters to fit. Skipping validation...")
                        prediction = [0.0, 0.0, 0.0]
                    except ValueError as e:
                        print("Problem with data. Skipping...")
                        prediction = [0.0, 0.0, 0.0]

                    bull_index = regime_setting["label_order"].index("Bull")
                    bear_index = regime_setting["label_order"].index("Bear")
                    regime_predictions[j].append(prediction[bull_index] - prediction[bear_index])

            for i in range(len(regime_predictions)):
                bars_df[f"regime_{i}"] = regime_predictions[i]
        else:
            for i in range(len(regime_settings)):
                bars_df[f"regime_{i}"] = 0.0

                regime_setting = regime_settings[i]
                try:
                    regime_setting["model"].fit(bars_df, regime_setting["features"], regime_setting["seed"])
                    predictions = regime_setting["model"].predict_probability(bars_df)
                    prediction = predictions[-1]
                except IndexError as e:
                    print("Too little clusters to fit. Skipping validation...")
                    prediction = [0.0, 0.0, 0.0]
                except ValueError as e:
                    print("Problem with data. Skipping...")
                    prediction = [0.0, 0.0, 0.0]

                bull_index = regime_setting["label_order"].index("Bull")
                bear_index = regime_setting["label_order"].index("Bear")
                bars_df[f"regime_{i}"].iloc[-1] = prediction[bull_index] - prediction[bear_index]

        bars_df.dropna(inplace=True)

    def preprocess_bars(self, bars_df):
        """
        Normalize features and convert to torch tensor.

        Args:
            bars_df: pandas DataFrame
            feature_cols: list of columns to use as input features

        Returns:
            torch.Tensor of shape [time, input_size]
        """
        features = bars_df[self.input_features].copy()

        # Normalize features independently (z-score normalization)
        features = (features - features.mean()) / (features.std() + 1e-8)

        features_tensor = torch.tensor(features.values, dtype=torch.float32)

        return features_tensor

    def evolve(self):
        '''# Choose survivors as the top n unique hidden sizes
        survivors = []
        survivor_sizes = set()
        i = 0
        while len(survivors) < self.num_survivors and i < self.population_size:
            if self.population[i].lstm.hidden_size not in survivor_sizes:
                survivors.append(self.population[i])
                survivor_sizes.add(self.population[i].lstm.hidden_size)
            i += 1'''

        survivors = self.population[:self.num_survivors]

        for i in range(0, self.population_size):
            '''if i < self.num_survivors:
                self.population[i] = survivors[i]  # Keep the survivors
            else:
                parent = random.choice(survivors)
                child = parent.clone_model()
                child.mutate(self.mutation_rate, self.network_mutate_rate, 4, 1)
                self.population[i] = child'''
            if i < self.num_survivors:
                self.population[i] = survivors[i]  # Keep the survivors
            elif self.population[i].age > self.train_splits:
                parent = random.choice(survivors)
                child = parent.clone_model()
                child.mutate(self.mutation_rate, self.network_mutate_rate, 4, 1)
                self.population[i] = child

    def validate(self):
        if len(self.population) <= 0:
            print("No population to validate.")
            population_path = os.path.join(POPULATION_DIR, input("Population filename> "))
            if os.path.exists(population_path):
                self.population = saving.SaveSystem.load_data(population_path)
            else:
                return

        #processed_bars = self.preprocess_bars(self.train_bars)
        #close_prices = self.train_bars["close"].tolist()

        processed_bars = self.preprocess_bars(self.val_bars)
        close_prices = self.val_bars["close"].tolist()

        results = []

        for model in tqdm(self.population, desc="Validating"):
            results.append(model.validation_simulation(processed_bars, close_prices))
        print("Validation finished.")

        return results

    def train(self, population_filename, model_filename):
        self.population.clear()
        population_path = os.path.join(POPULATION_DIR, population_filename)
        model_path = os.path.join(GENOME_DIR, model_filename)
        if os.path.exists(population_path):
            self.population = saving.SaveSystem.load_data(population_path)
            print(f"Resuming training from {population_path}")

            if len(self.population) < self.population_size:
                diff = self.population_size - len(self.population)
                for i in range(diff):
                    model = TradingLSTM(input_size=len(self.input_features),
                                        hidden_size=random.randint(self.hidden_size_range[0],
                                                                   self.hidden_size_range[1]),
                                        num_layers=random.randint(self.num_layers_range[0], self.num_layers_range[1]),
                                        device="cuda")
                    self.population.append(model)
                print(f"Added {diff} additional models to the population")
            self.population.sort(key=lambda model: model.fitness, reverse=True)
            self.evolve()
        else:
            for i in range(self.population_size):
                model = TradingLSTM(input_size=len(self.input_features),
                                    hidden_size=random.randint(self.hidden_size_range[0], self.hidden_size_range[1]),
                                    num_layers=random.randint(self.num_layers_range[0], self.num_layers_range[1]),
                                    device="cuda")
                self.population.append(model)
            print(f"Initialized {self.population_size} models for training")

        best_model = None
        if os.path.exists(model_path):
            best_model = saving.SaveSystem.load_data(model_path)
            self.population[-1] = best_model
            print(f"Loaded best model from {model_path}")

        bars_per_split = self.train_bars.shape[0] // self.train_splits
        processed_bars = []
        close_prices = []
        for i in range(self.train_splits):
            if i == self.train_splits - 1:
                sliced_bars = self.train_bars[i * bars_per_split:]
            else:
                sliced_bars = self.train_bars[i * bars_per_split:(i + 1) * bars_per_split - 1]
            processed_bars.append(self.preprocess_bars(sliced_bars))
            close_prices.append(sliced_bars["close"].tolist())

        generation = 0
        while self.max_generations == -1 or generation < self.max_generations:
            start_time = time.time()

            '''# Randomly vary size of training data every generation
            rand_start = random.randint(0, 100)
            rand_end = random.randint(self.train_bars.shape[0] - 101, self.train_bars.shape[0] - 1)
            sliced_bars = self.train_bars[rand_start:rand_end].copy()
            processed_bars = self.preprocess_bars(sliced_bars)
            close_prices = sliced_bars["close"].tolist()

            print(f"Training on {sliced_bars.shape[0]} bars from {sliced_bars.index[0]} to {sliced_bars.index[-1]}")
'''
            total_fitness = 0.0

            for model in tqdm(self.population, total=self.population_size, desc=f"Generation {generation}"):
                split_index = generation % self.train_splits
                model.training_simulation(processed_bars[split_index], close_prices[split_index], self.train_splits, split_index)
                model.fitness = sum(model.fitnesses)
                total_fitness += model.fitness
                model.age += 1

            self.population.sort(key=lambda model: model.fitness, reverse=True)

            print(f"---Generation {generation} finished in {time.time() - start_time:.2f} seconds---")
            mean_fitness = total_fitness / self.population_size
            std_sum = 0.0
            print(f"| \tAge\t | Hidden Size | \t# Layers\t | \tFitness\t |")
            for model in self.population:
                #print(f"| \t{model.age}\t | \t{model.lstm.hidden_size}\t | \t{model.lstm.num_layers}\t | \t{model.fitness:.4f}\t |")
                print(f"| \t{model.age}\t | \t{model.lstm.hidden_size}\t | \t{model.lstm.num_layers}\t | \t{model.fitness:.4f}\t | \t{model.fitnesses}\t |")

                std_sum += (model.fitness - mean_fitness) ** 2
            print(f"Max fitness: {self.population[0].fitness}")
            print(f"Min fitness: {self.population[-1].fitness}")
            print(f"Average fitness: {mean_fitness}")
            print(f"Standard deviation: {math.sqrt(std_sum / self.population_size)}")
            if best_model is None or self.population[0].fitness > best_model.fitness:
                best_model = self.population[0]
                saving.SaveSystem.save_data(best_model, model_path)
                print(f"New best model saved to {model_path}")
            else:
                print(f"All time best fitness: {best_model.fitness}")

            self.evolve()

            saving.SaveSystem.save_data(self.population, population_path)
            generation += 1

        #pool.close()
        #pool.join()

        print(f"Training finished. Best model: {best_model.fitness}")

    def plot_simulation(self, actions, title):
        plt.figure(figsize=(12, 6))

        action_labels = ["Buy", "Hold", "Sell"]
        action_colors = ["green", "yellow", "red"]
        #plt.plot(self.train_bars.index, self.train_bars["close"], color="blue", label="close", alpha=0.7)
        plt.plot(self.val_bars.index, self.val_bars["close"], color="blue", label="close", alpha=0.7)

        #val_bars_copy = self.train_bars.copy()
        val_bars_copy = self.val_bars.copy()
        val_bars_copy["action"] = actions

        for i in range(len(action_labels)):
            plt.fill_between(
                val_bars_copy.index,
                val_bars_copy["close"].min(),
                val_bars_copy["close"].max(),
                where=val_bars_copy["action"] == i,
                color=action_colors[i],
                alpha=0.3,
                label=action_labels[i]
            )

        plt.legend()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Feature")
        plt.show()


if __name__ == "__main__":
    with open(SETTINGS_PATH) as file:
        settings = json.load(file)
    alpaca_api = REST(settings["profiles"][0]["public_key"], settings["profiles"][0]["secret_key"],
                      base_url=URL("https://paper-api.alpaca.markets"))

    symbol = input("Symbol> ")

    bars_path = TRAINING_DIR + f"\\{symbol}_bars.gz"

    unit_map = {"minute": TimeFrameUnit.Minute, "day": TimeFrameUnit.Day, "week": TimeFrameUnit.Week,
                "month": TimeFrameUnit.Month, "hour": TimeFrameUnit.Hour}
    if os.path.exists(bars_path):
        bars_df = saving.SaveSystem.load_data(bars_path)
        print(f"Loaded bars from {bars_path}")
    else:
        start = input("Enter start date (YYYY-MM-DD)> ")
        end = input("Enter end date (YYYY-MM-DD)> ")
        interval = int(input("Enter interval (1, 5, 15, 30)> "))
        unit_input = input("Enter interval unit (minute, day, week, month, hour)> ")
        start_date = dt.datetime.strptime(start, "%Y-%m-%d").replace(hour=9, minute=30, tzinfo=pytz.timezone("US/Eastern"))
        end_date = dt.datetime.strptime(end, "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=pytz.timezone("US/Eastern"))

        bars_df = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, interval,
                                                         start_date - dt.timedelta(days=500), end_date, 500000,
                                                         unit_map[unit_input])
        Trainer.augment_bars(bars_df)
        bars_df = bars_df.copy()  # De-frag in memory

        regime_settings_path = TRAINING_DIR + f"\\{symbol}_regime_settings.gz"

        if os.path.exists(regime_settings_path):
            regime_settings = saving.SaveSystem.load_data(regime_settings_path)
            print(f"Loaded regime settings from {regime_settings_path}")
        else:
            regime_settings = []
            for stock in settings["profiles"][0]["stocks"]:
                if stock["symbol"] == symbol:
                    labels = ["", "", ""]
                    for key in stock["regime_settings"]["label_order"]:
                        labels[key] = stock["regime_settings"]["label_order"][key]
                    regime_settings.append({"features": ast.literal_eval(stock["regime_settings"]["features"]), "seed": stock["regime_settings"]["seed"], "label_order": labels, "model": HMMRegimePrediction()})
                    break
            #regime_features = input("Regime features> ")
            #regime_settings = []
            #while regime_features != "quit":
            #    seed = int(input("Seed> "))
            #    label_order = ast.literal_eval(input("Label order (['Bull', 'Bear', 'Choppy'])> "))
            #    regime_settings.append({"features": ast.literal_eval(regime_features), "seed": seed, "label_order": label_order, "model": HMMRegimePrediction()})
            #    regime_features = input("Regime features> ")

            saving.SaveSystem.save_data(regime_settings, regime_settings_path)

        Trainer.add_regime_features(bars_df, regime_settings, full=True)

        saving.SaveSystem.save_data(bars_df, bars_path)
        print(f"Saved bars from {bars_df.index[0]} to {bars_df.index[-1]}")

    trainer = Trainer(input_features=ast.literal_eval(input("LSTM features> ")),
                      hidden_size_range=(int(input("Hidden size min> ")), int(input("Hidden size max> "))),
                      num_layers_range=(int(input("Num layers min> ")), int(input("Num layers max> "))),
                      num_survivors=(int(input("Num survivors> "))),
                      population_size=int(input("Population size> ")),
                      mutation_rate=float(input("Mutation rate> ")),
                      network_mutate_rate=float(input("Network mutate rate> ")),
                      bars_df=bars_df,
                      max_generations=int(input("Max generations (-1 for infinite)> ")))

    user_cmd = ""
    while user_cmd != "quit":
        user_cmd = input("Enter command (train, validate)> ")
        if user_cmd == "train":
            trainer.train(input("Enter population filename> "), input("Enter model filename> "))
        elif user_cmd == "validate":
            results = trainer.validate()

            stock_change = ((trainer.val_bars["close"].iloc[-1] - trainer.val_bars["close"].iloc[0]) / trainer.val_bars["close"].iloc[0]) * 100
            #stock_change = ((trainer.train_bars["close"].iloc[-1] - trainer.train_bars["close"].iloc[0]) / trainer.train_bars["close"].iloc[0]) * 100
            print(f"Stock Change: {stock_change:.2f}%")

            print(f"Top {trainer.num_survivors} models from training:")
            for i in range(trainer.num_survivors):
                print(f" {i+1}: Model {i} | Fitness {results[i][0]} | Market Beat {(results[i][0] * 100) - stock_change:.2f}%")
                trainer.plot_simulation(results[i][1], f"Model {i} over time\nProfit: {results[i][0] * 100:.2f}%")

            sorted_indices = sorted(range(len(results)), key=lambda j: results[j][0], reverse=True)
            print(f"Top {trainer.num_survivors} models from validation:")
            for i in range(trainer.num_survivors):
                model_index = sorted_indices[i]
                print(f" {i+1}: Model {model_index} | Fitness {results[model_index][0]} | Market Beat {(results[model_index][0] * 100) - stock_change:.2f}%")
                trainer.plot_simulation(results[model_index][1], f"Model {model_index} over time\nProfit: {results[model_index][0] * 100:.2f}%")
        elif user_cmd == "now":
            model_filename = input("Enter model filename> ")
            model = saving.SaveSystem.load_data(os.path.join(GENOME_DIR, model_filename))
            interval = int(input("Enter interval (1, 5, 15, 30)> "))
            unit_input = input("Enter interval unit (minute, day, week, month, hour)> ")
            now_date = dt.datetime.now(pytz.timezone("US/Eastern"))
            bars_df = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, interval,
                                                             now_date - dt.timedelta(days=1500), now_date - dt.timedelta(minutes=16), 500000,
                                                             unit_map[unit_input])
            Trainer.augment_bars(bars_df)
            regime_settings_path = TRAINING_DIR + f"\\{symbol}_regime_settings.gz"
            if os.path.exists(regime_settings_path):
                regime_settings = saving.SaveSystem.load_data(regime_settings_path)
                Trainer.add_regime_features(bars_df, regime_settings, full=False)
                processed_data = trainer.preprocess_bars(bars_df)
                action, quantity = model.predict(processed_data[-1])
                print(f"Buy: {action[0][0].item()}, Sell: {action[0][2].item()}, Hold: {action[0][1].item()}, ")
                print("Quantity:", quantity.item())
            else:
                print("No regime settings found.")
