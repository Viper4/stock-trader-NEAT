import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss, CrossEntropy
import numpy as np
import talib
import ast
import Managers.base_manager
from constants import SETTINGS_PATH, PROJECT_DIR
from alpaca_trade_api.rest import REST, URL, TimeFrameUnit
import datetime as dt
import pytz
import json
import os
import saving
from tqdm import tqdm
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
#import warnings
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import torch


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
    bars_df["sma_a_pc"] = bars_df["sma_a"].pct_change(fill_method=None)
    bars_df["sma_b_pc"] = bars_df["sma_b"].pct_change(fill_method=None)
    bars_df["sma_c_pc"] = bars_df["sma_c"].pct_change(fill_method=None)
    bars_df["sma_d_pc"] = bars_df["sma_d"].pct_change(fill_method=None)
    bars_df["sma_a_dst"] = (bars_df["sma_a"] - bars_df["close"]) / bars_df["close"]
    bars_df["sma_b_dst"] = (bars_df["sma_b"] - bars_df["close"]) / bars_df["close"]
    bars_df["sma_c_dst"] = (bars_df["sma_c"] - bars_df["close"]) / bars_df["close"]
    bars_df["sma_d_dst"] = (bars_df["sma_d"] - bars_df["close"]) / bars_df["close"]

    bars_df["ema_a"] = talib.EMA(bars_df["close"], timeperiod=10)
    bars_df["ema_b"] = talib.EMA(bars_df["close"], timeperiod=30)
    bars_df["ema_c"] = talib.EMA(bars_df["close"], timeperiod=50)
    bars_df["ema_d"] = talib.EMA(bars_df["close"], timeperiod=200)
    bars_df["ema_a_pc"] = bars_df["ema_a"].pct_change(fill_method=None)
    bars_df["ema_b_pc"] = bars_df["ema_b"].pct_change(fill_method=None)
    bars_df["ema_c_pc"] = bars_df["ema_c"].pct_change(fill_method=None)
    bars_df["ema_d_pc"] = bars_df["ema_d"].pct_change(fill_method=None)
    bars_df["ema_a_dst"] = (bars_df["ema_a"] - bars_df["close"]) / bars_df["close"]
    bars_df["ema_b_dst"] = (bars_df["ema_b"] - bars_df["close"]) / bars_df["close"]
    bars_df["ema_c_dst"] = (bars_df["ema_c"] - bars_df["close"]) / bars_df["close"]
    bars_df["ema_d_dst"] = (bars_df["ema_d"] - bars_df["close"]) / bars_df["close"]

    bars_df["bb_upper"], bars_df["bb_middle"], bars_df["bb_lower"] = talib.BBANDS(bars_df["close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    bars_df["bb_width"] = (bars_df["bb_upper"] - bars_df["bb_lower"]) / bars_df["bb_lower"]
    bars_df["bb_upper_pc"] = bars_df["bb_upper"].pct_change(fill_method=None)
    bars_df["bb_middle_pc"] = bars_df["bb_middle"].pct_change(fill_method=None)
    bars_df["bb_lower_pc"] = bars_df["bb_lower"].pct_change(fill_method=None)
    bars_df["bb_upper_dst"] = (bars_df["bb_upper"] - bars_df["close"]) / bars_df["close"]
    bars_df["bb_middle_dst"] = (bars_df["bb_middle"] - bars_df["close"]) / bars_df["close"]
    bars_df["bb_lower_dst"] = (bars_df["bb_lower"] - bars_df["close"]) / bars_df["close"]

    bars_df["linearreg"] = talib.LINEARREG(bars_df["close"], timeperiod=14)
    bars_df["linearreg_angle"] = talib.LINEARREG_ANGLE(bars_df["close"], timeperiod=14) / 90
    bars_df["linearreg_pc"] = bars_df["linearreg"].pct_change(fill_method=None)
    bars_df["linearreg_dst"] = (bars_df["linearreg"] - bars_df["close"]) / bars_df["close"]

    bars_df["atr"] = talib.ATR(bars_df["high"], bars_df["low"], bars_df["close"], timeperiod=14)
    bars_df["natr"] = talib.NATR(bars_df["high"], bars_df["low"], bars_df["close"], timeperiod=14)
    bars_df["tr"] = talib.TRANGE(bars_df["high"], bars_df["low"], bars_df["close"])
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

    bars_df["adx"] = talib.ADX(bars_df["high"], bars_df["low"], bars_df["close"], timeperiod=14)

    bars_df["ht_trendline"] = talib.HT_TRENDLINE(bars_df["close"])

    bars_df["kama"] = talib.KAMA(bars_df["close"], timeperiod=30)
    bars_df["kama_pc"] = bars_df["kama"].pct_change(fill_method=None)
    bars_df["kama_dst"] = (bars_df["kama"] - bars_df["close"]) / bars_df["close"]

    bars_df["mama"], bars_df["fama"] = talib.MAMA(bars_df["close"], fastlimit=0.5, slowlimit=0.05)
    bars_df["mama_pc"] = bars_df["mama"].pct_change(fill_method=None)
    bars_df["fama_pc"] = bars_df["fama"].pct_change(fill_method=None)
    bars_df["mama_dst"] = (bars_df["mama"] - bars_df["close"]) / bars_df["close"]
    bars_df["fama_dst"] = (bars_df["fama"] - bars_df["close"]) / bars_df["close"]

    bars_df["sar"] = talib.SAR(bars_df["high"], bars_df["low"], acceleration=0.02, maximum=0.2)

    bars_df["volatility"] = bars_df["close_pc"].rolling(window=30).std()

    bars_df["macd"], bars_df["macdsignal"], bars_df["macdhist"] = talib.MACD(bars_df["close"], fastperiod=12,
                                                                             slowperiod=26, signalperiod=9)

    bars_df.replace(np.nan, 0.0, inplace=True)
    bars_df.replace(np.inf, 1.0, inplace=True)
    bars_df.replace(-np.inf, -1.0, inplace=True)


class TFTTrainer(object):
    def __init__(self, symbol, latent_bars, features, model_filename, max_epochs):
        self.symbol = symbol
        self.latent_bars = latent_bars
        self.feature_cols = features
        self.normalizer = None
        self.max_epochs = max_epochs
        torch.set_float32_matmul_precision('medium')

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=8, verbose=False, mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=PROJECT_DIR + "\\TFT\\models",
            filename=model_filename,
            verbose=False,
            auto_insert_metric_name=False,
            enable_version_counter=False
        )

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="cuda",
            enable_model_summary=True,
            gradient_clip_val=0.222,
            limit_train_batches=50,  # Comment in for training, running validation every 30 batches
            # fast_dev_run=True,  # Comment in to check that networker dataset has no serious bugs
            callbacks=[early_stop_callback, checkpoint_callback]
        )

        self.model_path = PROJECT_DIR + f"\\TFT\\models\\{model_filename}.ckpt"
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = TemporalFusionTransformer.load_from_checkpoint(self.model_path)
        else:
            self.model = None

    def prepare_data(self, bars_df, batch_size=128):
        bars_df["time_idx"] = range(bars_df.shape[0])
        bars_df["asset"] = self.symbol
        bars_df["target"] = bars_df["close"].shift(-1)

        # Split into first 80% training and last 20% validation
        train_size = int(bars_df.shape[0] * 0.8)
        train_bars_df = bars_df[:train_size].copy()
        val_bars_df = bars_df[train_size+1:-1].copy()  # Exclude last bar due to NaN from shifting

        # Create training dataset
        training = TimeSeriesDataSet(
            train_bars_df,
            time_idx="time_idx",
            target="target",
            group_ids=["asset"],
            max_encoder_length=self.latent_bars,
            max_prediction_length=3,
            static_categoricals=["asset"],
            static_reals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=self.feature_cols,
            target_normalizer=GroupNormalizer(groups=["asset"], transformation="softplus"),
            #target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        self.normalizer = training.target_normalizer

        # Create validation dataset using the same parameters
        validation = TimeSeriesDataSet.from_dataset(
            training,
            val_bars_df,
            predict=False,  # Necessary to have many prediction points during training
            stop_randomization=True)

        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size * 10, num_workers=0
        )

        # Save normalizer
        saving.SaveSystem.save_data(training.target_normalizer, PROJECT_DIR + f"\\TFT\\{self.symbol}_normalizer.gz")

        return train_bars_df, val_bars_df, training, validation, train_dataloader, val_dataloader

    def validate(self, training_dataset, val_bars, plot=False):
        # Slide over validation data
        output_index = []
        outputs = []
        actuals = []
        errors = []
        #correct = 0
        required_length = self.latent_bars + 3  # max_encoder_length + max_prediction_length
        for i in tqdm(range(required_length, val_bars.shape[0]), desc="Validating: ", unit="bars"):
            # Slice the data to use latent_bars up to i
            window = val_bars[i - required_length:i].copy()

            # Build a 1-step prediction dataset
            predict_ds = TimeSeriesDataSet.from_dataset(
                training_dataset,
                window,
                predict=True,
                stop_randomization=True
            )

            dataloader = predict_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
            prediction = self.model.predict(dataloader, return_y=True, trainer_kwargs=dict(accelerator="cuda"))
            pred_y = prediction.output[0][0].item()
            actual_y = prediction.y[0][0][0].item()

            output_index.append(val_bars.index[i])
            outputs.append(pred_y)
            actuals.append(actual_y)

            '''if pred_y == actual_y:
                correct += 1'''
            errors.append(abs(pred_y - actual_y))

        if plot:
            plt.figure(figsize=(12, 6))

            plt.plot(output_index, actuals, color="blue", label="actual", alpha=0.7)

            plt.plot(output_index, outputs, color="orange", label="predicted", alpha=1.0)

            plt.legend()
            #plt.title(f"Validation results (Score {correct}/{len(outputs)})")
            plt.title(f"Validation results (Error {np.mean(errors)})")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.grid(True)
            plt.show()

        #return pd.DataFrame(columns=["output", "actual"], data=zip(outputs, actuals)), correct
        return pd.DataFrame(columns=["output", "actual", "error"], data=zip(outputs, actuals, errors))

    def baseline(self, val_bars):
        # Predict next values as previous value
        outputs = []
        actuals = []
        errors = []
        '''correct = 0
        for i in range(self.latent_bars, val_bars.shape[0]):
            prev_change = val_bars.iloc[i - 1].change
            change = val_bars.iloc[i].change
            outputs.append(prev_change.item())
            actuals.append(change.item())
            if change == prev_change:
                correct += 1

        return pd.DataFrame(columns=["output", "actual"], data=zip(outputs, actuals)), correct'''

        for i in range(self.latent_bars, val_bars.shape[0]):
            prev_close = val_bars.iloc[i - 1].close
            close = val_bars.iloc[i].close
            outputs.append(prev_close.item())
            actuals.append(close.item())
            errors.append(np.abs(prev_close - close))
        return pd.DataFrame(columns=["output", "actual", "error"], data=zip(outputs, actuals, errors))

    def train(self, train_dataset, train_dataloader, val_dataloader, val_bars):
        pl.seed_everything(42)  # For reproducibility

        if self.model is None:
            self.model = TemporalFusionTransformer.from_dataset(
                train_dataset,
                learning_rate=0.002,
                hidden_size=100,
                attention_head_size=4,
                dropout=0.294,
                hidden_continuous_size=86,
                loss=QuantileLoss(),
                #output_size=3,
                log_interval=10,
                optimizer="ranger",
                reduce_on_plateau_patience=4
            )

        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        baseline_results, correct = self.baseline(val_bars)
        tft_results, tft_correct = self.validate(train_dataset, val_bars, True)

        #pd.set_option('display.max_rows', None)
        print("Baseline Results:")
        print(baseline_results)
        print(f"Baseline Score: {correct}/{baseline_results.shape[0]}")
        #print("Baseline Average Error:", baseline_results["error"].mean())

        print("TFT Results:")
        print(tft_results)
        print(f"TFT Score: {tft_correct}/{tft_results.shape[0]}")
        #print("TFT Average Error:", tft_results["error"].mean())

    def find_learning_rate(self, dataset, train_dataloader, val_dataloader):
        pl.seed_everything(42)  # For reproducibility
        trainer = pl.Trainer(
            accelerator="cuda",
            # Clipping gradients is a hyperparameter and important to prevent
            # divergence of the gradient for RNNs
            gradient_clip_val=0.222,
            max_epochs=1000,
        )

        searching_model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=0.002,  # Not meaningful for finding the learning rate but otherwise very important
            hidden_size=100,  # Most important hyperparameter apart from learning rate
            attention_head_size=4,  # Number of attention heads. Set to up to 4 for large datasets
            dropout=0.294,  # Between 0.1 and 0.3 are good values
            hidden_continuous_size=86,  # Set to <= hidden_size
            loss=QuantileLoss(),
            #output_size=3,
            optimizer="ranger",
        )

        result = Tuner(trainer).lr_find(
            searching_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            max_lr=10.0,
            min_lr=1e-6,
        )

        print(f"Suggested learning rate: {result.suggestion()}")
        fig = result.plot(show=True, suggest=True)
        fig.show()

    @staticmethod
    def tuning(train_dataloader, val_dataloader):
        n_trials = 200

        study = optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_path="optuna_test",
            n_trials=n_trials,
            max_epochs=50,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=8,
            use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
        )

        # save study results - also we can resume tuning at a later point in time
        saving.SaveSystem.save_data(study, PROJECT_DIR + "\\TFT\\test_study.gz")

        # show best hyperparameters
        print("Trial #", study.best_trial.number)
        print("Value:", study.best_trial.value)
        print(study.best_trial.params)


class TFTAgent(object):
    def __init__(self, symbol, latent_bars, features, model_filename):
        self.symbol = symbol
        self.latent_bars = latent_bars
        self.feature_cols = features
        self.model = TemporalFusionTransformer.load_from_checkpoint(PROJECT_DIR + f"\\TFT\\models\\{model_filename}.ckpt")
        self.normalizer = saving.SaveSystem.load_data(PROJECT_DIR + f"\\TFT\\{self.symbol}_normalizer.gz")
        torch.set_float32_matmul_precision('medium')

    def interpret(self, bars_df):
        bars_df.dropna(inplace=True)
        raw_predictions = self.predict(bars_df, mode="raw")
        interpretation = self.model.interpret_output(raw_predictions.output, reduction="sum")
        self.model.plot_interpretation(interpretation)
        plt.show()

    def predict(self, bars_df, batch_size=128, mode="prediction"):
        # Need to manually apply normalization from saved training
        dataset = TimeSeriesDataSet(
            bars_df,
            time_idx="time_idx",
            target="target",
            group_ids=["asset"],
            max_encoder_length=self.latent_bars,
            max_prediction_length=3,
            static_categoricals=["asset"],
            static_reals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=self.feature_cols,
            target_normalizer=self.normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            predict_mode=True,
            randomize_length=False
        )

        dataloader = dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )
        predictions = self.model.predict(
            dataloader, return_x=True, mode=mode, trainer_kwargs=dict(accelerator="cuda")
        )
        return predictions


if __name__ == "__main__":
    #warnings.filterwarnings("ignore")

    with open(SETTINGS_PATH) as file:
        settings = json.load(file)
    alpaca_api = REST(settings["profiles"][0]["public_key"], settings["profiles"][0]["secret_key"],
                      base_url=URL("https://paper-api.alpaca.markets"))

    symbol = input("Symbol> ")

    latent_bars = int(input("Latent bars> "))
    features = ast.literal_eval(input("Features> "))

    bars_path = PROJECT_DIR + f"\\TFT\\{symbol}_bars.gz"
    if os.path.exists(bars_path):
        bars_df = saving.SaveSystem.load_data(bars_path)
    else:
        start = input("Enter start date (YYYY-MM-DD)> ")
        end = input("Enter end date (YYYY-MM-DD)> ")
        interval = int(input("Enter interval (1, 5, 15, 30)> "))
        unit_input = input("Enter interval unit (minute, day, week, month, hour)> ")
        start_date = dt.datetime.strptime(start, "%Y-%m-%d").replace(hour=9, minute=30,
                                                                     tzinfo=pytz.timezone("US/Eastern"))
        end_date = dt.datetime.strptime(end, "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=pytz.timezone("US/Eastern"))

        unit_map = {"minute": TimeFrameUnit.Minute, "day": TimeFrameUnit.Day, "week": TimeFrameUnit.Week,
                    "month": TimeFrameUnit.Month, "hour": TimeFrameUnit.Hour}

        bars_df = Managers.base_manager.Manager.get_bars(symbol, alpaca_api, interval, start_date, end_date, 500000,
                                                         unit_map[unit_input])
        augment_bars(bars_df)
        bars_df = bars_df.copy()  # De-frag in memory
        print(f"Got bars from {bars_df.index[0]} to {bars_df.index[0]}")
        saving.SaveSystem.save_data(bars_df, bars_path)

    model_filename = input("Model filename> ")
    tft_trainer = TFTTrainer(symbol, latent_bars, features, model_filename, int(input("Max epochs: ")))
    train_bars, val_bars, train_dataset, val_dataset, train_dataloader, val_dataloader = tft_trainer.prepare_data(bars_df, 128)

    user_cmd = ""
    while user_cmd != "quit":
        user_cmd = input("Enter a command (train, flr, predict, interpret, tuning)> ")
        if user_cmd == "flr":
            tft_trainer.find_learning_rate(train_dataset, train_dataloader, val_dataloader)
        elif user_cmd == "train":
            tft_trainer.train(train_dataset, train_dataloader, val_dataloader, val_bars)
        elif user_cmd == "validate":
            tft_trainer.validate(train_dataset, val_bars, True)
        elif user_cmd == "predict":
            agent = TFTAgent(symbol, latent_bars, features, model_filename)
            print(agent.predict(bars_df).output)
        elif user_cmd == "interpret":
            agent = TFTAgent(symbol, latent_bars, features, model_filename)
            agent.interpret(bars_df)
        elif user_cmd == "tuning":
            tft_trainer.tuning(train_dataloader, val_dataloader)
        elif user_cmd == "tuning results":
            study = saving.SaveSystem.load_data(PROJECT_DIR + "\\TFT\\test_study.gz")

            print("Trial #", study.best_trial.number)
            print("Value:", study.best_trial.value)
            print(study.best_trial.params)
        elif user_cmd == "plot":
            plt.figure(figsize=(12, 6))

            colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white", "orange", "purple"]
            column = input("Column> ")
            i = 0
            while column != "quit":
                plt.plot(bars_df.index, bars_df[column], color=colors[i], label=column, alpha=0.7)
                column = input("Column> ")
                i += 1
                if i >= len(colors):
                    i = 0

            plt.legend()
            plt.title(f"Data over time")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.grid(True)
            plt.show()
