# stock-trader-NEAT
Repo for my AI stock trader project.

## Description
This is an implementation of various models used for different functions to make predictions and trade on the stock market:

<br>A Gaussian Hidden Markov Model is fit to historical stock data on a specific stock using various combinations of indicator data to predict the market regime for the next day.
<br>A Temporal Fusion Transformer is also trained on historical stock data to forecast the future stock price.
<br>News sentiment for stocks is estimated using an NLP model called finBERT (https://huggingface.co/ProsusAI/finbert).
<br>A RNN and LSTM are trained using a genetic algorithm using historical stock data, regime predictions from the HMM, and news sentiments as inputs.

---

## HMM backtest validation results
Accuracy percentage obtained from rewarding 1 point if the prediction was "Bull" and the price went up the next day, 1 point if "Bear" and price went down next day, 1 point if "Choppy" and price change was within 0.1x of the standard deviation of the stock price.

Profit percentage obtained from strategy of buying using all cash on green and selling everything on red.

Profit is a far better indicator of performance than accuracy as you can see from the graphs.

### VOO from 5/30/2023 to 4/17/2025
Stock change 28.98%
<br>Beat market by 31.35%
![VOO_RegimeBacktest](https://github.com/user-attachments/assets/51516c3e-e567-4dfb-8f2b-56e45aa93e92)

### QQQ from 5/30/2025 to 4/17/2025
Stock change 28.51%
<br>Beat market by 56.24%
![QQQ_RegimeBacktest](https://github.com/user-attachments/assets/354c2d90-0c30-4989-9464-1ec0439acd20)

### TSLA from 10/8/2023 to 5/20/2025
Stock change 32.57%
<br>Beat market by 158.75%
![TSLA_HMM_Backtest](https://github.com/user-attachments/assets/5988e339-f7a3-45ed-b3c0-f2cb91b78961)

### NVDA from 5/30/2023 to 4/17/2025
Stock change 153.16%
<br>Beat market by 281.69%
![NVDA_RegimeBacktest](https://github.com/user-attachments/assets/0762a5c9-d8ec-4032-a1dc-6b3c9a03a7fc)


## TFT backtest validation results
The TFT model lags behind the price, is slow to react to momentum shifts, and overreacts to sharp changes in price.

Trying to predict price changes instead of raw price values causes the TFT to underfit and constantly predict the safest value (0 change) at every time step.

Due to subpar performance, I didn't run many validation tests for the TFT and focused on other models.

### TSLA from 11/1/2023 to 4/25/2025
![TFT_TSLA_Validation-9](https://github.com/user-attachments/assets/c181f172-c094-441c-a0d8-aaaaaf0de829)
