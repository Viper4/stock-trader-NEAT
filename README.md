# stock-trader-NEAT
Repo for my AI stock trader project.

## Description
This is an implementation of various models used for different functions to make predictions and trade on the stock market:

<br>A Gaussian Hidden Markov Model is fit to historical stock data on a specific stock using various combinations of indicator data to predict the market regime for the next day.
<br>A Temporal Fusion Transformer is also trained on historical stock data to forecast the future stock price.
<br>News sentiment for stocks is estimated using an NLP model called finBERT (https://huggingface.co/ProsusAI/finbert).
<br>A RNN and LSTM are trained in parallel with multiprocessing using a genetic algorithm (using the NEAT library to mutate weights, biases, and number of nodes) using historical stock data, regime predictions from the HMM, and news sentiments as inputs.
