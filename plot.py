import plotly.graph_objects as go
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import time
import datetime as dt
import saving
import os
from constants import LOG_DIR, TRAINING_DIR


def plot_bars(bars, lines=None):
    fig = go.Figure(data=[go.Candlestick(x=bars.index, open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"])])

    if lines is not None:
        for line in lines:
            fig.add_trace(go.Scatter(x=bars.index, y=bars[line], mode="lines", name=line))

    fig.update_layout(title=f"Bars", xaxis_rangeslider_visible=False, xaxis_title="Time", yaxis_title="Price ($)")
    fig.show()


def plot_log(alpaca_api, symbol, log, interval, print_profit=False):
    log_start = log[0]["datetime"]
    log_end = log[-1]["datetime"]

    if log_start.date() == log_end.date():
        start_time = dt.datetime(log_start.year, log_start.month, log_start.day, 9, 30, tzinfo=log_start.tzinfo)
        end_time = dt.datetime(log_end.year, log_end.month, log_end.day, 16, 0, tzinfo=log_end.tzinfo)
    else:
        start_time = log_start
        end_time = log_end

    shares = 0
    profit = 0
    cost = 0
    annotations = []
    for i in range(len(log)):
        if i > 2500:
            print("Too many actions. Plotting only last 2500 actions.")
            break
        action = log[i]
        if "solid_cash" in action:
            action["settled_cash"] = action["solid_cash"]
        if "liquid_cash" in action:
            action["unsettled_cash"] = action["liquid_cash"]
        if "type" not in action:
            action["type"] = "long"
        text = f"{i} {action['type']} {action['side'][0]} {round(action['quantity'], 2)} ${round(action['price'], 2)}<br> S|U: {round(action['settled_cash'], 1)}|{round(action['unsettled_cash'], 1)}"
        color = "green"
        if action["type"] == "long":
            if action["side"] == "Sell":
                shares -= action["quantity"]
                profit += action["profit"]
                text += f"<br>P/L: {round(action['profit'], 2)}"
                cost -= (action["price"] * action["quantity"] - action["profit"])
                color = "red"
            elif action["side"] == "Buy":
                shares += action["quantity"]
                cost += action["price"] * action["quantity"]
        elif action["type"] == "short":
            if action["side"] == "Buy":
                shares += action["quantity"]
                profit += action["profit"]
                text += f"<br>P/L: {round(action['profit'], 2)}"
                cost -= (action["price"] * action["quantity"] - action["profit"])
                color = "red"
            elif action["side"] == "Sell":
                shares += action["quantity"]
                cost += action["price"] * action["quantity"]

        annotations.append(dict(x=action["datetime"].isoformat(),
                                y=action["price"],
                                xref="x",
                                yref="y",
                                text=text,
                                showarrow=True,
                                arrowhead=1,
                                arrowcolor=color,
                                arrowsize=2,
                                ))

    # Alpaca doesn't allow getting recent 15 minute data so wait if needed
    now_date = dt.datetime.now(tz=log_start.tzinfo)
    time_since = (now_date - end_time).total_seconds() / 60
    if time_since < 16:
        wait_time = 16 - time_since
        print(f"{symbol}: Waiting {wait_time} minutes before logging")
        time.sleep(wait_time * 60)
    bars_df = alpaca_api.get_bars(
        symbol=symbol,
        timeframe=TimeFrame(interval, TimeFrameUnit.Minute),
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        limit=500000,
        sort="asc",
        adjustment="all").df.tz_convert("US/Eastern").between_time("9:30", "16:00")

    if print_profit:
        last_bar = bars_df.iloc[-1]
        print(f"{symbol} realized profit: ${profit}")
        print(f"{symbol} unrealized profit: ${round(shares * last_bar['close'] - cost, 2)}")

    candlestick_fig = go.Figure(data=[go.Candlestick(x=bars_df.index,
                                                     open=bars_df["open"],
                                                     high=bars_df["high"],
                                                     low=bars_df["low"],
                                                     close=bars_df["close"])])
    candlestick_fig.update_layout(
        title=f"{symbol} {interval}m bars",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        annotations=annotations)
    candlestick_fig.show()


if __name__ == "__main__":
    filename = input("Enter file name: ")

    bars_df = saving.SaveSystem.load_data(os.path.join(TRAINING_DIR, f"{filename}.gz"))
    plot_bars(bars_df, ["vwap",
                        "rsi",
                        "slow_k",
                        "slow_d",
                        "atr",
                        "ema_k",
                        "ema_d",
                        "sma_30",
                        "sma_60",
                        "sma_200",
                        "sentiment",
                        "sentiment_spy",
                        "sentiment_qqq"])

    '''log_path = f"{settings['save_path']}\\Logs"
    logs = saving.SaveSystem.load_data(os.path.join(log_path, f"{filename}.gz"))
    for symbol in logs:
        if len(logs[symbol]) > 0:
            if input(f"Plot {symbol}? (y/n): ") == "y":
                plot_log(alpaca_api, symbol, logs[symbol], int(input("Enter interval: ")), True)
        else:
            print(f" {symbol} log is empty. Skipping")'''
