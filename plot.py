import plotly.graph_objects as go
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import time
import datetime as dt
import saving
import os
import manager


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
        text = f"{i} {action['type']} {action['side']} {round(action['quantity'], 2)} ${round(action['price'], 2)}<br>Cash S|U: {round(action['settled_cash'], 1)}|{round(action['unsettled_cash'], 1)}"
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
        adjustment="all").df.tz_convert("US/Eastern")
    bars_df = bars_df.between_time("9:30", "16:00")

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
        title=f"Candlestick chart for {symbol} at {interval}m interval",
        xaxis_title="Date",
        yaxis_title="Price ($USD)",
        annotations=annotations)
    candlestick_fig.show()


if __name__ == "__main__":
    settings, alpaca_api = manager.Manager.get_settings_and_alpaca(0)
    log_path = f"{settings['save_path']}\\Logs"
    filename = input("Enter file name: ")
    logs = saving.SaveSystem.load_data(os.path.join(log_path, f"{filename}.gz"))
    for symbol in logs:
        if len(logs[symbol]) > 0:
            if input(f"Plot {symbol}? (y/n): ") == "y":
                plot_log(alpaca_api, symbol, logs[symbol], int(input("Enter interval: ")), True)
        else:
            print(f" {symbol} log is empty. Skipping")
