import plotly.graph_objects as go
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import time
import datetime as dt
import pytz
import saving
import os
import main


def plot_log(alpaca_api, symbol, log, interval, print_profit=False):
    log_start = log[0]["datetime"]
    log_end = log[-1]["datetime"]

    if log_start.date() == log_end.date():
        start_time = dt.datetime(log_start.year, log_start.month, log_start.day, 9, 30, tzinfo=pytz.timezone("US/Eastern"))
        end_time = dt.datetime(log_end.year, log_end.month, log_end.day, 16, 0, tzinfo=pytz.timezone("US/Eastern"))
    else:
        start_time = log_start
        end_time = log_end

    shares = 0
    profit = 0
    annotations = []
    for i in range(len(log)):
        action = log[i]
        text = f"{i} {action['side']} {round(action['quantity'], 2)} ${round(action['price'], 2)}<br>Cash S|L: {round(action['solid_cash'], 1)}|{round(action['liquid_cash'], 1)}"
        color = "green"
        if action["side"] == "Sell":
            shares -= action["quantity"]
            profit += action["profit"]
            text += f"<br>P/L: {round(action['profit'], 2)}"
            color = "red"
        elif action["side"] == "Buy":
            shares += action["quantity"]

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
    now_date = dt.datetime.now(tz=pytz.timezone("US/Eastern"))
    time_since = (now_date - end_time).total_seconds() / 60
    if time_since < 16:
        wait_time = 16 - time_since
        # start_time and end_time are -05:51 timezone but now_time is -05:00
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
        print(f"Total {symbol} profit: ${profit + (shares * last_bar['close'])}")

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
    settings, alpaca_api = main.get_settings_and_alpaca(0)
    log_path = f"{settings['save_path']}\\Logs"
    filename = input("Enter file name: ")
    logs = saving.SaveSystem.load_data(os.path.join(log_path, f"{filename}.gz"))
    for symbol in logs:
        if input(f"Plot {symbol}? (y/n): ") == "y":
            plot_log(alpaca_api, symbol, logs[symbol], int(input("Enter interval: ")), True)
