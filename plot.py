import plotly.graph_objects as go
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import time
import datetime as dt
import pytz
import saving


def fix_logs(logs):
    for symbol in logs:
        for i in range(len(logs[symbol])):
            logs[symbol][i]["price"] = logs[symbol][i]["price"] / logs[symbol][i]["quantity"]


def plot_log(alpaca_api, symbol, log, interval):
    log_start = log[0]["datetime"]
    log_end = log[-1]["datetime"]

    if log_start.date() == log_end.date():
        start_time = dt.datetime(log_start.year, log_start.month, log_start.day, 9, 30, tzinfo=pytz.timezone("US/Eastern"))
        end_time = dt.datetime(log_end.year, log_end.month, log_end.day, 16, 0, tzinfo=pytz.timezone("US/Eastern"))
    else:
        start_time = log_start
        end_time = log_end

    annotations = []
    for i in range(len(log)):
        action = log[i]
        text = f"{i} {action['side']} {round(action['quantity'], 2)} ${round(action['price'], 2)}<br>Cash S|L: {round(action['solid_cash'], 1)}|{round(action['liquid_cash'], 1)}"
        color = "green"
        if action["side"] == "Sell":
            text += f"<br>P/L: {round(action['profit'], 2)}"
            color = "red"

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
        print("start_time: " + str(start_time))
        print("end_time: " + str(end_time))
        print("now_time: " + str(now_date))
        print("time_since: " + str(time_since))
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

    candlestick_fig = go.Figure(data=[go.Candlestick(x=bars_df.index,
                                                     open=bars_df["open"],
                                                     high=bars_df["high"],
                                                     low=bars_df["low"],
                                                     close=bars_df["close"])])
    candlestick_fig.update_layout(
        title=f"Candlestick chart for {symbol}",
        xaxis_title="Date",
        yaxis_title="Price ($USD)",
        annotations=annotations)
    candlestick_fig.show()


if __name__ == "__main__":
    path = input("Enter file path to fix: ")
    file_logs, balance_change, bought_shares = saving.SaveSystem.load_data(path)
    fix_logs(file_logs)
    saving.SaveSystem.save_data((file_logs, balance_change, bought_shares), path)
