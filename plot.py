import plotly.graph_objects as go
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import time
import datetime as dt
import pytz


class Plot(object):
    @staticmethod
    def plot_log(alpaca_api, symbol, log, interval):
        log_start = log[0]["datetime"]
        log_end = log[-1]["datetime"]
        if log_start.date() == log_end.date():
            start_time = dt.datetime(log_start.year, log_start.month, log_start.day, 8, 30, tzinfo=pytz.timezone("US/Central"))
            end_time = dt.datetime(log_start.year, log_start.month, log_start.day, 15, 0, tzinfo=pytz.timezone("US/Central"))
        else:
            start_time = log_start
            end_time = log_end

        annotations = []
        for action in log:
            text = f"{action['side']} {round(action['quantity'], 2)} ${round(action['price'], 2)}<br>Cash S|L: {round(action['solid_cash'], 1)}|{round(action['liquid_cash'], 1)}"
            color = "green"
            if action["side"] == "Sell":
                text += f"<br>P/L: {round(action['profit'], 2)}"
                color = "red"

            annotations.append(dict(x=action["datetime"].strftime("%Y-%m-%d %H:%M:%S%z"),
                                    y=action["price"] / action["quantity"],
                                    xref="x",
                                    yref="y",
                                    text=text,
                                    showarrow=True,
                                    arrowhead=1,
                                    arrowcolor=color,
                                    arrowsize=2,
                                    ))
        # Alpaca doesn't allow getting recent 15 minute data so wait if needed
        time_since = (dt.datetime.now(pytz.timezone("US/Central")) - end_time).total_seconds() / 60
        if time_since < 16:
            wait_time = 16 - time_since
            print(f"{symbol}: Waiting {wait_time} minutes before logging")
            time.sleep(wait_time * 60)
        bars_df = alpaca_api.get_bars(
            symbol=symbol,
            timeframe=TimeFrame(int(interval), TimeFrameUnit.Minute),
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            limit=100000,
            sort="asc").df.tz_convert("US/Central")

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
