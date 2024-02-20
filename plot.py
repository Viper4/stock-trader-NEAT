import plotly.graph_objects as go
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit


def plot_log(alpaca_api, symbol, log, interval):
    start_time = log[0]["datetime"]
    end_time = log[-1]["datetime"]
    annotations = []
    for action in log:
        text = f"{action['side']} {round(action['quantity'], 2)}<br>Cash S|L: {round(action['solid_cash'], 1)}|{round(action['liquid_cash'], 1)}"
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

    bars_df = alpaca_api.get_bars(
        symbol=symbol,
        timeframe=TimeFrame(int(interval), TimeFrameUnit.Minute),
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        limit=100000,
        sort="asc").df.tz_convert("US/Central")
    bars_df = bars_df.between_time("8:30", "15:00")

    candlestick_fig = go.Figure(data=[go.Candlestick(x=bars_df.index,
                                                     open=bars_df['open'],
                                                     high=bars_df['high'],
                                                     low=bars_df['low'],
                                                     close=bars_df['close'])])
    candlestick_fig.update_layout(
        title=f"Candlestick chart for {symbol}",
        xaxis_title="Date",
        yaxis_title="Price ($USD)",
        annotations=annotations)
    candlestick_fig.show()


def plot_logs(alpaca_api, log, interval):
    for symbol in log:
        plot_log(alpaca_api, symbol, log, interval)

