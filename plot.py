import plotly.graph_objects as go
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit


def plot_log(alpaca_api, log, interval):
    for symbol in log:
        start_time = log[symbol][0]["time"]
        end_time = log[symbol][-1]["time"]
        annotations = []
        for action in log[symbol]:
            text = f"{action['side']} {round(action['quantity'], 2)}\nCash: {round(action['cash'], 1)}"
            color = "green"
            if action["side"] == "Sell":
                text += f"\nP/L: {round(action['profit'], 2)}"
                color = "red"

            annotations.append(dict(x=action["time"],
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
            timeframe=TimeFrame(interval, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time,
            limit=30000,
            sort="asc").df.tz_convert("US/Central")
        bars_df = bars_df.between_time("8:30", "15:00")

        candlestick_fig = go.Figure(data=[go.Candlestick(x=bars_df.index,
                                                         open=bars_df['open'],
                                                         high=bars_df['high'],
                                                         low=bars_df['low'],
                                                         close=bars_df['close'],
                                                         volume=bars_df['volume'])])
        candlestick_fig.update_layout(
            title=f"Candlestick chart for {symbol}",
            xaxis_title="Date",
            yaxis_title="Price ($USD)",
            annotations=annotations)
        candlestick_fig.show()
