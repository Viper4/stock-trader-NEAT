import plotly.graph_objects as go
import time
import datetime as dt
import saving
import os
from constants import LOG_DIR, TRAINING_DIR, VALIDATION_DIR
from tqdm import tqdm


def plot_bars(bars, lines=None, fills=None, log=None, title="Bars"):
    fig = go.Figure(
        data=[go.Candlestick(x=bars.index, open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"])])

    if lines is not None:
        for line in lines:
            fig.add_trace(go.Scatter(x=bars.index, y=bars[line], mode="lines", name=line))

    if fills is not None:
        for fill in fills:
            print(f"Adding fill: {fill}")
            for row in tqdm(bars.itertuples(), total=bars.shape[0]):
                value = getattr(row, fill)
                time_start = row.Index
                time_end = row.Index + dt.timedelta(minutes=15)

                # Choose color based on value
                if value > 0.2:
                    color = "rgba(0, 255, 0, 0.1)"  # Light green
                elif value < -0.2:
                    color = "rgba(255, 0, 0, 0.1)"  # Light red
                else:
                    color = "rgba(255, 255, 0, 0.1)"  # Yellow (neutral)

                fig.add_trace(
                    go.Scatter(
                        x=[time_start, time_start, time_end, time_end, time_start],
                        y=[row.close - 10, row.close + 10, row.close + 10, row.close - 10, row.close - 10],
                        name=fill,
                        fill="toself",
                        fillcolor=color,
                        line_color=color,
                        mode='lines',
                        showlegend=False
                    )
                )

    if log is not None:
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

        print(f"Realized profit: ${profit}")
        print(f"Unrealized profit: ${round(shares * bars.iloc[-1].close - cost, 2)}")

        fig.update_layout(
            annotations=annotations)

    fig.update_layout(title=title, xaxis_rangeslider_visible=False, xaxis_title="Time", yaxis_title="Price ($)")
    fig.show()


if __name__ == "__main__":
    data_type = input("Enter data type (1 for training, 2 for validation): ")
    filename = input("Enter file name: ")
    log_filename = input("Enter log file name: ")
    fill = input("Enter fill: ")

    if data_type == "1":
        directory = TRAINING_DIR
    elif data_type == "2":
        directory = VALIDATION_DIR
    else:
        directory = TRAINING_DIR

    if log_filename != "":
        log = saving.SaveSystem.load_data(os.path.join(directory, f"{log_filename}.gz"))
    else:
        log = None

    bars_df = saving.SaveSystem.load_data(os.path.join(directory, f"{filename}.gz"))
    plot_bars(bars_df,
              lines=[
                  "vwap"
              ],
              fills=[fill],
              log=log,
              title=filename)
