import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
import os
import json
import manager
import finbert_news
import plot
import saving

ALPACA_PUB = "PKHSGZJ9CBQPWGLS63YP"
ALPACA_SEC = "NBZQMDH1JflJTiUnt4BUHezhwmEKCfpVju2BBblf"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper: https://paper-api.alpaca.markets | Live: https://api.alpaca.markets


if __name__ == "__main__":
    alpaca_api = alpaca.REST(ALPACA_PUB, ALPACA_SEC, base_url=URL(ALPACA_BASE_URL))

    local_dir = os.path.dirname(__file__)
    settings_path = os.path.join(local_dir, "agent-settings.json")
    with open(settings_path) as file:
        settings = json.load(file)

    if input("Plot logs? (y/n): ") == "y":
        logs = {}
        log_path = settings["save_path"] + "/Logs"
        for filename in os.listdir(log_path):
            filepath = os.path.join(log_path, filename)
            file_logs, balance_change, bought_shares = saving.SaveSystem.load_data(filepath)
            print(f"{filename}\n Bal Change: {balance_change}\n Bought Shares: {bought_shares}\n")
            for ticker in file_logs:
                if ticker in logs:
                    logs[ticker].extend(file_logs[ticker])
                else:
                    logs[ticker] = file_logs[ticker]
        for ticker in logs:
            if len(logs[ticker]) > 0:
                plot.Plot.plot_log(alpaca_api, ticker, logs[ticker], 1)

    finbert = finbert_news.FinBERTNews(alpaca_api)

    if settings["trading_mode"]:
        trader = manager.Trader(settings, alpaca_api, finbert)
        if input("Start trading? (y/n): ") == "y":
            trader.start_trading()
    else:
        trainer = manager.Trainer(settings, alpaca_api, finbert)
        if input("Start training? (y/n): ") == "y":
            trainer.start_training()
