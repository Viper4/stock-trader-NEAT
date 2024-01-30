import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
import os
import json
import agent
import trainer

ALPACA_PUB = ""
ALPACA_SEC = ""
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper: https://paper-api.alpaca.markets | Live: https://api.alpaca.markets


if __name__ == "__main__":
    alpaca_api = alpaca.REST(ALPACA_PUB, ALPACA_SEC, base_url=URL(ALPACA_BASE_URL))

    local_dir = os.path.dirname(__file__)
    settings_path = os.path.join(local_dir, "agent-settings.json")
    with open(settings_path) as file:
        settings = json.load(file)

    #finbert = finbert_news.FinbertNews(alpaca_api)
    finbert = None

    if settings["training_mode"]:
        trainer = trainer.Trainer(settings, alpaca_api, finbert)
        if input("Start training? (y/n): ") == "y":
            trainer.start_training()
    else:
        trader = agent.Trader(settings, alpaca_api, finbert)
        if input("Run trader agent? (y/n): ") == "y":
            trader.run()
