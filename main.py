import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
import os
import json
import agent
import trainer
import finbert_news

ALPACA_PUB = "PK6ESOPPA9A0OQBPA92K"
ALPACA_SEC = "EbIB8DYgl19gEGV1A6KeAQ4vYmsx6wXYRvuFk6vD"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper: https://paper-api.alpaca.markets | Live: https://api.alpaca.markets


if __name__ == "__main__":
    alpaca_api = alpaca.REST(ALPACA_PUB, ALPACA_SEC, base_url=URL(ALPACA_BASE_URL))

    finbert = finbert_news.FinBERTNews(alpaca_api)

    local_dir = os.path.dirname(__file__)
    settings_path = os.path.join(local_dir, "agent-settings.json")
    with open(settings_path) as file:
        settings = json.load(file)

    if settings["trading_mode"]:
        trader = agent.Trader(settings, alpaca_api, finbert)
        if input("Run trader agent? (y/n): ") == "y":
            trader.run()
    else:
        trainer = trainer.Trainer(settings, alpaca_api, finbert)
        if input("Start training? (y/n): ") == "y":
            trainer.start_training()
