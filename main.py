import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
import os
import json
import finnhub
import agent

FINN_KEY = "cmo8j01r01qj3malarpgcmo8j01r01qj3malarq0"

ALPACA_PUB = "PK2JXXYFS6WPQ62MWDAK"
ALPACA_SEC = "csb66Ud3tenU867Ywhr7ROCNZX41OZw7Uf0hmOgv"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper: https://paper-api.alpaca.markets | Live: https://api.alpaca.markets


if __name__ == "__main__":
    finn_client = finnhub.Client(api_key=FINN_KEY)
    alpaca_client = alpaca.REST(ALPACA_PUB, ALPACA_SEC, base_url=URL(ALPACA_BASE_URL))

    local_dir = os.path.dirname(__file__)
    settings_path = os.path.join(local_dir, "agent-settings.json")
    with open(settings_path) as file:
        settings = json.load(file)

    if settings["training_mode"]:
        agent = agent.Training(settings, finn_client, alpaca_client)
    else:
        agent = agent.Trader(settings, finn_client, alpaca_client)

    if input("Run agent? (y/n): ") == "y":
        agent.run()
