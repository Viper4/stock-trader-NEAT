import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
import os
import json
import finnhub
import agent

FINN_KEY = "cmm50ppr01qqjtfnvon0cmm50ppr01qqjtfnvong"

ALPACA_SEC = "uWBUnZaL9SwHZYqp70DGAHKankMua1l238ePtgGQ"
ALPACA_PUB = "PKQ7AH5OADHDBBE6TSC8"
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
        agent.run(local_dir)
