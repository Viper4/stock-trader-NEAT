import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
import os
import json
import manager
import finbert_news

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    settings_path = os.path.join(local_dir, "settings.json")
    with open(settings_path) as file:
        settings = json.load(file)

    first_account = settings["accounts"][0]
    base_url = URL("https://paper-api.alpaca.markets") if first_account["paper"] else URL("https://api.alpaca.markets")
    alpaca_api = alpaca.REST(first_account["public_key"], first_account["secret_key"], base_url=base_url)

    finbert = finbert_news.FinBERTNews(alpaca_api)
    modes = {
        "trading": manager.Trader,
        "training": manager.Trainer,
        "validation": manager.Validator
    }

    selected = input(f"Enter a mode ({', '.join(modes.keys())}): ")
    if selected in modes:
        instance = modes[selected](settings, finbert)
        instance.start()
    else:
        print(f"Invalid mode '{selected}'")
