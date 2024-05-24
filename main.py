import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
import os
import json
import manager
import finbert_news
import schwab


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    settings_path = os.path.join(local_dir, "settings.json")
    with open(settings_path) as file:
        settings = json.load(file)

    # Alpaca API
    first_account = settings["accounts"][0]
    alp_base_url = URL("https://paper-api.alpaca.markets") if first_account["paper"] else URL("https://api.alpaca.markets")
    alpaca_api = alpaca.REST(first_account["public_key"], first_account["secret_key"], base_url=alp_base_url)

    finbert = finbert_news.FinBERTNews(alpaca_api)
    modes = {"trading": manager.Trader,
             "paper trading": manager.PaperTrader,
             "training": manager.Trainer,
             "validation": manager.Validator}

    test_schwab = schwab.Schwab()
    print(test_schwab.get_account())

    selected = input(f"Enter a mode ({', '.join(modes.keys())}): ")
    if selected in modes:
        if selected == "trading":
            settings["schwab"] = schwab.Schwab()

        instance = modes[selected](settings, finbert)
        instance.start()
    else:
        print(f"Invalid mode '{selected}'")
