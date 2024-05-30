import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
import os
import json
import manager
import finbert_news
import schwab


def get_settings_and_alpaca(acc_index):
    local_dir = os.path.dirname(__file__)
    settings_path = os.path.join(local_dir, "settings.json")
    with open(settings_path) as file:
        _settings = json.load(file)

    # Alpaca API
    first_profile = _settings["profiles"][acc_index]
    alp_base_url = URL("https://paper-api.alpaca.markets") if first_profile["paper"] else URL("https://api.alpaca.markets")
    return _settings, alpaca.REST(first_profile["public_key"], first_profile["secret_key"], base_url=alp_base_url)


if __name__ == "__main__":
    settings, alpaca_api = get_settings_and_alpaca(0)

    finbert = finbert_news.FinBERTNews(alpaca_api)
    modes = {"trading": manager.Trader,
             "paper trading": manager.PaperTrader,
             "training": manager.Trainer,
             "validation": manager.Validator}

    selected = input(f"Enter a mode ({', '.join(modes.keys())}): ")
    if selected in modes:
        if selected == "trading":
            settings["schwab"] = schwab.Schwab()

        instance = modes[selected](settings, finbert)
        instance.start()
    else:
        print(f"Invalid mode '{selected}'")
