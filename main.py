from Managers.trainer import Trainer
from Managers.trader import Trader
from Managers.paper_trader import PaperTrader
from Managers.validator import Validator
import finbert_news
import schwab
from alpaca_trade_api.rest import REST, URL
import json
import saving
from constants import *

if __name__ == "__main__":
    saving.SaveSystem.make_dir(SAVE_DIR)
    saving.SaveSystem.make_dir(GENOME_DIR)
    saving.SaveSystem.make_dir(LOG_DIR)
    saving.SaveSystem.make_dir(POPULATION_DIR)
    saving.SaveSystem.make_dir(TRAINING_DIR)
    saving.SaveSystem.make_dir(VALIDATION_DIR)
    saving.SaveSystem.make_dir(VALUES_DIR)

    with open(SETTINGS_PATH) as file:
        settings = json.load(file)

    alpaca_api = REST(settings["profiles"][0]["public_key"], settings["profiles"][0]["secret_key"],
                      base_url=URL("https://paper-api.alpaca.markets"))

    finbert = finbert_news.FinBERTNews(alpaca_api)
    modes = {"trading": Trader,
             "paper trading": PaperTrader,
             "training": Trainer,
             "validation": Validator}

    selected = input(f"Enter a mode ({', '.join(modes.keys())}): ")
    if selected in modes:
        if selected == "trading":
            settings["schwab"] = schwab.Schwab()

        instance = modes[selected](settings, finbert)
        instance.start()
    else:
        print(f"Invalid mode '{selected}'")
