from base_manager import Manager
from trainer import Trainer
from trader import Trader
from paper_trader import PaperTrader
from validator import Validator
import finbert_news
import schwab


if __name__ == "__main__":
    settings, alpaca_api = Manager.get_settings_and_alpaca(0)

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
