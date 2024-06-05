import manager
import finbert_news
import schwab


if __name__ == "__main__":
    settings, alpaca_api = manager.Manager.get_settings_and_alpaca(0)

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
