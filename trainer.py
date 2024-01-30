import agent
import time
import datetime as dt
import pytz


class Trainer(object):
    def __init__(self, settings, alpaca_api, finbert):
        self.running = False
        self.visualize = settings["visualize"]
        self.print_stats = settings["print_stats"]

        now_date = dt.datetime.now(pytz.UTC)

        start_date = now_date - dt.timedelta(days=settings["backtest_days"])
        end_date = now_date - dt.timedelta(minutes=16)  # Cant get recent 15 minute data with free alpaca acc

        #symbols = [option["symbol"] for option in settings["ticker_options"]]
        #finbert.save_news(symbols, start_date, end_date)

        self.agents = []
        self.best_genomes = []
        if len(settings["ticker_options"]) == 1 and settings["gen_stagger"] != 0:
            print("Only training 1 agent. Setting gen_stagger to 0.")
            settings["gen_stagger"] = 0
        for i in range(len(settings["ticker_options"])):
            self.agents.append(agent.Training(settings, alpaca_api, finbert, i, start_date, end_date))
        print("Created trainer with settings: {0}\n".format(settings))

    def start_training(self):
        self.running = True
        if len(self.agents) > 1:
            i = 0
            while self.running:
                self.agents[i].run()
                while self.agents[i].running:
                    time.sleep(1)
                if self.visualize:
                    self.agents[i].plot()
                if self.print_stats:
                    time.sleep(5)  # Wait for console printout

                i += 1
                if i >= len(self.agents):
                    i = 0
        else:
            self.agents[0].run()

    def stop_training(self):
        self.running = False
        for i in range(len(self.agents)):
            self.best_genomes[i] = self.agents[i].best_genome
            self.agents[i].running = False
