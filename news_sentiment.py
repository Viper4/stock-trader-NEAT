#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#import torch

class NewsSentiment:
    def __init__(self, alpaca_client, symbols, start_date, end_date):
        self.alpaca_client = alpaca_client
        news_entity = self.alpaca_client.get_news(
            symbol=symbols,
            start=start_date.isoformat(),
            end=end_date.isoformat())
        print("Entity: " + news_entity)
        print("Entity raw: " + str(news_entity._raw))
        for ev in news_entity:
            print("ev: " + ev)
            print("ev raw: " + ev.__dict__["_raw"])
        self.news = news_entity._raw

    def get_sentiments(self):
        pass
