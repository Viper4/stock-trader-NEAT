import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime as dt


class FinBERTNews(object):
    def __init__(self, alpaca_api):
        self.alpaca_api = alpaca_api

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")  # Divides text by whitespace
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)

        self.saved_news = {}
        self.last_news = []
        self.last_sentiment = []

    def save_news(self, symbols, start_date, end_date):
        self.saved_news.clear()
        news_entity = self.alpaca_api.get_news(
            symbol=symbols,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            limit=20000)

        for entity in news_entity:
            news_dict = entity.__dict__["_raw"]

            for symbol in news_dict["symbols"]:
                news_obj = {"headline": news_dict["headline"], "timestamp": news_dict["updated_at"]}
                if symbol not in self.saved_news:
                    self.saved_news[symbol] = [news_obj]
                elif news_obj not in self.saved_news[symbol]:
                    self.saved_news[symbol].append(news_obj)

        print("Finbert: Cached {0} news involving {1} with a total of {2} unique symbols".format(len(news_entity), symbols, len(self.saved_news)))

    def estimate_sentiment(self, news):
        if len(news) == 0:
            return [0, 0, 0]
        else:
            if np.array_equal(news, self.last_news):
                return self.last_sentiment
            tokens = self.tokenizer(news, return_tensors="pt", padding=True).to(self.device)
            sentiment = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
            sentiment = torch.nn.functional.softmax(torch.sum(sentiment, 0), dim=-1)

            if self.device == "cuda:0":
                # Garbage collection on GPU
                sentiment_list = sentiment.cpu().detach().tolist()  # python list is faster than numpy and tensor
                del tokens, sentiment
                torch.cuda.empty_cache()
            else:
                sentiment_list = sentiment.detach().tolist()

            self.last_news = news
            self.last_sentiment = sentiment_list
            return sentiment_list

    def get_api_sentiment(self, symbol, start_date, end_date):
        news_entity = self.alpaca_api.get_news(symbol=symbol,
                                               start=start_date.isoformat(),
                                               end=end_date.isoformat(),
                                               limit=20)
        news = [ev.__dict__["_raw"]["headline"] for ev in news_entity]
        return self.estimate_sentiment(news)

    def get_saved_sentiment(self, symbol, start_date, end_date):
        news = []
        if symbol in self.saved_news:
            for news_obj in self.saved_news[symbol]:
                news_date = dt.datetime(year=int(news_obj["timestamp"][0:4]),
                                        month=int(news_obj["timestamp"][5:7]),
                                        day=int(news_obj["timestamp"][8:10]),
                                        hour=int(news_obj["timestamp"][11:13]),
                                        minute=int(news_obj["timestamp"][14:16]),
                                        second=int(news_obj["timestamp"][17:19]),
                                        tzinfo=dt.timezone.utc).replace(tzinfo=start_date.tzinfo)
                if start_date <= news_date <= end_date:
                    news.append(news_obj["headline"])
        return self.estimate_sentiment(news)
