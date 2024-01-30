from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime

class FinbertNews(object):
    def __init__(self, alpaca_api):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")  # Divides text by whitespace
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

        self.alpaca_api = alpaca_api
        self.saved_news = {}

    def save_news(self, symbols, start_date, end_date):
        news_entity = self.alpaca_api.get_news(
            symbol=symbols,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            limit=20000)

        print("Saved {0} unique news for {1}".format(len(news_entity), symbols))

        for entity in news_entity:
            news_dict = entity.__dict__["_raw"]

            for symbol in news_dict["symbols"]:
                if symbol not in self.saved_news:
                    self.saved_news[symbol] = [news_dict]
                elif news_dict not in self.saved_news[symbol]:
                    self.saved_news[symbol].append(news_dict)

        print("Sorted news for {0} symbols\n".format(len(self.saved_news)))

    def estimate_sentiment(self, news):
        if len(news) == 0:
            return [0, 0, 0]
        else:
            tokens = self.tokenizer(news, return_tensors="pt", padding=True).to(self.device)

            result = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
            result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
            return result

    def get_api_sentiment(self, symbol, start_date, end_date):
        news_entity = self.alpaca_api.get_news(symbol=symbol,
                                               start=start_date.date().isoformat(),
                                               end=end_date.date().isoformat(),
                                               limit=20)
        news = [ev.__dict__["_raw"]["headline"] for ev in news_entity]
        return self.estimate_sentiment(news)

    def get_saved_sentiment(self, symbol, start_date, end_date):
        news = []
        if symbol in self.saved_news:
            for news_dict in self.saved_news[symbol]:
                news_date = datetime.strptime(news_dict["updated_at"], "%Y-%m-%dT%H:%M:%SZ")
                if start_date < news_date < end_date:
                    news.append(news_dict["headline"])
        return self.estimate_sentiment(news)
