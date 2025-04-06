import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime as dt
from Managers.base_manager import Manager


class FinBERTNews(object):
    def __init__(self, alpaca_api):
        self.alpaca_api = alpaca_api

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)

        self.saved_news = {}
        self.last_news = []
        self.last_sentiment = 0

    def get_news(self, symbols, start_date, end_date, limit=50):
        tries = 1
        while True:
            try:
                news_entity = self.alpaca_api.get_news(symbol=symbols,
                                                       start=start_date.isoformat(),
                                                       end=end_date.isoformat(),
                                                       limit=limit,
                                                       sort="asc",
                                                       )
                return news_entity
            except Exception as e:
                Manager.check_internet_connection()
                print(f"Error getting news: '{e}'. Retrying in 5 seconds... ({tries})")
                tries += 1
                time.sleep(5)

    def save_news(self, symbols, start_date, end_date):
        if torch.cuda.is_available():
            print(f"Finbert CUDA: Saving news for {symbols} from {start_date} to {end_date}")
        else:
            print(f"Finbert CPU: Saving news for {symbols} from {start_date} to {end_date}")
        self.saved_news.clear()
        # 200 calls/minute API limit makes this slow; Gets 50 news entities per request.
        news_entity = self.get_news(symbols, start_date, end_date, 500000)

        for entity in news_entity:
            news_dict = entity.__dict__["_raw"]

            for symbol in news_dict["symbols"]:
                news_obj = {"headline": news_dict["headline"], "timestamp": news_dict["updated_at"]}

                if symbol not in self.saved_news:
                    self.saved_news[symbol] = [news_obj]
                elif news_obj not in self.saved_news[symbol]:
                    self.saved_news[symbol].append(news_obj)

        if torch.cuda.is_available():
            print(f"Finbert CUDA: Cached {len(news_entity)} news involving {symbols} with a total of {len(self.saved_news)} unique symbols")
        else:
            print(f"Finbert CPU: Cached {len(news_entity)} news involving {symbols} with a total of {len(self.saved_news)} unique symbols")

    def free_gpu_memory(self, threshold):
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        reserved_memory = torch.cuda.memory_reserved(self.device)
        if reserved_memory / total_memory > threshold:
            torch.cuda.empty_cache()

    def estimate_sentiment(self, news):
        if len(news) == 0:
            return 0

        if np.array_equal(news, self.last_news):
            return self.last_sentiment

        if self.device == "cpu":
            with torch.no_grad():  # Don't need gradients since we aren't training
                tokens = self.tokenizer(news, return_tensors="pt", padding=True).to(self.device)

                with torch.amp.autocast(self.device):  # Enable mixed precision
                    sentiment_probs = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
                    sentiment_probs = torch.nn.functional.softmax(torch.sum(sentiment_probs, 0), dim=-1).to(torch.float64).numpy()
                    sentiment = sentiment_probs[0] - sentiment_probs[1]  # positive% - negative%
        else:
            with torch.no_grad():  # Don't need gradients since we aren't training
                tokens = self.tokenizer(news, return_tensors="pt", padding=True).to(self.device)

                with torch.amp.autocast(self.device):  # Enable mixed precision
                    sentiment_probs = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
                    sentiment_probs = torch.nn.functional.softmax(torch.sum(sentiment_probs, 0), dim=-1).detach().cpu().numpy()
                    sentiment = sentiment_probs[0] - sentiment_probs[1]  # positive% - negative%

            torch.cuda.synchronize()
            del tokens, sentiment_probs
            self.free_gpu_memory(0.5)

        self.last_news = news
        self.last_sentiment = sentiment
        return sentiment

    def get_api_sentiment(self, symbol, start_date, end_date):
        news_entity = self.get_news(symbol, start_date, end_date)
        news = [ev.__dict__["_raw"]["headline"] for ev in news_entity]
        return self.estimate_sentiment(news)

    def find_saved_news_index(self, symbol, date):
        low = 0
        high = len(self.saved_news[symbol]) - 1
        while low <= high:
            mid = (low + high) // 2
            news_date = dt.datetime.fromisoformat(self.saved_news[symbol][mid]["timestamp"]).replace(tzinfo=date.tzinfo)

            if news_date < date:
                low = mid + 1
            elif news_date > date:
                high = mid - 1
            else:
                return mid  # Exact match
        return max(0, high)  # Closest index before "date"

    def get_saved_sentiment(self, symbol, start_date, end_date):
        news = []
        if symbol in self.saved_news:
            start_index = self.find_saved_news_index(symbol, start_date)

            for i in range(start_index, len(self.saved_news[symbol])):
                news_obj = self.saved_news[symbol][i]
                news_date = dt.datetime.fromisoformat(news_obj["timestamp"]).replace(tzinfo=start_date.tzinfo)

                if start_date <= news_date <= end_date:
                    news.append(news_obj["headline"])
                if news_date > end_date:
                    break

        return self.estimate_sentiment(news)
