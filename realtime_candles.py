'''from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import time'''
from zenrows import ZenRowsClient


client = ZenRowsClient("8ec8eac2101f2994a26ad21add88e0a27b4cb85a")


'''def get_dynamic_soup(url: str) -> BeautifulSoup:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        soup = BeautifulSoup(page.content(), "lxml")
        browser.close()
        return soup

def candle_scraper(ticker):
    print("Getting " + ticker)
    url = f"https://www.tradingview.com/chart/?symbol=NASDAQ%3A{ticker}"
    data = {}

    try:
        web_content = get_dynamic_soup(url)

        divs = web_content.find_all("div", class_=re.compile(r"^valueItem-l31H9iuA"))
        for div in divs:
            title = div.find_next("div", class_=re.compile(r"^valueTitle-l31H9iuA"))
            value = div.find_next("div", class_=re.compile(r"^valueValue-l31H9iuA"))
            if title is not None and value is not None:
                data[title.get_text()] = value.get_text()
        return data
    except ConnectionError as e:
        print("Connection Error: " + str(e))
        return data
        
# candle_scraper("AMD")
print(candle_scraper("TSLA"))
'''


def get_latest_candles(symbol, interval="5m"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}"

    response = client.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()

        # Extract OHLC data from the response
        try:
            quote_data = data["chart"]["result"][0]["indicators"]["quote"][0]
            formatted_data = []
            for i in range(len(quote_data["open"])):
                if quote_data["open"][i] != quote_data["high"][i] and quote_data["volume"][i] != 0:  # Sometimes when market is closed last element is weird
                    formatted_data.append({"open": quote_data["open"][i],
                                           "high": quote_data["high"][i],
                                           "low": quote_data["low"][i],
                                           "close": quote_data["close"][i],
                                           "volume": quote_data["volume"][i]})
            return formatted_data
        except (KeyError, IndexError):
            print("Error: Unable to extract data from the response.")
            return None
    else:
        print(f"Error: Unable to fetch data. Status Code: {response.status_code}")
        return None
