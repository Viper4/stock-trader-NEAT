import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import URL
import os
import json
import manager
import finbert_news
import base64
import requests

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    settings_path = os.path.join(local_dir, "settings.json")
    with open(settings_path) as file:
        settings = json.load(file)
    '''
    # Charles Schwab API
    public_schwab = "0NaypZvd2wdQaNMpq6I3WAMKTq9UxC6K"
    secret_schwab = "wsOkMnsJ9nWxTUVp"
    s_base_url = "https://api.schwabapi.com/trader/v1"

    auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={public_schwab}&redirect_uri=https://127.0.0.1"
    print("Click to authenticate charles schwab: " + auth_url)
    returned_link = input("Paste the redirect URL here: ")
    code = f"{returned_link[returned_link.index('code=') + 5:returned_link.index('%40')]}@"

    headers = {"Authorization": f"Basic {base64.b64encode(bytes(f'{public_schwab}:{secret_schwab}', 'utf-8')).decode('utf-8')}",
               "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "authorization_code", 'code': code,
            "redirect_uri": "https://127.0.0.1"}  # gets access and refresh tokens using authorization code

    tokens = requests.post('https://api.schwabapi.com/v1/oauth/token', headers=headers, data=data).json()

    response = requests.get(f"{s_base_url}/accounts/accountNumbers", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    print(response.json())
    account_number = input("Enter account number to trade with: ")
    schwab_account = None
    for item in response.json():
        if item["accountNumber"] == account_number:
            schwab_account = item
'''
    # Alpaca API
    first_account = settings["accounts"][0]
    alp_base_url = URL("https://paper-api.alpaca.markets") if first_account["paper"] else URL("https://api.alpaca.markets")
    alpaca_api = alpaca.REST(first_account["public_key"], first_account["secret_key"], base_url=alp_base_url)

    finbert = finbert_news.FinBERTNews(alpaca_api)
    modes = {"trading": manager.Trader,
             "training": manager.Trainer,
             "validation": manager.Validator}

    selected = input(f"Enter a mode ({', '.join(modes.keys())}): ")
    if selected in modes:
        instance = modes[selected](settings, finbert)
        instance.start()
    else:
        print(f"Invalid mode '{selected}'")
