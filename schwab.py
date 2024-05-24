import base64
import requests
import time


class Schwab:
    def __init__(self):
        self.public_key = "0NaypZvd2wdQaNMpq6I3WAMKTq9UxC6K"
        self.secret_key = "wsOkMnsJ9nWxTUVp"
        self.base_url = "https://api.schwabapi.com/trader/v1"
        self.account_info = None
        self.account = [None, 0]

        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={self.public_key}&redirect_uri=https://127.0.0.1"
        print("Click to authenticate charles schwab: " + auth_url)
        returned_link = input("Paste the redirect URL here: ")
        code = f"{returned_link[returned_link.index('code=') + 5:returned_link.index('%40')]}@"

        headers = {
            "Authorization": f"Basic {base64.b64encode(bytes(f'{self.public_key}:{self.secret_key}', 'utf-8')).decode('utf-8')}",
            "Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "authorization_code", 'code': code,
                "redirect_uri": "https://127.0.0.1"}  # gets access and refresh tokens using authorization code

        self.tokens = requests.post('https://api.schwabapi.com/v1/oauth/token', headers=headers, data=data).json()

        response = requests.get(url=f"{self.base_url}/accounts/accountNumbers",
                                headers={"Authorization": f"Bearer {self.tokens['access_token']}"})
        print(response.json())
        account_number = input("Enter account number to trade with: ")
        for item in response.json():
            if item["accountNumber"] == account_number:
                print("Here " + str(account_number) + " " + str(item))
                self.account_info = item

    def get_account(self):
        if self.account[0] is None or time.time() - self.account[1] > 1:
            tries = 1
            while True:
                response = requests.get(url=f"{self.base_url}/accounts/{self.account_info['hashValue']}?fields=positions",
                                        headers={"Authorization": f"Bearer {self.tokens['access_token']}"})
                if response.status_code == 200:
                    self.account[0] = response.json()["securitiesAccount"]
                    self.account[1] = time.time()
                    return self.account[0]
                else:
                    print(f"Error getting account: '{response.status_code}: {response.content}'. Retrying in 5 seconds... ({tries})")
                    time.sleep(5)
                    tries += 1
        return self.account[0]

    def get_position(self, symbol):
        account = self.get_account()
        if "positions" in account:
            for position in account["positions"]:
                if symbol == position["instrument"]["symbol"]:
                    return position
        return None

    def submit_order(self, symbol, quantity, side):
        order = {
            "complexOrderStrategyType": "NONE",
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": side,
                    "quantity": quantity,
                    "instrument":
                        {
                            "symbol": symbol,
                            "assetType": "EQUITY"
                        }
                }
            ]
        }
        requests.post(url=f"{self.base_url}/accounts/{self.account_info['hashValue']}/orders",
                      headers={"Authorization": f"Bearer {self.tokens['access_token']}"},
                      data=order)
