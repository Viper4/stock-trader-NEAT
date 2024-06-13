import base64
import requests
import time
import encryption
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import threading


class Schwab:
    def __init__(self):
        self.credentials = encryption.load_saved_data()
        self.base_url = "https://api.schwabapi.com/trader/v1"
        self.token_url = "https://api.schwabapi.com/v1/oauth/token"
        self.account_hash = None
        self.account = [None, 0]
        self.tokens = None
        self.authorizing = False
        self.refresh_thread = None

        self.authorize()

    def authorize(self):
        if self.authorizing:
            while self.authorizing:
                time.sleep(1)
            return

        print("Authorizing Charles Schwab client...")
        self.authorizing = True

        if self.refresh_thread is not None:
            self.refresh_thread.join()
            self.refresh_thread = None

        options = Options()

        options.add_experimental_option("detach", True)
        options.add_argument("--no-sandbox")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_argument("disable-infobars")

        user_agents = []
        with open(r"Saves\WebScraping\user_agents.txt", "r") as f:
            for i, line in enumerate(f):
                if i > 0:
                    user_agents.append(line.strip())
        options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3")

        service = Service(executable_path="C:\\Users\\vpr16\\PythonProjects\\StockTraderNEAT\\edgedriver_win64\\msedgedriver.exe")
        driver = webdriver.Edge(options, service)
        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={self.credentials['public_key']}&redirect_uri=https://127.0.0.1"

        #print(auth_url)
        #returned_link = input("Enter code link: ")
        driver.get(auth_url)

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "loginIdInput")))
        login_input = driver.find_element(By.ID, "loginIdInput")
        login_input.send_keys(self.credentials["username"])

        password_input = driver.find_element(By.ID, "passwordInput")
        password_input.send_keys(self.credentials["password"])

        driver.find_element(By.ID, "btnLogin").click()

        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.ID, "acceptTerms")))
        driver.find_element(By.ID, "acceptTerms").click()
        driver.find_element(By.ID, "submit-btn").click()
        driver.find_element(By.ID, "agree-modal-btn-").click()

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "form-container")))
        accounts_checkboxes = driver.find_element(By.ID, "form-container").find_element(By.TAG_NAME, "div").find_elements(By.TAG_NAME, "label")

        for checkbox in accounts_checkboxes:
            if checkbox.text[-3:] == self.credentials["account_number"]:
                checkbox_input = checkbox.find_element(By.TAG_NAME, "input")
                if not checkbox_input.is_selected():
                    checkbox_input.click()
                break

        driver.find_element(By.ID, "submit-btn").click()

        WebDriverWait(driver, 10).until(EC.url_contains("confirmation"))
        driver.find_element(By.ID, "cancel-btn").click()

        WebDriverWait(driver, 10).until(EC.url_contains("code="))
        returned_link = driver.current_url
        code = f"{returned_link[returned_link.index('code=') + 5:returned_link.index('%40')]}@"

        headers = {"Authorization": f"Basic {base64.b64encode(bytes(self.credentials['public_key'] + ':' + self.credentials['secret_key'], 'utf-8')).decode('utf-8')}",
                   "Content-Type": "application/x-www-form-urlencoded"}
        payload = {"grant_type": "authorization_code", 'code': code,
                   "redirect_uri": "https://127.0.0.1"}  # gets access and refresh tokens using authorization code

        self.tokens = requests.post('https://api.schwabapi.com/v1/oauth/token', headers=headers, data=payload).json()

        response = requests.get(url=f"{self.base_url}/accounts/accountNumbers",
                                headers={"Authorization": f"Bearer {self.tokens['access_token']}"})
        for item in response.json():
            if item["accountNumber"][-3:] == self.credentials["account_number"]:
                self.account_hash = item["hashValue"]
        try:
            driver.close()
        except Exception as e:
            print(f"Failed to close driver: {e}")

        self.authorizing = False

        self.refresh_thread = threading.Thread(target=self.refresh_token_loop)
        self.refresh_thread.start()

    def refresh_token_loop(self):
        sleep_time = self.tokens["expires_in"] - 5
        while not self.authorizing:
            time.sleep(sleep_time)
            headers = {"Authorization": f"Basic {base64.b64encode(bytes(self.credentials['public_key'] + ':' + self.credentials['secret_key'], 'utf-8')).decode('utf-8')}",
                       "Content-Type": "application/x-www-form-urlencoded"}
            payload = {'grant_type': 'refresh_token',
                       'refresh_token': self.tokens["refresh_token"]}

            response = requests.post(self.token_url,
                                     headers=headers,
                                     data=payload)

            if response.status_code == 200 or response.status_code == 201:
                print("Refreshed Charles Schwab tokens.")
                self.tokens = response.json()
                sleep_time = self.tokens["expires_in"] - 10
            elif response.status_code == 401:
                self.authorize()
                return
            else:
                print(f"{response.status_code} error refreshing tokens: {response.content}")
                sleep_time = 5

    def get_account(self):
        if self.account[0] is None or time.time() - self.account[1] > 1:
            tries = 1
            while True:
                try:
                    response = requests.get(url=f"{self.base_url}/accounts/{self.account_hash}?fields=positions",
                                            headers={"Authorization": f"Bearer {self.tokens['access_token']}"})

                    if response.status_code == 200 or response.status_code == 201:
                        self.account[0] = response.json()["securitiesAccount"]
                        self.account[1] = time.time()
                        return self.account[0]
                    elif response.status_code == 401:
                        self.authorize()
                        tries += 1
                    else:
                        print(f"Error getting account: '{response.status_code}: {response.content}'. Retrying in 5 seconds... ({tries})")
                        time.sleep(5)
                        tries += 1
                except ConnectionError as e:
                    print(f"Error getting account: '{e}'. Retrying in 5 seconds... ({tries})")
                    time.sleep(5)
                    tries += 1
        return self.account[0]

    def get_position(self, symbol):
        account = self.get_account()
        if "positions" in account:
            for position in account["positions"]:
                if symbol == position["instrument"]["symbol"]:
                    return position
        return {
            "longQuantity": 0,
            "instrument":
            {
                "symbol": symbol,
            },
            "longOpenProfitLoss": 0,
        }

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
        tries = 1
        while True:
            response = requests.post(url=f"{self.base_url}/accounts/{self.account_hash}/orders",
                                     headers=
                                     {
                                         "Authorization": f"Bearer {self.tokens['access_token']}",
                                         "Content-Type": "application/json"
                                     },
                                     json=order)
            if response.status_code == 200 or response.status_code == 201:
                print(f"Order submitted to {side} {quantity} shares of {symbol}")
                return
            elif response.status_code == 401:
                self.authorize()
                tries += 1
            else:
                print(f"Error submitting order: '{response.status_code}: {response.content}'. Retrying in 5 seconds... ({tries})")
                time.sleep(5)
                tries += 1
