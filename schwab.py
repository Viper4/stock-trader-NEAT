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

class Schwab:
    def __init__(self):
        self.credentials = encryption.load_saved_data()
        self.base_url = "https://api.schwabapi.com/trader/v1"
        self.account_hash = None
        self.account = [None, 0]
        self.tokens = None

        self.authorize()

    def authorize(self):
        options = Options()
        options.add_experimental_option("detach", True)
        options.add_argument("--no-sandbox")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_argument("disable-infobars")
        options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3")

        service = Service(executable_path="C:\\Users\\vpr16\\PythonProjects\\StockTraderNEAT\\edgedriver_win64\\msedgedriver.exe")
        driver = webdriver.Edge(options, service)
        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={self.credentials['public_key']}&redirect_uri=https://127.0.0.1"
        driver.get(auth_url)

        login_input = driver.find_element(By.ID, "loginIdInput")
        login_input.send_keys(self.credentials["username"])

        password_input = driver.find_element(By.ID, "passwordInput")
        password_input.send_keys(self.credentials["password"])

        driver.find_element(By.ID, "btnLogin").click()

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "acceptTerms")))
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
        data = {"grant_type": "authorization_code", 'code': code,
                "redirect_uri": "https://127.0.0.1"}  # gets access and refresh tokens using authorization code

        self.tokens = requests.post('https://api.schwabapi.com/v1/oauth/token', headers=headers, data=data).json()

        response = requests.get(url=f"{self.base_url}/accounts/accountNumbers",
                                headers={"Authorization": f"Bearer {self.tokens['access_token']}"})
        for item in response.json():
            if item["accountNumber"][-3:] == self.credentials["account_number"]:
                self.account_hash = item["hashValue"]
        driver.close()

    def get_account(self):
        if self.account[0] is None or time.time() - self.account[1] > 1:
            tries = 1
            while True:
                response = requests.get(url=f"{self.base_url}/accounts/{self.account_hash}?fields=positions",
                                        headers={"Authorization": f"Bearer {self.tokens['access_token']}"})
                if response.status_code == 200:
                    self.account[0] = response.json()["securitiesAccount"]
                    self.account[1] = time.time()
                    return self.account[0]
                elif response.status_code == 401:
                    print("Schwab client not authorized. Re authorizing...")
                    self.authorize()
                    tries += 1
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
            if response.status_code == 201:
                print(f"Order submitted to {side} {quantity} shares of {symbol}")
                return
            elif response.status_code == 401:
                print("Schwab client not authorized. Re authorizing...")
                self.authorize()
                tries += 1
            else:
                print(f"Error submitting order: '{response.status_code}: {response.content}'. Retrying in 5 seconds... ({tries})")
                time.sleep(5)
                tries += 1
