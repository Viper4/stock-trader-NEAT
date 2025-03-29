import base64
import saving


def load_saved_data():
    encoded_data = saving.SaveSystem.load_data("C:\\Users\\vpr16\\PythonProjects\\StockTraderNEAT\\Saves\\schwab_info.gz")

    return {"public_key": base64.b64decode(bytes(encoded_data[0], 'utf-8')).decode('utf-8'),
            "secret_key": base64.b64decode(bytes(encoded_data[1], 'utf-8')).decode('utf-8'),
            "username": base64.b64decode(bytes(encoded_data[2], 'utf-8')).decode('utf-8'),
            "password": base64.b64decode(bytes(encoded_data[3], 'utf-8')).decode('utf-8'),
            "account_number": base64.b64decode(bytes(encoded_data[4], 'utf-8')).decode('utf-8')}


if __name__ == "__main__":
    public_key = base64.b64encode(bytes(f'{input("Enter public key: ")}', 'utf-8')).decode('utf-8')
    secret_key = base64.b64encode(bytes(f'{input("Enter secret key: ")}', 'utf-8')).decode('utf-8')
    username = base64.b64encode(bytes(f'{input("Enter username: ")}', 'utf-8')).decode('utf-8')
    password = base64.b64encode(bytes(f'{input("Enter password: ")}', 'utf-8')).decode('utf-8')
    account_number = base64.b64encode(bytes(f'{input("Enter last 3 digits of account number: ")}', 'utf-8')).decode('utf-8')

    saving.SaveSystem.save_data((public_key, secret_key, username, password, account_number), "C:\\Users\\vpr16\\PythonProjects\\StockTraderNEAT\\Saves\\schwab_info.gz")
