import tkinter as tk
from tkinter import filedialog
import json


class JsonEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JSON Editor")

        self.json_data = {"accounts": []}
        self.selected_account_index = 0
        self.selected_stock_index = 0

        # Load JSON Button
        self.load_button = tk.Button(root, text="Load JSON", command=self.load_json)
        self.load_button.pack(pady=5)

        # Save JSON Button
        self.save_button = tk.Button(root, text="Save JSON", command=self.save_json)
        self.save_button.pack(pady=5)

        self.update_json_button = tk.Button(root, text="Update JSON", command=self.update_json_data)
        self.update_json_button.pack(pady=5)

        self.reset_button = tk.Button(root, text="Reset", command=self.reset_input_fields)
        self.reset_button.pack(pady=5)

        # Top-Level Settings Frame
        self.top_level_frame = tk.Frame(root)
        self.top_level_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.create_top_level_settings()

        # Account Settings Frame
        self.account_frame = tk.Frame(root)
        self.account_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.create_account_settings()

        # Stock Settings Frame
        self.stock_frame = tk.Frame(root)
        self.stock_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        self.create_stock_settings()

    def create_top_level_settings(self):
        self.create_label_entry(self.top_level_frame, "Config Path:", 0, 40, "config_path")
        self.create_label_entry(self.top_level_frame, "Save Path:", 1, 40, "save_path")
        self.create_label_entry(self.top_level_frame, "Training Reset:", 2, 10, "training_reset")
        self.create_label_entry(self.top_level_frame, "Gen Stagger:", 3, 10, "gen_stagger")
        self.create_label_entry(self.top_level_frame, "Processes:", 4, 10, "processes")
        self.create_checkbox(self.top_level_frame, "Print Stats", 5, "print_stats")
        self.create_checkbox(self.top_level_frame, "Log Training", 6, "log_training")
        self.create_checkbox(self.top_level_frame, "Visualize", 7, "visualize")

    def create_account_settings(self):
        # Account Selection Dropdown
        self.account_names = tk.StringVar(root)
        self.account_dropdown = tk.OptionMenu(self.account_frame, self.account_names, "")
        self.account_dropdown.grid(row=0, columnspan=2, pady=10)

        self.create_label_entry(self.account_frame, "Account Name:", 1, 40, "name")
        self.create_label_entry(self.account_frame, "Public Key:", 2, 40, "public_key")
        self.create_label_entry(self.account_frame, "Secret Key:", 3, 40, "secret_key")
        self.create_checkbox(self.account_frame, "Paper", 4, "paper")
        self.create_label_entry(self.account_frame, "Backtest Days:", 5, 10, "backtest_days")
        self.create_label_entry(self.account_frame, "Profit Window:", 6, 10, "profit_window")
        self.create_label_entry(self.account_frame, "Interval:", 7, 10, "interval")
        self.create_label_entry(self.account_frame, "Cash Limit:", 8, 10, "cash_limit")

        # Center the Add Account button
        self.add_account_button = tk.Button(self.account_frame, text="Add Account", command=self.add_account)
        self.add_account_button.grid(row=9, columnspan=2, pady=10)

    def create_stock_settings(self):
        # Stock listbox
        self.stocks_label = tk.Label(self.stock_frame, text="Stocks:")
        self.stocks_label.grid(row=0, column=0, padx=5, pady=5)
        self.stocks_listbox = tk.Listbox(self.stock_frame, selectmode=tk.SINGLE, height=10, width=60)
        self.stocks_listbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.stocks_listbox.bind('<<ListboxSelect>>', self.on_stock_select)

        self.create_label_entry(self.stock_frame, "Symbol:", 2, 10, "symbol")
        self.create_label_entry(self.stock_frame, "Cash at Risk:", 4, 10, "cash_at_risk")
        self.create_label_entry(self.stock_frame, "Population Filename:", 5, 40, "population_filename")
        self.create_label_entry(self.stock_frame, "Genome Filename:", 6, 40, "genome_filename")
        self.create_label_entry(self.stock_frame, "Training Filename:", 7, 40, "training_filename")

        # Buttons
        self.add_stock_button = tk.Button(self.stock_frame, text="Add Stock", command=self.add_stock)
        self.add_stock_button.grid(row=8, columnspan=2, pady=10)

        self.delete_stock_button = tk.Button(self.stock_frame, text="Delete Stock", command=self.delete_stock)
        self.delete_stock_button.grid(row=9, columnspan=2, pady=10)

    def create_label_entry(self, frame, label_text, row, width, setting_key):
        label = tk.Label(frame, text=label_text)
        entry = tk.Entry(frame, width=width)
        label.grid(row=row, column=0, padx=5, pady=5)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        setattr(self, f"{setting_key}_entry", entry)

    def create_checkbox(self, frame, label_text, row, setting_key):
        var = tk.BooleanVar()
        checkbox = tk.Checkbutton(frame, text=label_text, variable=var)
        checkbox.grid(row=row, column=0, padx=5, pady=5)
        setattr(self, f"{setting_key}_var", var)

    def update_top_level_settings(self):
        top_level_inputs = [
            "config_path", "save_path", "training_reset",
            "gen_stagger", "processes",
        ]

        for setting_key in top_level_inputs:
            entry = getattr(self, f"{setting_key}_entry", None)
            if entry:
                entry.delete(0, tk.END)
                entry.insert(0, self.json_data.get(setting_key, ""))

        top_level_checkboxes = [
            "print_stats", "log_training", "visualize"
        ]

        for setting_key in top_level_checkboxes:
            checkbox = getattr(self, f"{setting_key}_var", None)
            if checkbox:
                checkbox.set(self.json_data.get(setting_key, False))

    def update_account_dropdown(self):
        accounts = self.json_data.get("accounts", [])
        account_names = [account["name"] for account in accounts]

        self.account_names.set("")  # Clear the default value
        self.account_dropdown['menu'].delete(0, 'end')
        for name in account_names:
            self.account_dropdown['menu'].add_command(label=name,
                                                      command=tk._setit(self.account_names, name, self.select_account))

        if account_names:
            self.account_names.set(account_names[self.selected_account_index])

    def update_stocks_listbox(self):
        self.stocks_listbox.delete(0, tk.END)
        accounts = self.json_data.get("accounts", [])
        stocks = accounts[self.selected_account_index].get("stocks", [])

        for i, stock in enumerate(stocks):
            self.stocks_listbox.insert(tk.END, f"Stock {i + 1}: {stock['symbol']}")

    def select_account(self, account_name):
        accounts = self.json_data.get("accounts", [])
        account_index = next((i for i, account in enumerate(accounts) if account["name"] == account_name), None)

        if account_index is not None:
            self.selected_account_index = account_index
            self.update_account_inputs()
            self.update_stocks_listbox()
            self.update_account_dropdown()

    def on_stock_select(self, event):
        selected_index = self.stocks_listbox.curselection()
        if selected_index:
            self.selected_stock_index = selected_index[0]
            self.update_stock_input_fields()

    def update_stock_input_fields(self):
        stocks = self.get_selected_stocks()
        if stocks:
            stock = stocks[self.selected_stock_index]
            stock_settings = [
                "symbol", "cash_at_risk",
                "population_filename", "genome_filename", "training_filename",
            ]

            for setting_key in stock_settings:
                entry = getattr(self, f"{setting_key}_entry", None)
                if entry:
                    entry.delete(0, tk.END)
                    entry.insert(0, stock.get(setting_key, ""))

    def update_account_inputs(self):
        accounts = self.json_data["accounts"]
        account = accounts[self.selected_account_index]

        account_settings = [
            "name", "public_key", "secret_key",
            "backtest_days", "profit_window", "interval",
            "cash_limit"
        ]

        for setting_key in account_settings:
            entry = getattr(self, f"{setting_key}_entry", None)
            if entry:
                entry.delete(0, tk.END)
                entry.insert(0, account.get(setting_key, ""))

        account_checkboxes = [
            "paper"
        ]

        for setting_key in account_checkboxes:
            checkbox = getattr(self, f"{setting_key}_var", None)
            if checkbox:
                checkbox.set(account.get(setting_key, False))

    def get_attribute(self, name, operation):
        attribute = getattr(self, name, None).get()
        if attribute == '':
            if operation == "str":
                return ""
            elif operation == "int":
                return 0
            elif operation == "float":
                return 0.0
            elif operation == "bool":
                return False
        else:
            if operation == "str":
                return attribute
            elif operation == "int":
                return int(attribute)
            elif operation == "float":
                return float(attribute)
            elif operation == "bool":
                return bool(attribute)

    def add_stock(self):
        stock = {
            "symbol": self.get_attribute("symbol_entry", "str"),
            "cash_at_risk": self.get_attribute("cash_at_risk_entry", "float"),
            "population_filename": self.get_attribute("population_filename_entry", "str"),
            "genome_filename": self.get_attribute("genome_filename_entry", "str"),
            "training_filename": self.get_attribute("training_filename_entry", "str"),
        }
        accounts = self.json_data.setdefault("accounts", [])
        accounts[self.selected_account_index]["stocks"].append(stock)

        self.update_stocks_listbox()

    def delete_stock(self):
        stocks = self.get_selected_stocks()
        if stocks:
            del_index = self.selected_stock_index
            accounts = self.json_data.get("accounts", [])
            accounts[self.selected_account_index]["stocks"].pop(del_index)

            self.update_stocks_listbox()

    def get_selected_stocks(self):
        accounts = self.json_data.get("accounts", [])
        stocks = accounts[self.selected_account_index].get("stocks", [])

        return stocks

    def add_account(self):
        accounts = self.json_data.setdefault("accounts", [])
        accounts.append({
            "name": self.get_attribute("name_entry", "str"),
            "public_key": self.get_attribute("public_key_entry", "str"),
            "secret_key": self.get_attribute("secret_key_entry", "str"),
            "paper": self.get_attribute("paper_var", "bool"),
            "backtest_days": self.get_attribute("backtest_days_entry", "int"),
            "profit_window": self.get_attribute("profit_window_entry", "int"),
            "interval": self.get_attribute("interval_entry", "int"),
            "cash_limit": self.get_attribute("cash_limit_entry", "float"),
            "stocks": [],
        })

        self.select_account(accounts[-1]["name"])
        self.update_stocks_listbox()
        self.update_account_inputs()

    def load_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as file:
                self.json_data = json.load(file)
            accounts = self.json_data.get("accounts", [])
            if len(accounts) > 0:
                self.select_account(accounts[0]["name"])
            self.update_stocks_listbox()
            self.update_top_level_settings()
            self.update_account_inputs()

    def save_json(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w') as file:
                json.dump(self.json_data, file, indent=2)

    def update_json_data(self):
        self.json_data["config_path"] = self.get_attribute("config_path_entry", "str")
        self.json_data["save_path"] = self.get_attribute("save_path_entry", "str")
        self.json_data["training_reset"] = self.get_attribute("training_reset_entry", "int")
        self.json_data["gen_stagger"] = self.get_attribute("gen_stagger_entry", "int")
        self.json_data["processes"] = self.get_attribute("processes_entry", "int")

        self.json_data["print_stats"] = self.get_attribute("print_stats_var", "bool")
        self.json_data["log_training"] = self.get_attribute("log_training_var", "bool")
        self.json_data["visualize"] = self.get_attribute("visualize_var", "bool")

        accounts = self.json_data.get("accounts", [])
        account = accounts[self.selected_account_index]

        account["name"] = self.get_attribute("name_entry", "str")
        account["public_key"] = self.get_attribute("public_key_entry", "str")
        account["secret_key"] = self.get_attribute("secret_key_entry", "str")
        account["paper"] = self.get_attribute("paper_var", "bool")
        account["backtest_days"] = self.get_attribute("backtest_days_entry", "int")
        account["profit_window"] = self.get_attribute("profit_window_entry", "int")
        account["interval"] = self.get_attribute("interval_entry", "int")
        account["cash_limit"] = self.get_attribute("cash_limit_entry", "float")

        stocks = self.get_selected_stocks()
        if stocks:
            stock = stocks[self.selected_stock_index]
            stock["symbol"] = self.get_attribute("symbol_entry", "str")
            stock["cash_at_risk"] = self.get_attribute("cash_at_risk_entry", "float")
            stock["population_filename"] = self.get_attribute("population_filename_entry", "str")
            stock["genome_filename"] = self.get_attribute("genome_filename_entry", "str")
            stock["training_filename"] = self.get_attribute("training_filename_entry", "str")

        # Print the updated JSON data for testing purposes.
        print(json.dumps(self.json_data, indent=2))
        self.update_stocks_listbox()
        self.update_top_level_settings()
        self.update_account_inputs()

    def reset_input_fields(self):
        self.update_account_inputs()
        self.update_stocks_listbox()
        self.update_top_level_settings()
        self.update_stock_input_fields()


if __name__ == "__main__":
    root = tk.Tk()
    app = JsonEditorApp(root)
    root.mainloop()
