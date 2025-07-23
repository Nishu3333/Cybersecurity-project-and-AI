# generate_individual_data.py

import pandas as pd
import random
from faker import Faker

fake = Faker()
random.seed(42)

NUM_CUSTOMERS = 5000
MIN_TRANSACTIONS = 25000
MAX_TRANSACTIONS = 30000
NUM_TRANSACTIONS = random.randint(MIN_TRANSACTIONS, MAX_TRANSACTIONS)

BRANCH_CODES = ['001', '002', '003', '004', '005']
CURRENCY_MAP = {'NPR': '1', 'USD': '2'}

def generate_account_number():
    branch_code = random.choice(BRANCH_CODES)
    currency_digit = random.choice(list(CURRENCY_MAP.values()))
    middle_digits = str(random.randint(1000000, 9999999))  # 7 digits
    return f"{branch_code}{middle_digits}{currency_digit}"

# Load static indicators
from load_static_indicators import occupation_risk, country_risk, account_opening_method_risk, identification_used_risk

occupations = list(occupation_risk.keys())
nationalities = ["Nepali", "Indian", "Other"]
income_sources = ["Salary", "Business", "Investment", "Remittance", "Other"]

customer_data = []
for i in range(NUM_CUSTOMERS):
    nationality = random.choice(nationalities)
    occupation = random.choice(occupations)
    pep_status = "Yes" if "GOVT." in occupation or "LEGISLATOR" in occupation else "No"
    income_source = random.choice(income_sources)
    account_number = generate_account_number()

    # Generate annual income as numeric value
    base_income = random.uniform(300000, 8000000)  # Between NPR 300,000 – 8,000,000
    annual_income = round(base_income, -3)         # Round to nearest thousand

    customer_data.append({
        'CustomerID': f'300{i+1:05d}',
        'Nationality': nationality,
        'Occupation': occupation,
        'PEP_Status': pep_status,
        'RiskRating': occupation_risk.get(occupation, "Medium"),
        'IncomeSource': income_source,
        'AnnualIncome': annual_income,
        'AccountOpeningMethod': random.choice(list(account_opening_method_risk.keys())),
        'IdentificationUsed': random.choice(list(identification_used_risk.keys())),
        'AccountNumber': account_number
    })

df_customers = pd.DataFrame(customer_data)
df_customers.to_csv('../datasets/customer_data.csv', index=False)

# Generate transaction data
transaction_data = []
accounts_list = df_customers['AccountNumber'].tolist()

for t_id in range(NUM_TRANSACTIONS):
    sender_acc = random.choice(accounts_list)
    receiver_acc = random.choice(accounts_list)
    while receiver_acc == sender_acc:
        receiver_acc = random.choice(accounts_list)

    amount = round(random.uniform(100, 1000000), 2)
    date = fake.date_between(start_date='-1y', end_date='today')
    time = fake.time()
    channel = random.choice(["Online", "ATM", "Branch"])
    transaction_type = random.choice(["Transfer", "Cash Withdrawal", "Deposit"])

    sender_row = df_customers[df_customers['AccountNumber'] == sender_acc]
    sender_nationality = sender_row['Nationality'].values[0] if not sender_row.empty else "Other"

    country = "Nepal" if sender_nationality == "Nepali" else random.choice([
        "India", "China", "USA", "Russia", "Pakistan"
    ])

    transaction_data.append({
        'TransactionID': f'TX{t_id+1}',
        'Amount': amount,
        'Date': date,
        'Time': time,
        'SenderAccount': sender_acc,
        'ReceiverAccount': receiver_acc,
        'Country': country,
        'TransactionType': transaction_type,
        'Currency': "NPR",
        'Channel': channel,
        'IsSuspicious': 0
    })

df_transactions = pd.DataFrame(transaction_data)
df_transactions.to_csv('../datasets/transaction_data.csv', index=False)

print("Synthetic datasets generated successfully.")