import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.1-8b-instant")


# Sample data
data = [
    {"transaction_id": 1, "amount": 50, "currency": "USD", "country": "US"},
    {"transaction_id": 2, "amount": 50000, "currency": "USD", "country": "US"},
    {"transaction_id": 3, "amount": 70, "currency": "EUR", "country": "DE"},
    {"transaction_id": 4, "amount": 600000, "currency": "USD", "country": "RU"}
]

df = pd.DataFrame(data)


# Function to ask LLM whether a row is anomalous
def check_anomaly(row):
    prompt = f"""
    You are a financial anomaly detection assistant.
    Analyze the following transaction and determine if it is anomalous.
    Respond only with 'Yes' if it's anomalous or 'No' if it's normal.
    No need to explain your reasoning. Just provide a simple one word answer.
    Any transaction with an amount greater than 10000 in USD or EUR may or may not be anomalous.

    Transaction:
    ID: {row['transaction_id']}
    Amount: {row['amount']}
    Currency: {row['currency']}
    Country: {row['country']}
    """

    response = llm.invoke(prompt)
    return response.content


# Apply to DataFrame
df["Anomalous"] = df.apply(check_anomaly, axis=1)

print(df)
