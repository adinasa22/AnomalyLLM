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

def explain_anomaly(row):
    prompt = f"""
    You are a financial anomaly detection assistant.
    Explain why the following transaction is considered anomalous or not.
    
    Transaction:
    ID: {row['transaction_id']}
    Amount: {row['amount']}
    Currency: {row['currency']}
    Country: {row['country']}
    
    Anomalous: {row['Anomalous']}
    """

    response = llm.invoke(prompt)
    return response.content

# Apply explanation to DataFrame
df["Explanation"] = df.apply(explain_anomaly, axis=1)
print(df[["transaction_id", "Anomalous", "Explanation"]])
