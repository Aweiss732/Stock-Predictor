import yfinance as yf
import pandas as pd
import os

raw_path = "./data/raw/AAPL.csv"
cleaned_path= "./data/cleaned/AAPL_cleaned.csv"
os.makedirs(os.path.dirname(raw_path), exist_ok=True)
os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)


ticker = "AAPL AMZN GOOG"
ticker = "AAPL"

data = yf.download(ticker, period="6mo", interval="1h")
data.to_csv(raw_path)

df = pd.read_csv(raw_path)
print(df.columns)

df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['price_change'] = df['Close'].pct_change()
df['future_up'] = (df['price_change'].shift(-1) > 0).astype(int)

df.dropna(inplace=True)

df.to_csv(cleaned_path, index=False)
print(f"Processed and cleaned data to {cleaned_path}")