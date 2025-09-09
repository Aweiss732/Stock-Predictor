import pandas as pd
import joblib
from pydantic import BaseModel
from fastapi import FastAPI

model_path = "../models/stock_model.pkl"

model, scaler, features_cols = joblib.load(model_path)

app = FastAPI()

class StockData(BaseModel):
    features: list

@app.post("/predict")
def predict_stock(data: StockData):
    df = pd.DataFrame([data.features], columns=features_cols)
    df_scaled = scaler.transform(df)
    predicted = model.predict(df_scaled)[0]
    return {"prediction": "up" if predicted == 1 else "down"}