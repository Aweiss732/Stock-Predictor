import pandas as pd
import joblib
import os
from pydantic import BaseModel
from fastapi import FastAPI

script_dir = os.path.dirname(__file__)

model_relative_path = os.path.join(script_dir, '../models/stock_model.pkl')

model, scaler, features_cols = joblib.load(model_relative_path)

model_path = "../models/stock_model.pkl"

#model, scaler, features_cols = joblib.load(model_path)

app = FastAPI()

class StockData(BaseModel):
    features: list

@app.post("/predict")
def predict_stock(data: StockData):
    df = pd.DataFrame([data.features], columns=features_cols)
    df_scaled = scaler.transform(df)
    predicted = model.predict(df_scaled)[0]
    return {"prediction": "up" if predicted == 1 else "down"}


@app.get("/")
async def root():
    return {"message": "Hello"}
