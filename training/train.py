import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data_file_path = "./data/cleaned/AAPL_cleaned.csv"
mlflow_path = "./mlflow"
store_model_folder = "./models"
store_model_path = store_model_folder + "/stock_model.pkl"

data = pd.read_csv(data_file_path)
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'price_change']
X = data[features]
Y = data['future_up']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # does it make sense to split this randomly?

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

mlflow.set_tracking_uri(mlflow_path)
mlflow.set_experiment("Stock-mover-predictor")

with mlflow.start_run():
    n_est = 200
    rf_model = RandomForestClassifier(n_estimators=n_est, random_state=1)
    rf_model.fit(x_train_scaled, y_train)
    accuracy = rf_model.score(x_test_scaled, y_test)

    mlflow.log_param("Model", "Random forest")
    mlflow.log_param("n_estimators", n_est)
    mlflow.log_param("Accuracy", accuracy)
    mlflow.sklearn.log_model(rf_model, "rf_stock_model")

    print(f"Model trained with {accuracy*100:.2f}% accuracy")

os.makedirs(store_model_folder, exist_ok=True)
joblib.dump((rf_model, scaler, features), store_model_path)
print(f"Model trained and stored to {store_model_path}")