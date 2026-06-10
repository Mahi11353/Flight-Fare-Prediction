import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from utils import preprocess_data

# Load dataset
df = pd.read_excel("dataset/Data_Train.xlsx")

# Preprocess
df = preprocess_data(df)

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Categorical columns
categorical_cols = ["Airline", "Source", "Destination"]

# Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

rf_preds = rf_pipeline.predict(X_test)

print("\nRandom Forest Results")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("RMSE:", mean_squared_error(y_test, rf_preds) ** 0.5)
print("R2:", r2_score(y_test, rf_preds))

# XGBoost
xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ))
])

xgb_pipeline.fit(X_train, y_train)

xgb_preds = xgb_pipeline.predict(X_test)

print("\nXGBoost Results")
print("MAE:", mean_absolute_error(y_test, xgb_preds))
print("RMSE:", mean_squared_error(y_test, xgb_preds) ** 0.5)
print("R2:", r2_score(y_test, xgb_preds))

# Save models
joblib.dump(rf_pipeline, "models/random_forest.pkl")
joblib.dump(xgb_pipeline, "models/xgboost.pkl")

print("\nModels saved successfully.")