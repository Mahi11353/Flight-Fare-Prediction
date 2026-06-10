
import pandas as pd
import joblib
import random

from datetime import timedelta

from utils import preprocess_data

# Load models
rf_model = joblib.load("models/random_forest.pkl")
xgb_model = joblib.load("models/xgboost.pkl")

# Load dataset
dataset = pd.read_excel("dataset/Data_Train.xlsx")


def generate_predictions(source, destination, departure_date):

    results = []

    departure_date = pd.to_datetime(departure_date)

    # Generate nearby dates
    nearby_dates = [
        departure_date + timedelta(days=i)
        for i in range(-2, 3)
    ]

    # Filter valid route
    filtered = dataset[
        (dataset["Source"].str.lower() == source.lower()) &
        (dataset["Destination"].str.lower() == destination.lower())
    ].copy()

    # No route found
    if filtered.empty:
        return []

    # Generate flights for each nearby date
    for date in nearby_dates:

        # Take random rows every time
        temp_df = filtered.sample(
            min(8, len(filtered)),
            replace=True
        ).copy()

        # Change date
        temp_df["Date_of_Journey"] = date.strftime("%d/%m/%Y")

        # Preprocess
        processed = preprocess_data(temp_df.copy())

        # Match indexes
        temp_df = temp_df.loc[processed.index]

        # Features
        X = processed.drop("Price", axis=1)

        # Predictions
        rf_prices = rf_model.predict(X)
        xgb_prices = xgb_model.predict(X)

        final_prices = ((rf_prices + xgb_prices) / 2).astype(int)

        temp_df["Predicted_Price"] = final_prices

        # Create varied flight options
        for _, row in temp_df.iterrows():

            fare = int(row["Predicted_Price"])

            # Random fare variation
            fare += random.randint(-2000, 2000)

            # Random departure time
            dep_hour = random.randint(0, 23)
            dep_min = random.randint(0, 59)

            # Random duration
            duration_hours = random.randint(2, 10)
            duration_mins = random.choice([0, 15, 30, 45])

            # Arrival calculation
            arr_hour = (
                dep_hour + duration_hours
            ) % 24

            arr_min = (dep_min + duration_mins) % 60

            dep_time = f"{dep_hour:02d}:{dep_min:02d}"
            arr_time = f"{arr_hour:02d}:{arr_min:02d}"

            # Stops
            stops = random.choice([
                "non-stop",
                "1 stop",
                "2 stops"
            ])

            duration = f"{duration_hours}h {duration_mins}m"

            results.append({

                "airline": row["Airline"],

                "source": row["Source"],

                "destination": row["Destination"],

                "departure": dep_time,

                "arrival": arr_time,

                "duration": duration,

                "stops": stops,

                "date": row["Date_of_Journey"],

                "fare": max(fare, 2500)
            })

    # Convert to dataframe
    results_df = pd.DataFrame(results)

    # Remove exact duplicates
    results_df = results_df.drop_duplicates()

    # Sort by cheapest fare
    results_df = results_df.sort_values(by="fare")

    # Keep only top 20
    results_df = results_df.head(20)

    return results_df.to_dict(orient="records")
