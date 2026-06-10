from flask import Flask, render_template, request

from predict import generate_predictions

app = Flask(__name__)


import pandas as pd

import pandas as pd
import json

@app.route("/")
def home():

    df = pd.read_excel("dataset/Data_Train.xlsx")

    route_map = {}

    for source in df["Source"].unique():

        destinations = df[df["Source"] == source]["Destination"].unique()

        route_map[source] = sorted(destinations.tolist())

    sources = sorted(route_map.keys())

    return render_template(
        "index.html",
        sources=sources,
        route_map=json.dumps(route_map)
    )

@app.route("/predict", methods=["POST"])
def predict():

    source = request.form["source"]
    destination = request.form["destination"]
    departure_date = request.form["departure_date"]

    flights = generate_predictions(
        source,
        destination,
        departure_date
    )

    return render_template(
        "results.html",
        flights=flights,
        source=source,
        destination=destination
    )


if __name__ == "__main__":
    app.run(debug=True)