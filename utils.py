import pandas as pd


def convert_duration(duration):
    hours = 0
    minutes = 0

    if 'h' in duration:
        hours = int(duration.split('h')[0].strip())

    if 'm' in duration:
        minutes = int(duration.split('m')[0].split()[-1])

    return hours * 60 + minutes


def preprocess_data(df):

    df.dropna(inplace=True)

    # Date
    df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.day
    df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.month

    # Departure
    df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
    df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute

    # Arrival
    df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
    df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute

    # Duration
    df["Duration_mins"] = df["Duration"].apply(convert_duration)

    # Stops
    df["Total_Stops"] = df["Total_Stops"].replace({
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4
    })

    # Drop unnecessary columns
    df.drop([
        "Date_of_Journey",
        "Dep_Time",
        "Arrival_Time",
        "Duration",
        "Route",
        "Additional_Info"
    ], axis=1, inplace=True)

    return df