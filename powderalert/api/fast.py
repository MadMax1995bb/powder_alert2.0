from fastapi import FastAPI
from powderalert.ml_logic.data import fetch_prediction_data, clean_data
from powderalert.ml_logic.preprocessor import preprocess
from powderalert.ml_logic.params import *
from powderalert.ml_logic.registry import load_model_temperature, load_model_snowdepth
from darts import TimeSeries
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

app = FastAPI()

app.state.model1 = load_model_temperature()
app.state.model2 = load_model_snowdepth()

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.get("/predict_temperature")
def predict(lat: float, long: float):
    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)

    breakpoint()

    X = cleaned_data.drop(columns = target1)
    y = cleaned_data[target1]
    y = y.reset_index()

    X_processed = preprocess(X)
    df = y.join(X_processed)

    first_predict_time = (pd.Timestamp(data.date.tail(1).values[0]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")

    last_48h = df.drop(columns = "date")
    last_48h = np.expand_dims(last_48h, axis=0)
    predictions = app.state.model1.predict(last_48h)
    predicted_temperatures = predictions[0]
    next_48h = [float(i) for i in predicted_temperatures]

    return {
        'first_predict_time': first_predict_time,
        'temperature_prediction': next_48h
    }

###################################################################################

@app.get("/predict_snowdepth")
def predict(lat: float, long: float):
    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)

    X = cleaned_data.drop(columns = target2)
    y = cleaned_data[target2]
    y = y.reset_index()

    X_processed = preprocess(X)
    df = y.join(X_processed)

    first_predict_time = (pd.Timestamp(data.date.tail(1).values[0]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")

    last_48h = np.expand_dims(df, axis=0)
    predictions = app.state.model2.predict(last_48h)
    predicted_temperatures = predictions[0]
    next_48h = [float(i) for i in predicted_temperatures]

    return {
        'first_predict_time': first_predict_time,
        'snowdepth_prediction': next_48h
    }

# @app.get("/predict_snowfall")
# def predict(lat: float, long: float):

#     data = fetch_prediction_data(lat,long)
#     cleaned_data = clean_data(data)

#     X = cleaned_data.drop(columns = target1)
#     y = cleaned_data[target1]
#     y = y.reset_index()

#     X_processed = preprocess(X)
#     df = y.join(X_processed)

#     #Get time information
#     first_predict_time = (pd.Timestamp(data.date.tail(1).values[0]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")

#     X_pred_columns = X_processed.columns.tolist()

#     snowfall_series = TimeSeries.from_dataframe(df, 'date', value_cols=['snowfall']).astype("float32")
#     feature_series = TimeSeries.from_dataframe(df, 'date', value_cols=X_pred_columns).astype("float32")

#     y_pred = app.state.model1.predict(series=snowfall_series, past_covariates=feature_series, n=48).values().flatten().tolist()
#     # ⚠️ fastapi only accepts simple Python data types as a return value
#     # among them dict, list, str, int, float, bool
#     # in order to be able to convert the api response to JSON
#     # uvicorn simple:app --reload
#     return {
#         'first_predict_time': first_predict_time,
#         'snowfall_prediction': y_pred
#     }
