from fastapi import FastAPI
from powderalert.ml_logic.data import fetch_prediction_data, clean_data
from powderalert.ml_logic.preprocessor import preprocess
from powderalert.ml_logic.params import *
from powderalert.ml_logic.registry import load_model_snowfall, load_model_temperature
from darts import TimeSeries
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

app = FastAPI()

app.state.model1 = load_model_snowfall()
app.state.model2 = load_model_temperature()

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.get("/predict_snowfall")
def predict(lat: float, long: float):

    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)
    X_processed = preprocess(cleaned_data)

    X_pred_columns = X_processed.drop(columns=['snowfall']).columns.tolist()

    snowfall_series = TimeSeries.from_dataframe(X_processed, value_cols=['snowfall']).astype("float32")
    feature_series = TimeSeries.from_dataframe(X_processed, value_cols=X_pred_columns).astype("float32")

    y_pred = app.state.model1.predict(series=snowfall_series, past_covariates=feature_series, n=48).values().flatten().tolist()

    print(y_pred)
    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    # uvicorn simple:app --reload
    return {
        'prediction': y_pred
    }

@app.get("/predict_temperature")
def predict(lat: float, long: float):

    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)
    df = preprocess(cleaned_data)

    #####################################################
    # The following individual steps where needed compared to the Darts model:
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    current_time = datetime.now()
    last_48h = df[(df['date'] <= current_time) & (df['date'] > current_time - timedelta(hours=48))]
    last_48h = last_48h.drop(columns='date')
    last_48h = np.expand_dims(last_48h, axis=0)
    #####################################################

    predictions = app.state.model2.predict(last_48h)
    predicted_temperatures = predictions[0]

    print(predicted_temperatures)
    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return {
        'prediction': predicted_temperatures
    }
