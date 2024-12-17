from fastapi import FastAPI
from powderalert.ml_logic.data import fetch_prediction_data, clean_data
from powderalert.ml_logic.preprocessor import preprocess
from powderalert.ml_logic.params import *
from powderalert.ml_logic.registry import load_model_snowfall, load_model_temperature, load_model_snowfall2, load_model_windspeed
from darts import TimeSeries
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

app = FastAPI()

app.state.model1 = load_model_snowfall()
app.state.model2 = load_model_temperature()
app.state.model3 = load_model_snowfall2()
app.state.model4 = load_model_windspeed()


@app.get("/")
def root():
    return dict(greeting="Hello")

@app.get("/predict_snowfall")
def predict(lat: float, long: float):

    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)
    X_processed = preprocess(cleaned_data)

    #Get time information
    first_predict_time = (pd.Timestamp(data.date.tail(1).values[0]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")

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
        'first_predict_time': first_predict_time,
        'snowfall_prediction': y_pred
    }

@app.get("/predict_temperature")
def predict(lat: float, long: float):
    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)
    df = preprocess(cleaned_data)

    first_predict_time = (pd.Timestamp(data.date.tail(1).values[0]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")

    last_48h = np.expand_dims(df, axis=0)
    predictions = app.state.model2.predict(last_48h)
    predicted_temperatures = predictions[0]
    next_48h = [float(i) for i in predicted_temperatures]

    return {
        'first_predict_time': first_predict_time,
        'temperature_prediction': next_48h
    }

###################################################################################
########################## Optional Snowfall predictions ##########################

@app.get("/predict_snowfall2")
def predict(lat: float, long: float):
    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)
    df = preprocess(cleaned_data)

    first_predict_time = (pd.Timestamp(data.date.tail(1).values[0]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")

    last_48h = np.expand_dims(df, axis=0)
    predictions = app.state.model3.predict(last_48h)
    predicted_temperatures = predictions[0]
    next_48h = [float(i) for i in predicted_temperatures]

    return {
        'first_predict_time': first_predict_time,
        'snowfall2_prediction': next_48h
    }

################################### Optional DL wind prediction ##############################

@app.get("/predict_windspeed")
def predict(lat: float, long: float):
    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)
    df = preprocess(cleaned_data)

    first_predict_time = (pd.Timestamp(data.date.tail(1).values[0]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")

    last_48h = np.expand_dims(df, axis=0)
    predictions = app.state.model4.predict(last_48h)
    predicted_temperatures = predictions[0]
    next_48h = [float(i) for i in predicted_temperatures]

    return {
        'first_predict_time': first_predict_time,
        'windspeed_prediction': next_48h
    }
