from fastapi import FastAPI
from powderalert.ml_logic.data import fetch_prediction_data, clean_data
from powderalert.ml_logic.preprocessor import preprocess_pred_temperature, preprocess_pred_snowdepth, preprocess_pred_windspeed
from powderalert.ml_logic.params import *
from powderalert.ml_logic.registry import load_model_temperature, load_model_snowdepth, load_model_windspeed
from darts import TimeSeries
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle
import dill


app = FastAPI()

###################################################################################

def label_encode_columns(cat_data, cat_columns):
    """Apply LabelEncoder to specified categorical columns."""
    for col in cat_columns:
        cat_data[col] = LabelEncoder().fit_transform(cat_data[col])
    return cat_data

def apply_label_encoding(data, cat_columns):
    return label_encode_columns(data, cat_columns)

###################################################################################

app.state.model1 = load_model_temperature()
app.state.model2 = load_model_snowdepth()
app.state.model3 = load_model_windspeed()

###################################################################################

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.get("/predict_temperature")
def predict(lat: float, long: float):

    data = fetch_prediction_data(lat,long)

    first_predict_time = (pd.Timestamp(data.date.tail(1).values[0]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")

    data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.tz_localize(None)

    X = data
    X['date'] = pd.to_datetime(X['date'])
    X.set_index('date', inplace=True)
    y = data.temperature_2m

    with open('models/preprocessor_temperature.dill', 'rb') as file:
        preprocessor = dill.load(file)

    X_processed = preprocess_pred_temperature(X, preprocessor=preprocessor)
    df = pd.concat([y.reset_index(drop=True), X_processed], axis=1)

    last_48h = np.expand_dims(df, axis=0)
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

    X = cleaned_data
    y = cleaned_data[target2]
    y = y.reset_index()

    with open('models/pipeline.dill', 'rb') as file:
        preprocessor = dill.load(file)

    X_processed = preprocess_pred_snowdepth(X, preprocessor=preprocessor)
    df = y.join(X_processed)
    df = df.drop(columns='date')

    first_predict_time = (pd.Timestamp(data.date.tail(1).values[0]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")

    X_pred_columns = X_processed.columns.tolist()

    snowdepth_series = TimeSeries.from_dataframe(df, value_cols=['snow_depth']).astype("float32")
    feature_series = TimeSeries.from_dataframe(df, value_cols=X_pred_columns).astype("float32")

    y_pred = app.state.model2.predict(series=snowdepth_series, past_covariates=feature_series, n=48).values().flatten().tolist()

    return {
        'first_predict_time': first_predict_time,
        'snowdepth_prediction': y_pred
    }

###################################################################################

@app.get("/predict_windspeed")
def predict(lat: float, long: float):
    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)

    # current_time = dt.now()
    # earliest_time = current_time - timedelta(hours=48)
    # df['datetime'] = pd.date_range(start=dt.now() - timedelta(hours=len(df)), periods=len(df), freq='H')
    # df = df[df['datetime'].between(earliest_time, current_time)]
    # df = df.drop(columns = 'datetime')

    X = cleaned_data.drop(columns = target3)
    y = cleaned_data[target3]
    y = y.reset_index()

    with open('models/pipeline.dill', 'rb') as file:
        preprocessor = dill.load(file)

    X_processed = preprocess_pred_windspeed(X, preprocessor=preprocessor)
    df = pd.concat([y.reset_index(drop=True), X_processed], axis=1)
    df = df.drop(columns='date')

    first_predict_time = (pd.Timestamp(data.date.tail(1).values[0]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")

    last_48h = np.expand_dims(df, axis=0)
    predictions = app.state.model3.predict(last_48h)
    predicted_temperatures = predictions[0]
    next_48h = [float(i) for i in predicted_temperatures]

    return {
        'first_predict_time': first_predict_time,
        'windspeed_prediction': next_48h
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
