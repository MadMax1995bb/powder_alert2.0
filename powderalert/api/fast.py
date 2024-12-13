import pandas as pd
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from darts.models import TransformerModel
from datetime import datetime
from powderalert.ml_logic.data import fetch_prediction_data, clean_data, time_features
from powderalert.ml_logic.preprocessor import preprocess, define_X
from powderalert.ml_logic.params import *
from powderalert.ml_logic import load_best_model

from darts import TimeSeries

app = FastAPI()

model_relative_path = 'models/my_model.pt'
app.state.model = TransformerModel.load(model_relative_path)

max_model_relative_path = 'models/max_fullDS_all_features.keras'
app.state.model2 = TransformerModel.load(max_model_relative_path)

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.get("/predict_snowfall")
def predict():

    model = app.state.model
    assert model is not None

    data = fetch_prediction_data(lat,long)
    data_engineered_cleaned = time_features(data)
    cleaned_data = clean_data(data_engineered_cleaned)
    X_pred = define_X(cleaned_data,target1)
    X_processed = preprocess(X_pred)

    data_darts = data.copy()

    snowfall_series = TimeSeries.from_dataframe(data_darts, 'date', 'snowfall')
    feature_series = TimeSeries.from_dataframe(X_processed, value_cols=X_processed.columns)

    print(X_processed)
    y_pred = model.predict(series =snowfall_series,past_covariates=feature_series,n=48)

    print(y_pred)
    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return {
        'key': 'value'
    }

@app.get("/predict_temp")
def predict():

    model = app.state.model2
    assert model is not None

    data = fetch_prediction_data(lat,long)
    data_engineered_cleaned = time_features(data)
    X_processed = preprocess(data_engineered_cleaned)

    df_topredict = X_processed.copy()


    best_model = load_best_model(model)
    y_pred = best_model.predict(df_topredict)

    print(y_pred)
    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return {
        'key': 'value'
    }
