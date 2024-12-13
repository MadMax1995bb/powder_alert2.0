import pandas as pd
# $WIPE_BEGIN

# $WIPE_END
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from darts.models import TransformerModel
from datetime import datetime
from powderalert.ml_logic.data import fetch_prediction_data, clean_data
from powderalert.ml_logic.preprocessor import preprocess, define_X
from powderalert.ml_logic.params import *
from darts import TimeSeries

app = FastAPI()

# model_relative_path = 'models/my_model.pt'
# app.state.model = TransformerModel.load(model_relative_path)

max_model_relative_path = 'models/max_fullDS_all_features.keras'
app.state.model = TransformerModel.load(max_model_relative_path)

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END


@app.get("/predict_temp")
def predict():

    model = app.state.model
    assert model is not None

    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)
    X_pred = define_X(cleaned_data,target)
    X_processed = preprocess(X_pred)

    data_darts = data.copy()
    data_darts['date'] = data_darts['date'].dt.tz_localize(None)

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
    # $CHA_END


# @app.get("/predict_snowfall")
# def predict():

#     model = app.state.model
#     assert model is not None

#     data = fetch_prediction_data(lat,long)
#     cleaned_data = clean_data(data)
#     X_pred = define_X(cleaned_data,target)
#     X_processed = preprocess(X_pred)

#     data_darts = data.copy()
#     data_darts['date'] = data_darts['date'].dt.tz_localize(None)

#     snowfall_series = TimeSeries.from_dataframe(data_darts, 'date', 'snowfall')
#     feature_series = TimeSeries.from_dataframe(X_processed, value_cols=X_processed.columns)

#     print(X_processed)
#     y_pred = model.predict(series =snowfall_series,past_covariates=feature_series,n=48)

#     print(y_pred)
#     # ⚠️ fastapi only accepts simple Python data types as a return value
#     # among them dict, list, str, int, float, bool
#     # in order to be able to convert the api response to JSON
#     return {
#         'key': 'value'
#     }
#     # $CHA_END
