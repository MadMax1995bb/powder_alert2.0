from fastapi import FastAPI
from powderalert.ml_logic.data import fetch_prediction_data, clean_data, time_features
from powderalert.ml_logic.preprocessor import preprocess, define_X
from powderalert.ml_logic.params import *
from powderalert.ml_logic.registry import load_model_snowfall, load_model_temperature
from darts import TimeSeries

app = FastAPI()

# model_relative_path = 'models/my_model.pt'
# app.state.model = TransformerModel.load(model_relative_path)
# app.state.model = load_model_snowfall(target1)

# max_model_relative_path = 'models/max_fullDS_all_features.keras'
# app.state.model2 = TransformerModel.load(max_model_relative_path)
# app.state.model = load_model_snowfall(target2)

app.state.model1 = load_model_snowfall()
app.state.model2 = load_model_temperature()

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.get("/predict_snowfall")
def predict(lat: float, long: float):

    # breakpoint()
    data = fetch_prediction_data(lat,long)
    cleaned_data = clean_data(data)
    cleaned_data['date'] = cleaned_data.index # might be updated later
    data_engineered_cleaned = time_features(cleaned_data)
    # X_pred = define_X(data_engineered_cleaned, target1)
    X_processed = preprocess(data_engineered_cleaned)

    snowfall_series = TimeSeries.from_dataframe(X_processed, value_cols=['snowfall'])
    feature_series = TimeSeries.from_dataframe(X_processed, value_cols=X_processed.columns)

    y_pred = app.state.model1.predict(series=snowfall_series, past_covariates=feature_series, n=48)

    print(y_pred)
    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    # uvicorn simple:app --reload
    return {
        'prediction': y_pred
    }

@app.get("/predict_temperature")
def predict():

    data = fetch_prediction_data(lat,long)
    data_engineered_cleaned = time_features(data)
    cleaned_data = clean_data(data_engineered_cleaned)
    X_pred = define_X(cleaned_data, target2)
    X_processed = preprocess(X_pred)

    y_pred = app.state.model2.predict(X_processed)

    print(y_pred)
    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return {
        'prediction': y_pred
    }
