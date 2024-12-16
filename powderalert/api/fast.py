from fastapi import FastAPI
from powderalert.ml_logic.data import fetch_prediction_data, clean_data
from powderalert.ml_logic.preprocessor import preprocess
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
    X_processed = preprocess(cleaned_data)

    y_pred = app.state.model2.predict(X_processed)

    print(y_pred)
    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return {
        'prediction': y_pred
    }
