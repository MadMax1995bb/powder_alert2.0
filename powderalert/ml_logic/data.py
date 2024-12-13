import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from google.cloud import bigquery
from colorama import Fore, Style
from powderalert.ml_logic.params import *


def fetch_weather_data(latitude, longitude, start_date, end_date, variables=None, models="best_match"):
    """
    Fetch hourly weather data from the Open-Meteo Archive API.

    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        variables (list): List of weather variables to fetch (default: all variables).
        models (str): Weather model (default: "best_match").

    Returns:
        pd.DataFrame: Hourly weather data as a Pandas DataFrame.
    """
    # Set up the Open-Meteo API client with cache and retry
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Default variables if not provided
    if variables is None:
        variables = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
            "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
            "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration",
            "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
            "wind_direction_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm", "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm", "sunshine_duration"
        ]

    # API parameters
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variables,
        "models": models
    }

    # Fetch the weather data
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]  # Process the first location

    # Process hourly data
    hourly = response.Hourly()
    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    # Assign hourly variables
    for i, variable in enumerate(variables):
        hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()

    # Create a DataFrame
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    print(f"✅ Model training data fetched")
    return hourly_dataframe

def fetch_prediction_data(latitude, longitude, variables=None, models="best_match"):
    """
    Fetch prediction data from the Open-Meteo Archive API (only forecast data available in the last 48hrs).

    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        variables (list): List of weather variables to fetch (default: all variables).
        models (str): Weather model (default: "best_match").

    Returns:
        pd.DataFrame: Hourly weather data as a Pandas DataFrame.
    """
    # Set up the Open-Meteo API client with cache and retry
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Default variables if not provided
    if variables is None:
        variables = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
            "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
            "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration",
            "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
            "wind_direction_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm", "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm", "sunshine_duration"
        ]

    # API parameters
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": prediction_length, #in days
        "forecast_days": 0,
        "hourly": variables,
        "models": models
    }

    # Fetch the weather data
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]  # Process the first location

    # Process hourly data
    hourly = response.Hourly()
    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    # Assign hourly variables
    for i, variable in enumerate(variables):
        hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()

    # Create a DataFrame
    prediction_dataframe = pd.DataFrame(data=hourly_data)
    print(f"✅ Prediction data fetched")
    return prediction_dataframe

def clean_data(df):
    df = df.set_index(['date'])
    df = df.drop_duplicates()
    print(f"✅ Data cleaned")
    return df

def load_data_to_bq(data: pd.DataFrame, gcp_project:str, bq_dataset:str,table: str,truncate: bool) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

train_df = fetch_weather_data(lat,long,start_date_hist,end_date_hist)
