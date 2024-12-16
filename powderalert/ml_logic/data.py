import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from requests.exceptions import RequestException
from google.cloud import bigquery
from colorama import Fore, Style
from powderalert.ml_logic.params import *
from powderalert.ml_logic.preprocessor import define_X, preprocess
from datetime import datetime as dt


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
    Fetch prediction data from the Open-Meteo API, processing hourly weather data for the last `prediction_length` days.

    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        prediction_length (int): Number of past days to fetch data for.
        models (str): Weather model (default: "best_match").

    Returns:
        pd.DataFrame: Hourly weather data as a Pandas DataFrame or None if the API request fails.
    """
    # Set up the Open-Meteo API client with cache and retry
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Weather variables
    variables = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
        "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl",
        "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
        "cloud_cover_high", "et0_fao_evapotranspiration", "vapour_pressure_deficit",
        "wind_speed_10m", "wind_speed_120m", "wind_direction_10m", "wind_direction_120m",
        "wind_gusts_10m", "sunshine_duration"
    ]

    # API parameters
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": prediction_length,  # Number of past days to fetch
        "forecast_days": prediction_length,  # Only fetch past data
        "hourly": variables,
        "models": models
    }

    try:
        # Fetch the weather data
        responses = openmeteo.weather_api(url, params=params)

        # Ensure the response is valid
        if not responses or len(responses) == 0:
            raise ValueError("No response received from the Open-Meteo API.")

        # Process first location
        response = responses[0]
        print(f"Coordinates: {response.Latitude()}°N, {response.Longitude()}°E")
        print(f"Elevation: {response.Elevation()} m asl")
        print(f"Timezone: {response.Timezone()} ({response.TimezoneAbbreviation()})")
        print(f"UTC Offset: {response.UtcOffsetSeconds()} seconds")

        # Process hourly data
        hourly = response.Hourly()
        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}

        # Assign hourly variables
        hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
        hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
        hourly_data["dew_point_2m"] = hourly.Variables(2).ValuesAsNumpy()
        hourly_data["precipitation"] = hourly.Variables(3).ValuesAsNumpy()
        hourly_data["rain"] = hourly.Variables(4).ValuesAsNumpy()
        hourly_data["snowfall"] = hourly.Variables(5).ValuesAsNumpy()
        hourly_data["snow_depth"] = hourly.Variables(6).ValuesAsNumpy()
        hourly_data["weather_code"] = hourly.Variables(7).ValuesAsNumpy()
        hourly_data["pressure_msl"] = hourly.Variables(8).ValuesAsNumpy()
        hourly_data["surface_pressure"] = hourly.Variables(9).ValuesAsNumpy()
        hourly_data["cloud_cover"] = hourly.Variables(10).ValuesAsNumpy()
        hourly_data["cloud_cover_low"] = hourly.Variables(11).ValuesAsNumpy()
        hourly_data["cloud_cover_mid"] = hourly.Variables(12).ValuesAsNumpy()
        hourly_data["cloud_cover_high"] = hourly.Variables(13).ValuesAsNumpy()
        hourly_data["et0_fao_evapotranspiration"] = hourly.Variables(14).ValuesAsNumpy()
        hourly_data["vapour_pressure_deficit"] = hourly.Variables(15).ValuesAsNumpy()
        hourly_data["wind_speed_10m"] = hourly.Variables(16).ValuesAsNumpy()
        hourly_data["wind_speed_100m"] = hourly.Variables(17).ValuesAsNumpy()  # Renamed for train dataset
        hourly_data["wind_direction_10m"] = hourly.Variables(18).ValuesAsNumpy()
        hourly_data["wind_direction_100m"] = hourly.Variables(19).ValuesAsNumpy()  # Renamed for train dataset
        hourly_data["wind_gusts_10m"] = hourly.Variables(20).ValuesAsNumpy()
        hourly_data["sunshine_duration"] = hourly.Variables(21).ValuesAsNumpy()

        # Create a DataFrame
        hourly_dataframe = pd.DataFrame(data=hourly_data)
        print(f"✅ Prediction data fetched successfully")
        print(f"Shape: {hourly_dataframe.shape}")
        print(hourly_dataframe.dtypes)
        return hourly_dataframe

    except RequestException as req_err:
        print(f"❌ API request failed due to a network issue: {req_err}")
    except ValueError as val_err:
        print(f"❌ API response error: {val_err}")
    except Exception as ex:
        print(f"❌ An unexpected error occurred: {ex}")

    # Return None if the data could not be fetched
    return None

def clean_data(df):
    df['date'] = df['date'].dt.tz_localize(None)
    df = df.set_index(['date'])
    df = df.drop_duplicates()
    print(f"✅ Data cleaned")
    return df

def time_features(df: pd.DataFrame):
    df = clean_data(df)
    df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['date'].dt.month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['date'].dt.month - 1) / 12)

    print(f"✅ time features engineered and saved into DataFrame")
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

def save_dataframe_to_csv(dataframe, file_path, index=True):
    """
    Save a Pandas DataFrame to a CSV file.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to save.
        file_path (str): The full file path including the file name and .csv extension.
        index (bool): Whether to include the DataFrame's index in the CSV file. Default is False.

    Returns:
        None
    """
    try:
        dataframe.to_csv(file_path, index=index)
        print(f"✅ DataFrame successfully saved to '{file_path}'")
    except Exception as e:
        print(f"❌ Failed to save DataFrame to CSV: {e}")
