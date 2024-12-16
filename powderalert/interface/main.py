import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from ml_logic.params import LOCAL_DATA_PATH
from ml_logic.data import *
from ml_logic.registry import *
from ml_logic.model import load_model, preprocess_features
from ml_logic.model2 import load_model, preprocess_features



def preprocess(table_name,min_date:str = start_date_hist, max_date:str = end_date_hist) -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    # Query raw data from BigQuery using `get_data_with_cache`
    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT *
        FROM `{GCP_PROJECT}`.{BQ_DATASET}.{table_name}
        WHERE date BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY date
    """

    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}.csv")  # Removed DATA_SIZE
    data_query = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    # Process data
    data_clean = clean_data(data_query)

    X = data_clean.drop("fare_amount", axis=1)
    y = data_clean[["fare_amount"]]

    X_processed = preprocess_features(X)

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()
    data_processed_with_timestamp = pd.DataFrame(np.concatenate((
        data_clean[["pickup_datetime"]],
        X_processed,
        y,
    ), axis=1))

    load_data_to_bq(
        data_processed_with_timestamp,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_{DATA_SIZE}',
        truncate=True
    )

    print("✅ preprocess() done \n")


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
            pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1],
        ))

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")

    return y_pred
