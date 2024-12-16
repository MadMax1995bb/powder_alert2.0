import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

# from ml_logic.params import *

def label_encode_columns(cat_data, cat_columns):
    """Apply LabelEncoder to specified categorical columns."""
    for col in cat_columns:
        cat_data[col] = LabelEncoder().fit_transform(cat_data[col])
    return cat_data

def preprocess(data):
    """
    Process the input data X by adding cylical feature, applying label encoding to categorical columns
    and standard scaling to numerical columns.
    Parameters:
        X (pd.DataFrame): Input dataframe to process.
    Returns:
        pd.DataFrame: Processed dataframe.
    """

    # Check if the DataFrame index is a datetime-like index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a datetime-like index (e.g., pd.DatetimeIndex). "
                        "Ensure your DataFrame has a datetime index using data.set_index().")

    #Add cyclical features
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)

    data['day_of_week_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)

    data['month_sin'] = np.sin(2 * np.pi * (data.index.month - 1) / 12)
    data['month_cos'] = np.cos(2 * np.pi * (data.index.month - 1) / 12)

    # Define categorical and numerical columns
    cat_columns = ['weather_code']
    num_columns = data.drop(columns=cat_columns).select_dtypes(include=['float64']).columns.tolist()

    # Helper function to generate column names for label-encoded columns
    def get_label_encoded_column_names(cat_columns):
        return [f"{col}_encoded" for col in cat_columns]

    # Define the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            # Apply FunctionTransformer to label encode categorical columns
            ('label_encoder', FunctionTransformer(lambda data: label_encode_columns(data, cat_columns), validate=False), cat_columns),

            # Apply StandardScaler to numerical columns
            ('standard_scaler', StandardScaler(), num_columns)
        ],
        remainder='passthrough'  # Keeps other columns as is
    )

    # Create the pipeline
    preprocess_pipe = make_pipeline(preprocessor)

    # Process and return the transformed data
    processed_data = preprocess_pipe.fit_transform(data)

    # Convert to DataFrame to maintain column names
    processed_columns = get_label_encoded_column_names(cat_columns) + num_columns + list(data.columns.difference(cat_columns + num_columns))

    print("✅ Processed data, with shape", processed_data.shape)

    # print(processed_columns)

    return pd.DataFrame(processed_data, columns=processed_columns)
