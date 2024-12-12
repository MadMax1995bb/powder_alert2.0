import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
# from ml_logic.params import *

def label_encode_columns(cat_data, cat_columns):
    """Apply LabelEncoder to specified categorical columns."""
    for col in cat_columns:
        cat_data[col] = LabelEncoder().fit_transform(cat_data[col])
    return cat_data

def define_X(df: pd.DataFrame, target:list):
    X = df.drop(columns=target, axis=1)
    return X

def preprocess(df):
    """
    Process the input data X by applying label encoding to categorical columns
    and standard scaling to numerical columns.

    Parameters:
        X (pd.DataFrame): Input dataframe to process.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    # Define categorical and numerical columns
    cat_columns = ['weather_code']
    num_columns = X.drop(columns=cat_columns).select_dtypes(include=['float64']).columns.tolist()

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
    processed_X = preprocess_pipe.fit_transform(X)

    # Convert to DataFrame to maintain column names
    processed_columns = get_label_encoded_column_names(cat_columns) + num_columns + list(X.columns.difference(cat_columns + num_columns))

    print("✅ X_processed, with shape", processed_X.shape)

    # print(processed_columns)

    return pd.DataFrame(processed_X, columns=processed_columns)



#Path to raw data for 1) Open weather historical data 2) Open-meteo API weather
filepath_hist_api = '/Users/torstenwrigley/code/MadMax1995bb/powder_alert2.0/raw_data/openmeteo_api_zentralstation.csv'
#CSV -> DF for api data
csv_file = filepath_hist_api
df = pd.read_csv(csv_file)



X = define_X(df,['snowfall'])

preprocess(X)
