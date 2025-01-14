import os

lat = 47.26580883196723
long = 11.84457426992035

start_date_hist = "2009-01-01"
end_date_hist = "2023-12-31"

prediction_length = 48 #

target1 = ['temperature_2m']
target2 = ['snow_depth']
target3 = ['wind_speed_10m']

MODEL_TARGET = 'local'
LOCAL_REGISTRY_PATH = os.path.join(os.getcwd()) # Change it without home direcot


##################  VARIABLES  ##################
# DATA_SIZE = os.environ.get("DATA_SIZE")
# CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
# MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
# GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
# INSTANCE = os.environ.get("INSTANCE")
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
# MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
# PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
# EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
# GAR_IMAGE = os.environ.get("GAR_IMAGE")
# GAR_MEMORY = os.environ.get("GAR_MEMORY")
