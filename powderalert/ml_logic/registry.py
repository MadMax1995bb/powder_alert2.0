import glob
import os

from colorama import Fore, Style
from tensorflow import keras
from darts.models import TransformerModel
from powderalert.ml_logic.params import MODEL_TARGET, LOCAL_REGISTRY_PATH
import torch

# def load_model():
#     model_snowfall_path = os.path.join(LOCAL_REGISTRY_PATH, 'models', 'snowfall')
#     test_path = '/Users/maxburger/code/MadMax1995bb/powder_alert2.0/models/snowfall/data.pkl'
#     model_temperature_path = os.path.join(LOCAL_REGISTRY_PATH, 'models', 'temperature')

#     model_snowfall = load_model() # change it to model_snowfall_path later
#     # model_temperature = load_model()

#     return {
#         'snowfall_model': model_snowfall,
#         'temperature_model': model_temperature}


def load_model_snowfall():
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> optional
    Return None (but do not Raise) if no model is found
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, 'models', 'snowfall')
        local_model_paths = glob.glob(f"{local_model_directory}/*.pt")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = max(local_model_paths, key=os.path.getmtime)

        latest_model = TransformerModel.load(most_recent_model_path_on_disk, map_location=torch.device('cpu'))

        print("✅ Model loaded from local disk")

        return latest_model

def load_model_temperature() -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> optional
    Return None (but do not Raise) if no model is found
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, 'models', 'temperature')
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = max(local_model_paths, key=os.path.getmtime)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

# LOCAL_REGISTRY_PATH = "/Users/maxburger/code/MadMax1995bb/powder_alert2.0"  # Replace with the actual registry path

# def load_model():
#     """
#     Load the latest snowfall_model and temperature_model from the local registry.
#     """
#     print(Fore.BLUE + "\nLoading latest models from local registry..." + Style.RESET_ALL)

#     # Define directories for snowfall and temperature models
#     snowfall_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", "snowfall")
#     temperature_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", "temperature")

#     # Helper function to load the latest model from a directory
#     def load_latest_model(model_directory):
#         model_paths = glob.glob(f"{model_directory}/*")
#         if not model_paths:
#             print(Fore.RED + f"❌ No models found in {model_directory}" + Style.RESET_ALL)
#             return None

#         most_recent_model_path = sorted(model_paths)[-1]
#         print(Fore.BLUE + f"Loading model from: {most_recent_model_path}" + Style.RESET_ALL)

#         model = keras.models.load_model(most_recent_model_path)
#         print(Fore.GREEN + f"✅ Model loaded from: {most_recent_model_path}" + Style.RESET_ALL)
#         return model

#     # Load the latest models
#     model_snowfall = load_latest_model(snowfall_model_directory)
#     model_temperature = load_latest_model(temperature_model_directory)

#     # Return both models as a dictionary
#     return {
#         "snowfall_model": model_snowfall,
#         "temperature_model": model_temperature
#     }
