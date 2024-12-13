import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries
from darts.models import TransformerModel
from sklearn.metrics import mean_absolute_error
from darts.metrics import rmse, mae
from pytorch_lightning.callbacks import EarlyStopping
from darts.utils.timeseries_generation import datetime_attribute_timeseries

def train_model(input_chunk_length=24,
                            output_chunk_length=48,
                            dropout=0.0641,
                            d_model=32,
                            batch_size=16,
                            activation="relu",
                            n_epochs=5,
                            random_state=42,
                            model_name="best_model",
                            train_series=None,
                            val_series=None):
    """
    Train a TransformerModel for time series forecasting using Darts.

    Parameters:
        input_chunk_length (int): Input sequence length.
        output_chunk_length (int): Output prediction length.
        dropout (float): Dropout rate.
        d_model (int): Dimension of the model.
        batch_size (int): Batch size for training.
        activation (str): Activation function (e.g., "relu").
        n_epochs (int): Number of epochs for training.
        random_state (int): Random seed for reproducibility.
        model_name (str): Name of the model (for saving).
        train_series (TimeSeries): Training dataset (Darts TimeSeries).
        val_series (TimeSeries): Validation dataset (Darts TimeSeries).

    Returns:
        model: The trained TransformerModel.
    """
    if train_series is None or val_series is None:
        raise ValueError("Both train_series and val_series must be provided.")

    # Define the Transformer model
    model = TransformerModel(
                            input_chunk_length=input_chunk_length,
                            output_chunk_length=output_chunk_length,
                            dropout=dropout,
                            d_model=d_model,
                            batch_size=batch_size,
                            activation=activation,
                            n_epochs=n_epochs,
                            random_state=random_state,
                            model_name=model_name
                            )

    # Fit the model
    history = model.fit(series=train_series, val_series=val_series, verbose=True)

    print("✅  Model training complete.")
    return model, history
