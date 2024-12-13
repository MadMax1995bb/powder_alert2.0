import numpy as np
from colorama import Fore, Style
from typing import Tuple
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import tensorflow as tf
from ml_logic.preprocessor import preprocess, label_encode_columns
from ml_logic.data import fetch_weather_data, clean_data
from ml_logic.params import lat, long, start_date_hist, end_date_hist_date_hist
from typing import Dict, List, Tuple, Sequence
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.layers import Normalization, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

################################################################################################################

def create_folds_sequences(label_encode_columns):
    FOLD_LENGTH = label_encode_columns.shape[0] # each fold will have the whole dataset --> only 1 fold in this model
    FOLD_STRIDE = 1 # sliding only on hour
    # Temporal Train-Test split
    TRAIN_TEST_RATIO = 0.66
    # Inputs
    N_FEATURES = df.shape[1]
    INPUT_LENGTH = 48 # 48 hours input = forecast the upcooming 48 hours
    # Outputs
    TARGET = ['temperature_2m']
    TARGET_COLUMN_IDX = 1 # 'temperature_2m' corresponds to the second column of the df
    N_TARGETS = 1
    OUTPUT_LENGTH = N_TARGETS*48 # - Predicting one target, the temperature - for two days with predictions every hour
    # Additional parameters
    HORIZON = 1 # - We are predicting next two days

    def get_folds(
        df: pd.DataFrame,
        fold_length: int,
        fold_stride: int) -> List[pd.DataFrame]:
        """
        This function slides through the Time Series dataframe of shape (n_timesteps, n_features) to create folds
        - of equal `fold_length`
        - using `fold_stride` between each fold
        Returns a list of folds, each as a DataFrame
        """
        folds = []
        for idx in range(0, len(df), fold_stride):
            # Exits the loop as soon as the last fold index would exceed the last index
            if (idx + fold_length) > len(df):
                break
            fold = df.iloc[idx:idx + fold_length, :]
            folds.append(fold)
        return folds

    folds = get_folds(df, FOLD_LENGTH, FOLD_STRIDE)
    fold = folds[0]

    def train_test_split(fold: pd.DataFrame,
        train_test_ratio: float,
        input_length: int,
        horizon: int) -> Tuple[pd.DataFrame]:
        '''
        Returns a train dataframe and a test dataframe (fold_train, fold_test)
        from which one can sample (X,y) sequences.
        df_train should contain all the timesteps until round(train_test_ratio * len(fold))
        '''
        last_train_idx = round(train_test_ratio * len(fold))
        fold_train = fold.iloc[0:last_train_idx, :]

        first_test_idx = last_train_idx - input_length
        fold_test = fold.iloc[first_test_idx:, :]

        return (fold_train, fold_test)

    fold_train, fold_test = train_test_split(fold,
                                            TRAIN_TEST_RATIO,
                                            INPUT_LENGTH,
                                            HORIZON)

    def get_Xi_yi(first_index: int,
                fold: pd.DataFrame,
                horizon: int,
                input_length: int,
                output_length: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        - extracts one sequence from a fold
        - returns a pair (Xi, yi) with:
            * len(Xi) = `input_length` and Xi starting at first_index
            * len(yi) = `output_length`
            * last_Xi and first_yi separated by the gap = horizon -1
        '''
        Xi_start = first_index
        Xi_last = Xi_start + input_length
        yi_start = Xi_last + horizon - 1
        yi_last = yi_start + output_length

        Xi = fold[Xi_start:Xi_last]
        yi = fold[yi_start:yi_last][TARGET]

        return (Xi, yi)

    def get_X_y(fold: pd.DataFrame,
                horizon: int,
                input_length: int,
                output_length: int,
                stride: int,
                shuffle=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        - Uses `data`, a 2D-array with axis=0 for timesteps, and axis=1 for (targets+covariates columns)
        - Returns a Tuple (X,y) of two ndarrays :
            * X.shape = (n_samples, input_length, n_covariates)
            * y.shape =
                (n_samples, output_length, n_targets) if all 3-dimensions are of size > 1
                (n_samples, output_length) if n_targets == 1
                (n_samples, n_targets) if output_length == 1
                (n_samples, ) if both n_targets and lenghts == 1
        - You can shuffle the pairs (Xi,yi) of your fold
        """

        X = []
        y = []

        for i in range(0, len(fold), stride):
            ## Extracting a sequence starting at index_i
            Xi, yi = get_Xi_yi(first_index=i,
                                fold=fold,
                                horizon=horizon,
                                input_length=input_length,
                                output_length=output_length)
        ## Exits loop as soon as we reach the end of the dataset
            if len(yi) < output_length:
                break
            X.append(Xi)
            y.append(yi)

        X = np.array(X)
        y = np.array(y)
        y = np.squeeze(y)

        if shuffle:
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

        return X, y

    X_train, y_train = get_X_y(fold=fold_train,
                            horizon=HORIZON,
                            input_length=INPUT_LENGTH,
                            output_length=OUTPUT_LENGTH,
                            stride=1)
    X_test, y_test = get_X_y(fold=fold_test,
                            horizon=HORIZON,
                            input_length=INPUT_LENGTH,
                            output_length=OUTPUT_LENGTH,
                            stride=1)

    print("Shapes for the training set:")
    print(f"X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
    print("Shapes for the test set:")
    print(f"X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

################################################################################################################

def only_use_relevant_features(df): # optional, depends if we rund the model on all variables or not
    correlation_matrix = df.corr()
    temperature_corr = correlation_matrix['temperature_2m']
    high_corr_features = temperature_corr[abs(temperature_corr) > 0.55]
    relevant_features = high_corr_features.index.tolist()
    features = [col for col in relevant_features if col in df.columns]
    df = df[features]

################################################################################################################

def initialize_base_model(X_train: tuple) -> Model:
    reg_l2 = regularizers.L2(0.1)

    # 1 - RNN architecture
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))

    # Recurrent Layer
    model.add(layers.LSTM(units=32, activation='tanh',return_sequences=True,
                        #   recurrent_dropout=0.3,dropout=0.3
                        ))

    # Hidden Dense Layer that we are regularizing
    model.add(layers.Dense(16, activation="relu",
                        #    kernel_regularizer = reg_l2
                        ))
    # model.add(layers.Dropout(rate=0.3))

    # Predictive Dense Layer
    model.add(layers.Dense(1, activation='linear'))

    # 2 - Compiler
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='mse', optimizer=optimizer, metrics=["mae"])

    return model

# 1) best_hps
# 2) load model
# assign a dictionary (kwargs, kargs)

################################################################################################################

def fit_model(model: tf.keras.Model, verbose=1, X_train, y_train) -> Tuple[tf.keras.Model, dict]:

    es = EarlyStopping(
        monitor="val_mae",
        patience=10,
        mode="min",
        restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_mae',
        factor=0.1,
        patience=5,
        min_lr=1e-6)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.3,
        shuffle=False,
        batch_size=64,
        epochs=100,
        callbacks=[es, reduce_lr],
        verbose=verbose)

    return model, history

################################################################################################################

def evaluate_model(
        model: Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """
    res = model.evaluate(
        X_test,
        y_test,
        batch_size=batch_size,
        verbose=1,
        return_dict=True)

    loss = res["mse"]
    mae = res["mae"]

    return loss, mae
