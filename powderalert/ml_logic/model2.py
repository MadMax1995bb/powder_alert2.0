import numpy as np
from typing import Tuple
from tensorflow import keras
from keras import Model, layers, regularizers
import pandas as pd
import numpy as np
import tensorflow as tf
from ml_logic.params import target2
from typing import List, Tuple
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel

################################################################################################################

def only_use_relevant_features(df): # optional, depends if we run the model on all variables or not
    correlation_matrix = df.corr()
    temperature_corr = correlation_matrix[target2]
    high_corr_features = temperature_corr[abs(temperature_corr) > 0.55]
    relevant_features = high_corr_features.index.tolist()
    features = [col for col in relevant_features if col in df.columns]
    df = df[features]

################################################################################################################

def create_folds_sequences(df):
    FOLD_LENGTH = df.shape[0] # each fold will have the whole dataset --> only 1 fold in this model
    FOLD_STRIDE = 1 # sliding only on hour
    # Temporal Train-Test split
    TRAIN_TEST_RATIO = 2/3
    # Inputs
    N_FEATURES = df.shape[1]
    INPUT_LENGTH = 48 # 48 hours input = forecast the upcooming 48 hours
    # Outputs
    TARGET = target2
    N_TARGETS = 1
    OUTPUT_LENGTH = N_TARGETS*48 # - Predicting one target, the temperature - for two days with predictions every hour
    # Additional parameters
    HORIZON = 1 # - We are predicting next two days

    def get_fold(
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
        fold = folds[0]
        return fold

    get_fold(df, FOLD_LENGTH, FOLD_STRIDE)

    def train_test_split(fold: pd.DataFrame,
        train_test_ratio: float,
        input_length: int) -> Tuple[pd.DataFrame]:
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

    fold_train, fold_test = train_test_split(get_fold(df, FOLD_LENGTH, FOLD_STRIDE),
                                            TRAIN_TEST_RATIO,
                                            INPUT_LENGTH)

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

    return X_train, y_train, X_test, y_test

################################################################################################################

def initialize_best_model(X_train: tuple) -> Model:
    class LSTMModel(HyperModel):
        def build(self, hp):
            reg_l2 = regularizers.L2(hp.Float('l2_reg', min_value=0.001, max_value=0.1, step=0.001))

            #========================================================================================

            model = models.Sequential()

            # Input Layer
            model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))

            # Recurrent Layer with tunable units and dropout
            model.add(layers.LSTM(
                units=hp.Int('units', min_value=16, max_value=128, step=16),
                activation='tanh',
                return_sequences=True,
                recurrent_dropout=hp.Float('recurrent_dropout', min_value=0.2, max_value=0.5, step=0.05),
                dropout=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.05)
            ))
            model.add(layers.LSTM(
                units=hp.Int('units', min_value=16, max_value=128, step=16),
                activation='tanh',
                return_sequences=True,
                recurrent_dropout=hp.Float('recurrent_dropout', min_value=0.2, max_value=0.5, step=0.05),
                dropout=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.05)
            ))

            # Hidden Dense Layer with tunable regularization
            model.add(layers.Dense(
                units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
                activation="relu",
                kernel_regularizer=reg_l2
            ))
            model.add(layers.Dropout(rate=hp.Float('dense_dropout', min_value=0.2, max_value=0.5, step=0.05)))

            # Output Layer
            model.add(layers.Dense(1, activation='linear'))

            # Compile the model
            model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')),
                metrics=["mae"]
            )

            return model

################################################################################################################

def fit_and_build_best_model(X_train, y_train, RandomSearch, LSTMModel, X_test, y_test):
    tuner = RandomSearch(
        LSTMModel(),
        objective='val_mae',
        max_trials=10,
        executions_per_trial=1,
        directory='models',
        project_name='temp_hyperparameters')

    tuner.search(
        X_train,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.3,  # Use a validation split
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=2)])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters: {best_hps.values}")

    # Build the best model with those hyperparameters
    best_model = tuner.hypermodel.build(best_hps)

    # Train the best model
    history = best_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=15)])

    test_results = best_model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_results[1]} Celsius degrees")

    return best_hps, best_model

################################################################################################################

def evaluate_model(
        best_model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """
    res = best_model.evaluate(
        X_test,
        y_test,
        batch_size=batch_size,
        verbose=1,
        return_dict=True)

    loss = res["mse"]
    mae = res["mae"]

    return loss, mae

################################################################################################################
################################################################################################################

def safe_best_model(best_model):
    models_folder = ('/Users/maxburger/code/MadMax1995bb/powder_alert2.0/models')
    save_as_keras = (models_folder, 'best_model_temp.keras')

    best_model.save(save_as_keras)

    print(f"✅ Temperature model safed in the local models folder!")

################################################################################################################

def load_best_model(save_as_keras):
    loaded_best_model = tf.keras.models.load_model(save_as_keras)

    print(f"✅ Temperature model loaded successfully!")
    return loaded_best_model
