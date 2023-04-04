import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, mean_squared_error as mse, mean_absolute_percentage_error as mape, mean_absolute_error as mae
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import regularizers
# from tpot import TPOTRegressor
from bayes_opt import BayesianOptimization
import absl.logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn import metrics
from statsmodels.tsa.seasonal import seasonal_decompose
from tempfile import TemporaryFile

import os
import time
import datetime
import re
import statistics
import random
import pandas as pd
import seaborn as sns
import keras_tuner as kt
from pandas_datareader import data as pdr
from datetime import date, timedelta
from copy import deepcopy
import yfinance as yf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

# global variables
timeframe = 9000
enable_pca = 0
standard_scaling = 0
win_size = 5
epochs = 10
batch_size = 128
lower = 0.7
upper = 0.8
future_window = 90


## we have a multi-index: let's collapse that so we have usable, single index column names
def collapse_columns(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.to_series().apply(lambda x: "__".join(x))
    return df


def set_verbosity():
    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(30)


# persistence model
def model_persistence(x):
    return x


def make_baselines(stocks, future):
    
    for stock in stocks:
        frame = load_frame(timeframe, stock).drop(['Adj. Close', 'Volume'], axis=1, errors='ignore')
        values = pd.DataFrame(pd.series.values)
        frame = pd.concat([values.shift(future_window), values], axis=1)
        frame.columns = ['t-1', 't+1']
        X = frame.values[:,0]
        y = frame.values[:,1]
        predictions = list()
        for x in X:
            yhat = model_persistence(x)
            predictions.append(yhat)



def make_persistence(frame, future_window):

    # Create lagged dataset
    values = pd.DataFrame(pd.series.values)
    frame = pd.concat([values.shift(future_window), values], axis=1)
    frame.columns = ['t-1', 't+1']

    # split into train and test sets

    X = frame.values[:,0]
    y = frame.values[:,1]

    # walk-forward validation
    predictions = list()
    for x in X:
        yhat = model_persistence(x)
        predictions.append(yhat)

    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = mean_squared_error(y, predictions,  squared=False)
    mape = mean_absolute_percentage_error(y, predictions)
    mae = mean_absolute_error(y, predictions)

    performance = pd.DataFrame(data={'metric': ['RMSE', 'MSE', 'MAPE', 'MAE', 'R2'], 'value': [rmse, mse, mape, mae, r2]})
    performance.to_csv('stocks_csvs/baseline_performance.csv')


def load_frame(days, stock):   
    end = date.today()
    start = end - timedelta(days=days)
    yf.pdr_override()

    data = yf.download(stock, start, end)


    data = data.resample('D').first() # ALWAYS resample before shifting so we don't get the wrong shift amount if there are missing rows/timestamps
    data = collapse_columns(data)
    data = data.dropna(how='any', axis='rows')

    assert data.isna().any().any() == False # Make sure there are no NaNs left

    return data


# There's alot of multi-collinearity in this data. Ideally, we should remove colinear features, as they will 
# Skew results
# After calling this function, simply remove these correlated columns from the dataset (Better to not have any of them)
# PCA is another option for removing it

def remove_correlation(data, threshold):
    correlated_cols = set()
    correlation_matrix = data.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_cols.add(colname)

    return correlated_cols


def remove_correlations_PCA(X):

    X_std = StandardScaler().fit_transform(X)
    pca = PCA().fit_transform(X_std)

    # Use these two indicators to see which variables are having the most effect on the system
    # Choose the high few impacts, and put them into the new PCA
    print(np.cumsum(pca.explained_variance_ratio))
    print(pca.explained_variance_ratio)

    # Change num_componenets to be the number of useful variables observed above
    pca = PCA(num_components=1).fit_transform(X_std)
    return pca


# This class is the final part of the preprocessing pipeline, and is used to remove columns that are unnecessary
class FeatureDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X.drop(['Volume', 'Adj Close'], axis=1, inplace=True, errors='ignore')
        if enable_pca:
            X = remove_correlations_PCA(X)
        return X


# This class is the final part of the preprocessing pipeline, and is used to remove columns that are unnecessary
class FeatureScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        if standard_scaling:
            open = StandardScaler(feature_range=(0, 1))
            high = StandardScaler(feature_range=(0, 1))
            low = StandardScaler(feature_range=(0, 1))
            close = StandardScaler(feature_range=(0, 1))

        else:
            open = MinMaxScaler(feature_range=(0, 1))
            high = MinMaxScaler(feature_range=(0, 1))
            low = MinMaxScaler(feature_range=(0, 1))
            close = MinMaxScaler(feature_range=(0, 1))

        X['Open'] = open.fit_transform(X[['Open']])
        X['High'] = high.fit_transform(X[['High']])
        X['Low'] = low.fit_transform(X[['Low']])
        X['Close'] = close.fit_transform(X[['Close']])
        
        return X, open, high, low, close


def create_dataset_tpot(data, future_window, win_size):
    
    np_data = data.to_numpy()
    X = []
    y = []
    future_X = []
    for i in range(len(np_data)-(win_size+future_window)):
        row = [r for r in np_data[i:i+win_size]]
        X.append(list(np.concatenate(row).flat))
        label = np_data[i+win_size+future_window][3]
        y.append(label)

    for i in range(len(np_data) - win_size):
        row = [r for r in np_data[i:i+win_size]]
        future_X.append(list(np.concatenate(row).flat))

    return np.array(X), np.array(y), np.array(future_X)


def create_dataset(data, future_window, win_size):
    
    np_data = data.to_numpy()
    X = []
    y = []
    future_X = []
    for i in range(len(np_data)-(win_size+future_window)):
        row = [r for r in np_data[i:i+win_size]]
        X.append(row)
        label = np_data[i+win_size+future_window]
        y.append(label)

    for i in range(len(np_data) - win_size):
        row = [r for r in np_data[i:i+win_size]]
        future_X.append(row)

    return np.array(X), np.array(y), np.array(future_X)


def kt_model(hp):

    pipe = Pipeline([('Dropper', FeatureDropper()), ('Scaler', FeatureScaler())])
    frame = load_frame(timeframe, 'AAPL')
    frame, open, high, low, close = pipe.fit_transform(frame)
    X, y, future_X = create_dataset(frame, future_window, win_size)

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_reg = hp.Float("reg", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_dropout = hp.Float("dropout", min_value=1e-3, max_value=0.5, sampling="linear")
    hp_neuron_pct = hp.Float('NeuronPct', min_value=1e-3, max_value=1.0, sampling='linear')
    hp_neuron_shrink = hp.Float('NeuronShrink', min_value=1e-3, max_value=1.0, sampling='linear')
    
    hp_l_layer_1 = hp.Int('l_layer_1', min_value=1, max_value=100, step=10)
    hp_l_layer_2 = hp.Int('l_layer_2', min_value=1, max_value=100, step=10)
    hp_max_neurons = hp.Int('neurons', min_value=10, max_value=200, step=10)

    neuron_count = int(hp_neuron_pct * hp_max_neurons)
    layers = 0

    model = Sequential()
    model.add(InputLayer((X.shape[1], X.shape[2])))
    model.add(LSTM(hp_l_layer_1, return_sequences=True, activity_regularizer=regularizers.l1(hp_reg)))
    model.add(Dropout(hp_dropout))
    model.add(LSTM(hp_l_layer_2, return_sequences=True, activity_regularizer=regularizers.l1(hp_reg)))
    model.add(Dropout(hp_dropout))
    model.add(Flatten())

    while neuron_count > 5 and layers < 5:

        model.add(Dense(units=neuron_count, activation=hp_activation))
        model.add(Dropout(hp_dropout))
        layers += 1
        neuron_count = int(neuron_count * hp_neuron_shrink)

    model.add(Dense(4, 'linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=hp_learning_rate), 
                metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])

    return model


def test_performance(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions,  squared=False)
    mape = mean_absolute_percentage_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    performance = pd.DataFrame(data={'metric': ['RMSE', 'MSE', 'MAPE', 'MAE', 'R2'], 'value': [rmse, mse, mape, mae, r2]})
    performance.to_csv('stocks_csvs/' + model_name '_performance.csv')

    
def train_model(stocks, future_window):

    model_name = 'model_' + future_window
    set_verbosity()
    pipe = Pipeline([('Dropper', FeatureDropper()), ('Scaler', FeatureScaler())])
    random.shuffle(stocks)
    cache = {}

    for stock in stocks:
        frame = load_frame(timeframe, stock)
        frame, open, high, low, close = pipe.fit_transform(frame)
        dates = frame.index[win_size:]
        X, y, future_X = create_dataset(frame, future_window, win_size)
        cache[stock] = [X, y,frame, open, high, low, close, future_X, dates]

    persistance_frame = load_frame(timeframe, stocks[int(len(stocks) * upper)])
    for i in range(int(len(stocks) * upper) + 1, len(stocks)):
        frame = load_frame(timeframe, stock)
        persistance_frame = np.concatenate([persistance_frame, load_frame(timeframe, stocks[i])], axis=0)
    persistance_frame = np.drop(['Open', 'Low', 'High', 'Volume', 'Adj. Close'], axis=1, errors='ignore')
    persistance_frame = persistance_frame.dropna()
    make_persistence(persistance_frame, future_window)

    X_train = cache.get(stocks[0])[0]
    y_train = cache.get(stocks[0])[1]
    X_val = cache.get(stocks[int(len(stocks) * lower)])[0]
    y_val = cache.get(stocks[int(len(stocks) * lower)])[1]
    X_test = cache.get(stocks[int(len(stocks) * upper)])[0]
    y_test = cache.get(stocks[int(len(stocks) * upper)])[1]

    for i in range(1, int(len(stocks) * lower)):
        X_train = np.concatenate([X_train, cache.get(stocks[i])[0]], axis=0)
        y_train = np.concatenate([y_train, cache.get(stocks[i])[1]], axis=0)

    for i in range(int(len(stocks) * lower) + 1, int(len(stocks) * upper)):
        X_val = np.concatenate([X_val, cache.get(stocks[i])[0]], axis=0)
        y_val = np.concatenate([y_val, cache.get(stocks[i])[1]], axis=0)

    for i in range(int(len(stocks) * upper) + 1, len(stocks)):
        X_test = np.concatenate([X_val, cache.get(stocks[i])[0]], axis=0)
        y_test = np.concatenate([y_val, cache.get(stocks[i])[1]], axis=0)

    tuner = kt.Hyperband(kt_model, objective='mean_squared_error', max_epochs=epochs, factor=3, directory='models/kt_dir', 
            project_name=model_name, overwrite=True)

    monitor = EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, verbose=0, mode='auto', 
                    restore_best_weights=True)

    tuner.search(cache.get(stocks[0])[0], cache.get(stocks[0])[1], verbose=1, epochs=epochs, batch_size=batch_size, callbacks=[monitor])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, verbose=1, epochs=epochs, validation_data=(X_val, y_val), callbacks=[monitor],
                    batch_size=batch_size)
    model.save('models/' + model_name)

    test_performance(model, X_test, y_test, model_name)


def generate_outputs(stocks):

    model_7 = load_model('models/model_7')
    model_30 = load_model('models/model_30')
    model_90 = load_model('models/model_90')
    models = {'model_7': model_7, 'model_30': model_30, 'model_90': model_90}

    pipe = Pipeline([('Dropper', FeatureDropper()), ('Scaler', FeatureScaler())])
    end = date.today()
    start = end - timedelta(days=timeframe)
    yf.pdr_override()

    for stock in stocks:

        data = yf.download(stock, start, end)
        data = data.resample('D').first() 
        data = data.dropna(how='any', axis='rows')
        frame, open, high, low, close = pipe.fit_transform(data)
        inverters = [open, high, low, close]
        X_7, y_7, future_X_7 = create_dataset(frame, 7, win_size)
        X_30, y_30, future_X_30 = create_dataset(frame, 30, win_size)
        X_90, y_90, future_X_90 = create_dataset(frame, 90, win_size)
        cache = {'data':data, 'model_7':future_X_7, 'model_30':future_X_30, 'model_90':future_X_90}

        prediction(models, cache, inverters, stock)


def prediction(models, cache, inverters, stock):

    cols = ['Open', 'High', 'Low', 'Close']

    for name, model in models.items():
        if name == 'model_7':
            window = 7
        elif name == 'model_30':
            window = 30
        else:
            window = 90

        dates = list(cache.get('data').index)
        last = dates[-1]
        for i in range(window):
                dates.append(last + pd.DateOffset(days=i+1))
        dates = dates[window+5:]
        output = pd.DataFrame(data={'Dates':dates})
        
        preds = model.predict(cache.get(name))
        for pred in range(4):
            output[cols[pred]] = inverters[pred].inverse_transform(preds[:,pred].reshape(-1, 1))
            
        output = output.set_index('Dates')
        output.to_csv('stocks_csvs/' + name + '_' + stock + '.csv')

# def train_model_tpot(stocks, future_window):

#     set_verbosity()
#     pipe = Pipeline([('Dropper', FeatureDropper()), ('Scaler', FeatureScaler())])
#     random.shuffle(stocks)
#     cache = {}

#     for stock in stocks:
#         frame = load_frame(timeframe, stock)
#         frame, open, high, low, close = pipe.fit_transform(frame)
#         dates = frame.index[win_size:]
#         X, y, future_X = create_dataset_tpot(frame, future_window, win_size)
#         cache[stock] = [X, y,frame, open, high, low, close, future_X, dates]

#     X_train = cache.get(stocks[0])[0]
#     y_train = cache.get(stocks[0])[1]
#     X_val = cache.get(stocks[int(len(stocks) * thresh)])[0]
#     y_val = cache.get(stocks[int(len(stocks) * thresh)])[1]

#     for i in range(1, int(len(stocks) * thresh)):
#         X_train = np.concatenate([X_train, cache.get(stocks[i])[0]], axis=0)
#         y_train = np.concatenate([y_train, cache.get(stocks[i])[1]], axis=0)

#     for i in range(int(len(stocks) * thresh), len(stocks)):
#         X_val = np.concatenate([X_val, cache.get(stocks[i])[0]], axis=0)
#         y_val = np.concatenate([y_val, cache.get(stocks[i])[1]], axis=0)

#     time = str(datetime.datetime.now())
#     time = re.sub("\s", "_", time)
#     time = re.sub(":", "_", time)
#     time = re.sub("-", "_", time)
#     time = re.sub("\.", "_", time)

#     teapot = TPOTRegressor(generations=5, population_size=20, cv=5, verbosity=2)
#     teapot.fit(cache.get(stocks[0])[0], cache.get(stocks[0])[1])
#     teapot.export('models/tpot_model_' + time + '.py')


if __name__=='__main__':
    stocks = ['NIO', 'SQ', 'F', 'PYPL', 'GE', 'INTC', 'BA', 'AMD', 'T', 'NFLX', 'VZ', 'DIS', 'CSCO', 'PFE', 'KO']
    real_stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'XOM', 'META', 'JNJ', 'JPM'] 
    train_model(stocks, 7)
    train_model(stocks, 30)
    train_model(stocks, 90)
    generate_outputs(real_stocks)
    make_baselines(real_stocks)
    # train_model_tpot(stocks, 7)
    # train_model(stocks, 30)
    # train_model_tpot(stocks, 30)
    # train_model(stocks, 90)
    # train_model_tpot(stocks, 90)