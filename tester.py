import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash  # pip install dash (version 2.0.0 or higher)
from dash import html
import dash_bootstrap_components as dbc
import yfinance as yf
from datetime import date, timedelta

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, mean_squared_error as mse, mean_absolute_percentage_error as mape, mean_absolute_error as mae
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import regularizers

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn import metrics
import pandas as pd
import numpy as np

standard_scaling = 0
enable_pca = 0
win_size = 5

# This class is the final part of the preprocessing pipeline, and is used to remove columns that are unnecessary
class FeatureDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X.drop(['Volume', 'Adj Close'], axis=1, inplace=True, errors='ignore')
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


if __name__=='__main__':

    model_7 = load_model('models/model_7')
    model_30 = load_model('models/model_30')
    model_90 = load_model('models/model_90')
    models = {'model_7': model_7, 'model_30': model_30, 'model_90': model_90}

    days = 9000

    stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'XOM', 'META', 'JNJ', 'JPM'] 
    pipe = Pipeline([('Dropper', FeatureDropper()), ('Scaler', FeatureScaler())])
    end = date.today()
    start = end - timedelta(days=days)
    yf.pdr_override()

    #mdrwmemphis1

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