import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime as dt

def build_model(ticker):
    # Getting data and preprocessing
    data = yf.download(ticker, period='60d')
    data.reset_index(inplace=True)
    data['Day'] = data.index
    days = list()
    for i in range(len(data.Day)): days.append([i])
    X = days
    y = data['Close']

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    # Search for best parameters
    parameters = {
        'C': [0.001, 0.01, 0.1, 1, 100, 1000], 
        'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 150, 1000], 
        'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]
    }
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(svr, parameters, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_train = y_train.values.ravel()


    # Get best parameters
    final_params = grid_search.best_params_

    # Build final model with best parameters
    svr_model = SVR(kernel='rbf', C=final_params['C'], epsilon=final_params['epsilon'], gamma=final_params['gamma'])
    svr_model.fit(X_train, y_train)

    # Get predictions
    # y_pred = svr_model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)

    # print(mse)
    # print(mae)
    return svr_model

def plot_predictions(model, ticker, n_days):
    data = yf.download(ticker, period='60d')
    data.reset_index(inplace=True)
    data['Day'] = data.index

    # Create a list of days for which we want to predict
    future_days = list()
    for i in range(len(data.Day), len(data.Day) + n_days):
        future_days.append([i])

    # Use the model to predict the future
    future_predictions = model.predict(future_days)

    # Create a new dataframe to hold the future predictions
    future_df = pd.DataFrame()
    start_date = pd.to_datetime(dt.datetime.today().date()) + dt.timedelta(days=1)
    future_df['Day'] = pd.date_range(start_date, periods=n_days, freq='B')
    future_df['Predicted Close'] = future_predictions

    # Plot the predicted data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_df['Day'], y=future_df['Predicted Close'], mode='lines+markers', name='Predicted Close'))
    fig.update_layout(title=f'Predicted Close Price for {n_days} Days in the Future', xaxis_title='Day', yaxis_title='Closing Price')

    return fig