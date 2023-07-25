from dash import Dash, html, dcc, Output, Input, State , exceptions
from datetime import datetime as dt
from model import build_model, plot_predictions
import yfinance as yf
import pandas as pd
import plotly.express as px

def get_stock_price_fig(df):
    fig = px.line(df, x='Date', y=['Open', 'Close'], title='Closing and Opening Price')
    return fig

def get_ema_fig(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,x= 'Date', y= 'EMA_20', title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig


app = Dash(__name__, external_stylesheets=['assests/styles.css'])
server = app.server

app.layout= html.Div([
    html.Div([
        html.H1("Welcome to Dash Stock Forecasting app", className="start"),

        html.Div([
            #stock code input
            html.Label('Input stock code'),
            html.Br(),
            dcc.Input(placeholder='Stock Name', id='stock-name', type='text'),
            html.Button('Submit', id='submit-button', n_clicks=0),
        ], className="stock-code"),

        html.Div([
            #date range picker input
            html.Label('Select a date range'),
            html.Br(),
            dcc.DatePickerRange(
                id='my-date-picker-range',
                initial_visible_month=dt.now(),
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                calendar_orientation='horizontal',
                clearable=True,
            ),
        ], className="date-range"),

        html.Div([
            #stock price button
            #indicators button
            #Number of days of forecast input
            #forecast button
            html.Button('Stock Price', id='stock-price-button'),
            html.Button('Indicators', id='indicators-button'),
            dcc.Input(type='number', id='forecast-period'),
            html.Button('Forecast', id='forecast-period-button'),
        ]),
    ], className="nav"
    ),

    html.Div([
        html.Div([
            #logo
            #company name
        ], className="header", id="header"),
        html.Div([
            #description
        ], className="description-ticker", id="description"),
        html.Div([
            #stock price plot
        ], id="graphs-content"),
        html.Div([
            #indicator plot
        ], id="main-content"),
        html.Div([
            #forecast plot        
        ], id="forecast-content")
    ], className="content"
    )

], className="container")

@app.callback(
    Output('header', 'children'),
    Output('description', 'children'),
    Input('submit-button', 'n_clicks'),
    State('stock-name', 'value')
)
def update_info(n_clicks, stock_name):
    if n_clicks is None or n_clicks == 0:
        raise exceptions.PreventUpdate
    elif n_clicks > 0:
        stock = yf.Ticker(stock_name)
        info = stock.info
        df = pd.DataFrame().from_dict(info, orient="index").T
        domain = df['website'].values[0]
        logo_url = f"https://logo.clearbit.com/{domain}"
        return [
            html.Img(src=logo_url, className="logo"),
            html.H3(df['shortName'].values[0])
        ], html.P(df['longBusinessSummary'].values[0])

@app.callback(
    Output('graphs-content', 'children'),
    Input('stock-price-button', 'n_clicks'),
    State('stock-name', 'value'),
    State('my-date-picker-range', 'start_date'),
    State('my-date-picker-range', 'end_date')
)
def update_stock_graph(n_clicks, stock_name, start_date, end_date):
    if n_clicks is None:
        raise exceptions.PreventUpdate
    elif n_clicks > 0:
        df = yf.download(stock_name, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        fig = get_stock_price_fig(df)
        return dcc.Graph(figure=fig)
    else:
        return None
    

@app.callback(
    Output('main-content', 'children'),
    Input('indicators-button', 'n_clicks'),
    State('stock-name', 'value'),
    State('my-date-picker-range', 'start_date'),
    State('my-date-picker-range', 'end_date')
)
def update_indicators_graph(n_clicks, stock_name, start_date, end_date):
    if n_clicks is None:
        raise exceptions.PreventUpdate
    elif n_clicks > 0:
        df = yf.download(stock_name, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        fig = get_ema_fig(df)
        return dcc.Graph(figure=fig)
    else:
        return None
    
@app.callback(
    Output('forecast-content', 'children'),
    Input('forecast-period-button', 'n_clicks'),
    State('stock-name', 'value'),
    State('forecast-period', 'value')
)
def update_forecast_graph(n_clicks, stock_name, forecast_period):
    if n_clicks is None:
        raise exceptions.PreventUpdate
    elif n_clicks > 0:
        fig = plot_predictions(build_model(stock_name), stock_name, forecast_period)
        return dcc.Graph(figure=fig)
    else:
        return None


if __name__ == "__main__":
    app.run(debug=True)
