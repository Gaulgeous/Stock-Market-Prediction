import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash  # pip install dash (version 2.0.0 or higher)
from dash import html
import dash_bootstrap_components as dbc
import yfinance as yf
from datetime import date, timedelta

import pandas as pd

days = 9000
text_colour = 'white'
title_colour = 'white'


stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'XOM', 'META', 'JNJ', 'JPM'] 
end = date.today()
start = end - timedelta(days=days)
yf.pdr_override()
cache = {}

for stock in stocks:

    data = yf.download(stock, start, end)
    data = data.resample('D').first() 
    data = data.dropna(how='any', axis='rows')
    cache[stock] = data

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ------------------------------------------------------------------------------
# App layout
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([
            html.H1("Stock Predictor Application", style={'text-align': 'center',
                    "color": title_colour})
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Label('Stock', style={"color": text_colour}),
            dash.dcc.Dropdown(id="stock_selection",
                options=[
                    {"label": "Apple", "value": "AAPL"},
                    {"label": "Microsoft", "value": "MSFT"},
                    {"label": "Google", "value": "GOOG"},
                    {"label": "Amazon", "value": "AMZN"},
                    {"label": "Tesla", "value": "TSLA"},
                    {"label": "NVIDIA", "value": "NVDA"},
                    {"label": "Exxon Mobil", "value": "XOM"},
                    {"label": "Meta Platforms", "value": "META"},
                    {"label": "Johnson & Johnson", "value": "JNJ"},
                    {"label": "JPMorgan", "value": "JPM"}],
                multi=False,
                value="AAPL",
                style={'width': "100%"},
                placeholder='Select a Stock'
            )
        ]), 

        dbc.Col([
            html.Label('Actual Data Type', style={"color": text_colour}),
            dash.dcc.Dropdown(id="actual_showing",
                        options=[
                            {"label": "Open", "value": "Open"},
                            {"label": "High", "value": "High"},
                            {"label": "Low", "value": "Low"},
                            {"label": "Close", "value": "Close"},
                            {"label": "Candlestick", "value": "Candlestick"}],
                        multi=False,
                        value="Close",
                        style={'width': "100%"},
                        placeholder='Select Display for Actual Data'
                        )]), 

        dbc.Col([
            html.Label('Predicted Data Type', style={"color": text_colour}),
            dash.dcc.Dropdown(id="predictor_showing",
                        options=[
                            {"label": "Open", "value": 'Open'},
                            {"label": "High", "value": 'High'},
                            {"label": "Low", "value": 'Low'},
                            {"label": "Close", "value": 'Close'}],
                        multi=False,
                        value="Close",
                        style={'width': "100%"},
                        placeholder='Select Display for Predicted Data'
                        )]), 

        dbc.Col([
            html.Label('Forecasting Period (Days)', style={"color": text_colour}),
            dash.dcc.Dropdown(id="period",
                        options=[
                            {"label": "7", "value": "model_7"},
                            {"label": "30", "value": "model_30"},
                            {"label": "90", "value": "model_90"}],
                        multi=False,
                        value="model_7",
                        style={'width': "100%"},
                        placeholder='Select Forecasting Period'
                        )
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id='output_container', children=[]),
        ])
    ]),
    # html.Br(),

    dbc.Row([
        dbc.Col([
            dash.dcc.Graph(id='stock_graph', figure={})
        ]),
        dbc.Col([
            dash.dcc.Graph(id='future_graph', figure={}),
            dash.dcc.Graph(id='performance_graph', figure={})
        ])
    ])


], style={'backgroundColor':'black'})


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [dash.Output(component_id='output_container', component_property='children'),
     dash.Output(component_id='stock_graph', component_property='figure'),
     dash.Output(component_id='future_graph', component_property='figure'),
     dash.Output(component_id='performance_graph', component_property='figure')],
    [dash.Input(component_id='stock_selection', component_property='value'),
    dash.Input(component_id='actual_showing', component_property='value'),
    dash.Input(component_id='predictor_showing', component_property='value'),
    dash.Input(component_id='period', component_property='value')]
)
def update_main_graph(stock, actual, pred, period):

    container = "Showing stock: {}".format(stock)

    frame = cache.get(stock).copy()
    pred_data = pd.read_csv('stocks_csvs/' + period + '_' + stock + '.csv')

    if period == 'model_7':
        partition = 70
        performance = pd.read_csv('stocks_csvs/model_7_performance.csv')
    elif period == 'model_30':
        partition = 300
        performance = pd.read_csv('stocks_csvs/model_30_performance.csv')
    else:
        partition = 900
        performance = pd.read_csv('stocks_csvs/model_90_performance.csv')
    

    if actual == "Candlestick":
        fig = go.Figure(go.Candlestick(x=frame.index,
                open=frame['Open'][-partition:], 
                high=frame['High'][-partition:],
                low=frame['Low'][-partition:], 
                close=frame['Close'][-partition:]))
        future_fig = go.Figure(go.Candlestick(x=frame.index,
                open=frame['Open'], 
                high=frame['High'],
                low=frame['Low'], 
                close=frame['Close']))

        fig.add_trace(go.Scatter(x=pred_data['Dates'][-partition:], y=pred_data[actual][-partition:], line_color='cyan', name='predicted'))
        future_fig.add_trace(go.Scatter(x=pred_data['Dates'], y=pred_data[actual], line_color='cyan', name='predicted'))

    else:
        # Plotly Express
        
        fig = go.Figure(go.Scatter(x=frame.index[-partition:], y=frame[actual][-partition:], line_color='red', name='Actual'))
        fig.add_trace(go.Scatter(x=pred_data['Dates'][-partition:], y=pred_data[actual][-partition:], line_color='cyan', name='predicted'))
        future_fig = go.Figure(go.Scatter(x=frame.index, y=frame[actual], line_color='red', name='Actual'))
        future_fig.add_trace(go.Scatter(x=pred_data['Dates'], y=pred_data[actual], line_color='cyan', name='predicted'))

    performance_fig = px.line_polar(performance, r='value', theta='metric', line_close=True, template='plotly_dark')
    
    fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', title=actual + " of " + stock + " (Zoomed)",
            xaxis_title="Date", yaxis_title="Price", paper_bgcolor='rgba(0,0,0,0)')

    future_fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', title=actual + " of " + stock,
            xaxis_title="Year", yaxis_title="Price", paper_bgcolor='rgba(0,0,0,0)')

    performance_fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', title="Model Performance",
            paper_bgcolor='rgba(0,0,0,0)')
        

    return container, fig, future_fig, performance_fig



if __name__=='__main__':
    app.run_server(debug=True)
