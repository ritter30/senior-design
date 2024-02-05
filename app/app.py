#%%
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import numpy as np
import plotly.express as px

#%%
# Incorporate Data
df = pd.read_csv("./data/noise_accel_data.csv", header=None)
df = df.iloc[:,:3]
df.columns = ['x_accel', 'y_accel', 'z_accel']
df.index

#%%
sin_df = pd.read_csv('./data/sine_wave.csv')
#%%
# initalize the app
app = Dash(
    __name__,
    assets_ignore='purdue.*'
    )

app.layout = html.Div([
    html.Div(
        className='app-header',
        children=[
            html.Div('My first app with data, graph, and controls',
                     className='app-header--title')
        ],
        style={'textAlign': 'center'}
             ),
    html.Hr(),
    # html.Div(className='row', children=[
    #     dcc.RadioItems(
    #         options=['x_accel', 'y_accel', 'z_accel'], 
    #         value='x_accel',
    #         id='controls-and-radio-item'
    #         )           
    # ]),
    html.Div(className='row', children=[
        html.H1('Table'),
        html.Div(className='six columns', children=[
            dash_table.DataTable(data=sin_df.to_dict('records'), page_size=6),
        ]),
        html.Div(className='six columns', children=[
            dcc.Graph(figure={}, id='line-chart-final'),
            dcc.Interval(
                id='live-update',
                interval=1*1000,
                n_intervals=0
            )
        ])
    ])
])

# add controls to build interaction
# @callback(
#     Output(component_id='line-chart-final', component_property='figure'),
#     Input(component_id='controls-and-radio-item', component_property='value')
# )
# def update_graph(col_chosen):
#     fig = px.line(df, x=df.index, y=col_chosen)
#     return fig

@callback(
    Output('line-chart-final', 'figure'),
    Input('live-update', 'n_intervals')
)
def live_graph(n):
    sin_df = pd.read_csv('./data/sine_wave.csv')

    sin_df.to_csv('./data/sine_wave.csv')

    fig = {
        'data': [
            {'x': sin_df.index, 'y': sin_df['sin'], 'type': 'line', 'name': 'Live Data'}
        ],
        'layout': {
            'margin': {'l': 30, 'r': 20, 'b': 20, 't': 20},
            'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'Sine'}
        }
    }

    return fig

if __name__ == '__main__':
    app.run(debug=True)

# %%