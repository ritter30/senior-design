#%%
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px

#%%
# Incorporate Data
df = pd.read_csv("./data/noise_accel_data.csv", header=None)
df = df.iloc[:,:3]
df.columns = ['x_accel', 'y_accel', 'z_accel']
df.index
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
    html.Div(className='row', children=[
        dcc.RadioItems(
            options=['x_accel', 'y_accel', 'z_accel'], 
            value='x_accel',
            id='controls-and-radio-item'
            )           
    ]),
    html.Div(className='row', children=[
        html.H1('Table'),
        html.Div(className='six columns', children=[
            dash_table.DataTable(data=df.to_dict('records'), page_size=6),
        ]),
        html.Div(className='six columns', children=[
            dcc.Graph(figure={}, id='line-chart-final')
        ])
    ])
])

# add controls to build interaction
@callback(
    Output(component_id='line-chart-final', component_property='figure'),
    Input(component_id='controls-and-radio-item', component_property='value')
)
def update_graph(col_chosen):
    fig = px.line(df, x=df.index, y=col_chosen)
    return fig

if __name__ == '__main__':
    app.run(debug=True)

# %%
