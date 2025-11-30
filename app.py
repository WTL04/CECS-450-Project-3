import dash
from dash import dcc, html, Input, Output
import os
from aoibargraph import build_bar_figure
from aoi_graph import build_main_figure

def build_bar_figure(aoi):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Successful", "Unsuccessful"], y=[1.0, 0.5]))
    fig.update_layout(title=f"Bar Chart - {aoi}")
    return fig

def build_main_figure(aoi):
    import plotly.graph_objects as go
    fig = go.Figure()
    # Dummy lines
    fig.add_trace(go.Scatter(x=[0,1,2], y=[aoi.count("A"), aoi.count("I"), aoi.count("O")], mode='lines+markers'))
    fig.update_layout(title=f"Main Chart - {aoi}")
    return fig


AOI_REGIONS = {
    'AI':        {'x0': 120, 'y0': 100, 'x1': 230, 'y1': 180},
    'Alt_VSI':   {'x0': 340, 'y0': 100, 'x1': 460, 'y1': 180},
    'ASI':       {'x0': 560, 'y0': 100, 'x1': 670, 'y1': 180},
    'SSI':       {'x0': 330, 'y0': 200, 'x1': 470, 'y1': 380},
    'TI_HSI':    {'x0': 50,  'y0': 0,   'x1': 750, 'y1': 80},
    'RPM':       {'x0': 140, 'y0': 300, 'x1': 210, 'y1': 370},
    'Window':    {'x0': 40,  'y0': 40,  'x1': 770, 'y1': 120},
    'No_AOI':    {'x0': 400, 'y0': 400, 'x1': 450, 'y1': 440}
}
AOI_LIST = list(AOI_REGIONS.keys())
IMG_PATH = "./images/cockpit.png"   # Put your cockpit image here!

app = dash.Dash(__name__)

def cockpit_figure(selected_aoi=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=IMG_PATH,
            xref="x", yref="y",
            x=0, y=0,
            sizex=800, sizey=449,
            layer="below"
        )
    )
    # Add AOI box
    for aoi, box in AOI_REGIONS.items():
        line_col = "red" if aoi != selected_aoi else "green"
        opac = 0.2 if aoi != selected_aoi else 0.4
        fig.add_shape(
            type="rect",
            x0=box['x0'], y0=box['y0'], x1=box['x1'], y1=box['y1'],
            line_color=line_col,
            fillcolor=line_col,
            opacity=opac,
        )
        fig.add_annotation(
            x=(box['x0'] + box['x1']) / 2,
            y=box['y1'] + 16,
            text=aoi,
            showarrow=False,
            font={'color': 'white', 'size': 11},
            bgcolor=line_col
        )
    fig.update_xaxes(visible=False, range=[0,800])
    fig.update_yaxes(visible=False, range=[0,449])
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title="Cockpit AOI Map (Click a box to update charts)",
        height=480,
        width=800,
        dragmode=False,
        clickmode='event+select'
    )
    return fig

app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='cockpit-image',
            figure=cockpit_figure(selected_aoi=AOI_LIST[0]),
            config={"staticPlot": False}
        ),
        html.H4("Click an AOI region above to update charts", style={'textAlign':'center'})
    ], style={'width':'48%', 'display':'inline-block', 'verticalAlign':'top'}),
    html.Div([
        dcc.Graph(id='bar-chart', figure=build_bar_figure(AOI_LIST[0])),
        dcc.Graph(id='main-chart', figure=build_main_figure(AOI_LIST[0])),
    ], style={'width':'48%', 'display':'inline-block', 'verticalAlign':'top'}),
], style={'display':'flex', 'gap':'12px'})

@app.callback(
    Output('bar-chart', 'figure'),
    Output('main-chart', 'figure'),
    Output('cockpit-image', 'figure'),
    Input('cockpit-image', 'clickData'),
)
def update_on_click(clickData):
    selected_aoi = AOI_LIST[0]
    if clickData and 'points' in clickData and len(clickData['points']) > 0:
        xval = clickData['points'][0].get('x')
        yval = clickData['points'][0].get('y')
        for aoi, box in AOI_REGIONS.items():
            if xval is not None and yval is not None \
                and box['x0'] <= xval <= box['x1'] \
                and box['y0'] <= yval <= box['y1']:
                selected_aoi = aoi
                break

    return build_bar_figure(selected_aoi), build_main_figure(selected_aoi), cockpit_figure(selected_aoi)

if __name__ == "__main__":
    app.run_server(debug=True)
