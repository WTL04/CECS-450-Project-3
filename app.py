import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

AOI_LIST = [
    "AI", "Alt_VSI", "ASI", "SSI", "TI_HSI", "RPM", "Window"
]

AOI_COORDS = {
    "AI":      {"x0": 195, "y0": 180, "x1": 225, "y1": 210, "label":"AI"},
    "Alt_VSI": {"x0": 253, "y0": 180, "x1": 285, "y1": 210, "label":"Alt_VSI"},
    "ASI":     {"x0": 195, "y0": 230, "x1": 225, "y1": 260, "label":"ASI"},
    "SSI":     {"x0": 253, "y0": 230, "x1": 285, "y1": 260, "label":"SSI"},
    "TI_HSI":  {"x0": 215, "y0": 275, "x1": 265, "y1": 305, "label":"TI_HSI"},
    "RPM":     {"x0": 349, "y0": 210, "x1": 395, "y1": 260, "label":"RPM"},
    "Window":  {"x0": 28, "y0": 28, "x1": 370, "y1": 120, "label":"Window"}
}
COCKPIT_IMAGE_PATH = "images/cockpit.png"  # update as needed

from aoibargraph import build_bar_figure
from aoi_graph import build_main_figure

app = dash.Dash(__name__)

def cockpit_figure(selected_aoi):
    # Put the cockpit image behind AOI overlays
    width, height = 400, 320  # actual image size (update as needed!)
    fig = go.Figure()

    # Add your cockpit image (Update path if needed!)
    fig.add_layout_image(
        dict(
            source="images/cockpit.png",  # Path relative to run location
            xref="x", yref="y",
            x=0, y=0,
            sizex=width, sizey=height,
            layer="below"
        )
    )
    # Overlay AOI rectangles and labels
    for aoi, coords in AOI_COORDS.items():
        boxcolor = "green" if aoi == selected_aoi else "red"
        opacity = 0.4 if aoi == selected_aoi else 0.18
        fig.add_shape(
            type="rect",
            x0=coords["x0"], y0=coords["y0"],
            x1=coords["x1"], y1=coords["y1"],
            line=dict(color=boxcolor, width=3 if aoi == selected_aoi else 2),
            fillcolor=boxcolor,
            opacity=opacity,
            layer="above"
        )
        fig.add_annotation(
            x=(coords["x0"] + coords["x1"]) / 2,
            y=coords["y0"] - 5,
            text=coords["label"],
            showarrow=False,
            font=dict(size=13, color='white' if aoi == selected_aoi else 'black'),
            bgcolor=boxcolor,
            opacity=opacity
        )
    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[0, height])
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=height+50,
        width=width,
        title="Cockpit AOI Map",
        paper_bgcolor="#f6f6fa"
    )
    return fig

# DASH LAYOUT
app.layout = html.Div([
    html.Div([
        html.H3("AOI Dashboard", style={"margin-bottom": "5px"}),
        html.Label("Select AOI:", style={"font-weight": "bold"}),
        dcc.Dropdown(
            id="aoi-dropdown",
            options=[{"label": AOI_COORDS[aoi]["label"], "value": aoi} for aoi in AOI_LIST],
            value=AOI_LIST[0],
            clearable=False,
            style={"width": "80%", "margin-bottom": "15px"}
        ),
        dcc.Graph(id="cockpit-image", figure=cockpit_figure(AOI_LIST[0])),
    ], style={'width': '44%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding':'15px 8px', "background":"#f6f6fa"}),
    html.Div([
        dcc.Graph(id="bar-chart", config={"displayModeBar": False}),
        dcc.Graph(id="main-chart", config={"displayModeBar": False}),
    ], style={'width':'54%', 'display':'inline-block', 'verticalAlign':'top', "padding":"15px 12px"}),
], style={"font-family":"Helvetica, Arial, sans-serif"})

@app.callback(
    Output('cockpit-image', 'figure'),
    Output('bar-chart', 'figure'),
    Output('main-chart', 'figure'),
    Input('aoi-dropdown', 'value')
)
def update_all(selected_aoi):
    # update image, bar, parallel charts
    return cockpit_figure(selected_aoi), build_bar_figure(selected_aoi), build_main_figure(selected_aoi)

if __name__ == "__main__":
    app.run(debug=True)
