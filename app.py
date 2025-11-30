import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# AOIs (same as other scripts)
AOI_LIST = [
    "AI", "Alt_VSI", "ASI", "SSI", "TI_HSI", "RPM", "Window"
]

# AOI bounding box coordinates for cockpit overlay 
AOI_COORDS = {
    "AI":      {"x0": 250, "y0": 400, "x1": 320, "y1": 470, "label":"AI"},
    "Alt_VSI": {"x0": 350, "y0": 400, "x1": 420, "y1": 470, "label":"Alt_VSI"},
    "ASI":     {"x0": 250, "y0": 300, "x1": 320, "y1": 370, "label":"ASI"},
    "SSI":     {"x0": 350, "y0": 300, "x1": 420, "y1": 370, "label":"SSI"},
    "TI_HSI":  {"x0": 280, "y0": 500, "x1": 390, "y1": 570, "label":"TI_HSI"},
    "RPM":     {"x0": 500, "y0": 400, "x1": 570, "y1": 470, "label":"RPM"},
    "Window":  {"x0": 30, "y0": 60, "x1": 620, "y1": 200, "label":"Window"}
}

COCKPIT_IMAGE_PATH = "images/cockpit.png"  

from aoibargraph import build_bar_figure
from aoi_graph import build_main_figure

app = dash.Dash(__name__)

def cockpit_figure(selected_aoi):
    # Uses the local file path; Dash loves base64 and PIL for local images, but here we use direct image inclusion for local dev
    fig = go.Figure()
    # Add cockpit image
    fig.add_layout_image(
        dict(
            source=COCKPIT_IMAGE_PATH,
            xref="x", yref="y",
            x=0, y=0,
            sizex=650, sizey=600,
            layer="below"
        )
    )
    # Draw all AOI rectangles, highlight selected
    for aoi, coords in AOI_COORDS.items():
        boxcolor = "green" if aoi == selected_aoi else "red"
        opacity = 0.4 if aoi == selected_aoi else 0.2
        fig.add_shape(
            type="rect",
            x0=coords["x0"], y0=coords["y0"],
            x1=coords["x1"], y1=coords["y1"],
            line=dict(color=boxcolor, width=3 if aoi == selected_aoi else 2),
            fillcolor=boxcolor,
            opacity=opacity,
        )
        fig.add_annotation(
            x=(coords["x0"] + coords["x1"]) / 2,
            y=coords["y0"] - 8,
            text=coords["label"],
            showarrow=False,
            font=dict(color='white' if aoi == selected_aoi else 'black'),
            bgcolor=boxcolor
        )
    fig.update_xaxes(visible=False, range=[0, 650])
    fig.update_yaxes(visible=False, range=[0, 600])
    fig.update_layout(
        title="Cockpit AOI Map",
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        width=650,
    )
    return fig

app.layout = html.Div([
    html.Div([
        html.H3("AOI Dashboard"),
        html.Label("Select AOI:"),
        dcc.Dropdown(
            id="aoi-dropdown",
            options=[{"label": AOI_COORDS[aoi]["label"], "value": aoi} for aoi in AOI_LIST],
            value="SSI",
            clearable=False
        ),
        dcc.Graph(id="cockpit-image", figure=cockpit_figure("SSI")),
    ], style={'width': '39%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Div([
        dcc.Graph(id="bar-chart"),
        dcc.Graph(id="main-chart"),
    ], style={'width':'59%', 'display':'inline-block', 'verticalAlign':'top'}),
])

@app.callback(
    Output('cockpit-image', 'figure'),
    Output('bar-chart', 'figure'),
    Output('main-chart', 'figure'),
    Input('aoi-dropdown', 'value')
)
def update_all(selected_aoi):
    return cockpit_figure(selected_aoi), build_bar_figure(selected_aoi), build_main_figure(selected_aoi)

if __name__ == "__main__":
    app.run_server(debug=True)
