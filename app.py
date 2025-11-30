import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import base64, os
import pandas as pd

# Config:
COCKPIT_IMAGE_PATH = "images/cockpit.png"
IMAGE_WIDTH, IMAGE_HEIGHT = 600, 360   # <-- Update to match your actual image size!

AOI_LIST = [
    "AI", "Alt_VSI", "ASI", "SSI", "TI_HSI", "RPM", "Window"
]
AOI_COORDS = {
    "AI":      {"x0": 170, "y0": 160, "x1": 220, "y1": 200, "label":"AI"},
    "Alt_VSI": {"x0": 250, "y0": 160, "x1": 300, "y1": 200, "label":"Alt_VSI"},
    "ASI":     {"x0": 170, "y0": 220, "x1": 220, "y1": 260, "label":"ASI"},
    "SSI":     {"x0": 250, "y0": 220, "x1": 300, "y1": 260, "label":"SSI"},
    "TI_HSI":  {"x0": 210, "y0": 280, "x1": 270, "y1": 320, "label":"TI_HSI"},
    "RPM":     {"x0": 440, "y0": 200, "x1": 520, "y1": 250, "label":"RPM"},
    "Window":  {"x0": 40, "y0": 20, "x1": 560, "y1": 100, "label":"Window"}
}
DATA_PATH = "./datasets/AOI_DGMs.csv"

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def cockpit_figure(selected_aoi):
    encoded_image = get_base64_image(COCKPIT_IMAGE_PATH)
    fig = go.Figure()

    # Show image
    if encoded_image:
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image}",
                xref="x", yref="y",
                x=0, y=0,
                sizex=IMAGE_WIDTH, sizey=IMAGE_HEIGHT,
                layer="below"
            )
        )
    else:
        fig.add_annotation(
            x=IMAGE_WIDTH/2, y=IMAGE_HEIGHT/2, text="Image file not found!",
            showarrow=False, font=dict(size=22, color="red")
        )
    # Overlays
    for aoi, coords in AOI_COORDS.items():
        boxcolor = "limegreen" if aoi == selected_aoi else "orangered"
        opacity = 0.3 if aoi == selected_aoi else 0.10
        fig.add_shape(
            type="rect",
            x0=coords["x0"], y0=coords["y0"],
            x1=coords["x1"], y1=coords["y1"],
            line=dict(color=boxcolor, width=3 if aoi == selected_aoi else 1),
            fillcolor=boxcolor,
            opacity=opacity,
            layer="above"
        )
        fig.add_annotation(
            x=(coords["x0"] + coords["x1"]) / 2,
            y=coords["y0"] - 7,
            text=coords["label"],
            showarrow=False,
            font=dict(size=14, color='white' if aoi == selected_aoi else 'black'),
            bgcolor=boxcolor,
            opacity=opacity
        )
    fig.update_xaxes(visible=False, range=[0, IMAGE_WIDTH])
    fig.update_yaxes(visible=False, range=[0, IMAGE_HEIGHT])
    fig.update_layout(
        autosize=False,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT+65,
        margin=dict(l=0, r=0, t=40, b=0),
        title="Cockpit AOI Map",
        paper_bgcolor="#f6f6fa"
    )
    return fig

#  BAR CHART
def build_bar_figure(selected_aoi):
    if not os.path.exists(DATA_PATH):
        return go.Figure().update_layout(title="CSV Not Found for Bar Chart")
    df = pd.read_csv(DATA_PATH)
    col = f"{selected_aoi} Proportion of fixations spent in AOI"
    if col not in df.columns or "pilot_success" not in df.columns:
        return go.Figure().update_layout(title=f"Bar chart: Column '{col}' or pilot_success missing!")
    data = df[[col, "pilot_success"]].dropna()
    success_mean = data[data["pilot_success"] == "Successful"][col].mean()
    unsuccess_mean = data[data["pilot_success"] == "Unsuccessful"][col].mean()
    fig = go.Figure(data=[
        go.Bar(x=["Successful", "Unsuccessful"], y=[success_mean, unsuccess_mean],
            marker_color=["royalblue", "firebrick"])
    ])
    fig.update_layout(
        title=f"{selected_aoi}: Mean Proportion of Fixations",
        xaxis_title="Pilot Success",
        yaxis_title="Mean Proportion"
    )
    return fig


def build_main_figure(selected_aoi):
    if not os.path.exists(DATA_PATH):
        return go.Figure().update_layout(title="CSV Not Found for Main Chart")
    df = pd.read_csv(DATA_PATH)
    cols = [f"{selected_aoi} Total Number of Fixations",
            f"{selected_aoi} Mean fixation duration s",
            f"{selected_aoi} Proportion of fixations spent in AOI",
            f"{selected_aoi} Proportion of fixations durations spent in AOI"]
    plot_cols = [c for c in cols if c in df.columns]
    if len(plot_cols) < 2 or "pilot_success" not in df.columns:
        return go.Figure().update_layout(title=f"Not enough data for AOI: {selected_aoi}")
    dims = []
    for c in plot_cols:
        dims.append(dict(label=c, values=pd.to_numeric(df[c], errors="coerce")))
    color_map = df["pilot_success"].map(lambda s: 1 if str(s).strip().lower() == "successful" else 0)
    fig = go.Figure(data=[go.Parcoords(
        line=dict(
            color=color_map,
            colorscale=[[0, 'firebrick'], [1, 'royalblue']],
            colorbar=dict(
                title="Pilot Success",
                tickvals=[0, 1],
                ticktext=["Unsuccessful", "Successful"]
            )
        ),
        dimensions=dims
    )])
    fig.update_layout(title=f"Parallel Coordinates – AOI: {selected_aoi}", margin=dict(t=60))
    return fig


app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H3("AOI Dashboard", style={"margin-bottom": "5px"}),
        html.Label("Select AOI:", style={"font-weight": "bold"}),
        dcc.Dropdown(
            id="aoi-dropdown",
            options=[{"label": AOI_COORDS[aoi]["label"], "value": aoi} for aoi in AOI_LIST],
            value=AOI_LIST[0],
            clearable=False,
            style={"width": "85%", "margin-bottom": "15px"}
        ),
        dcc.Graph(id="cockpit-image"),
    ], style={'flex': '1', 'min-width': "410px", 'padding': '20px', "background":"#f6f6fa"}),
    html.Div([
        dcc.Graph(id="bar-chart", config={"displayModeBar": False}),
        dcc.Graph(id="main-chart", config={"displayModeBar": False}),
    ], style={'flex': '2', 'padding': '20px'}),
], style={"display": "flex", "flexDirection": "row", "height":"100vh"})

@app.callback(
    Output('cockpit-image', 'figure'),
    Output('bar-chart', 'figure'),
    Output('main-chart', 'figure'),
    Input('aoi-dropdown', 'value')
)
def update_all(selected_aoi):
    return (
        cockpit_figure(selected_aoi),
        build_bar_figure(selected_aoi),
        build_main_figure(selected_aoi)
    )


import webbrowser, threading, time
if __name__ == "__main__":
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:8050/")
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
