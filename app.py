import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import base64, os
import pandas as pd


COCKPIT_IMAGE_PATH = "images/cockpit.png"
IMAGE_WIDTH, IMAGE_HEIGHT = 1199, 675


AOI_TITLES = {
    "AI": "Attitude Indicator (AI)",
    "Alt_VSI": "Altitude/Vert. Speed (Alt-VSI)",
    "ASI": "Airspeed Indicator (ASI)",
    "SSI": "Standby Instruments (SSI)",
    "TI_HSI": "Turn Indicator/HSI (TI-HSI)",
    "RPM": "RPM Gauge",
    "Window": "Window"
}
def pretty_aoi(aoi):
    return AOI_TITLES.get(aoi, aoi.replace("_", " ").title())


AOI_LIST = list(AOI_TITLES.keys())
AOI_COORDS = {
    "AI":      {"x0": 540, "y0": 420, "x1": 620, "y1": 500, "label":AOI_TITLES["AI"]},
    "Alt_VSI": {"x0": 640, "y0": 420, "x1": 720, "y1": 500, "label":AOI_TITLES["Alt_VSI"]},
    "ASI":     {"x0": 540, "y0": 520, "x1": 620, "y1": 600, "label":AOI_TITLES["ASI"]},
    "SSI":     {"x0": 640, "y0": 520, "x1": 720, "y1": 600, "label":AOI_TITLES["SSI"]},
    "TI_HSI":  {"x0": 590, "y0": 640, "x1": 670, "y1": 720, "label":AOI_TITLES["TI_HSI"]},
    "RPM":     {"x0": 950, "y0": 420, "x1": 1080, "y1": 540, "label":AOI_TITLES["RPM"]},
    "Window":  {"x0": 50, "y0": 50, "x1": 1150, "y1": 200, "label":AOI_TITLES["Window"]}
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
    # Draw image background
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
    # AOI overlays
    for aoi, coords in AOI_COORDS.items():
        boxcolor = "#32CD32" if aoi == selected_aoi else "#FF6347"
        opacity = 0.32 if aoi == selected_aoi else 0.14
        fig.add_shape(
            type="rect",
            x0=coords["x0"], y0=coords["y0"],
            x1=coords["x1"], y1=coords["y1"],
            line=dict(color=boxcolor, width=4 if aoi == selected_aoi else 2),
            fillcolor=boxcolor,
            opacity=opacity,
        )
        fig.add_annotation(
            x=(coords["x0"] + coords["x1"]) / 2,
            y=(coords["y0"] + coords["y1"]) / 2,
            text=coords["label"].split("(")[0].replace("/","<br>"),  # only main label
            showarrow=False,
            font=dict(size=18, color='white' if aoi == selected_aoi else 'black', family="Arial"),
            bgcolor=boxcolor,
            opacity=0.95,
        )
    fig.update_xaxes(visible=False, range=[0, IMAGE_WIDTH])
    fig.update_yaxes(visible=False, range=[0, IMAGE_HEIGHT])
    fig.update_layout(
        autosize=False,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT+70,
        margin=dict(l=0, r=0, t=40, b=0),
        title="Cockpit AOI Map"
    )
    return fig

def build_bar_figure(selected_aoi):
    if not os.path.exists(DATA_PATH):
        return go.Figure().update_layout(title="CSV Not Found for Bar Chart")
    df = pd.read_csv(DATA_PATH)
    col = f"{selected_aoi}_Proportion_of_fixations_spent_in_AOI"
    if col not in df.columns or "pilot_success" not in df.columns:
        return go.Figure().update_layout(title=f"Bar chart: Column '{col}' or pilot_success missing!")
    data = df[[col, "pilot_success"]].dropna()
    success_mean = data[data["pilot_success"] == "Successful"][col].mean()
    unsuccess_mean = data[data["pilot_success"] == "Unsuccessful"][col].mean()
    fig = go.Figure(data=[
        go.Bar(x=["Successful", "Unsuccessful"], y=[success_mean, unsuccess_mean],
            marker_color=["#1f77b4", "#d62728"])
    ])
    fig.update_layout(
        title=f"{pretty_aoi(selected_aoi)}: Mean Proportion of Fixations",
        xaxis_title="Pilot Success Group",
        yaxis_title="Mean Proportion",
        font=dict(size=16)
    )
    return fig

def build_main_figure(selected_aoi):
    if not os.path.exists(DATA_PATH):
        return go.Figure().update_layout(title="CSV Not Found for Main Chart")
    df = pd.read_csv(DATA_PATH)
    cols = [
        f"{selected_aoi}_Total_Number_of_Fixations",
        f"{selected_aoi}_Mean_fixation_duration_s",
        f"{selected_aoi}_Proportion_of_fixations_spent_in_AOI",
        f"{selected_aoi}_Proportion_of_fixations_durations_spent_in_AOI"
    ]
    plot_cols = [c for c in cols if c in df.columns]
    if len(plot_cols) < 2 or "pilot_success" not in df.columns:
        return go.Figure().update_layout(title=f"Not enough data for AOI: {pretty_aoi(selected_aoi)}")
    dims = []
    for c in plot_cols:
        dims.append(dict(label=c.replace("_", " "), values=pd.to_numeric(df[c], errors="coerce")))
    color_map = df["pilot_success"].map(lambda s: 1 if str(s).strip().lower() == "successful" else 0)
    fig = go.Figure(data=[go.Parcoords(
        line=dict(
            color=color_map,
            colorscale=[[0, '#d62728'], [1, '#1f77b4']],
            colorbar=dict(
                title="Pilot Success",
                tickvals=[0, 1],
                ticktext=["Unsuccessful", "Successful"]
            )
        ),
        dimensions=dims
    )])
    fig.update_layout(title=f"Parallel Coordinates – {pretty_aoi(selected_aoi)}", margin=dict(t=55), font=dict(size=15))
    return fig

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H3("AOI Dashboard", style={"margin-bottom": "5px"}),
        html.Label("Select AOI:", style={"font-weight": "bold"}),
        dcc.Dropdown(
            id="aoi-dropdown",
            options=[{"label": AOI_TITLES[aoi], "value": aoi} for aoi in AOI_LIST],
            value=AOI_LIST[0],
            clearable=False,
            style={"width": "85%", "margin-bottom": "15px", "fontWeight": "bold", "fontSize":"1.1rem"}
        ),
        dcc.Graph(id="cockpit-image"),
    ], style={'flex': '1', 'min-width': "450px", 'padding': '20px', "background":"#f6f6fa"}),
    html.Div([
        dcc.Graph(id="bar-chart", config={"displayModeBar": False}),
        dcc.Graph(id="main-chart", config={"displayModeBar": False}),
    ], style={'flex': '2', 'padding': '20px'}),
], style={"display": "flex", "flexDirection": "row", "height":"100vh", "fontFamily":"Arial"})

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
