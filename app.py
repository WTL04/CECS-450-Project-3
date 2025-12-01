import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import base64, os
import pandas as pd
import numpy as np

COCKPIT_IMAGE_PATH = "images/cockpit.png"
IMAGE_WIDTH, IMAGE_HEIGHT = 900, 504

AOI_TITLES = {
    "AI": "Attitude Indicator (AI)",
    "Alt_VSI": "Altitude/Vert. Speed (Alt-VSI)",
    "ASI": "Airspeed Indicator (ASI)",
    "SSI": "Standby Instruments (SSI)",
    "TI_HSI": "Turn Indicator/HSI (TI-HSI)",
    "RPM": "RPM Gauge",
    "Window": "Window"
}
def pretty_aoi(aoi): return AOI_TITLES.get(aoi, aoi.replace("_", " ").title())

AOI_LIST = list(AOI_TITLES.keys())
AOI_COORDS = {
    "AI":      {"x0": 383, "y0": 110, "x1": 491, "y1": 160, "label":AOI_TITLES["AI"]},
    "Alt_VSI": {"x0": 497, "y0": 67, "x1": 568, "y1": 200, "label":AOI_TITLES["Alt_VSI"]},
    "ASI":     {"x0": 323, "y0": 75, "x1": 376, "y1": 191, "label":AOI_TITLES["ASI"]},
    "SSI":     {"x0": 383, "y0": 165, "x1": 491, "y1": 200, "label":AOI_TITLES["SSI"]},
    "TI_HSI":  {"x0": 383, "y0": 0, "x1": 492, "y1": 105, "label":AOI_TITLES["TI_HSI"]},
    "RPM":     {"x0": 790, "y0": 150, "x1": 855, "y1": 220, "label":AOI_TITLES["RPM"]},
    "Window":  {"x0": 0, "y0": 700, "x1": 1000, "y1": 280, "label":AOI_TITLES["Window"]},
}
DATA_PATH = "./datasets/AOI_DGMs.csv"
PATTERN_DATA_PATH = "./datasets/Collapsed Patterns (Group).xlsx"

# Consistent color scheme for success/unsuccessful groups
SUCCESS_COLOR = "#2ecc71"  # green
UNSUCCESS_COLOR = "#e74c3c"  # red

# AOI mapping for scan patterns
PATTERN_AOI_MAP = {
    'B': 'Alt_VSI',
    'C': 'AI',
    'D': 'TI_HSI',
    'E': 'SSI',
    'F': 'ASI',
    'G': 'RPM'
}

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def cockpit_figure(selected_aoi):
    SCALE = 0.7 # scale figure size
    encoded_image = get_base64_image(COCKPIT_IMAGE_PATH)
    fig = go.Figure()

    scaled_w = IMAGE_WIDTH * SCALE
    scaled_h = IMAGE_HEIGHT * SCALE

    # background image
    if encoded_image:
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image}",
                xref="x",
                yref="y",
                x=0,
                y=scaled_h,
                sizex=scaled_w,
                sizey=scaled_h,
                xanchor="left",
                yanchor="top",
                sizing="stretch",
                layer="below"
            )
        )

    # AOI boxes, scaled
    for aoi, coords in AOI_COORDS.items():
        x0 = coords["x0"] * SCALE
        y0 = coords["y0"] * SCALE
        x1 = coords["x1"] * SCALE
        y1 = coords["y1"] * SCALE

        boxcolor = "#32CD32" if aoi == selected_aoi else "#FF6347"
        opacity = 0.32 if aoi == selected_aoi else 0.14

        fig.add_shape(
            type="rect",
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            line=dict(color=boxcolor, width=4 if aoi == selected_aoi else 2),
            fillcolor=boxcolor,
            opacity=opacity,
        )

    fig.update_xaxes(visible=False, range=[0, scaled_w])
    fig.update_yaxes(visible=False, range=[0, scaled_h])
    fig.update_layout(
        autosize=False,
        width=scaled_w,
        height=scaled_h + 40,
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
            marker_color=[SUCCESS_COLOR, UNSUCCESS_COLOR])
    ])
    fig.update_layout(
        title=f"{pretty_aoi(selected_aoi)}: Mean Proportion of Fixations",
        xaxis_title="Pilot Success Group",
        yaxis_title="Mean Proportion",
        margin=dict(t=60)
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
        dims.append(dict(
            label=c.replace("_", " ").title(),
            values=pd.to_numeric(df[c], errors="coerce"),
        ))
    color_map = df["pilot_success"].map(lambda s: 1 if str(s).strip().lower() == "successful" else 0)
    fig = go.Figure(data=[go.Parcoords(
        line=dict(
            color=color_map,
            colorscale=[[0, UNSUCCESS_COLOR], [1, SUCCESS_COLOR]],
            showscale=True,
            colorbar=dict(
                title="Pilot Success",
                tickvals=[0, 1],
                ticktext=["Unsuccessful", "Successful"],
            )
        ),
        dimensions=dims,
    )])
    fig.update_layout(
        title=f"Parallel Coordinates – {pretty_aoi(selected_aoi)}",
        margin=dict(t=95)
    )
    return fig

def build_saccade_metrics_comparison_figure():
    """2x2 grid of box plots comparing saccade metrics"""
    if not os.path.exists(DATA_PATH):
        return go.Figure().update_layout(title="CSV Not Found")
    df = pd.read_csv(DATA_PATH)

    # Add success category
    if 'Approach_Score' not in df.columns:
        return go.Figure().update_layout(title="Missing Approach_Score column")

    df['Success_Category'] = df['Approach_Score'].apply(
        lambda x: 'Successful' if pd.notna(x) and x >= 0.7 else 'Unsuccessful'
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Mean Saccade Duration',
            'Mean Saccade Length',
            'Peak Saccade Velocity',
            'Fixation to Saccade Ratio'
        )
    )

    metrics_to_plot = [
        ('mean_saccade_duration', 1, 1, 'Duration (s)'),
        ('mean_saccade_length', 1, 2, 'Length (px)'),
        ('Average_Peak_Saccade_Velocity', 2, 1, 'Velocity (°/s)'),
        ('fixation_to_saccade_ratio', 2, 2, 'Ratio')
    ]

    for metric, row, col, ylabel in metrics_to_plot:
        if metric not in df.columns:
            continue

        for category, color in [('Successful', SUCCESS_COLOR), ('Unsuccessful', UNSUCCESS_COLOR)]:
            data = df[df['Success_Category'] == category][metric]

            fig.add_trace(
                go.Box(
                    y=data,
                    name=category,
                    marker_color=color,
                    showlegend=(row == 1 and col == 1),
                    boxmean='sd'
                ),
                row=row, col=col
            )

        fig.update_yaxes(title_text=ylabel, row=row, col=col)

    fig.update_layout(
        title='Saccade Metrics Comparison: Successful vs Unsuccessful Pilots',
        height=800,
        template='plotly_white'
    )
    return fig

def build_saccade_count_by_aoi_figure():
    if not os.path.exists(DATA_PATH):
        return go.Figure().update_layout(title="CSV Not Found for Saccade Count by AOI")
    df = pd.read_csv(DATA_PATH)

    # Check if Approach_Score exists
    if 'Approach_Score' not in df.columns:
        return go.Figure().update_layout(title="Missing Approach_Score column")

    # Add success category
    df['Success_Category'] = df['Approach_Score'].apply(
        lambda x: 'Successful' if pd.notna(x) and x >= 0.7 else 'Unsuccessful'
    )

    # Prepare data for AOI comparison
    aoi_data = []
    for aoi in AOI_LIST:
        col_name = f'{aoi}_total_number_of_saccades'
        if col_name in df.columns:
            # Convert column to numeric, replacing #NULL! and other invalid values with NaN
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

            for category in ['Successful', 'Unsuccessful']:
                mean_count = df[df['Success_Category'] == category][col_name].mean()
                aoi_data.append({
                    'AOI': pretty_aoi(aoi),
                    'Category': category,
                    'Mean_Saccades': mean_count
                })

    if not aoi_data:
        return go.Figure().update_layout(title="No saccade count data found")

    aoi_df = pd.DataFrame(aoi_data)

    fig = go.Figure()

    for category, color in [('Successful', SUCCESS_COLOR), ('Unsuccessful', UNSUCCESS_COLOR)]:
        data = aoi_df[aoi_df['Category'] == category]

        fig.add_trace(go.Bar(
            x=data['AOI'],
            y=data['Mean_Saccades'],
            name=category,
            marker_color=color,
            text=data['Mean_Saccades'].round(1),
            textposition='auto'
        ))

    fig.update_layout(
        title='Mean Saccade Count by Area of Interest',
        xaxis_title='Area of Interest',
        yaxis_title='Mean Number of Saccades',
        barmode='group',
        template='plotly_white',
        margin=dict(t=60)
    )
    return fig

def build_scan_transition_differences_figure():
    """Horizontal diverging bar showing transition differences"""
    if not os.path.exists(PATTERN_DATA_PATH):
        return go.Figure().update_layout(title="Pattern data file not found")

    try:
        collapsed_xls = pd.ExcelFile(PATTERN_DATA_PATH)
        successful_collapsed = pd.read_excel(collapsed_xls, sheet_name='Succesful Excluding No AOI(A)')
        unsuccessful_collapsed = pd.read_excel(collapsed_xls, sheet_name='Unsuccesful Excluding No AOI(A)')
    except Exception as e:
        return go.Figure().update_layout(title=f"Error loading pattern data: {str(e)}")

    def get_transition_differences(succ_df, unsucc_df):
        pattern_col = 'Pattern String'
        freq_col = 'Proportional Pattern Frequency'
        succ_trans, unsucc_trans = {}, {}

        for df, trans_dict in [(succ_df, succ_trans), (unsucc_df, unsucc_trans)]:
            for _, row in df.iterrows():
                pattern = str(row[pattern_col]).upper()
                frequency = row[freq_col]
                if pattern == 'nan':
                    continue
                for i in range(len(pattern) - 1):
                    key = (pattern[i], pattern[i+1])
                    trans_dict[key] = trans_dict.get(key, 0) + frequency

        succ_total = sum(succ_trans.values())
        unsucc_total = sum(unsucc_trans.values())
        succ_trans = {k: v/succ_total*100 for k, v in succ_trans.items()}
        unsucc_trans = {k: v/unsucc_total*100 for k, v in unsucc_trans.items()}

        all_transitions = set(list(succ_trans.keys()) + list(unsucc_trans.keys()))
        differences = []
        for trans in all_transitions:
            succ_pct = succ_trans.get(trans, 0)
            unsucc_pct = unsucc_trans.get(trans, 0)
            diff = succ_pct - unsucc_pct
            if abs(diff) > 0.5:
                from_aoi, to_aoi = trans
                differences.append({
                    'Transition': f"{PATTERN_AOI_MAP.get(from_aoi, from_aoi)} → {PATTERN_AOI_MAP.get(to_aoi, to_aoi)}",
                    'Difference': diff,
                    'Successful': succ_pct,
                    'Unsuccessful': unsucc_pct
                })
        return sorted(differences, key=lambda x: abs(x['Difference']), reverse=True)[:12]

    diff_data = get_transition_differences(successful_collapsed, unsuccessful_collapsed)
    diff_df = pd.DataFrame(diff_data)

    colors = [SUCCESS_COLOR if d > 0 else UNSUCCESS_COLOR for d in diff_df['Difference']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=diff_df['Transition'],
        x=diff_df['Difference'],
        orientation='h',
        marker_color=colors,
        text=[f'{d:+.1f}pp' for d in diff_df['Difference']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' +
                      'Successful: %{customdata[0]:.1f}%<br>' +
                      'Unsuccessful: %{customdata[1]:.1f}%<br>' +
                      'Difference: %{x:+.1f} pp<br>' +
                      '<extra></extra>',
        customdata=diff_df[['Successful', 'Unsuccessful']].values
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
    fig.update_layout(
        title='Scanning Behaviors: Success vs Failure<br>' +
              '<sub>Green = Successful do MORE | Red = Unsuccessful do MORE</sub>',
        xaxis_title='Difference in Transition Frequency (pp)',
        yaxis_title='',
        template='plotly_white',
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def build_scan_instrument_attention_figure():
    """Side-by-side view of instrument attention distribution and differences"""
    if not os.path.exists(PATTERN_DATA_PATH):
        return go.Figure().update_layout(title="Pattern data file not found")

    try:
        collapsed_xls = pd.ExcelFile(PATTERN_DATA_PATH)
        successful_collapsed = pd.read_excel(collapsed_xls, sheet_name='Succesful Excluding No AOI(A)')
        unsuccessful_collapsed = pd.read_excel(collapsed_xls, sheet_name='Unsuccesful Excluding No AOI(A)')
    except Exception as e:
        return go.Figure().update_layout(title=f"Error loading pattern data: {str(e)}")

    def calculate_instrument_attention(pattern_df):
        pattern_col = 'Pattern String'
        freq_col = 'Proportional Pattern Frequency'
        appearances, total = {}, 0

        for _, row in pattern_df.iterrows():
            pattern = str(row[pattern_col]).upper()
            frequency = row[freq_col]
            if pattern == 'nan':
                continue
            for aoi in pattern:
                appearances[aoi] = appearances.get(aoi, 0) + frequency
                total += frequency
        return {k: v/total*100 for k, v in appearances.items()}

    succ_attention = calculate_instrument_attention(successful_collapsed)
    unsucc_attention = calculate_instrument_attention(unsuccessful_collapsed)

    attention_data = []
    for aoi in PATTERN_AOI_MAP.keys():
        attention_data.append({
            'Instrument': PATTERN_AOI_MAP[aoi],
            'Successful': succ_attention.get(aoi, 0),
            'Unsuccessful': unsucc_attention.get(aoi, 0)
        })

    attention_df = pd.DataFrame(attention_data)
    attention_df['Difference'] = attention_df['Successful'] - attention_df['Unsuccessful']
    attention_df = attention_df.sort_values('Successful', ascending=False)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Instrument Attention Distribution', 'Difference (Successful - Unsuccessful)'),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.15,
        column_widths=[0.55, 0.45]
    )

    fig.add_trace(go.Bar(
        x=attention_df['Instrument'],
        y=attention_df['Successful'],
        name='Successful',
        marker_color=SUCCESS_COLOR,
        text=[f'{v:.1f}%' for v in attention_df['Successful']],
        textposition='outside'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=attention_df['Instrument'],
        y=attention_df['Unsuccessful'],
        name='Unsuccessful',
        marker_color=UNSUCCESS_COLOR,
        text=[f'{v:.1f}%' for v in attention_df['Unsuccessful']],
        textposition='outside'
    ), row=1, col=1)

    colors = [SUCCESS_COLOR if d > 0 else UNSUCCESS_COLOR for d in attention_df['Difference']]
    fig.add_trace(go.Bar(
        x=attention_df['Instrument'],
        y=attention_df['Difference'],
        marker_color=colors,
        showlegend=False,
        text=[f'{d:+.1f}pp' for d in attention_df['Difference']],
        textposition='outside'
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2, line_width=2)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="% of Fixations", row=1, col=1)
    fig.update_yaxes(title_text="Percentage Points", row=1, col=2)

    fig.update_layout(
        title='Which Instruments Get The Most Attention?',
        height=500,
        template='plotly_white',
        barmode='group'
    )
    return fig

def build_scan_efficiency_metrics_figure():
    """Grouped bar chart showing scan path efficiency metrics"""
    if not os.path.exists(PATTERN_DATA_PATH):
        return go.Figure().update_layout(title="Pattern data file not found")

    try:
        collapsed_xls = pd.ExcelFile(PATTERN_DATA_PATH)
        successful_collapsed = pd.read_excel(collapsed_xls, sheet_name='Succesful Excluding No AOI(A)')
        unsuccessful_collapsed = pd.read_excel(collapsed_xls, sheet_name='Unsuccesful Excluding No AOI(A)')
    except Exception as e:
        return go.Figure().update_layout(title=f"Error loading pattern data: {str(e)}")

    def analyze_scan_efficiency(pattern_df):
        pattern_col = 'Pattern String'
        freq_col = 'Proportional Pattern Frequency'
        total_backtracks, total_transitions = 0, 0
        pattern_lengths, unique_instruments = [], []

        for _, row in pattern_df.iterrows():
            pattern = str(row[pattern_col]).upper()
            frequency = row[freq_col]
            if pattern == 'nan':
                continue

            backtracks = sum(1 for i in range(len(pattern) - 2)
                           if pattern[i] == pattern[i+2] and pattern[i] != pattern[i+1])
            total_backtracks += backtracks * frequency
            total_transitions += (len(pattern) - 1) * frequency
            pattern_lengths.extend([len(pattern)] * int(frequency * 1000))
            unique_instruments.extend([len(set(pattern))] * int(frequency * 1000))

        backtrack_rate = (total_backtracks / total_transitions * 100) if total_transitions > 0 else 0
        avg_pattern_len = np.mean(pattern_lengths)
        avg_unique = np.mean(unique_instruments)
        efficiency = (avg_unique / avg_pattern_len * 100) if avg_pattern_len > 0 else 0

        return {
            'backtrack_rate': backtrack_rate,
            'avg_pattern_length': avg_pattern_len,
            'avg_unique_instruments': avg_unique,
            'efficiency_score': efficiency
        }

    succ_eff = analyze_scan_efficiency(successful_collapsed)
    unsucc_eff = analyze_scan_efficiency(unsuccessful_collapsed)

    metrics = ['Backtrack Rate (%)', 'Avg Pattern Length', 'Avg Unique Instruments', 'Efficiency Score (%)']
    succ_values = [succ_eff['backtrack_rate'], succ_eff['avg_pattern_length'],
                   succ_eff['avg_unique_instruments'], succ_eff['efficiency_score']]
    unsucc_values = [unsucc_eff['backtrack_rate'], unsucc_eff['avg_pattern_length'],
                     unsucc_eff['avg_unique_instruments'], unsucc_eff['efficiency_score']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics,
        y=succ_values,
        name='Successful',
        marker_color=SUCCESS_COLOR,
        text=[f'{v:.2f}' for v in succ_values],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        x=metrics,
        y=unsucc_values,
        name='Unsuccessful',
        marker_color=UNSUCCESS_COLOR,
        text=[f'{v:.2f}' for v in unsucc_values],
        textposition='outside'
    ))

    fig.update_layout(
        title='Scan Path Efficiency Comparison<br>' +
              '<sub>Lower backtrack = more decisive | Higher efficiency = better coverage</sub>',
        yaxis_title='Value',
        xaxis_title='',
        barmode='group',
        template='plotly_white',
        height=500
    )
    return fig

app = dash.Dash(__name__)

app.layout = html.Div([
    # ========== SECTION 1: AOI-SPECIFIC DASHBOARD ==========
    html.Div([
        html.Div([
            html.H3("AOI-Specific Analysis", style={"margin-bottom": "5px"}),
            html.Label("Select Area of Interest:", style={"font-weight": "bold"}),
            dcc.Dropdown(
                id="aoi-dropdown",
                options=[{"label": AOI_TITLES[aoi], "value": aoi} for aoi in AOI_LIST],
                value=AOI_LIST[0],
                clearable=False,
                style={"width": "85%", "margin-bottom": "15px", "fontWeight": "bold"}
            ),
            dcc.Graph(id="cockpit-image"),
        ], style={'flex': '1', 'min-width': "700px", 'padding': '20px', "background":"#f6f6fa"}),
        html.Div([
            dcc.Tabs(
                id="aoi-viz-tabs",
                value="tab-bar",
                children=[
                    dcc.Tab(label="Mean Fixation Bar Chart", value="tab-bar"),
                    dcc.Tab(label="Parallel Coordinates", value="tab-main"),
                ],
                style={"margin-bottom": "5px", "margin-top" : "5px"}
            ),
            dcc.Graph(id="aoi-graph", config={"displayModeBar": False}, style={"height": "600px"})
        ], style={'flex': '2', 'padding': '50px', 'overflow': 'auto'}),
    ], style={"display": "flex", "flexDirection": "row", "min-height":"50vh", "fontFamily":"Arial", "border-bottom": "3px solid #ccc"}),

    # ========== SECTION 2: CUMULATIVE METRICS DASHBOARD ==========
    html.Div([
        html.H2("Overall Performance Metrics", style={"text-align": "center", "margin": "30px 0 20px 0", "color": "#333"}),
        dcc.Tabs(
            id="cumulative-tabs",
            value="tab-saccade-metrics",
            children=[
                dcc.Tab(label="Saccade Metrics Comparison", value="tab-saccade-metrics"),
                dcc.Tab(label="Saccade Count by AOI", value="tab-saccade-count"),
                dcc.Tab(label="Scan Transition Differences", value="tab-scan-transitions"),
                dcc.Tab(label="Instrument Attention", value="tab-instrument-attention"),
                dcc.Tab(label="Scan Efficiency Metrics", value="tab-scan-efficiency"),
            ],
            style={"margin-bottom": "20px"}
        ),
        dcc.Graph(id="cumulative-graph", config={"displayModeBar": False}, style={"height": "700px"})
    ], style={"padding": "20px 50px 50px 50px", "background": "#fafafa", "min-height":"50vh"}),
], style={"fontFamily":"Arial"})

# Callback for AOI-specific dashboard
@app.callback(
    Output('cockpit-image', 'figure'),
    Output('aoi-graph', 'figure'),
    Input('aoi-dropdown', 'value'),
    Input('aoi-viz-tabs', 'value'),
)
def update_aoi_dashboard(selected_aoi, selected_tab):
    cockpit_fig = cockpit_figure(selected_aoi)

    if selected_tab == "tab-bar":
        viz_fig = build_bar_figure(selected_aoi)
    elif selected_tab == "tab-main":
        viz_fig = build_main_figure(selected_aoi)
    else:
        viz_fig = build_bar_figure(selected_aoi)

    return cockpit_fig, viz_fig

# Callback for cumulative metrics dashboard
@app.callback(
    Output('cumulative-graph', 'figure'),
    Input('cumulative-tabs', 'value'),
)
def update_cumulative_metrics(selected_tab):
    if selected_tab == "tab-saccade-metrics":
        return build_saccade_metrics_comparison_figure()
    elif selected_tab == "tab-saccade-count":
        return build_saccade_count_by_aoi_figure()
    elif selected_tab == "tab-scan-transitions":
        return build_scan_transition_differences_figure()
    elif selected_tab == "tab-instrument-attention":
        return build_scan_instrument_attention_figure()
    elif selected_tab == "tab-scan-efficiency":
        return build_scan_efficiency_metrics_figure()
    else:
        return build_saccade_metrics_comparison_figure()

if __name__ == "__main__":
    app.run(debug=True)
