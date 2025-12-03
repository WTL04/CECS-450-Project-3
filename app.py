import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
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
COLLAPSED_FILE = "./datasets/Collapsed Patterns (Group).xlsx"
EXPANDED_FILE = "./datasets/Expanded Patterns (Group).xlsx"

SUCCESS_COLOR = "#2ecc71" 
UNSUCCESS_COLOR = "#e74c3c" 
COLOR_MAP = {
    "Successful": SUCCESS_COLOR,
    "Unsuccessful": UNSUCCESS_COLOR,
}

COLOR_MAP_DIFF = {
    "Successful > Unsuccessful": SUCCESS_COLOR,
    "Unsuccessful > Successful": UNSUCCESS_COLOR,
}

PATTERN_AOI_MAP = {
    'B': 'Alt_VSI',
    'C': 'AI',
    'D': 'TI_HSI',
    'E': 'SSI',
    'F': 'ASI',
    'G': 'RPM'
}

AOI_LEGEND = {
    "A": "No AOI",
    "B": "Alt_VSI",
    "C": "AI",
    "D": "TI_HSI",
    "E": "SSI",
    "F": "ASI",
    "G": "RPM",
    "H": "Window",
}

def load_patterns(excel_path: str, pattern_type: str) -> pd.DataFrame:
    xls = pd.ExcelFile(excel_path)
    frames = []

    for sheet in xls.sheet_names:
        df = xls.parse(sheet)

        if "Succesful" in sheet:
            group = "Successful"
        else:
            group = "Unsuccessful"

        if "Excluding" in sheet:
            aoi_filter = "Exclude No AOI (A)"
        else:
            aoi_filter = "All AOIs"
        df["Group"] = group
        df["AOI_Filter"] = aoi_filter
        df["PatternType"] = pattern_type
        df["Pattern Length"] = df["Pattern String"].astype(str).str.len()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def top_patterns_bar(df: pd.DataFrame, title: str, top_n: int = 15):
    totals = (
        df.groupby("Pattern String")["Frequency"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    top_patterns = list(totals.index)

    plot_df = df[df["Pattern String"].isin(top_patterns)].copy()
    plot_df["Pattern String"] = pd.Categorical(
        plot_df["Pattern String"],
        categories=top_patterns,
        ordered=True,
    )

    fig = px.bar(
        plot_df,
        x="Pattern String",
        y="Proportional Pattern Frequency",
        color="Group",
        barmode="group",
        title=title,
        labels={
            "Pattern String": "AOI Pattern",
            "Proportional Pattern Frequency": "Proportional Pattern Frequency",
            "Group": "Approach Outcome",
        },
    )
    for trace in fig.data:
        group_name = trace.name
        if group_name in COLOR_MAP:
            trace.marker.color = COLOR_MAP[group_name]

    fig.update_layout(
        xaxis_tickangle=-45,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig

def difference_bar(df: pd.DataFrame, title: str, top_each_side: int = 10):
    pivot = df.pivot_table(
        index="Pattern String",
        columns="Group",
        values="Proportional Pattern Frequency",
        aggfunc="mean",
        fill_value=0.0,
    )

    if "Successful" not in pivot.columns or "Unsuccessful" not in pivot.columns:
        return px.bar(title=title)

    pivot["Difference"] = pivot["Successful"] - pivot["Unsuccessful"]
    pivot = pivot[pivot["Difference"] != 0]
    top_pos = pivot.sort_values("Difference", ascending=False).head(top_each_side)
    top_neg = pivot.sort_values("Difference", ascending=True).head(top_each_side)

    subset = pd.concat([top_pos, top_neg]).reset_index()
    subset["Direction"] = subset["Difference"].apply(
        lambda v: "Successful > Unsuccessful" if v > 0 else "Unsuccessful > Successful"
    )

    fig = px.bar(
        subset,
        x="Pattern String",
        y="Difference",
        color="Direction",
        title=title,
        labels={
            "Pattern String": "AOI Pattern",
            "Difference": "Proportional Frequency (Success - Unsuccess)",
            "Direction": "Which Group Looks Here More",
        },
        color_discrete_map=COLOR_MAP_DIFF,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig

def length_hist(df: pd.DataFrame, title: str, top_n: int = 10):
    totals = (
        df.groupby(["Pattern Length", "Group"])["Frequency"]
        .sum()
        .reset_index(name="TotalFrequency")
    )

    df_sorted = df.sort_values(
        ["Pattern Length", "Group", "Frequency"],
        ascending=[True, True, False]
    )

    top_patterns = (
        df_sorted.groupby(["Pattern Length", "Group"], group_keys=False)
        .head(top_n)
        .groupby(["Pattern Length", "Group"])
        .apply(lambda g: "<br>".join(
            [f"{row['Pattern String']} ({row['Frequency']})"
             for _, row in g.iterrows()]
        ))
        .reset_index(name="PatternDetails")
    )

    merged = totals.merge(
        top_patterns,
        on=["Pattern Length", "Group"],
        how="left"
    )

    success_df = merged[merged["Group"] == "Successful"]
    unsuccess_df = merged[merged["Group"] == "Unsuccessful"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Successful",
        x=success_df["Pattern Length"],
        y=success_df["TotalFrequency"],
        customdata=success_df["PatternDetails"],
        marker_color=COLOR_MAP["Successful"],
        hovertemplate=(
            "<b>Group:</b> Successful<br>"
            "<b>Pattern Length:</b> %{x}<br>"
            "<b>Total Occurrences:</b> %{y}<br><br>"
            "<b>Top Patterns:</b><br>%{customdata}"
            "<extra></extra>"
        )
    ))

    fig.add_trace(go.Bar(
        name="Unsuccessful",
        x=unsuccess_df["Pattern Length"],
        y=unsuccess_df["TotalFrequency"],
        customdata=unsuccess_df["PatternDetails"],
        marker_color=COLOR_MAP["Unsuccessful"],
        hovertemplate=(
            "<b>Group:</b> Unsuccessful<br>"
            "<b>Pattern Length:</b> %{x}<br>"
            "<b>Total Occurrences:</b> %{y}<br><br>"
            "<b>Top Patterns:</b><br>%{customdata}"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Pattern Length (# AOIs in pattern)",
        yaxis_title="Total Pattern Occurrences",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig

def build_aoi_legend_dash():
    table_rows = [
        html.Tr([
            html.Td(html.Strong(letter)),
            html.Td(desc)
        ])
        for letter, desc in AOI_LEGEND.items()
    ]

    return html.Div(
        style={
            "display": "flex",
            "gap": "32px",
            "flexWrap": "wrap",
            "alignItems": "flex-start",
            "marginTop": "40px",
            "borderTop": "1px solid #ccc",
            "paddingTop": "20px",
        },
        children=[
            html.Div([
                html.H3("AOI Letters"),
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Letter"),
                            html.Th("AOI"),
                        ])
                    ),
                    html.Tbody(table_rows),
                ], style={
                    "borderCollapse": "collapse",
                    "maxWidth": "400px",
                    "border": "1px solid #ddd",
                })
            ]),
            html.Div([
                html.H3("Pattern Metrics"),
                html.Ul([
                    html.Li([
                        html.B("Proportional Pattern Frequency"),
                        ": share of all patterns for a group that match a specific pattern. ",
                        html.Br(),
                        "Example: 0.08 for pattern ",
                        html.B("ADA"),
                        " means 8% of all patterns for that group are ADA."
                    ]),
                    html.Li([
                        html.B("Proportional Frequency (Success - Unsuccess)"),
                        ": how much more common a pattern is for successful pilots.",
                        html.Br(),
                        "Example: +0.03 for ",
                        html.B("ADA"),
                        " means ADA is 3 percentage points more common in successful than unsuccessful approaches ",
                        "(green bars = more in successful, red = more in unsuccessful)."
                    ], style={"marginTop": "8px"}),
                ], style={"maxWidth": "420px"})
            ])
        ]
    )

def build_pattern_figures():
    figs = []

    if not os.path.exists(COLLAPSED_FILE) or not os.path.exists(EXPANDED_FILE):
        placeholder = go.Figure().update_layout(
            title="Pattern analysis files not found in datasets/"
        )
        return [placeholder]

    collapsed_df = load_patterns(COLLAPSED_FILE, pattern_type="Collapsed")
    expanded_df = load_patterns(EXPANDED_FILE, pattern_type="Expanded")

    collapsed_all = collapsed_df[collapsed_df["AOI_Filter"] == "All AOIs"].copy()
    expanded_all = expanded_df[expanded_df["AOI_Filter"] == "All AOIs"].copy()

    figs.append(
        top_patterns_bar(
            collapsed_all,
            "Top 15 Collapsed Patterns by Proportional Frequency (All AOIs)",
        )
    )
    figs.append(
        top_patterns_bar(
            expanded_all,
            "Top 15 Expanded Patterns by Proportional Frequency (All AOIs)",
        )
    )
    figs.append(
        difference_bar(
            collapsed_all,
            "Collapsed Patterns: Largest Differences in Proportional Frequency "
            "(Successful − Unsuccessful)",
        )
    )
    figs.append(
        difference_bar(
            expanded_all,
            "Expanded Patterns: Largest Differences in Proportional Frequency "
            "(Successful − Unsuccessful)",
        )
    )
    figs.append(
        length_hist(
            collapsed_all,
            "Collapsed Patterns: Pattern Length Distribution (Weighted by Frequency)",
        )
    )
    figs.append(
        length_hist(
            expanded_all,
            "Expanded Patterns: Pattern Length Distribution (Weighted by Frequency)",
        )
    )
    return figs

PATTERN_FIGURES = build_pattern_figures()

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def cockpit_figure(selected_aoi):
    SCALE = 0.7
    encoded_image = get_base64_image(COCKPIT_IMAGE_PATH)
    fig = go.Figure()

    scaled_w = IMAGE_WIDTH * SCALE
    scaled_h = IMAGE_HEIGHT * SCALE

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
    if not os.path.exists(DATA_PATH):
        return go.Figure().update_layout(title="CSV Not Found")
    df = pd.read_csv(DATA_PATH)

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
    if 'Approach_Score' not in df.columns:
        return go.Figure().update_layout(title="Missing Approach_Score column")

    df['Success_Category'] = df['Approach_Score'].apply(
        lambda x: 'Successful' if pd.notna(x) and x >= 0.7 else 'Unsuccessful'
    )
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

def build_comprehensive_metrics_heatmap():
    if not os.path.exists(DATA_PATH):
        return go.Figure().update_layout(title="CSV Not Found for Heatmap")

    df = pd.read_csv(DATA_PATH)
    aois = ["AI", "Alt_VSI", "ASI", "SSI", "TI_HSI", "RPM", "Window"]
    metrics = [
        ("Proportion_of_fixations_spent_in_AOI", "Attention Distribution"),
        ("Proportion_of_fixations_durations_spent_in_AOI", "Attention Duration"),
        ("Mean_fixation_duration_s", "Mean Fixation Time (s)"),
        ("Total_Number_of_Fixations", "Fixation Count"),
        ("fixation_to_saccade_ratio", "Fixation/Saccade Ratio"),
        ("mean_saccade_duration", "Saccade Duration (s)"),
    ]

    successful = df[df["pilot_success"] == "Successful"]
    unsuccessful = df[df["pilot_success"] == "Unsuccessful"]

    heatmap_data = []
    y_labels = []

    for metric_suffix, metric_label in metrics:
        for success_group, group_label in [(successful, "Successful"), (unsuccessful, "Unsuccessful")]:
            row_data = []
            for aoi in aois:
                col_name = f"{aoi}_{metric_suffix}"
                if col_name in df.columns:
                    mean_val = pd.to_numeric(success_group[col_name], errors="coerce").mean()
                    row_data.append(mean_val)
                else:
                    row_data.append(np.nan)

            heatmap_data.append(row_data)
            y_labels.append(f"{metric_label} - {group_label}")
        diff_row = []
        succ_row = heatmap_data[-2]
        unsucc_row = heatmap_data[-1]

        for s_val, u_val in zip(succ_row, unsucc_row):
            if not np.isnan(s_val) and not np.isnan(u_val):
                diff_row.append(s_val - u_val)
            else:
                diff_row.append(np.nan)

        heatmap_data.append(diff_row)
        y_labels.append(f"{metric_label} - Δ (S - U)")
    heatmap_array = np.array(heatmap_data)
    hover_text = []
    for i, row in enumerate(heatmap_data):
        hover_row = []
        for j, val in enumerate(row):
            if np.isnan(val):
                hover_row.append("N/A")
            else:
                hover_row.append(f"{val:.4f}")
        hover_text.append(hover_row)

    x_labels = [pretty_aoi(aoi) for aoi in aois]

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlGn',
        text=hover_text,
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="<b>%{x}</b><br>%{y}<br>Value: %{text}<extra></extra>",
        colorbar=dict(
            title=dict(text="Metric Value", side="right"),
            len=0.7
        ),
        zmin=-np.nanmax(np.abs(heatmap_array)), 
        zmax=np.nanmax(np.abs(heatmap_array)),
    ))

    shapes = []
    for i in range(3, len(y_labels), 3):
        shapes.append(dict(
            type='line',
            x0=-0.5, x1=len(x_labels)-0.5,
            y0=i-0.5, y1=i-0.5,
            line=dict(color='black', width=2)
        ))

    fig.update_layout(
        title='Comprehensive AOI Performance Metrics: Successful vs Unsuccessful Pilots<br>' +
              '<sub>Rows show: metric for successful pilots, unsuccessful pilots, and difference (Δ = S - U)<br>' +
              'Green Δ = successful pilots higher | Red Δ = unsuccessful pilots higher</sub>',
        xaxis_title="Area of Interest",
        yaxis_title="",
        height=900,
        width=1200,
        margin=dict(l=250, r=150, t=120, b=80),
        font=dict(size=10),
        template='plotly_white',
        shapes=shapes
    )

    fig.update_yaxes(autorange="reversed")

    return fig

app = dash.Dash(__name__)

app.layout = html.Div([
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

    html.Div([
        html.H2("Overall Performance Metrics", style={"text-align": "center", "margin": "30px 0 20px 0", "color": "#333"}),
        dcc.Tabs(
            id="cumulative-tabs",
            value="tab-comprehensive-heatmap",
            children=[
                dcc.Tab(label="Comprehensive Metrics Heatmap", value="tab-comprehensive-heatmap"),
                dcc.Tab(label="Saccade Metrics Comparison", value="tab-saccade-metrics"),
                dcc.Tab(label="Saccade Count by AOI", value="tab-saccade-count"),
                dcc.Tab(label="Scan Transition Differences", value="tab-scan-transitions"),
                dcc.Tab(label="Instrument Attention", value="tab-instrument-attention"),
                dcc.Tab(label="Scan Efficiency Metrics", value="tab-scan-efficiency"),
            ],
            style={"margin-bottom": "20px"}
        ),
        dcc.Graph(id="cumulative-graph", config={"displayModeBar": True}, style={"height": "900px"})
    ], style={"padding": "20px 50px 50px 50px", "background": "#fafafa", "min-height":"50vh"}),

    html.Div([
        html.H2(
            "Scan Pattern Analysis",
            style={"textAlign": "center", "margin": "30px 0 10px 0", "color": "#333"}
        ),
        html.P(
            "These charts show which scan patterns are most common, how they differ between "
            "successful and unsuccessful approaches, and how long those patterns are.",
            style={"textAlign": "center", "maxWidth": "900px", "margin": "0 auto 30px auto", "color": "#555"}
        ),
        dcc.Graph(
            id="pattern-fig-1",
            figure=PATTERN_FIGURES[0] if len(PATTERN_FIGURES) > 0 else go.Figure(),
            style={"height": "550px", "marginBottom": "30px"}
        ),
        dcc.Graph(
            id="pattern-fig-2",
            figure=PATTERN_FIGURES[1] if len(PATTERN_FIGURES) > 1 else go.Figure(),
            style={"height": "550px", "marginBottom": "30px"}
        ),
        dcc.Graph(
            id="pattern-fig-3",
            figure=PATTERN_FIGURES[2] if len(PATTERN_FIGURES) > 2 else go.Figure(),
            style={"height": "550px", "marginBottom": "30px"}
        ),
        dcc.Graph(
            id="pattern-fig-4",
            figure=PATTERN_FIGURES[3] if len(PATTERN_FIGURES) > 3 else go.Figure(),
            style={"height": "550px", "marginBottom": "30px"}
        ),
        dcc.Graph(
            id="pattern-fig-5",
            figure=PATTERN_FIGURES[4] if len(PATTERN_FIGURES) > 4 else go.Figure(),
            style={"height": "550px", "marginBottom": "30px"}
        ),
        dcc.Graph(
            id="pattern-fig-6",
            figure=PATTERN_FIGURES[5] if len(PATTERN_FIGURES) > 5 else go.Figure(),
            style={"height": "550px", "marginBottom": "30px"}
        ),
        build_aoi_legend_dash(),
    ], style={"padding": "20px 50px 60px 50px", "background": "#ffffff", "min-height":"50vh"}),
], style={"fontFamily":"Arial"})

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

@app.callback(
    Output('cumulative-graph', 'figure'),
    Input('cumulative-tabs', 'value'),
)
def update_cumulative_metrics(selected_tab):
    if selected_tab == "tab-comprehensive-heatmap":
        return build_comprehensive_metrics_heatmap()
    elif selected_tab == "tab-saccade-metrics":
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
        return build_comprehensive_metrics_heatmap()

if __name__ == "__main__":
    app.run(debug=True)
