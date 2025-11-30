import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# AOI labels 
AOI_PREFIXES = {
    "NoAOI": "A - No AOI",
    "Alt_VSI": "B - Alt_VSI",
    "AI": "C - AI",
    "TI_HSI": "D - TI_HSI",
    "SSI": "E - SSI",
    "ASI": "F - ASI",
    "RPM": "G - RPM",
    "Window": "H - Window",
}

def normalize_success(value) -> str:
    """
    Convert pilot_success / flags into 'Successful' or 'Unsuccessful'.
    """
    s = str(value).strip().lower()
    if s in {"1", "successful", "success", "true", "yes"}:
        return "Successful"
    return "Unsuccessful"

def build_bar_figure(csv_path: str = "./datasets/AOI_DGMs.csv", success_filter: str = "all") -> go.Figure:
    """
    Build the grouped bar chart:
        x = AOI
        y = mean proportion of fixations
        color = Successful vs Unsuccessful
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_file)
    success_filter = success_filter.lower()
    records = []

    for prefix, label in AOI_PREFIXES.items():
        col = f"{prefix}_Proportion_of_fixations_spent_in_AOI"
        if col not in df.columns:
            continue

        for _, row in df.iterrows():
            status = normalize_success(row.get("pilot_success"))
            if success_filter in {"successful", "success"} and status != "Successful":
                continue
            if success_filter in {"unsuccessful", "fail", "failed"} and status != "Unsuccessful":
                continue

            records.append(
                {
                    "AOI": label,
                    "pilot_success": status,
                    "proportion_fixations": row[col],
                }
            )

    viz_df = pd.DataFrame(records)
    if viz_df.empty:
        raise ValueError("No data matched filter or no numeric proportion values found.")

    viz_df["proportion_fixations"] = (
        viz_df["proportion_fixations"]
        .astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
    )
    viz_df["proportion_fixations"] = pd.to_numeric(
        viz_df["proportion_fixations"], errors="coerce"
    )
    viz_df = viz_df.dropna(subset=["proportion_fixations"])

    summary_df = (
        viz_df
        .groupby(["AOI", "pilot_success"], as_index=False)["proportion_fixations"]
        .mean()
    )

    title_filter_part = "All" if success_filter == "all" else success_filter.capitalize()

    fig_bar = px.bar(
        summary_df,
        x="AOI",
        y="proportion_fixations",
        color="pilot_success",
        barmode="group",
        title=f"AOI Attention Distribution (Proportion of Fixations) – {title_filter_part}",
        labels={"proportion_fixations": "Mean Proportion of Fixations"},
    )

    fig_bar.update_traces(
        hovertemplate=(
            "AOI: %{x}<br>"
            "Group: %{marker.color}<br>"
            "Mean Proportion of Fixations: %{y:.3f}<br>"
            "<extra></extra>"
        )
    )

    return fig_bar

def select_dgms(df: pd.DataFrame, keywords: list, aoi: str) -> pd.DataFrame:
    """
    Select columns that contain ANY of the given keywords,
    restricted to a single AOI prefix (e.g. 'AI_').
    """
    keyword_pattern = "|".join(keywords)
    if aoi:
        full_pattern = rf"^{aoi}_.*({keyword_pattern})"
    else:
        full_pattern = rf"({keyword_pattern})"
    return df.filter(regex=full_pattern, axis=1)

def filter_success(df: pd.DataFrame):
    """
    Split root dataframe into two dataframes, successful and unsuccessful.
    """
    df_successful = df[df["pilot_success"] == "Successful"].copy()
    df_unsuccessful = df[df["pilot_success"] == "Unsuccessful"].copy()
    return df_successful, df_unsuccessful

def build_parcoords_figure(csv_path: str = "./datasets/AOI_DGMs.csv") -> go.Figure:
    """
    Build a single parallel coordinates figure with a dropdown
    that switches between AOIs, using the same logic as para_coords.
    """
    df = pd.read_csv(csv_path)

    df_successful, df_unsuccessful = filter_success(df)

    fixation_keywords = [
        "Proportion_of_fixations_spent_in_AOI",
        "Proportion_of_fixations_durations_spent_in_AOI",
        "Total_Number_of_Fixations",
        "Mean_fixation_duration",
    ]

    aoi_list = ["AI", "Alt_VSI", "ASI", "SSI", "TI_HSI", "RPM", "Window", "NoAOI"]
    aoi_traces = []
    trace_visibility_map = []

    for aoi in aoi_list:
        fix_df_successful = select_dgms(df_successful, fixation_keywords, aoi=aoi).copy()
        fix_df_unsuccessful = select_dgms(df_unsuccessful, fixation_keywords, aoi=aoi).copy()

        if fix_df_successful.empty and fix_df_unsuccessful.empty:
            continue

        fix_df_successful["pilot_success"] = df_successful["pilot_success"].values
        fix_df_unsuccessful["pilot_success"] = df_unsuccessful["pilot_success"].values

        result = pd.concat(
            [fix_df_successful, fix_df_unsuccessful],
            axis=0,
            ignore_index=True,
        )

        result["pilot_success_code"] = (result["pilot_success"] == "Successful").astype(int)

        numeric_cols = result.select_dtypes(include=np.number).columns.tolist()
        dims_cols = [c for c in numeric_cols if c != "pilot_success_code"]
        if not dims_cols:
            continue

        dimensions = [
            dict(label=col.replace("_", " "), values=result[col]) for col in dims_cols
        ]

        aoi_traces.append(
            {
                "aoi": aoi,
                "dimensions": dimensions,
                "color": result["pilot_success_code"].tolist(),
            }
        )
        trace_visibility_map.append(aoi)

    if not aoi_traces:
        raise ValueError("No AOI traces could be built from the CSV.")

    first = aoi_traces[0]
    fig_par = go.Figure()
    fig_par.add_trace(
        go.Parcoords(
            line=dict(
                color=first["color"],
                colorscale=[[0, "firebrick"], [1, "royalblue"]],
                showscale=False,
            ),
            dimensions=first["dimensions"],
        )
    )

    buttons = []
    for item in aoi_traces:
        buttons.append(
            dict(
                label=item["aoi"],
                method="update",
                args=[
                    {
                        "dimensions": [item["dimensions"]],
                        "line.color": [item["color"]],
                    },
                    {
                        "title": {
                            "text": f"Parallel Coordinates – AOI: {item['aoi']}"
                        }
                    },
                ],
            )
        )

    fig_par.update_layout(
        title=f"Parallel Coordinates – AOI: {trace_visibility_map[0]}",
        updatemenus=[
            dict(
                type="dropdown",
                active=0,
                x=1.05,
                y=1.15,
                xanchor="left",
                buttons=buttons,
            )
        ],
    )

    return fig_par

def build_dashboard(csv_path: str = "./datasets/AOI_DGMs.csv") -> go.Figure:
    """
    Create a single figure with:
      - left: parallel coordinates with AOI dropdown
      - right: grouped bar chart with hover tooltips
    """
    fig_par = build_parcoords_figure(csv_path)
    fig_bar = build_bar_figure(csv_path, success_filter="all")

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "parcoords"}, {"type": "xy"}]],
        subplot_titles=("", ""),
        horizontal_spacing=0.12,
    )

    fig.add_trace(fig_par.data[0], row=1, col=1)
    for trace in fig_bar.data:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(
        title_text="AOI Gaze Dashboard: Parallel Coordinates (left) + AOI Bar Chart (right)",
        title_font_size=22,
        title_y=0.96,  # Move title a little lower
        height=600,
        width=1200,
        showlegend=True,
        margin=dict(t=120, l=40, r=40, b=40)  # Increase top margin to add space
    )

    fig.add_annotation(
        text="Select AOI from dropdown above the left plot.<br />Hover over right bars to view tooltip metrics.",
        x=0.5, y=1.02, xref="paper", yref="paper",  # Place just below main title
        showarrow=False,
        font=dict(size=14),
        align="center"
    )
    
    return fig

if __name__ == "__main__":
    dash_fig = build_dashboard("./datasets/AOI_DGMs.csv")
    dash_fig.show()
    Path("outputs").mkdir(exist_ok=True)
    dash_fig.write_html("outputs/aoi_dashboard.html")
    print("Saved dashboard to outputs/aoi_dashboard.html")
