import pandas as pd
import plotly.graph_objs as go

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
    s = str(value).strip().lower()
    if s in {"1", "successful", "success", "true", "yes"}:
        return "Successful"
    return "Unsuccessful"

def build_bar_figure(aoi: str, csv_path: str = "./datasets/AOI_DGMs.csv") -> go.Figure:
    """
    Returns a bar chart (Plotly Figure) showing mean proportion of fixations
    for the requested AOI, for successful/unsuccessful pilots.
    """
    df = pd.read_csv(csv_path)
    prefix_label = AOI_PREFIXES.get(aoi, aoi)
    col = f"{aoi}_Proportion_of_fixations_spent_in_AOI"
    if col not in df.columns:
        # Handle missing AOI column gracefully
        fig = go.Figure()
        fig.update_layout(title=f"{prefix_label}: No data column found")
        return fig

    # Collect data per pilot/success status
    success_vals = []
    unsuccess_vals = []
    for _, row in df.iterrows():
        status = normalize_success(row.get("pilot_success"))
        value = row.get(col)
        # Clean up string percent/NaN/etc
        try:
            value = float(str(value).replace("%", "").strip())
        except Exception:
            continue
        if status == "Successful":
            success_vals.append(value)
        elif status == "Unsuccessful":
            unsuccess_vals.append(value)

    y_success = sum(success_vals) / len(success_vals) if success_vals else 0
    y_unsuccess = sum(unsuccess_vals) / len(unsuccess_vals) if unsuccess_vals else 0

    fig = go.Figure([
        go.Bar(
            x=["Successful", "Unsuccessful"],
            y=[y_success, y_unsuccess],
            marker_color=["royalblue", "firebrick"]
        )
    ])

    fig.update_layout(
        title=f"{prefix_label}: Mean Proportion of Fixations",
        yaxis_title="Mean Proportion",
        xaxis_title="Pilot Success"
    )
    return fig
