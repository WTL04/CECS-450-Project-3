import pandas as pd
import plotly.graph_objects as go

# List of AOIs for the dashboard/app dropdowns and checks
AOI_LIST = [
    "AI", "Alt_VSI", "ASI", "SSI", "TI_HSI", "RPM", "Window", "NoAOI"
]

def normalize_success(value) -> int:
    """
    Convert pilot_success to integer: Successful (1, blue), Unsuccessful (0, red)
    Supports values like "successful", "unsuccessful", 1/0, yes/no, true/false.
    """
    s = str(value).strip().lower()
    if s in {"1", "successful", "success", "true", "yes"}:
        return 1
    return 0

def build_main_figure(aoi: str, csv_path: str = "./datasets/AOI_DGMs.csv") -> go.Figure:
    """
    Creates a parallel coordinates Plotly figure for the selected AOI.
    Color-codes lines: blue for Successful, red for Unsuccessful pilots.
    Dimensions shown:
        - Total Number of Fixations
        - Mean fixation duration s
        - Proportion of fixations spent in AOI
        - Proportion of fixations durations spent in AOI
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error loading CSV: {e}")
        return fig

    # Define expected AOI-specific columns 
    columns = [
        f"{aoi} Total Number of Fixations",
        f"{aoi} Mean fixation duration s",
        f"{aoi} Proportion of fixations spent in AOI",
        f"{aoi} Proportion of fixations durations spent in AOI"
    ]

    # Only include columns that exist in the data
    plot_cols = [c for c in columns if c in df.columns]
    if len(plot_cols) < 2:
        fig = go.Figure()
        fig.update_layout(title=f"Not enough data for AOI: {aoi}")
        return fig

    # labels and values
    dims = []
    for c in plot_cols:
        vec = pd.to_numeric(df[c], errors="coerce")
        dims.append(dict(
            label=c,
            values=vec
        ))

    # fill missing as Unsuccessful
    if "pilot_success" in df.columns:
        color_vec = df["pilot_success"].map(normalize_success).fillna(0).values
    else:
        color_vec = [0] * len(df)  # default to Unsuccessful if missing

    fig = go.Figure()
    fig.add_trace(go.Parcoords(
        line=dict(
            color=color_vec,
            colorscale=[
                [0, 'firebrick'],   # Unsuccessful (red)
                [1, 'royalblue']    # Successful (blue)
            ],
            colorbar=dict(
                title="Pilot Success",
                tickvals=[0, 1],
                ticktext=["Unsuccessful", "Successful"]
            )
        ),
        dimensions=dims
    ))

    fig.update_layout(
        title=f"Parallel Coordinates – AOI: {aoi}",
        margin=dict(t=60)
    )
    return fig


if __name__ == "__main__":
    # Show plot for the default AOI (e.g., "SSI")
    fig = build_main_figure("SSI")
    fig.show()



