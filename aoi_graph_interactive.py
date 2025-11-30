import pandas as pd
import networkx as nx
import plotly.graph_objects as go


def load_aoi_summary(csv_path: str = "datasets/AOI_summary.csv") -> pd.DataFrame:
    """
    Load the AOI summary file created by summary_aoi.py.

    Expected columns:
        - AOI  (e.g., 'Alt_VSI', 'AI', 'TI_HSI', 'ASI', 'SSI', 'RPM', 'Window', 'NoAOI')
        - Group ('Successful' or 'Unsuccessful')
        - prop_fixations (float)

    Returns
    -------
    df : pd.DataFrame
    """
    df = pd.read_csv(csv_path)

    required_cols = {"AOI", "Group", "prop_fixations"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"AOI_summary.csv is missing columns: {missing}")

    return df


def build_hover_text(summary_df: pd.DataFrame) -> dict:
    """
    Build a mapping from AOI -> hover text string.

    Hover text example:

        AOI TI_HSI
        Successful: 0.2009
        Unsuccessful: 0.1708

    Returns
    -------
    hover_by_aoi : dict[str, str]
    """
    hover_by_aoi: dict[str, str] = {}

    for aoi, group_df in summary_df.groupby("AOI"):
        lines = [f"<b>{aoi}</b>"]
        for _, row in group_df.iterrows():
            group_name = row["Group"]
            value = row["prop_fixations"]
            lines.append(f"<br>{group_name}: {value:.4f}")
        hover_by_aoi[aoi] = "".join(lines)

    return hover_by_aoi


def build_aoi_graph(summary_df: pd.DataFrame) -> go.Figure:
    """
    Construct an AOI network visualization using NetworkX and Plotly.

    Nodes: AOIs
    Edges: simple ring connections between AOIs (visual only)
    """
    hover_by_aoi = build_hover_text(summary_df)

    # List AOIs
    aois = sorted(summary_df["AOI"].unique().tolist())

    # undirected graph
    G = nx.Graph()
    for aoi in aois:
        G.add_node(aoi)

    # (AOI[i] -> AOI[i+1]) + last -> firstt
    if len(aois) > 1:
        for i in range(len(aois)):
            src = aois[i]
            tgt = aois[(i + 1) % len(aois)]
            G.add_edge(src, tgt)

    # AOIs form a ring
    pos = nx.circular_layout(G)

    # Build edge traces
    edge_x = []
    edge_y = []
    for src, tgt in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
        showlegend=False,
    )

    # nodetrace 
    node_x = []
    node_y = []
    node_text = []
    node_hover = []

    for aoi in G.nodes():
        x, y = pos[aoi]
        node_x.append(x)
        node_y.append(y)
        node_text.append(aoi)
        node_hover.append(hover_by_aoi.get(aoi, aoi))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        hovertext=node_hover,
        marker=dict(
            size=20,
            line=dict(width=2),
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="AOI Graph – Proportion of Fixations (Successful vs Unsuccessful)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        plot_bgcolor="white",
    )

    return fig


def main():
    # Load the summary 
    summary_df = load_aoi_summary("datasets/AOI_summary.csv")

    # interactive AOI graph
    fig = build_aoi_graph(summary_df)

    
    fig.show()


if __name__ == "__main__":
    main()
