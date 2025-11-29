import networkx as nx
import plotly.graph_objects as go

from summary_aoi import load_and_summarize


def build_aoi_hover_text(summary_df):
    """
    Build a dict: AOI -> hover text string with stats for each group.
    """
    hover = {}

    for aoi, group_df in summary_df.groupby("AOI"):
        # Start with AOI name
        lines = [f"<b>{aoi}</b>"]

        # Add info for each group sSuccessful Unsuccessful
        for _, row in group_df.iterrows():
            grp = row["Group"]
            pf = row["prop_fixations"]
            pdur = row["prop_fix_dur"]

            lines.append(
                f"<br><b>{grp}</b>"
                f"<br>Prop. fixations: {pf:.3f}"
                f"<br>Prop. fixation duration: {pdur:.3f}"
            )

        hover[aoi] = "".join(lines)

    return hover


def build_aoi_graph(csv_path):
    """
    Make a simple AOI network with one node per AOI.
    Hovering a node shows how much attention each group spends there.
    """
    # Get AOI summary table from the CSV
    summary = load_and_summarize(csv_path)

    # Create directed graph
    G = nx.DiGraph()

    # Add one node for each AOI
    for aoi in sorted(summary["AOI"].unique()):
        G.add_node(aoi)

    aoi_list = sorted(summary["AOI"].unique())
    for i in range(len(aoi_list)):
        src = aoi_list[i]
        tgt = aoi_list[(i + 1) % len(aoi_list)]
        G.add_edge(src, tgt)

    # Layout positions
    pos = nx.circular_layout(G)

    # Build edge traces for Plotly
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
        hoverinfo="none",
        line=dict(width=1),
    )

    # Build node traces
    hover_text_by_aoi = build_aoi_hover_text(summary)

    node_x = []
    node_y = []
    node_text = []
    hover_text = []

    for aoi in G.nodes():
        x, y = pos[aoi]
        node_x.append(x)
        node_y.append(y)
        node_text.append(aoi)
        hover_text.append(hover_text_by_aoi.get(aoi, aoi))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        hovertext=hover_text,
        marker=dict(
            size=20,
            line=dict(width=2),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="AOI Network – Hover for Successful vs Unsuccessful Attention",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig


if __name__ == "__main__":
    csv_path = "datasets/AOI_DGMs.csv"
    fig = build_aoi_graph(csv_path)
    fig.show()
