import networkx as nx
import plotly.graph_objects as go
import pandas as pd

from summary_aoi import load_and_summarize


def build_hover_text(summary_df):
    """
    Create the hover text for each AOI using the summary statistics.
    """
    hover = {}

    for aoi, group_df in summary_df.groupby("AOI"):
        text_lines = [f"<b>AOI: {aoi}</b>"]

        for group_name, row in group_df.set_index("Group").iterrows():
            text_lines.append(f"<br><b>{group_name}</b>")

            if "fixation_count" in row:
                text_lines.append(f"<br>Avg fixations: {row['fixation_count']:.2f}")

            if "mean_fix_dur" in row:
                text_lines.append(f"<br>Avg fixation duration: {row['mean_fix_dur']:.3f}")

            if "prop_fixations" in row:
                text_lines.append(f"<br>Fixation proportion: {row['prop_fixations']:.3f}")

            if "prop_fix_dur" in row:
                text_lines.append(f"<br>Duration proportion: {row['prop_fix_dur']:.3f}")

        hover[aoi] = "".join(text_lines)

    return hover


def build_aoi_graph(csv_path):
    """
    Build a simple AOI network graph with hover text for each node.
    """
    # Load AOI-level summary data.
    summary = load_and_summarize(csv_path)
    hover_text = build_hover_text(summary)

    # Create a graph with one node per AOI.
    G = nx.Graph()
    for aoi in summary["AOI"].unique():
        G.add_node(aoi)

    # Layout the nodes.
    pos = nx.circular_layout(G)

    # Extract node positions.
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    texts = [hover_text.get(node, f"AOI: {node}") for node in G.nodes()]

    # Build Plotly scatter for nodes.
    node_trace = go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        hoverinfo="text",
        hovertext=texts,
        marker=dict(size=20, line=dict(width=2))
    )

    fig = go.Figure(data=[node_trace])
    fig.update_layout(
        title="AOI Graph with Hover Details",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return fig


if __name__ == "__main__":
    # Update this path to match your dataset.
    aoi_csv = "datasets/OPTION_A_AOI_DGMs.csv"
    fig = build_aoi_graph(aoi_csv)
    fig.show()
# builds the AOI network graph and adds hover text for each AOI.
# The hover text shows summary for successful and unsuccessful pilots.
