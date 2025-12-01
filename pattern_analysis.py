import os
import pandas as pd
import plotly.express as px
import plotly.io as pio

COLOR_MAP = {
    "Successful": "#2ecc71",
    "Unsuccessful": "#e74c3c",
}

COLOR_MAP_DIFF = {
    "Successful > Unsuccessful": "#2ecc71",
    "Unsuccessful > Successful": "#e74c3c",
}

COLLAPSED_FILE = "datasets/Collapsed Patterns (Group).xlsx"
EXPANDED_FILE = "datasets/Expanded Patterns (Group).xlsx"
OUTPUT_DIR = "outputs"
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "pattern_analysis_dashboard.html")

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
    )
    return fig


def length_hist(df: pd.DataFrame, title: str, top_n: int = 10):
    import plotly.graph_objects as go
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
        hovermode="closest"
    )

    return fig

def build_aoi_legend_html() -> str:
    """
    AOI mapping + short explanations of Proportional Pattern Frequency
    and Difference, with simple examples.
    """

    rows = []
    for letter, desc in AOI_LEGEND.items():
        rows.append(f"<tr><td><strong>{letter}</strong></td><td>{desc}</td></tr>")

    html = f"""
    <div style="display:flex; gap:32px; flex-wrap:wrap; align-items:flex-start;">
      <div>
        <h3>AOI Letters</h3>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-collapse:collapse; max-width:400px;">
          <thead>
            <tr>
              <th>Letter</th>
              <th>AOI</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
      <div style="max-width:420px;">
        <h3>Pattern Metrics</h3>
        <ul>
          <li>
            <b>Proportional Pattern Frequency</b>: share of all patterns for a group
            that match a specific pattern.<br>
            Example: 0.08 for pattern <b>ADA</b> means 8% of all patterns
            for that group are ADA.
          </li>
          <li style="margin-top:8px;">
            <b>Proportional Frequency</b> (Success - Unsuccess): how much more
            common a pattern is for successful pilots.<br>
            Example: +0.03 for <b>ADA</b> means ADA is 3 percentage
            points more common in successful than unsuccessful approaches
            (green bars = more in successful, red = more in unsuccessful).
          </li>
        </ul>
      </div>

    </div>
    """
    return html

def main():
    if not os.path.exists(COLLAPSED_FILE):
        raise FileNotFoundError(f"Missing file: {COLLAPSED_FILE}")
    if not os.path.exists(EXPANDED_FILE):
        raise FileNotFoundError(f"Missing file: {EXPANDED_FILE}")

    collapsed_df = load_patterns(COLLAPSED_FILE, pattern_type="Collapsed")
    expanded_df = load_patterns(EXPANDED_FILE, pattern_type="Expanded")

    collapsed_all = collapsed_df[collapsed_df["AOI_Filter"] == "All AOIs"].copy()
    expanded_all = expanded_df[expanded_df["AOI_Filter"] == "All AOIs"].copy()

    figs = []

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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    html_parts = []
    for i, fig in enumerate(figs):
        include_js = "cdn" if i == 0 else False
        fragment = pio.to_html(
            fig,
            include_plotlyjs=include_js,
            full_html=False,
        )
        html_parts.append(fragment)

    legend_html = build_aoi_legend_html()

    full_html = (
        "<html><head><meta charset='utf-8'>"
        "<title>Pattern Analysis Dashboard</title></head><body>\n"
    )
    full_html += "<h1>Pattern Analysis: Successful vs Unsuccessful Approaches</h1>\n"
    full_html += legend_html
    full_html += "<hr>\n"
    full_html += "<hr>\n".join(html_parts)
    full_html += "\n</body></html>"

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved dashboard to: {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
