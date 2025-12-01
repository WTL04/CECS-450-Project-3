import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def load_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={"pattern string": "Pattern"})
    df["Metric"] = df["proportional pattern frequency"].astype(float)
    df = df[~df["Pattern"].str.fullmatch(r"A+")]

    return df

def plot_top(df_s: pd.DataFrame, df_u: pd.DataFrame, title: str) -> str:
    df_s2 = df_s.copy()
    df_s2["Group"] = "Successful"

    df_u2 = df_u.copy()
    df_u2["Group"] = "Unsuccessful"

    both = pd.concat([df_s2, df_u2], ignore_index=True)
    both = both.sort_values("Metric", ascending=False).head(10)

    fig = px.bar(
        both,
        x="Pattern",
        y="Metric",
        color="Group",
        barmode="group",
        title=title,
        hover_data={"Metric": ":.6f"}
    )

    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f7fafc",
        xaxis_title="Pattern",
        yaxis_title="Proportional Pattern Frequency",
        hovermode="closest",
        title_x=0.5,
        showlegend=True,
        legend=dict(
            x=0.95,           
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=0.5
        ),
        margin=dict(r=140)
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def plot_diff(df_s: pd.DataFrame, df_u: pd.DataFrame, title: str) -> str:
    s = df_s[["Pattern", "Metric"]].rename(columns={"Metric": "Success"})
    u = df_u[["Pattern", "Metric"]].rename(columns={"Metric": "Unsuccessful"})

    merged = pd.merge(s, u, on="Pattern", how="outer").fillna(0.0)
    merged["Diff"] = merged["Success"] - merged["Unsuccessful"]
    pos = merged[merged["Diff"] > 0].sort_values("Diff", ascending=False).head(5)
    neg = merged[merged["Diff"] < 0].sort_values("Diff", ascending=True).head(5)

    final = pd.concat([pos, neg], ignore_index=True)
    final["Color"] = final["Diff"].apply(
        lambda x: "#2f855a" if x > 0 else "#e53e3e"
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=final["Pattern"],
        y=final["Diff"],
        marker_color=final["Color"],
        text=final["Diff"].round(6),
        textposition="outside",
        customdata=final[["Success", "Unsuccessful"]],
        hovertemplate=(
            "<b>Pattern:</b> %{x}<br>"
            "<b>Difference:</b> %{y:.6f}<extra></extra>"
        ),
        showlegend=False
    ))
    fig.update_layout(
        title=title,
        yaxis_title="Difference (Success – Unsuccessful)",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f7fafc",
        hovermode="closest",
        title_x=0.5,
        showlegend=True,
        legend=dict(
            x=0.98,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1
        ),
        margin=dict(r=150)
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")

    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color="#2f855a",
        name="Successful > Unsuccessful",
        showlegend=True
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color="#e53e3e",
        name="Unsuccessful > Successful",
        showlegend=True
    ))

    return fig.to_html(full_html=False, include_plotlyjs=False)

def run_mode(path: Path, collapsed: bool, exclude_noaoi: bool, html_parts: list):
    mode_label = "Collapsed" if collapsed else "Expanded"
    aoi_label = "Excluding No AOI" if exclude_noaoi else "Including No AOI"

    sheet_s = "Succesful Excluding No AOI(A)" if exclude_noaoi else "Succesful"
    sheet_u = "Unsuccesful Excluding No AOI(A)" if exclude_noaoi else "Unsuccesful"

    df_s = load_sheet(path, sheet_s)
    df_u = load_sheet(path, sheet_u)

    html_parts.append(f"<h2>{mode_label} ({aoi_label})</h2>")

    html_parts.append(f"<h3>Top 10 Patterns</h3>")
    html_parts.append(
        plot_top(
            df_s,
            df_u,
            f"Top 10 Patterns — {mode_label} ({aoi_label})"
        )
    )

    html_parts.append(f"<h3>Pattern Differences</h3>")
    html_parts.append(
        plot_diff(
            df_s,
            df_u,
            f"Pattern Differences — {mode_label} ({aoi_label})"
        )
    )

def main():
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    html = []
    html.append("<html><head><title>Pattern Analysis Dashboard</title>")

    html.append('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')

    html.append("</head><body>")
    html.append("<h1>Pattern Analysis Dashboard</h1>")

    COLLAPSED = Path("datasets") / "Collapsed Patterns (Group).xlsx"
    EXPANDED = Path("datasets") / "Expanded Patterns (Group).xlsx"

    run_mode(COLLAPSED, collapsed=True,  exclude_noaoi=False, html_parts=html)
    run_mode(COLLAPSED, collapsed=True,  exclude_noaoi=True,  html_parts=html)
    run_mode(EXPANDED,  collapsed=False, exclude_noaoi=False, html_parts=html)
    run_mode(EXPANDED,  collapsed=False, exclude_noaoi=True,  html_parts=html)

    html.append("</body></html>")

    output_file = outdir / "pattern_analysis_dashboard.html"
    output_file.write_text("\n".join(html), encoding="utf-8")

    print(f"Dashboard saved to {output_file}")


if __name__ == "__main__":
    main()
