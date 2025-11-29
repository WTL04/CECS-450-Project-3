from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



DATA_DIR = Path("datasets")
OUTPUT_DIR = Path("outputs")

COLLAPSED_XLSX = DATA_DIR / "Collapsed Patterns (Group).xlsx"
EXPANDED_XLSX = DATA_DIR / "Expanded Patterns (Group).xlsx"

TOP_N = 10


def _find_sheet_name(xls: pd.ExcelFile, target: str) -> str | None:
    """
    Map expected target names to the actual sheet names.
    Handles the misspellings "Succesful" and "Unsuccesful".
    """

    mapping = {
        "successful": "Succesful",
        "unsuccessful": "Unsuccesful",
        "successfulexcludingnoaoia": "Succesful Excluding No AOI(A)",
        "unsuccessfulexcludingnoaoia": "Unsuccesful Excluding No AOI(A)",
    }

    key = (
        target.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("(a)", "a")
    )

    if key in mapping:
        return mapping[key]

    for s in xls.sheet_names:
        s_clean = (
            s.lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("(a)", "a")
        )
        if s_clean == key:
            return s

    return None



def load_pattern_sheet(xlsx_path: Path, group: str, exclude_no_aoi: bool) -> pd.DataFrame:
    """
    Load the correct sheet based on group + No AOI condition.
    """

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing file: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)

    if exclude_no_aoi:
        base = f"{group} Excluding No AOI(A)"
    else:
        base = group

    sheet_name = _find_sheet_name(xls, base)

    if sheet_name is None:
        raise ValueError(
            f"Could not find sheet for: {base}\n"
            f"Available sheets: {xls.sheet_names}"
        )

    df = pd.read_excel(xls, sheet_name=sheet_name)

    df.columns = [c.strip() for c in df.columns]

    return df



def select_metric_column(df: pd.DataFrame) -> str:
    """
    Choose the best metric column for pattern ranking.
    """
    candidates = [
        "Proportional Pattern Frequency",
        "Proportional pattern frequency",
        "Sequence Support",
        "Sequence support",
        "Frequency",
        "Pattern Frequency",
    ]

    normalized = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in normalized:
            return normalized[cand.lower()]

    numeric_cols = df.select_dtypes("number").columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric metric columns found.")
    return numeric_cols[0]


def find_pattern_column(df: pd.DataFrame) -> str:
    """
    Identify the pattern string column.
    """
    candidates = [
        "Pattern String",
        "pattern string",
        "Pattern",
        "pattern",
        "AOI Pattern",
        "aoi pattern",
    ]

    normalized = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in normalized:
            return normalized[cand.lower()]

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        raise ValueError("No pattern string column found.")
    return obj_cols[0]


def prepare_top_patterns(df: pd.DataFrame, group: str, metric: str, pattern_col: str) -> pd.DataFrame:
    tmp = df[[pattern_col, metric]].dropna()
    tmp = tmp.sort_values(metric, ascending=False).head(TOP_N)
    tmp["Group"] = group
    tmp.rename(columns={pattern_col: "Pattern", metric: "Metric"}, inplace=True)
    return tmp


def compute_difference_table(df_succ: pd.DataFrame, df_unsucc: pd.DataFrame,
                             metric_col: str, pattern_col: str) -> pd.DataFrame:

    s = df_succ[[pattern_col, metric_col]].copy()
    s.rename(columns={metric_col: "Metric_Success"}, inplace=True)

    u = df_unsucc[[pattern_col, metric_col]].copy()
    u.rename(columns={metric_col: "Metric_Unsuccess"}, inplace=True)

    merged = pd.merge(s, u, on=pattern_col, how="outer").fillna(0.0)
    merged.rename(columns={pattern_col: "Pattern"}, inplace=True)

    merged["Diff"] = merged["Metric_Success"] - merged["Metric_Unsuccess"]
    merged["AbsDiff"] = merged["Diff"].abs()

    merged = merged.sort_values("AbsDiff", ascending=False)

    return merged

def plot_top_patterns(df: pd.DataFrame, title: str, output_file: Path):
    fig = px.bar(
        df,
        x="Pattern",
        y="Metric",
        color="Group",
        barmode="group",
        text="Metric",
        title=title,
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, uniformtext_minsize=8)

    fig.write_html(str(output_file))
    print("[OK] Saved:", output_file)


def plot_difference_patterns(df: pd.DataFrame, title: str, output_file: Path):
    top = df.head(TOP_N).sort_values("Diff", ascending=False)

    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in top["Diff"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["Pattern"],
        y=top["Diff"],
        marker_color=colors,
        text=top["Diff"].round(3),
        textposition="outside",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="black")

    fig.update_layout(
        title=title,
        xaxis_title="Pattern",
        yaxis_title="Success - Unsuccess",
        xaxis_tickangle=-45,
        template="plotly_white",
    )

    fig.write_html(str(output_file))
    print("[OK] Saved:", output_file)



def analyze_patterns(xlsx_path: Path, collapsed: bool, exclude_no_aoi: bool):
    mode = "Collapsed" if collapsed else "Expanded"
    noaoi = "Excluding_NoAOI" if exclude_no_aoi else "Including_NoAOI"

    print(f"\n=== {mode} patterns ({noaoi}) from {xlsx_path.name} ===")

    df_succ = load_pattern_sheet(xlsx_path, "Successful", exclude_no_aoi)
    df_unsucc = load_pattern_sheet(xlsx_path, "Unsuccessful", exclude_no_aoi)

    metric_col = select_metric_column(df_succ)
    pattern_col = find_pattern_column(df_succ)

    print("[INFO] Metric column:", metric_col)
    print("[INFO] Pattern column:", pattern_col)

    top_s = prepare_top_patterns(df_succ, "Successful", metric_col, pattern_col)
    top_u = prepare_top_patterns(df_unsucc, "Unsuccessful", metric_col, pattern_col)

    top_all = pd.concat([top_s, top_u])

    diff_df = compute_difference_table(df_succ, df_unsucc, metric_col, pattern_col)

    base = f"{mode.lower()}_{noaoi.lower()}"

    top_html = OUTPUT_DIR / f"{base}_top_patterns.html"
    diff_html = OUTPUT_DIR / f"{base}_pattern_differences.html"

    plot_top_patterns(top_all, f"Top {TOP_N} {mode} Patterns ({noaoi})", top_html)
    plot_difference_patterns(diff_df, f"Top {TOP_N} Pattern Differences ({mode}, {noaoi})", diff_html)

    (OUTPUT_DIR / f"{base}_top_patterns.csv").write_text(top_all.to_csv(index=False))
    (OUTPUT_DIR / f"{base}_pattern_differences.csv").write_text(diff_df.to_csv(index=False))

    print("[OK] CSVs saved for", base)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    analyze_patterns(COLLAPSED_XLSX, collapsed=True, exclude_no_aoi=False)
    analyze_patterns(COLLAPSED_XLSX, collapsed=True, exclude_no_aoi=True)
    analyze_patterns(EXPANDED_XLSX, collapsed=False, exclude_no_aoi=False)
    analyze_patterns(EXPANDED_XLSX, collapsed=False, exclude_no_aoi=True)

    print("\nAll pattern analysis done.")
    print("Outputs in ./outputs/")


if __name__ == "__main__":
    main()
