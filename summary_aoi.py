import pandas as pd

# file loading the AOI-level CSV so that itcan be used to show hover information in the AOI graph.


def load_aoi_data(csv_path):
    df = pd.read_csv(csv_path)

    # Try to identify the AOI column
    if "AOI" in df.columns:
        pass
    elif "AOI_Label" in df.columns:
        df.rename(columns={"AOI_Label": "AOI"}, inplace=True)
    elif "AOI_Name" in df.columns:
        df.rename(columns={"AOI_Name": "AOI"}, inplace=True)
    else:
        raise ValueError("No AOI column found. Add or rename the AOI column.")

    # Create pilot group (based on Approach_Score)
    if "Group" not in df.columns:
        if "Approach_Score" not in df.columns:
            raise ValueError("Need either Group or Approach_Score in the CSV.")
        df["Group"] = df["Approach_Score"].apply(
            lambda s: "Successful" if s >= 0.7 else "Unsuccessful"
        )

    return df


def compute_aoi_summary(df):
    # Try to detect common column names
    # You may need to rename these depending on your actual CSV
    fix_count = None
    fix_dur = None
    prop_fix = None
    prop_dur = None
    transition_col = None

    for c in df.columns:
        name = c.lower()
        if "fixation count" in name and fix_count is None:
            fix_count = c
        if "fixation duration" in name and fix_dur is None:
            fix_dur = c
        if "proportion of fixations" in name and prop_fix is None:
            prop_fix = c
        if "proportion of fixation duration" in name and prop_dur is None:
            prop_dur = c
        if "transition" in name and transition_col is None:
            transition_col = c

    agg = {}
    if fix_count:
        agg["fixation_count"] = (fix_count, "mean")
    if fix_dur:
        agg["mean_fix_dur"] = (fix_dur, "mean")
    if prop_fix:
        agg["prop_fixations"] = (prop_fix, "mean")
    if prop_dur:
        agg["prop_fix_dur"] = (prop_dur, "mean")
    if transition_col:
        agg["mean_transition_count"] = (transition_col, "mean")

    if not agg:
        raise ValueError("Couldn't detect fixation/transition columns. Check names.")

    summary = (
        df.groupby(["AOI", "Group"])
        .agg(**agg)
        .reset_index()
    )

    return summary


def load_and_summarize(csv_path):
    df = load_aoi_data(csv_path)
    return compute_aoi_summary(df)


if __name__ == "__main__":
    # For testing the file directly
    test_path = "datasets/your_aoi_file.csv"
    summary = load_and_summarize(test_path)
    print(summary.head())
