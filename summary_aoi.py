import pandas as pd

def load_data(path):
    """Load dataset and ensure success label exists."""
    df = pd.read_csv(path)

    # Use existing label
    if "pilot_success" in df.columns:
        df["Group"] = df["pilot_success"].replace({1: "Successful", 0: "Unsuccessful"})
    else:
        df["Group"] = df["Approach_Score"].apply(lambda x: "Successful" if x >= 0.7 else "Unsuccessful")

    return df


def find_aoi_columns(df):
    """
    Automatically detect AOI names by column prefixes.
    Example columns:
        AI_Proportion_of_fixations_spent_in_AOI
        ASI_Proportion_of_fixations_spent_in_AOI
        RPM_Proportion_of_fixations_spent_in_AOI
    """
    aoi_cols = {}

    for col in df.columns:
        if col.endswith("Proportion_of_fixations_spent_in_AOI"):
            aoi = col.split("_")[0]  # AOI prefix before the first underscore
            aoi_cols.setdefault(aoi, {})
            aoi_cols[aoi]["fix_prop"] = col

        if col.endswith("Proportion_of_fixations_durations_spent_in_AOI"):
            aoi = col.split("_")[0]
            aoi_cols.setdefault(aoi, {})
            aoi_cols[aoi]["dur_prop"] = col

    return aoi_cols


def summarize(df, aoi_cols):
    """Compute mean proportions for each AOI, separately for each group."""
    rows = []

    for aoi, cols in aoi_cols.items():
        fix_col = cols.get("fix_prop")
        dur_col = cols.get("dur_prop")

        if fix_col is None and dur_col is None:
            continue

        for group, group_df in df.groupby("Group"):
            row = {
                "AOI": aoi,
                "Group": group,
                "prop_fixations": group_df[fix_col].mean() if fix_col else None,
                "prop_fix_dur": group_df[dur_col].mean() if dur_col else None
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    csv_path = "datasets/AOI_DGMs.csv"

    df = load_data(csv_path)
    aoi_cols = find_aoi_columns(df)
    summary_df = summarize(df, aoi_cols)

    print(summary_df)
    print("\nTotal rows:", len(summary_df))


if __name__ == "__main__":
    main()

    print("\nTotal rows:", len(summary))


if __name__ == "__main__":
    main()

