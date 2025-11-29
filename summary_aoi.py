import pandas as pd

CSV_PATH = "datasets/AOI_DGMs.csv"

def extract_aoi_prefix(col_name: str):
    """
    Get the AOI prefix from a column like:
    'AI_Proportion_of_fixations_spent_in_AOI'
    Returns 'AI'
    """
    if "_Proportion_of_fixations" in col_name:
        return col_name.split("_Proportion_of_fixations")[0]
    return None


def summarize_aoi_metrics(df: pd.DataFrame):
    """
    Create a clean AOI summary table using the wide-format dataset.
    Output DataFrame columns:
        AOI, Group, prop_fixations, prop_fix_durations
    """

    
    fix_cols = [c for c in df.columns if "Proportion_of_fixations_spent_in_AOI" in c]
    dur_cols = [c for c in df.columns if "Proportion_of_fixations_durations_spent_in_AOI" in c]

    
    aois = sorted(set(extract_aoi_prefix(c) for c in fix_cols if extract_aoi_prefix(c)))

    records = []

    for aoi in aois:
        fix_col = f"{aoi}_Proportion_of_fixations_spent_in_AOI"
        dur_col = f"{aoi}_Proportion_of_fixations_durations_spent_in_AOI"

        
        if fix_col not in df.columns or dur_col not in df.columns:
            continue

        
        for group, group_df in df.groupby("pilot_success"):
            records.append({
                "AOI": aoi,
                "Group": "Successful" if group == 1 else "Unsuccessful",
                "prop_fixations": group_df[fix_col].mean(),
                "prop_fix_durations": group_df[dur_col].mean()
            })

    return pd.DataFrame(records)


def main():
    df = pd.read_csv(CSV_PATH)

    summary = summarize_aoi_metrics(df)

    print(summary)
    print("\nTotal rows:", len(summary))


if __name__ == "__main__":
    main()

