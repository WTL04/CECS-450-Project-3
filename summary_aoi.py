import pandas as pd

AOI_NAMES = [
    "AI",
    "Alt_VSI",
    "ASI",
    "SSI",
    "TI_HSI",
    "RPM",
    "Window",
]


def load_and_summarize(csv_path: str) -> pd.DataFrame:
    """
    Load AOI_DGMs.csv and build a summary table with one row
    per (AOI, Group), where Group is Successful or Unsuccessful.
    """
    df = pd.read_csv(csv_path)

    # Decides which column tells us if a pilot is successful
    if "pilot_success" in df.columns:
        group_col = "pilot_success"
    else:
        if "Approach_Score" not in df.columns:
            raise ValueError(
                "Expected a 'pilot_success' or 'Approach_Score' column "
                "to determine successful vs unsuccessful pilots."
            )
        # Make a labell
        df["pilot_success"] = df["Approach_Score"].apply(
            lambda s: "Successful" if s >= 0.7 else "Unsuccessful"
        )
        group_col = "pilot_success"

    rows = []

    
    for aoi in AOI_NAMES:
        # Column names we care about for this AOI
        prop_fix_col = f"{aoi}_Proportion_of_fixations_spent_in_AOI"
        prop_dur_col = f"{aoi}_Proportion_of_fixations_durations_spent_in_AOI"
        fix_count_col = f"{aoi}_total_number_of_fixations"  # may or may not exist

        for group_value, group_df in df.groupby(group_col):
            row = {
                "AOI": aoi,
                "Group": group_value,
            }

            if prop_fix_col in group_df.columns:
                row["prop_fixations"] = group_df[prop_fix_col].mean()

            if prop_dur_col in group_df.columns:
                row["prop_fix_dur"] = group_df[prop_dur_col].mean()

            if fix_count_col in group_df.columns:
                row["fixation_count"] = group_df[fix_count_col].mean()

            rows.append(row)

    summary = pd.DataFrame(rows)
    return summary


if __name__ == "__main__":
    # Quick manual test when you run: python summary_aoi.py
    summary_df = load_and_summarize("datasets/AOI_DGMs.csv")
    print(summary_df.head())
