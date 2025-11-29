import pandas as pd

AOI_LIST = [
    "AI",
    "Alt_VSI",
    "ASI",
    "SSI",
    "TI_HSI",
    "RPM",
    "Window",
    "NoAOI"
]

def load_and_summarize(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure we have a success label
    if "pilot_success" in df.columns:
        df["Group"] = df["pilot_success"].apply(
            lambda x: "Successful" if x == 1 else "Unsuccessful"
        )
    else:
        df["Group"] = df["Approach_Score"].apply(
            lambda x: "Successful" if x >= 0.7 else "Unsuccessful"
        )

    rows = []

    for aoi in AOI_LIST:
        fix_col = f"{aoi}_Proportion_of_fixations_spent_in_AOI"
        dur_col = f"{aoi}_Proportion_of_fixations_durations_spent_in_AOI"

        if fix_col not in df.columns or dur_col not in df.columns:
            print(f"Skipping AOI {aoi}: missing columns")
            continue

        for group_name, group_df in df.groupby("Group"):
            row = {
                "AOI": aoi,
                "Group": group_name,
                "prop_fixations": group_df[fix_col].mean(),
                "prop_fix_dur": group_df[dur_col].mean()
            }
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    print(summary_df)
    print("\nTotal rows in summary:", len(summary_df))
    return summary_df

if __name__ == "__main__":
    csv_path = "datasets/AOI_DGMs.csv"
    summary_df = load_and_summarize(csv_path)

