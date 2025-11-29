import pandas as pd

def load_aoi_data(csv_path: str) -> pd.DataFrame:
    """
    Load the AOI_DGMs.csv file and set up the Group column.
    """
    df = pd.read_csv(csv_path)

    # Create a Group column using Approach_Score
    df['Group'] = df['Approach_Score'].apply(
        lambda x: 'Successful' if x >= 0.7 else 'Unsuccessful'
    )

    return df


def compute_aoi_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AOI-level summaries for proportion of fixations and fixation duration.
    Each AOI column already contains its own fixation proportion in AOI_DGMs.csv.
    Example columns:
       - AI_Proportion_of_fixations_spent_in_AOI
       - AI_Proportion_of_fixations_durations_spent_in_AOI
    """

    
    proportion_cols = [
        col for col in df.columns
        if col.endswith("_Proportion_of_fixations_spent_in_AOI")
    ]

    aoi_names = [col.split("_")[0] for col in proportion_cols]

    summary_rows = []

    for aoi in aoi_names:
        fix_col = f"{aoi}_Proportion_of_fixations_spent_in_AOI"
        dur_col = f"{aoi}_Proportion_of_fixations_durations_spent_in_AOI"

        for group in ["Successful", "Unsuccessful"]:
            group_df = df[df["Group"] == group]

            summary_rows.append({
                "AOI": aoi,
                "Group": group,
                "prop_fixations": group_df[fix_col].mean(),
                "prop_fix_dur": group_df[dur_col].mean()
            })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def load_and_summarize(csv_path: str) -> pd.DataFrame:
    """
    Wrapper to load data and produce AOI summary table.
    """
    df = load_aoi_data(csv_path)
    summary = compute_aoi_group_summary(df)
    return summary



# Manual test

if __name__ == "__main__":
    csv_path = "datasets/AOI_DGMs.csv"
    summary_df = load_and_summarize(csv_path)

    print(summary_df.head())
    print("\nTotal rows in summary:", len(summary_df))
