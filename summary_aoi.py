import pandas as pd


def load_aoi_data(csv_path: str) -> pd.DataFrame:
    """
    Load the AOI_DGMs.csv file and create a Group column
    based on Approach_Score.

    Group rules (from project spec):
    - Successful: Approach_Score >= 0.7
    - Unsuccessful: Approach_Score < 0.7
    """
    # Treat '#NULL!' as a missing value
    df = pd.read_csv(csv_path, na_values=["#NULL!"])

    if "Approach_Score" not in df.columns:
        raise ValueError("Expected an 'Approach_Score' column in the CSV.")

    # group label for each pilot
    df["Group"] = df["Approach_Score"].apply(
        lambda s: "Successful" if s >= 0.7 else "Unsuccessful"
    )

    return df


def find_aoi_columns(df: pd.DataFrame) -> dict:
    """
    Find all AOI columns of the form:

        <AOI_NAME>_Proportion_of_fixations_spent_in_AOI

    and return a dictionary mapping:
        AOI_NAME -> column_name
    """
    aoi_cols = {}

    for col in df.columns:
        if col.endswith("_Proportion_of_fixations_spent_in_AOI"):
            # Everything before '_Proportion_of_fixations_spent_in_AOI'
            aoi_name = col.replace("_Proportion_of_fixations_spent_in_AOI", "")
            aoi_cols[aoi_name] = col

    if not aoi_cols:
        raise ValueError(
            "Could not find any AOI proportion columns. "
            "Expected columns ending with '_Proportion_of_fixations_spent_in_AOI'."
        )

    return aoi_cols


def load_and_summarize(csv_path: str) -> pd.DataFrame:
    """
    Load the AOI data and compute, for each AOI and each Group,
    the mean proportion of fixations spent in that AOI.

    Returns a DataFrame with columns:
        - AOI
        - Group
        - prop_fixations
    """
    df = load_aoi_data(csv_path)
    aoi_cols = find_aoi_columns(df)

    rows = []

    for aoi_name, col_name in aoi_cols.items():
        # Convert this AOI column to numeric
        series_all = pd.to_numeric(df[col_name], errors="coerce")

        # Attach back to df so we can group by Group
        df["_current_aoi_prop"] = series_all

        # Group by Success Unsuccessful
        for group_name, group_df in df.groupby("Group"):
            mean_prop = group_df["_current_aoi_prop"].mean()

            rows.append(
                {
                    "AOI": aoi_name,
                    "Group": group_name,
                    "prop_fixations": mean_prop,
                }
            )

    
    if "_current_aoi_prop" in df.columns:
        df.drop(columns=["_current_aoi_prop"], inplace=True)

    summary = pd.DataFrame(rows)
    return summary


def main():
    csv_path = "datasets/AOI_DGMs.csv"
    summary_df = load_and_summarize(csv_path)

    print(summary_df)
    print(f"\nTotal rows in summary: {len(summary_df)}")

    summary_df.to_csv("datasets/AOI_summary.csv", index=False)


if __name__ == "__main__":
    main()
