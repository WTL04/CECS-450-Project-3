import pandas as pd

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Convert "#NULL!" → NaN
    df = df.replace("#NULL!", pd.NA)

    # Convert entire dataset to numeric where possible
    df = df.apply(pd.to_numeric, errors="ignore")

    # Derive success group if needed
    if "pilot_success" in df.columns:
        df["Group"] = df["pilot_success"].apply(
            lambda x: "Successful" if x == 1 else "Unsuccessful"
        )
    else:
        df["Group"] = df["Approach_Score"].apply(
            lambda x: "Successful" if x >= 0.7 else "Unsuccessful"
        )

    return df


def extract_aoi_prefix(colname):
    """
    Example:
    'AI_Proportion_of_fixations_spent_in_AOI' → 'AI'
    'Alt_Proportion_of_fixations_spent_in_AOI' → 'Alt'
    """
    return colname.split("_")[0]


def summarize(df):
    # Identify proportion columns
    prop_cols = [
        c for c in df.columns
        if "Proportion_of_fixations_spent_in_AOI" in c
    ]

    # Extract AOI  from prefixes
    aoi_list = sorted({extract_aoi_prefix(c) for c in prop_cols})

    summary_rows = []

    for aoi in aoi_list:
        # Find the correct column for this AOI
        fix_col = [
            c for c in prop_cols if c.startswith(aoi + "_")
        ][0]

        # Compute means for success & unsuccessful
        for group in ["Successful", "Unsuccessful"]:
            group_df = df[df["Group"] == group]

            mean_prop = pd.to_numeric(group_df[fix_col], errors="coerce").mean()

            summary_rows.append({
                "AOI": aoi,
                "Group": group,
                "mean_prop_fixations": mean_prop
            })

    return pd.DataFrame(summary_rows)


def main():
    csv_path = "datasets/AOI_DGMs.csv"

    df = load_data(csv_path)
    summary_df = summarize(df)

    print(summary_df)
    print("\nTotal rows in summary:", len(summary_df))


if __name__ == "__main__":
    main()
