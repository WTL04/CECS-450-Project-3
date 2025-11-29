import pandas as pd


def load_and_summarize(csv_path="datasets/AOI_DGMs.csv"):
    """
    Build a small AOI x Group summary table from AOI_DGMs.csv.

    For each AOI (AI, Alt_VSI, ASI, SSI, TI_HSI, RPM, Window, etc.)
    and each pilot_success group, we compute:

    - prop_fixations: average proportion of fixations spent in that AOI
    - prop_fix_dur: average proportion of fixation duration spent in that AOI
    """

   
    df = pd.read_csv(csv_path)

    if "pilot_success" not in df.columns:
        raise ValueError("Expected a 'pilot_success' column in AOI_DGMs.csv")

  
    aoi_prefixes = set()
    for col in df.columns:
        if col.endswith("_Proportion_of_fixations_spent_in_AOI"):
            prefix = col.split("_")[0]
            aoi_prefixes.add(prefix)

    if not aoi_prefixes:
        raise ValueError(
            "Could not find any columns ending with "
            "'_Proportion_of_fixations_spent_in_AOI'. "
            "Check the AOI_DGMs column names."
        )

    aoi_prefixes = sorted(aoi_prefixes)

    rows = []

    # For each AOI & success group 
    for aoi in aoi_prefixes:
        col_fix_prop = f"{aoi}_Proportion_of_fixations_spent_in_AOI"
        col_dur_prop = f"{aoi}_Proportion_of_fixations_durations_spent_in_AOI"

        if col_fix_prop not in df.columns or col_dur_prop not in df.columns:
            # Skip AOIs that don't have both proportion columns
            continue

        # Split by pilot success 
        for group_name, group_df in df.groupby("pilot_success"):
            row = {
                "AOI": aoi,
                "Group": str(group_name),
            }

            # Convert to numeric, ignore '#NULL!' and other text
            fix_vals = pd.to_numeric(group_df[col_fix_prop], errors="coerce")
            dur_vals = pd.to_numeric(group_df[col_dur_prop], errors="coerce")

            row["prop_fixations"] = fix_vals.mean()
            row["prop_fix_dur"] = dur_vals.mean()

            rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df


if __name__ == "__main__":
    # Small manual test: print the first few rows of the summary
    summary = load_and_summarize("datasets/AOI_DGMs.csv")
    print(summary.head())
    print(f"\nTotal rows in summary: {len(summary)}")
