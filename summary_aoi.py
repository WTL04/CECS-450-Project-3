import pandas as pd


def load_aoi_dgms(csv_path: str) -> pd.DataFrame:
    """
    Load the AOI_DGMs table and add a 'Group' label for each pilot.

    One row = one pilot.
    Many columns = AOI-specific measures (Alt_VSI_..., AI_..., RPM_..., etc.).
    """

    df = pd.read_csv(csv_path)

    # --- Build a clean 'Group' column (Successful / Unsuccessful) ---

    if "Group" in df.columns:
        # Already labeled; just keep it
        group = df["Group"].astype(str)
    elif "pilot_success" in df.columns:
        # pilot_success likely 1/0 or True/False
        def label_from_flag(x):
            return "Successful" if x else "Unsuccessful"

        group = df["pilot_success"].apply(label_from_flag)
    elif "Approach_Score" in df.columns:
        # Fallback: use Approach_Score >= 0.7 as success
        def label_from_score(score):
            return "Successful" if score >= 0.7 else "Unsuccessful"

        group = df["Approach_Score"].apply(label_from_score)
    else:
        raise ValueError(
            "Could not find Group, pilot_success, or Approach_Score in AOI_DGMs.csv."
        )

    df = df.copy()
    df["Group"] = group

    return df


def summarize_aoi_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the wide AOI_DGMs table into a tidy summary:

    Columns in the result:
      - AOI  (e.g., 'AI', 'Alt_VSI', 'RPM', 'Window', ...)
      - Group ('Successful' or 'Unsuccessful')
      - prop_fixations  (mean proportion of fixations in that AOI)
      - prop_fix_dur    (mean proportion of fixation duration in that AOI)
    """

    suffix_fix = "_Proportion_of_fixations_spent_in_AOI"
    suffix_dur = "_Proportion_of_fixations_durations_spent_in_AOI"

    # Find all AOIs by looking for the *_Proportion_of_fixations_spent_in_AOI columns
    fix_cols = [c for c in df.columns if c.endswith(suffix_fix)]

    if not fix_cols:
        raise ValueError(
            "No columns ending with "
            f"'{suffix_fix}'. Check AOI_DGMs.csv column names."
        )

    # Build a list of AOI names (strip the suffix off)
    aoi_names = [c.replace(suffix_fix, "") for c in fix_cols]

    rows = []

    for aoi in aoi_names:
        fix_col = aoi + suffix_fix
        dur_col = aoi + suffix_dur if (aoi + suffix_dur) in df.columns else None

        # Group by Successful / Unsuccessful
        grouped = df.groupby("Group")

        for group_name, gdf in grouped:
            prop_fix_mean = gdf[fix_col].mean()

            if dur_col is not None:
                prop_dur_mean = gdf[dur_col].mean()
            else:
                prop_dur_mean = float("nan")

            rows.append(
                {
                    "AOI": aoi,
                    "Group": group_name,
                    "prop_fixations": prop_fix_mean,
                    "prop_fix_dur": prop_dur_mean,
                }
            )

    summary = pd.DataFrame(rows)
    return summary


def load_and_summarize(csv_path: str) -> pd.DataFrame:
    """
    Convenience function used by aoi_graph_interactive.py:

    1. Load AOI_DGMs.csv
    2. Add Group labels
    3. Summarize proportions per AOI and Group
    """
    df = load_aoi_dgms(csv_path)
    summary = summarize_aoi_proportions(df)
    return summary


if __name__ == "__main__":
    # Quick manual test: run `python summary_aoi.py` from the repo root.
    test_path = "datasets/AOI_DGMs.csv"
    s = load_and_summarize(test_path)
    print(s.head())
