import pandas as pd
import numpy as np
import plotly.express as px


def select_dgms(df: pd.DataFrame, keywords: list, aoi: str):
    """
    Select columns that contain ANY of the given keywords,
    optionally restricted to specific AOI prefixes.

    df : pandas DataFrame
    keywords : list of strings to match in column names
    aois : list of AOI prefixes (e.g. ["AI", "ASI", "SSI", ...])
           If None, match keywords across ALL columns.
    """

    # Build keyword regex block: (key1|key2|key3...)
    keyword_pattern = "|".join(keywords)

    if aoi:
        # anchor to beginning-of-column and match only that AOI
        full_pattern = rf"^{aoi}_.*({keyword_pattern})"
    else:
        full_pattern = rf"({keyword_pattern})"

    return df.filter(regex=full_pattern, axis=1)


def filter_success(df: pd.DataFrame):
    # filter successful vs unsuccessful pilots into subset dfs
    df_successful = df[df["pilot_success"] == "Successful"].copy()
    df_unsuccessful = df[df["pilot_success"] == "Unsuccessful"].copy()
    return df_successful, df_unsuccessful


def create_parallel_coords(result: pd.DataFrame, dims: list):
    fig = px.parallel_coordinates(
        result,
        dimensions=dims,
        color="pilot_success_code",
        color_continuous_scale=["firebrick", "royalblue"],
        labels={col: col.replace("_", " ") for col in result.columns},
    )

    fig.show()


def main():
    df = pd.read_csv("./datasets/AOI_DGMs.csv")

    # filter successful vs unsuccessful pilots into subset dfs
    df_successful, df_unsuccessful = filter_success(df)

    # filter out fixation information from both successful vs uncessesful dfs
    fixation_keywords = [
        "Proportion_of_fixations_spent_in_AOI",  # how often pilot checked AOI
        "Proportion_of_fixations_durations_spent_in_AOI",  # duration of how long pilot looked at AOI
        "Total_Number_of_Fixations",
        "Mean_fixation_duration",
    ]

    # list of all AOIs
    # TODO: RPM and NO_AOI may not exist in the dataset, double check this and edit accordinly
    aoi_list = ["AI", "Alt_VSI", "ASI", "SSI", "TI_HSI", "RPM", "Window", "No_AOI"]

    # create a figure for all AOIs
    # TODO: turn all indivisual figures into one website with a filter
    for aoi in aoi_list:
        # select fixation DGMs
        fix_df_successful = select_dgms(
            df_successful, fixation_keywords, aoi=aoi
        ).copy()
        fix_df_unsuccessful = select_dgms(
            df_unsuccessful, fixation_keywords, aoi=aoi
        ).copy()

        # reappend pilot_success column to each df
        fix_df_successful["pilot_success"] = df_successful["pilot_success"].values
        fix_df_unsuccessful["pilot_success"] = df_unsuccessful["pilot_success"].values

        # concatinate the two dfs for comparision
        result = pd.concat([fix_df_successful, fix_df_unsuccessful], axis=0)

        # numeric code for color
        result["pilot_success_code"] = (result["pilot_success"] == "Successful").astype(
            int
        )

        # automatically choose dimensions: all numeric except the color column
        numeric_cols = result.select_dtypes(include=np.number).columns.tolist()
        dims = [c for c in numeric_cols if c != "pilot_success_code"]

        create_parallel_coords(result, dims)


main()
