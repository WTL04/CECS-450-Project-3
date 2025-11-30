import pandas as pd
import numpy as np
import plotly.graph_objects as go


def select_dgms(df: pd.DataFrame, keywords: list, aoi: str):
    """
    Select columns that contain ANY of the given keywords,
    restricted to a single AOI prefix (e.g. "AI_").
    """
    keyword_pattern = "|".join(keywords)

    if aoi:
        full_pattern = rf"^{aoi}_.*({keyword_pattern})"
    else:
        full_pattern = rf"({keyword_pattern})"

    return df.filter(regex=full_pattern, axis=1)


def filter_success(df: pd.DataFrame):
    """
    Split root dataframe into two dataframes, successful and unsuccessful
    """
    df_successful = df[df["pilot_success"] == "Successful"].copy()
    df_unsuccessful = df[df["pilot_success"] == "Unsuccessful"].copy()
    return df_successful, df_unsuccessful


def main():
    df = pd.read_csv("./datasets/AOI_DGMs.csv")

    # split by success
    df_successful, df_unsuccessful = filter_success(df)

    fixation_keywords = [
        "Proportion_of_fixations_spent_in_AOI",  # how often AOI checked
        "Proportion_of_fixations_durations_spent_in_AOI",  # how long AOI viewed
        "Total_Number_of_Fixations",
        "Mean_fixation_duration",
    ]

    aoi_list = ["AI", "Alt_VSI", "ASI", "SSI", "TI_HSI", "RPM", "Window", "No_AOI"]
    aoi_traces = []
    trace_visibility_map = []

    for aoi in aoi_list:
        # select fixation DGMs for this AOI
        fix_df_successful = select_dgms(
            df_successful, fixation_keywords, aoi=aoi
        ).copy()
        fix_df_unsuccessful = select_dgms(
            df_unsuccessful, fixation_keywords, aoi=aoi
        ).copy()

        # skip AOIs that don't exist in the data
        if fix_df_successful.empty and fix_df_unsuccessful.empty:
            continue

        # append pilot_success
        fix_df_successful["pilot_success"] = df_successful["pilot_success"].values
        fix_df_unsuccessful["pilot_success"] = df_unsuccessful["pilot_success"].values

        # combine
        result = pd.concat(
            [fix_df_successful, fix_df_unsuccessful], axis=0, ignore_index=True
        )

        # numeric code for color
        result["pilot_success_code"] = (result["pilot_success"] == "Successful").astype(
            int
        )

        # choose dimensions: all numeric except the color column
        numeric_cols = result.select_dtypes(include=np.number).columns.tolist()
        dims_cols = [c for c in numeric_cols if c != "pilot_success_code"]

        if not dims_cols:
            continue  # nothing to plot for this AOI

        # build dimensions list for go.Parcoords
        dimensions = [
            dict(label=col.replace("_", " "), values=result[col]) for col in dims_cols
        ]

        aoi_traces.append(
            {
                "aoi": aoi,
                "dimensions": dimensions,
                # store the color array (one value per row) as a plain Python list
                "color": result["pilot_success_code"].tolist(),
            }
        )

        trace_visibility_map.append(aoi)

    if not aoi_traces:
        # nothing to show
        return

    # create a single Parcoords trace with the first AOI's data
    first = aoi_traces[0]
    fig = go.Figure()
    fig.add_trace(
        go.Parcoords(
            line=dict(
                color=first["color"],
                colorscale=[[0, "firebrick"], [1, "royalblue"]],
                showscale=False,
            ),
            dimensions=first["dimensions"],
        )
    )

    # Build buttons that restyle the single trace's dimensions and line color,
    # and update the title. Using one trace avoids overlayed axes/labels.
    buttons = []
    for item in aoi_traces:
        buttons.append(
            dict(
                label=item["aoi"],
                method="update",
                args=[
                    {"dimensions": [item["dimensions"]], "line.color": [item["color"]]},
                    {"title": {"text": f"Parallel Coordinates – AOI: {item['aoi']}"}},
                ],
            )
        )

    fig.update_layout(
        title=f"Parallel Coordinates – AOI: {trace_visibility_map[0]}",
        updatemenus=[
            dict(
                type="dropdown",
                active=0,
                x=1.05,
                y=1.15,
                xanchor="left",
                buttons=buttons,
            )
        ],
    )
    

    fig.show()
if __name__ == "__main__":
    main()
