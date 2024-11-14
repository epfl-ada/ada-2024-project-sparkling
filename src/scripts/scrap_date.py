import requests
import pandas as pd
import numpy as np


def wikidata_from_wikipedia_id(wikipedia_id, language="en"):
    """
    Returns wikidata link and wikidata id of the given wikipedia id.
    """

    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "pageprops",
        "pageids": wikipedia_id,
        "format": "json",
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            wikidata_id = data["query"]["pages"][str(wikipedia_id)]["pageprops"][
                "wikibase_item"
            ]
            wikidata_link = f"https://www.wikidata.org/wiki/{wikidata_id}"
            return wikidata_link, wikidata_id
        except KeyError:
            # Wikidata ID not found
            return None, None
    else:
        # Error retrieving data
        return None, None


def get_release_date(wikidata_id):
    """
    Returns the release date of the movie with the given wikidata ID.
    """
    url = f"https://www.wikidata.org/w/api.php"
    params = {"action": "wbgetentities", "ids": wikidata_id, "format": "json"}

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            # Access 'claims' section to find property P577 corresponding to publication date
            publication_claims = data["entities"][wikidata_id]["claims"].get("P577")
            if publication_claims:
                # There can be multiple publication dates but we get the first one
                publication_date = publication_claims[0]["mainsnak"]["datavalue"][
                    "value"
                ]["time"]
                return publication_date
            else:
                # P577 not found
                return None
        except KeyError:
            # Case of unexpected data structure
            return None
    else:
        return None
        # Error retrieving data


def format_date_numeric(date_str):
    if date_str == None:
        return "", ""
    year_month = date_str[1:8]  # Extract "+YYYY-MM"
    return year_month[:4], year_month[5:]


def scrap_years_months_movies(df_movies, path_to_csv="dates_scraped.csv"):
    """
    Scrap years and months of the publication date of the movies and save the scrapped values in path_to_csv

    Arguments:
        - df_movies: Dataframe containing the movie wikipedia_ID along
        with their publication date (release_year and release_month)

    """

    # Create a copy of the dataframe of movies
    df_movies_scraped = df_movies.copy()

    df_movies_scraped = df_movies_scraped.rename(
        columns={"release_year": "release_year_x", "release_month": "release_month_x"}
    )

    df_movies_scraped = df_movies_scraped[
        ["wikipedia_ID", "release_year_x", "release_month_x"]
    ].copy()

    # add empty columns that are going to be fill by scraping the dates
    df_movies_scraped["release_year_y"] = ""
    df_movies_scraped["release_month_y"] = ""

    for idx, row in df_movies_scraped.iterrows():
        w_id = row.wikipedia_ID
        year, month = format_date_numeric(
            get_release_date(wikidata_from_wikipedia_id(w_id)[1])
        )
        df_movies_scraped.loc[idx, "release_year_y"] = year
        df_movies_scraped.loc[idx, "release_month_y"] = month

        # save the file with the new scraped dates
        # We are saving it at each iteration as it may take time to run.
        df_movies_scraped.to_csv(path_to_csv, index=False)


def date_choice(dates):
    """
    Arguments:
        dates: list [year in original dataset, month in original dataset, year scraped, month scraped]
    Returns:
        final_date: Series with final year and month we chose. One of them could be nan, it
                    will be processed in get_final_dates.
    """
    year = float(dates[0])
    month = float(dates[1])
    year_scr = float(dates[2])
    month_scr = float(dates[3])

    final_date = [0, 0]

    if np.isnan(year):
        if np.isnan(month):
            final_date = [year_scr, month_scr]
        else:
            if np.isnan(month_scr):
                final_date = [year_scr, month]
            else:
                if month_scr == 1.0:
                    final_date = [year_scr, month]
                else:
                    final_date = [year_scr, month_scr]
    else:
        if year == year_scr:
            if np.isnan(month):
                final_date = [year, month_scr]
            else:
                if np.isnan(month_scr) or month_scr == 1.0 or month_scr == 0:
                    final_date = [year, month]
                else:
                    final_date = [year, month_scr]
        else:
            if np.isnan(month):
                final_date = [year_scr, month_scr]
            else:
                final_date = [year, month]

    return pd.Series(
        final_date
    )  # pd.DataFrame(data={'release_year': [final_date[0]], 'release_month': [final_date[1]]})


def get_final_dates(df):
    """
    Arguments:
        df: dataframe of original dates and scraped dates.
    Returns:
        final_df: dataframe with 3 columns ['wikipedia_ID', 'release_year', 'release_month']
                  with only the movies for which we have found a way to get both their year
                  and month of release (no NaN).
    """
    final_df = pd.DataFrame(data={"wikipedia_ID": list(df.wikipedia_ID)})
    final_df[["release_year", "release_month"]] = df[
        ["release_year_x", "release_month_x", "release_year_y", "release_month_y"]
    ].apply(lambda x: date_choice(list(x)), axis=1)
    final_df = final_df.drop(
        (
            final_df[
                np.isnan(final_df["release_year"])
                | np.isnan(final_df["release_month"])
                | (final_df["release_month"] == 0)
            ]
        ).index
    )
    return final_df
