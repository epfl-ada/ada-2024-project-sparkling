import os
import pandas as pd
from src.data.load_data import DATASETS_DIR
from src.data.data_wrangling import load_and_clean_movies_df
from src.data.load_data_reviews_kaggle import load_movie_reviews_kaggle
from src.data.load_data_non_commercial_imdb import (
    load_imdb_id_wikipedia_id,
    load_imdb_average_reviews,
)

FINAL_DATA_DIR = "FinalData"


def load_final_files(filename):
    """
    Load a csv file from the FinalData folder

    Returns a dataframe containing the data
    """
    path = os.path.join(DATASETS_DIR, FINAL_DATA_DIR, filename)
    df = pd.read_csv(path)

    return df


def load_final_dates():
    """
    Load the final date file

    Returns a dataframe containing the data
    """
    df_final_dates = load_final_files("final_dates.csv")
    df_final_dates["release_year"] = df_final_dates["release_year"].astype("int64")
    df_final_dates["release_month"] = df_final_dates["release_month"].astype("int64")

    return df_final_dates


def load_scrapped_imdb_reviews():
    """
    Load the scrapped imdb reviews file

    Returns a dataframe containing the data
    """
    return load_final_files("scrapped_imdb_reviews.csv")


def remove_not_used_one_hot_encoding(df):
    """
    Given a dataframe containing an one hot encoding, return the same dataframe but removing the columns
    where the one hot encoding is always equal to False

    Argument:
        - df: Dataframe containing the one hot encoding

    Return:
        The same Dataframe without the columns where the one hot encoding is always equal to False
    """
    column_to_remove_boolean = df.sum() == 0
    column_to_remove_list = column_to_remove_boolean[column_to_remove_boolean].index

    return df.drop(column_to_remove_list, axis=1)


def load_final_movies_and_reviews(
    save_to_load_from_movies="final_movies.csv",
    save_to_load_from_reviews="final_reviews.csv",
):
    """
    Load the final datas for our project

    Note: If you already have the files 'final_movies.csv/save_to_load_from_movies' and 'final_reviews.csv/save_to_load_from_reviews'
    you also need the movie.metadata.tsv from the original CMU movie summary dataset (More information in the README)

    Argument:
        - save_to_load_from_movies: File to save or load data from for movies (None -> do not save nor load the data) -> Default: final_movies.csv
        - save_to_load_from_reviews: File to save or load data from for reviews (None -> do not save nor load the data) -> Default: final_reviews.csv

    Returns a list of 5 dataframes.
    - A dataframe containing the main metadata of the movies along with its plot and average reviews
    - A dataframe containing the language of the movie in one hot encoding, associated to the wikipedia_ID of the movie
    - A dataframe containing the countries of the movie in one hot encoding, associated to the wikipedia_ID of the movie
    - A dataframe containing the genres of the movie in one hot encoding, associated to the wikipedia_ID of the movie
    - A dataframe containing the reviews of the movies associated to the wikipedia_ID of the movie

    If the file 'final_movies.csv/save_to_load_from_movies' and/or 'final_reviews.csv/save_to_load_from_reviews' exists,
    then load it, otherwise create the files
    """
    MINIMUM_SENTENCES_PLOT = 10
    MINIMUM_REVIEWS_MOVIES = 10
    MAXIMUM_REVIEWS_MOVIES = 50

    df_final_movies = None
    df_final_reviews = None

    if save_to_load_from_movies is not None:
        path_csv_movies = os.path.join(
            DATASETS_DIR, FINAL_DATA_DIR, save_to_load_from_movies
        )
        if os.path.isfile(path_csv_movies):
            df_final_movies = pd.read_csv(path_csv_movies)

    if save_to_load_from_reviews is not None:
        path_csv_reviews = os.path.join(
            DATASETS_DIR, FINAL_DATA_DIR, save_to_load_from_reviews
        )
        if os.path.isfile(path_csv_reviews):
            df_final_reviews = pd.read_csv(path_csv_reviews)

    # Load movie metadata
    df_movies_metadata, df_movies_language, df_movies_countries, df_movies_genres = (
        load_and_clean_movies_df()
    )

    if df_final_movies is None or df_final_reviews is None:
        # Load final dates
        df_final_dates = load_final_dates()

        df_movies_improved_dates = df_movies_metadata.merge(
            df_final_dates, on="wikipedia_ID", suffixes=("_old", "")
        )
        df_movies_improved_dates = df_movies_improved_dates.drop(
            [
                "release_year_old",
                "release_month_old",
                "release_day",
                "revenue",
                "runtime",
            ],
            axis=1,
        )

        # Load movie reviews from kaggle
        df_reviews_kaggle = load_movie_reviews_kaggle(df_movies_improved_dates)

        # Load mapping IMDB id to wikipedia ID
        df_mapping_imdb_id_wikipedia_id = load_imdb_id_wikipedia_id(
            df_movies_improved_dates
        )

        # Load movie average reviews from the non-commercial IMDB
        df_imdb_average_reviews = load_imdb_average_reviews(
            df_mapping_imdb_id_wikipedia_id
        )

        # Load IMDB reviews
        df_imdb_reviews = load_scrapped_imdb_reviews()

        # Apply minimum conditions

        # The movie should at least have 10 sentences in its plot
        plot_number_sentences_minimum = (
            df_movies_improved_dates["plot"].apply(lambda x: len(x.split(".")))
            >= MINIMUM_SENTENCES_PLOT
        )
        df_movies_plot_number_minimum = df_movies_improved_dates[
            plot_number_sentences_minimum
        ]

        # Merge the scrapped reviews and the reviews from kaggle
        df_scrapped_reviews_movies_merged = df_movies_plot_number_minimum[
            ["wikipedia_ID", "name", "release_year"]
        ].merge(df_imdb_reviews, on="wikipedia_ID")

        # Our version of the scrapped dataset also have a column "2", we fixed this in the new scrapping code
        if "2" in df_scrapped_reviews_movies_merged.columns:
            df_scrapped_reviews_movies_merged = df_scrapped_reviews_movies_merged.drop(
                ["2"], axis=1
            )
        df_scrapped_reviews_movies_merged = df_scrapped_reviews_movies_merged.drop(
            ["tconst"], axis=1
        )
        df_scrapped_reviews_movies_merged = df_scrapped_reviews_movies_merged.rename(
            {"review_summary": "review_detail"}, axis=1
        )

        df_all_reviews = pd.concat(
            [df_scrapped_reviews_movies_merged, df_reviews_kaggle], axis=0
        )

        # The movie should at least have MINIMUM_REVIEWS_MOVIES reviews
        number_reviews_movies_min = (
            df_all_reviews["wikipedia_ID"].value_counts() >= MINIMUM_REVIEWS_MOVIES
        )
        wikipedia_id_minimum_reviews = number_reviews_movies_min[
            number_reviews_movies_min
        ].index

        df_final_movies = df_movies_plot_number_minimum[
            df_movies_plot_number_minimum["wikipedia_ID"].isin(
                wikipedia_id_minimum_reviews
            )
        ]
        df_final_movies = df_final_movies.merge(
            df_imdb_average_reviews[["wikipedia_ID", "averageRating"]],
            on="wikipedia_ID",
        )

        ## Create final reviews dataframe
        df_all_reviews = df_all_reviews[
            df_all_reviews["wikipedia_ID"].isin(df_final_movies["wikipedia_ID"])
        ]
        number_reviews_movies = df_all_reviews["wikipedia_ID"].value_counts()
        number_reviews_movies_above_max = number_reviews_movies > MAXIMUM_REVIEWS_MOVIES
        number_reviews_movies_below_max = ~number_reviews_movies_above_max

        wikipedia_id_above_max = number_reviews_movies_above_max[
            number_reviews_movies_above_max
        ].index
        wikipedia_id_below_max = number_reviews_movies_below_max[
            number_reviews_movies_below_max
        ].index

        df_reviews_above_max = df_all_reviews[
            df_all_reviews["wikipedia_ID"].isin(wikipedia_id_above_max)
        ]
        df_reviews_below_max = df_all_reviews[
            df_all_reviews["wikipedia_ID"].isin(wikipedia_id_below_max)
        ]

        df_subset_above_max_reviews = df_reviews_above_max.groupby(
            "wikipedia_ID"
        ).sample(n=MAXIMUM_REVIEWS_MOVIES, replace=False, random_state=42)

        df_final_reviews = pd.concat(
            [df_subset_above_max_reviews, df_reviews_below_max], axis=0
        )

        # Save the dataframes
        if save_to_load_from_movies is not None and df_final_movies is not None:
            df_final_movies.to_csv(path_csv_movies, index=False)
        if save_to_load_from_reviews is not None and df_final_reviews is not None:
            df_final_reviews.to_csv(path_csv_reviews, index=False)

    df_final_language = df_movies_language[
        df_movies_language["wikipedia_ID"].isin(df_final_movies["wikipedia_ID"].values)
    ]
    df_final_countries = df_movies_countries[
        df_movies_countries["wikipedia_ID"].isin(df_final_movies["wikipedia_ID"].values)
    ]
    df_final_genres = df_movies_genres[
        df_movies_genres["wikipedia_ID"].isin(df_final_movies["wikipedia_ID"].values)
    ]

    df_final_language = remove_not_used_one_hot_encoding(df_final_language)
    df_final_countries = remove_not_used_one_hot_encoding(df_final_countries)
    df_final_genres = remove_not_used_one_hot_encoding(df_final_genres)

    return (
        df_final_movies,
        df_final_language,
        df_final_countries,
        df_final_genres,
        df_final_reviews,
    )
