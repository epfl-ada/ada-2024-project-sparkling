import os
import pandas as pd
from src.data.load_data import DATASETS_DIR

IMDB_DIR = "IMDB"


def load_imdb_movie_metadata():
    """
    Load the movie metadata file from the IMDB non-commercial dataset

    Returns a dataframe containing the data
    """
    path = os.path.join(DATASETS_DIR, IMDB_DIR, "title.basics.tsv")
    df = pd.read_csv(path, sep="\t")

    return df


def load_raw_imdb_average_reviews():
    """
    Load the movie reviews file from the IMDB non-commercial dataset

    Returns a dataframe containing the data
    """
    path = os.path.join(DATASETS_DIR, IMDB_DIR, "title.ratings.tsv")
    df = pd.read_csv(path, sep="\t")

    return df


def load_imdb_id_wikipedia_id(df_movies):
    """
    Load the IMDB IDs and merge it the right wikipedia_ID.

    Arguments:
        - df_movies: Dataframe containing our movies with at least the columns wikipedia_ID, name and release_year

    Returns a Dataframe containing the wikipedia_ID, tconst (IDMB ID)
    """

    # Load IMDB data
    imdb_id = load_imdb_movie_metadata()

    def startYear_mapping(element):
        """
        Transform element from string start year to int when possible

        Return the start year as int or return NA
        """
        try:
            return int(element)
        except:
            return pd.NA

    # To have the titles in the same format
    imdb_id["primaryTitle"] = (
        imdb_id["primaryTitle"].str.lower().str.replace(" ", "", regex=True)
    )
    imdb_id["originalTitle"] = (
        imdb_id["originalTitle"].str.lower().str.replace(" ", "", regex=True)
    )
    imdb_id["startYear"] = imdb_id["startYear"].apply(startYear_mapping).astype("Int64")
    df_movies = df_movies.copy()
    df_movies["name"] = df_movies["name"].str.lower().str.replace(" ", "", regex=True)

    # Let the imdb id only for movies ans tvMovies
    imdb_id = imdb_id[
        (imdb_id["titleType"] == "movie") | (imdb_id["titleType"] == "tvMovie")
    ]

    merge_primary = pd.merge(
        df_movies,
        imdb_id,
        left_on=["name", "release_year"],
        right_on=["primaryTitle", "startYear"],
        how="inner",
    )
    merge_original = pd.merge(
        df_movies,
        imdb_id,
        left_on=["name", "release_year"],
        right_on=["originalTitle", "startYear"],
        how="inner",
    )

    df_mapping_imdb_id_wikipedia_id = pd.concat(
        [merge_primary, merge_original]
    ).drop_duplicates(subset=["wikipedia_ID"])
    df_mapping_imdb_id_wikipedia_id = df_mapping_imdb_id_wikipedia_id[
        ["wikipedia_ID", "tconst"]
    ]

    return df_mapping_imdb_id_wikipedia_id


def load_imdb_average_reviews(df_mapping_imdb_id_wikipedia_id):
    """
    Load the reviews and merge with our movies.

    Arguments:
        - df_mapping_imdb_id_wikipedia_id: Dataframe containing the mapping between the wikipedia_ID and the IMDB ID (tconst)

    Returns a Dataframe containing the wikipedia_ID, tconst and the averageRating
    """
    # Load IMDB data
    imdb_ratings = load_raw_imdb_average_reviews()

    # Merge with average rating
    ratings = pd.merge(
        df_mapping_imdb_id_wikipedia_id, imdb_ratings, on="tconst", how="inner"
    )[["wikipedia_ID", "tconst", "averageRating"]]

    return ratings
