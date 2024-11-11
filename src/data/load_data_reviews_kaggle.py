import os
import pandas as pd
from src.data.load_data import DATASETS_DIR

IMDB_REVIEW_1 = "IMDBReviewsKaggle"


def load_movie_reviews_kaggle(df_movies, save_to_load_from="IMDBReviewsKaggle.csv"):
    """
    Load and return a dataframe containing the reviews relevant with our movie data set (df_movies).
    A movie is matched on its name and year of publication (To be sure that it is the right movie).

    The different parts of the dataset should be downloaded from: https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset/data (See README)

    Argument:
        - df_movies: Dataframe containing the movie name along with their publication date
        - save_to_load_from: File to save or load data from (None -> Load from parts and do not save) -> Default: IMDBReviewsKaggle.csv

    If the file 'IMDBReviewsKaggle/save_to_load_from' exists, then load it, otherwise load the data from the different parts,
    create the csv file (~ 2.5 GB) and return the dataframe
    """
    if save_to_load_from is not None:
        path_csv = os.path.join(DATASETS_DIR, IMDB_REVIEW_1, save_to_load_from)
        if os.path.isfile(path_csv):
            return pd.read_csv(path_csv)

    def loadfile(id):
        file_name = f"part-0{id}.json"
        path = os.path.join(DATASETS_DIR, IMDB_REVIEW_1, file_name)
        if not os.path.isfile(path):
            raise Exception(
                f"The file {file_name} is missing -> Please download the dataset from https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset/data (See README)"
            )

        print(f"Loading reviews: {file_name}")
        return pd.read_json(path)

    # If some movies have the same name and year, we cannot differentiate, then we remove them
    # On the original dataset, this correspond to 103 movies
    df_movies = df_movies.drop_duplicates(subset=["name", "release_year"], keep=False)

    df_datas = []
    for i in range(1, 7):

        df_data = loadfile(i)

        df_data = df_data.drop(
            [
                "review_id",
                "reviewer",
                "review_summary",
                "review_date",
                "spoiler_tag",
                "helpful",
            ],
            axis=1,
        )

        # We first want to remove the Season, Episode count from the end.
        # And remove Episode #<int>.<int> from movies names
        # Explaination for this regex:
        # So we have two format that we want to match
        # -> Season <something>, Episode <something> -> Season .+, Episode .+
        #   Before the format we have in the movie names: either a new line (\\R), a space or nothing
        #   After the format we have either a space or nothing
        #   And the $ at the end specify that our match to be at the end of the string
        # -> Episode #<int>.<int>
        #
        # Note that we seperated both format for readability

        # First format:
        pattern_season_episode = r"(\\R| |)Season .+, Episode .+( |)$"
        df_data["movie_name"] = df_data["movie"].str.replace(
            pattern_season_episode, "", regex=True
        )

        # Second format:
        pattern_episode = r" Episode #\d+.\d+ "
        df_data["movie_name"] = df_data["movie_name"].str.replace(
            pattern_episode, "", regex=True
        )

        # We want to extract the year
        # Here the format is:
        #   <year containing 4 integers> -> (\d{4}) -> In a group to be extracted
        #   Before the year we have either ':(' or ' ('
        #   After the year we sometimes has some '–' and then we check that we have a closing paranthesis and a
        #   $ to say that we want to last paranthese (We saw that some films had 2 parantheses with years inside)

        # Moreover we don't want pandas to extract every groups, so we have to remove unwanted groups with:
        # https://stackoverflow.com/questions/3378773/can-i-use-an-or-in-regex-without-capturing-whats-enclosed
        pattern_year = r"(?::| )\((\d{4})(?:[^\)]+|)\)$"
        df_data["movie_year"] = (
            df_data["movie_name"].str.extract(pattern_year).astype("Int64")
        )
        df_data["movie_name"] = df_data["movie_name"].str.replace(
            pattern_year, "", regex=True
        )

        df_data = df_data.drop(["movie"], axis=1)

        # Match the movies with the reviews on their name and publication date
        merged_df = df_movies[["wikipedia_ID", "name", "release_year"]].merge(
            df_data,
            how="inner",
            left_on=["name", "release_year"],
            right_on=["movie_name", "movie_year"],
        )

        merged_df = merged_df.drop(["movie_name", "movie_year"], axis=1)

        df_datas.append(merged_df)

    final_df = pd.concat(df_datas)
    if save_to_load_from is not None:
        final_df.to_csv(path_csv, index=False)
    return final_df
