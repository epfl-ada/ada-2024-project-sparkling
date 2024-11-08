import os
import pandas as pd

DATASETS_DIR = "data"
MOVIE_SUMMARIES_DIR = "MovieSummaries"

def load_movie_metadata():
    """
    Load the movie metadata file

    Returns a dataframe containing the data
    """
    movie_metadata_path = os.path.join(DATASETS_DIR, MOVIE_SUMMARIES_DIR, "movie.metadata.tsv")
    df_movies_metadata = pd.read_csv(movie_metadata_path, sep='\t', header=None, names=["wikipedia_ID", "freebase_ID", "name", "release_date", "revenue", "runtime", "languages", "countries", "genres"])

    return df_movies_metadata

def load_movie_plot():
    """
    Load the movie plot file

    Returns a dataframe containing the data
    """
    movie_plot_path = os.path.join(DATASETS_DIR, MOVIE_SUMMARIES_DIR, "plot_summaries.txt")
    df_movies_plot = pd.read_csv(movie_plot_path, sep='\t', header=None, names=["wikipedia_ID", "plot"])

    return df_movies_plot

def load_character_metadata():
    """
    Load the character metadata file

    Returns a dataframe containing the data
    """
    path = os.path.join(DATASETS_DIR, MOVIE_SUMMARIES_DIR, "character.metadata.tsv")
    df = pd.read_csv(path, sep='\t', header=None, names=["wikipedia_ID", "freebase_ID", "movie_release_date", "name", "actor_birth_date", "actor_gender", "actor_height", "freebase_ID_actor_ethnicity", "actor_name", "actor_age", "freebase_ID_character_actor_map", "freebase_ID_character", "freebase_ID_actor"])

    return df